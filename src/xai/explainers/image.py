"""Image explainers: Grad-CAM, Gradient SHAP and LIME.

Each returns a 224x224 saliency map aligned to the input image, together with
summary statistics. The raw maps are handed back as arrays; persisting and
rendering them is the caller's job (see ``xai.render``).

All three attribute through ``ImageClassifier.forward_pixels``, so every map
is an attribution over **pixels**, not over normalised network inputs. That
makes the three directly comparable and makes the overlays honest.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import torch
import torch.nn as nn

from ..config import TaskConfig
from ..models import ImageClassifier
from .base import Explanation, timed


class _PixelWrapper(nn.Module):
    """Expose ``forward_pixels`` as an nn.Module for Captum."""

    def __init__(self, clf: ImageClassifier):
        super().__init__()
        self.clf = clf
        self.model = clf.model  # so Captum sees the parameters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.clf.forward_pixels(x)


def _saliency_stats(sal: np.ndarray) -> dict:
    """Summarise a saliency map so the report can compare maps numerically."""
    a = np.abs(sal)
    total = a.sum() + 1e-12
    flat = np.sort(a.ravel())[::-1]
    n = flat.size
    peak = np.unravel_index(np.argmax(a), a.shape)
    return {
        # Fraction of total attribution mass held by the strongest 5% of
        # pixels: high means a focused explanation, low means a diffuse one.
        "top5pct_mass": round(float(flat[: max(1, n // 20)].sum() / total), 4),
        "peak_yx": [int(peak[0]), int(peak[1])],
        "mean_abs": round(float(a.mean()), 8),
        "max_abs": round(float(a.max()), 8),
    }


# --------------------------------------------------------------------------
# Grad-CAM
# --------------------------------------------------------------------------


class GradCAM:
    """Class-discriminative localisation from the last conv block.

    Gradients are captured with a hook on the activation tensor itself rather
    than with ``register_backward_hook``, which fires unreliably on modules
    with multiple inputs and was the source of the original implementation's
    silently-wrong maps.
    """

    name = "gradcam"

    def __init__(self, model: ImageClassifier, task: TaskConfig):
        self.model = model
        self.task = task
        self.size = task.image_size
        self.layer = model.target_layer

    def explain(self, x: torch.Tensor, pred_id: int) -> tuple[Explanation, dict]:
        exp = Explanation(method=self.name, kind="saliency")
        cam = np.zeros((self.size, self.size), dtype=np.float32)
        with timed(exp):
            try:
                acts: dict[str, torch.Tensor] = {}
                grads: dict[str, torch.Tensor] = {}

                def fwd_hook(_m, _i, out):
                    acts["v"] = out
                    out.register_hook(lambda g: grads.__setitem__("v", g))

                handle = self.layer.register_forward_hook(fwd_hook)
                try:
                    self.model.model.zero_grad(set_to_none=True)
                    inp = x.unsqueeze(0).to(self.model.device).requires_grad_(True)
                    logits = self.model.forward_pixels(inp)
                    logits[0, pred_id].backward()
                finally:
                    handle.remove()

                a = acts["v"].detach()[0]  # (C, h, w)
                g = grads["v"].detach()[0]
                weights = g.mean(dim=(1, 2))  # global-average-pooled gradients
                raw = torch.relu((weights[:, None, None] * a).sum(0))

                cam_t = torch.nn.functional.interpolate(
                    raw[None, None], size=(self.size, self.size),
                    mode="bilinear", align_corners=False,
                )[0, 0]
                cam = cam_t.float().cpu().numpy()
                rng = cam.max() - cam.min()
                cam = (cam - cam.min()) / rng if rng > 1e-12 else np.zeros_like(cam)

                exp.stats = {
                    "layer": self.task.gradcam_layer,
                    "feature_map": list(a.shape),
                    **_saliency_stats(cam),
                }
            except Exception as err:  # noqa: BLE001
                exp.error = f"{type(err).__name__}: {err}"
        return exp, {self.name: cam}


# --------------------------------------------------------------------------
# Gradient SHAP
# --------------------------------------------------------------------------


class ShapImage:
    """Expected gradients between the input and a blurred baseline.

    A Gaussian-blurred copy of the image is used as the reference rather than
    a black image: it removes high-frequency detail such as spectacle frames
    while preserving pose and illumination, so the attribution isolates the
    evidence for the class instead of re-describing the whole face.
    """

    name = "shap"

    def __init__(self, model: ImageClassifier, task: TaskConfig):
        from captum.attr import GradientShap

        self.model = model
        self.task = task
        self.wrapped = _PixelWrapper(model)
        self.gs = GradientShap(self.wrapped)
        self.n_samples = task.shap_n_samples
        self.stdevs = task.shap_stdevs

    def explain(self, x: torch.Tensor, pred_id: int) -> tuple[Explanation, dict]:
        exp = Explanation(method=self.name, kind="saliency")
        sal = np.zeros((self.task.image_size, self.task.image_size), dtype=np.float32)
        with timed(exp):
            try:
                import torchvision.transforms.functional as TF

                inp = x.unsqueeze(0).to(self.model.device)
                blurred = TF.gaussian_blur(x.cpu(), kernel_size=11)
                # Include the blurred image and a black frame so the baseline
                # distribution has some spread, as GradientShap expects.
                baselines = torch.stack([blurred, torch.zeros_like(blurred)]).to(
                    self.model.device
                )
                attr, delta = self.gs.attribute(
                    inp,
                    baselines=baselines,
                    target=int(pred_id),
                    n_samples=self.n_samples,
                    stdevs=self.stdevs,
                    return_convergence_delta=True,
                )
                # Sum over colour channels: the question is which *pixels*
                # mattered, not which colour channel carried the signal.
                sal = attr[0].sum(0).detach().float().cpu().numpy()
                exp.stats = {
                    "n_samples": self.n_samples,
                    "stdevs": self.stdevs,
                    # How far the attributions are from summing to the
                    # difference in model output: the method's own error bar.
                    "convergence_delta": round(float(delta.abs().mean()), 6),
                    **_saliency_stats(sal),
                }
            except Exception as err:  # noqa: BLE001
                exp.error = f"{type(err).__name__}: {err}"
        return exp, {self.name: sal}


# --------------------------------------------------------------------------
# LIME
# --------------------------------------------------------------------------


class LimeImage:
    """Superpixel occlusion with a local linear surrogate."""

    name = "lime"

    def __init__(self, model: ImageClassifier, task: TaskConfig):
        from lime import lime_image
        from skimage.segmentation import slic

        self.model = model
        self.task = task
        self.num_samples = task.lime_num_samples
        self.num_features = task.lime_num_features
        self.explainer = lime_image.LimeImageExplainer(verbose=False, random_state=42)
        # SLIC with a fixed segment count keeps superpixel granularity
        # comparable across images; quickshift's defaults vary wildly with
        # face size and made the original explanations hard to compare.
        self._segment = lambda img: slic(
            img, n_segments=80, compactness=10.0, sigma=1.0, start_label=0
        )

    def explain(self, x: torch.Tensor, pred_id: int) -> tuple[Explanation, dict]:
        exp = Explanation(method=self.name, kind="saliency")
        size = self.task.image_size
        sal = np.zeros((size, size), dtype=np.float32)
        segs = np.zeros((size, size), dtype=np.int16)
        with timed(exp):
            try:
                img = x.permute(1, 2, 0).cpu().numpy().astype(np.float64)
                # This lime build has no switch for its progress bar and writes
                # it to stderr on every call; redirect rather than drown the log.
                with contextlib.redirect_stderr(io.StringIO()):
                    res = self.explainer.explain_instance(
                        img,
                        self.model.predict_proba,
                        labels=(pred_id,),
                        top_labels=None,
                        hide_color=0,
                        num_samples=self.num_samples,
                        segmentation_fn=self._segment,
                        batch_size=self.model.batch_size,
                    )
                segments = res.segments
                # Paint each superpixel with its surrogate weight, producing a
                # dense map comparable with the gradient-based ones.
                segs = segments.astype(np.int16)
                weights = dict(res.local_exp[pred_id])
                for seg_id, w in weights.items():
                    sal[segments == seg_id] = w
                # The full per-superpixel weight vector, not just the top few.
                exp.tokens = [(str(int(k)), float(v)) for k, v in sorted(weights.items())]

                pos = sorted(
                    ((w, s) for s, w in weights.items() if w > 0), reverse=True
                )[: self.num_features]
                # lime_image exposes ``score`` as a bare float when a single
                # label was explained and as a per-label mapping otherwise.
                score = res.score
                if not isinstance(score, (int, float)):
                    score = score[pred_id]
                exp.stats = {
                    "num_samples": self.num_samples,
                    "n_superpixels": int(segments.max() + 1),
                    "surrogate_r2": round(float(score), 4),
                    "top_superpixels": [int(s) for _, s in pos],
                    "n_weights": len(weights),
                    **_saliency_stats(sal),
                }
            except Exception as err:  # noqa: BLE001
                exp.error = f"{type(err).__name__}: {err}"
        # The segmentation is stored alongside the map so the superpixel
        # weights above can be mapped back onto the image exactly.
        return exp, {self.name: sal, "lime_segments": segs}


__all__ = ["GradCAM", "ShapImage", "LimeImage"]
