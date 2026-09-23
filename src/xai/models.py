"""Unified model wrappers.

Each wrapper exposes the same two things an explainer ever needs:

    predict_proba(inputs) -> np.ndarray of shape (n, n_classes)
    labels                -> list[str]

plus, where relevant, ``embed`` for prototype retrieval and ``module`` for
gradient-based methods that need the raw ``nn.Module``.

The prediction functions are batched. LIME calls ``predict_proba`` with
hundreds to thousands of perturbations per sample, so a per-item loop here
would dominate the entire pipeline's runtime.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import TaskConfig

warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------


class TextClassifier:
    """A HuggingFace sequence classifier behind a batched numpy interface."""

    def __init__(self, task: TaskConfig, device: torch.device):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.task = task
        self.device = device
        self.labels: list[str] = task.labels
        self.max_length: int = task.max_length
        self.batch_size: int = task.batch_size

        if task.get("lora_dir"):
            # PEFT stores the fine-tuned classification head inside the adapter
            # checkpoint under a ``base_model.model.`` prefix. Loading the
            # adapter folder with AutoModelForSequenceClassification instead of
            # PeftModel silently drops those keys and leaves a randomly
            # initialised head -- the bug in the original notebooks.
            from peft import PeftModel

            base_name = task.base_model
            base = AutoModelForSequenceClassification.from_pretrained(
                base_name, num_labels=len(self.labels)
            )
            model = PeftModel.from_pretrained(base, str(task.path("lora_dir")))
            self.tokenizer = AutoTokenizer.from_pretrained(base_name)
            self._verify_lora_head(model, task.path("lora_dir"))
        else:
            src = str(task.path("model_dir"))
            model = AutoModelForSequenceClassification.from_pretrained(src)
            self.tokenizer = AutoTokenizer.from_pretrained(src)

        self.model = model.eval().to(device)
        self._n_calls = 0

    @staticmethod
    def _verify_lora_head(model, adapter_dir: Path) -> None:
        """Fail loudly if the fine-tuned classifier head did not survive loading."""
        from safetensors import safe_open

        ckpt = adapter_dir / "adapter_model.safetensors"
        if not ckpt.exists():
            return
        with safe_open(str(ckpt), "pt") as fh:
            key = "base_model.model.classifier.weight"
            if key not in fh.keys():
                return
            saved = fh.get_tensor(key)
        live = model.base_model.model.classifier.weight.detach().cpu()
        if not torch.allclose(saved, live, atol=1e-5):
            raise RuntimeError(
                "LoRA classification head did not load; explanations would "
                "describe a randomly initialised classifier."
            )

    @torch.no_grad()
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Softmax probabilities for a list of strings."""
        texts = [str(t) for t in texts]
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(
                texts[i : i + self.batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**batch).logits
            out.append(F.softmax(logits, dim=-1).float().cpu().numpy())
        self._n_calls += len(texts)
        return np.concatenate(out, axis=0)

    @torch.no_grad()
    def embed(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """[CLS] embeddings from the encoder, used for prototype retrieval.

        Taken from the encoder rather than the classification head so that
        neighbours reflect the model's learned representation of the whole
        sequence, not just its final two-dimensional decision score.
        """
        bs = batch_size or self.batch_size
        encoder = self._encoder()
        out = []
        for i in range(0, len(texts), bs):
            batch = self.tokenizer(
                [str(t) for t in texts[i : i + bs]],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            hidden = encoder(**batch).last_hidden_state
            out.append(hidden[:, 0, :].float().cpu().numpy())
        return np.concatenate(out, axis=0)

    def _encoder(self):
        """Return the underlying BERT encoder for either a plain or PEFT model."""
        m = self.model
        if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
            return m.base_model.model.bert  # PEFT-wrapped
        return m.bert

    @property
    def hf_pipeline(self):
        """A HuggingFace pipeline, which SHAP's text explainer consumes directly."""
        from transformers import pipeline

        if not hasattr(self, "_pipe"):
            self._pipe = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                top_k=None,
                function_to_apply="softmax",
                device=self.device,
                truncation=True,
                max_length=self.max_length,
            )
        return self._pipe


# --------------------------------------------------------------------------
# Image
# --------------------------------------------------------------------------


def build_resnet(ckpt: Path, n_labels: int) -> torch.nn.Module:
    """Load a ResNet-18 from either a pickled module or a state_dict checkpoint.

    The original project saved the whole module via ``torch.save(model)``;
    the retrained model saves a state_dict plus metadata, which is portable
    across torch and torchvision versions.
    """
    from torchvision.models import resnet18

    obj = torch.load(ckpt, map_location="cpu", weights_only=False)

    if isinstance(obj, dict):
        state = obj.get("state_dict", obj)
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, n_labels)
        model.load_state_dict(state)
        return model

    return obj  # already an nn.Module


class ImageClassifier:
    """A ResNet glasses classifier behind a batched numpy interface."""

    def __init__(self, task: TaskConfig, device: torch.device):
        self.task = task
        self.device = device
        self.labels: list[str] = task.labels
        self.size: int = task.image_size
        self.batch_size: int = task.batch_size

        ckpt = task.path("checkpoint")
        self.model = build_resnet(ckpt, len(self.labels)).eval().to(device)

        # Whether the network expects ImageNet-normalised input. This is applied
        # inside the forward pass rather than in ``transform`` so that every
        # explainer sees the same raw pixel space: LIME perturbs pixels, SHAP
        # attributes to pixels, and Grad-CAM overlays onto pixels.
        self.normalize: bool = bool(task.get("normalize", False))
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self._mean, self._std = mean.to(device), std.to(device)

    def transform(self):
        """Pixel-space transform: resize to the network's input size, no more."""
        from torchvision import transforms

        return transforms.Compose(
            [transforms.Resize((self.size, self.size)), transforms.ToTensor()]
        )

    def load_tensor(self, path: Path) -> torch.Tensor:
        """Load one image as a CHW float tensor in [0, 1]."""
        from PIL import Image

        with Image.open(path) as im:
            return self.transform()(im.convert("RGB"))

    def forward_pixels(self, x: torch.Tensor) -> torch.Tensor:
        """Logits from an NCHW tensor of pixels in [0, 1], differentiably.

        Gradient-based explainers (Grad-CAM, GradientSHAP) attribute through
        this function, so the normalisation step is part of the explained
        computation rather than hidden preprocessing.
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if self.normalize:
            x = (x - self._mean) / self._std
        return self.model(x)

    @torch.no_grad()
    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        """Probabilities for a batch of HWC float images in [0, 1].

        This is the signature LIME's image explainer expects.
        """
        x = torch.as_tensor(np.asarray(images)).float()
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        out = []
        for i in range(0, x.shape[0], self.batch_size):
            logits = self.forward_pixels(x[i : i + self.batch_size].to(self.device))
            out.append(F.softmax(logits, dim=-1).float().cpu().numpy())
        return np.concatenate(out, axis=0)

    @torch.no_grad()
    def predict_proba_tensor(self, x: torch.Tensor) -> np.ndarray:
        """Probabilities for an already-batched NCHW pixel tensor."""
        logits = self.forward_pixels(x.to(self.device))
        return F.softmax(logits, dim=-1).float().cpu().numpy()

    @property
    def target_layer(self):
        """The convolutional block Grad-CAM hooks."""
        return getattr(self.model, self.task.gradcam_layer)


def load_model(task: TaskConfig, device: torch.device):
    """Build the right wrapper for a task."""
    if task.kind == "text":
        return TextClassifier(task, device)
    if task.kind == "image":
        return ImageClassifier(task, device)
    raise ValueError(f"unknown task kind: {task.kind}")


__all__ = ["TextClassifier", "ImageClassifier", "load_model"]
