"""Text explainers: LIME, SHAP and prototype retrieval.

All three answer the same question -- "why this label for this text?" -- from
three different angles:

    LIME       perturbs words and fits a local linear surrogate
    SHAP       assigns each token its Shapley value under a partition masker
    prototype  retrieves the nearest training examples in embedding space

The first two are token-level and signed toward the predicted class; the
third is example-level and has no token signal at all. Reporting them
together is the point: where they agree, the evidence is strong; where they
disagree, the prediction deserves scrutiny.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import layout
from ..config import TaskConfig
from ..models import TextClassifier
from .base import Explanation, timed, top_k


# --------------------------------------------------------------------------
# LIME
# --------------------------------------------------------------------------


class LimeText:
    """Local surrogate explanation over word-presence features."""

    name = "lime"

    # Passed as num_features so LIME keeps every word rather than a top slice.
    # Its 'highest_weights' selector tolerates a bound larger than the feature
    # count and simply returns all of them.
    ALL_FEATURES = 10**6

    def __init__(self, model: TextClassifier, task: TaskConfig, top_k_tokens: int = 10):
        from lime.lime_text import LimeTextExplainer

        self.model = model
        self.task = task
        self.top_k_tokens = top_k_tokens  # display only; every token is stored
        self.num_samples = task.lime_num_samples
        # bow=False keeps repeated words as distinct positions, which matters
        # for negation ("not good" vs "good ... not").
        self.explainer = LimeTextExplainer(
            class_names=model.labels, bow=False, random_state=42
        )

    def explain(self, text: str, pred_id: int) -> Explanation:
        exp = Explanation(method=self.name, kind="tokens")
        with timed(exp):
            try:
                res = self.explainer.explain_instance(
                    text,
                    self.model.predict_proba,
                    labels=(pred_id,),
                    num_features=self.ALL_FEATURES,
                    num_samples=self.num_samples,
                )
                # Weights are already signed toward `pred_id`. The complete
                # vector is stored; renderers take their own top slice.
                exp.tokens = [(str(t), float(w)) for t, w in res.as_list(label=pred_id)]
                exp.stats = {
                    "num_samples": self.num_samples,
                    "n_tokens": len(exp.tokens),
                    # R^2 of the local surrogate: how well the linear model
                    # actually reproduces the classifier near this input. A low
                    # value means the explanation itself is unreliable.
                    "surrogate_r2": round(float(res.score), 4),
                    "local_pred": round(float(res.local_pred[0]), 6),
                }
            except Exception as err:  # noqa: BLE001 - recorded, run continues
                exp.error = f"{type(err).__name__}: {err}"
        return exp


# --------------------------------------------------------------------------
# SHAP
# --------------------------------------------------------------------------


class ShapText:
    """Partition SHAP over the tokenizer's own segmentation."""

    name = "shap"

    def __init__(self, model: TextClassifier, task: TaskConfig, top_k_tokens: int = 10):
        import shap

        self.model = model
        self.task = task
        self.top_k_tokens = top_k_tokens
        self.max_evals = task.shap_max_evals

        # Handing SHAP a plain batched function rather than a HuggingFace
        # pipeline object. A pipeline evaluates its inputs one at a time
        # unless explicitly batched, which made SHAP ~6x slower than LIME on
        # identical inputs; this routes every masked variant through the same
        # batched predict_proba the other explainers use.
        def f(texts):
            return self.model.predict_proba([str(t) for t in texts])

        # Masking with the tokenizer keeps masked inputs inside the model's
        # own vocabulary rather than producing arbitrary word deletions.
        self.explainer = shap.Explainer(
            f,
            masker=shap.maskers.Text(model.tokenizer),
            output_names=model.labels,
            silent=True,
        )

    def explain(self, text: str, pred_id: int) -> Explanation:
        exp = Explanation(method=self.name, kind="tokens")
        with timed(exp):
            try:
                values = self.explainer(
                    [text], max_evals=self.max_evals, batch_size=self.task.batch_size
                )
                v = values[0]
                # v.values has shape (n_tokens, n_classes); take the predicted class.
                arr = np.asarray(v.values)
                col = arr[:, pred_id] if arr.ndim == 2 else arr
                toks = [str(t) for t in v.data]
                pairs = [
                    (t, float(w))
                    for t, w in zip(toks, col)
                    if t.strip()  # drop the whitespace tokens SHAP emits
                ]
                # Store every token's Shapley value. Truncating here would
                # break the additivity the method guarantees and would discard
                # most of the explanation on longer inputs.
                exp.tokens = pairs
                base = np.asarray(v.base_values)
                exp.base_value = float(base[pred_id] if base.ndim else base)
                exp.stats = {
                    "max_evals": self.max_evals,
                    "n_tokens": len(pairs),
                    # Shapley values are additive: base + sum(values) should
                    # equal the model output. Recording the total lets the
                    # report verify that rather than assert it.
                    "sum_values": round(float(col.sum()), 6),
                }
            except Exception as err:  # noqa: BLE001
                exp.error = f"{type(err).__name__}: {err}"
        return exp


# --------------------------------------------------------------------------
# Prototypes
# --------------------------------------------------------------------------


class PrototypeText:
    """Nearest training examples in [CLS] embedding space.

    The training embeddings are computed once and reused for every test
    sample, so retrieval costs one matrix product per explanation.
    """

    name = "prototype"

    def __init__(
        self,
        model: TextClassifier,
        task: TaskConfig,
        train_texts: list[str],
        train_labels: list[int],
        k: int = 3,
        cache_path=None,
    ):
        self.model = model
        self.task = task
        self.k = k
        self.train_texts = train_texts
        self.train_labels = np.asarray(train_labels)

        # The cache is a directory of .npy, not an .npz — same reason as the
        # saliency maps: an archive cannot be inspected without unpacking it.
        emb = None
        if cache_path is not None:
            cached = layout.load_arrays(Path(cache_path), names=("emb",))
            emb = cached.get("emb")
            if emb is not None and emb.shape[0] != len(train_texts):
                emb = None  # stale cache, recompute
        if emb is None:
            emb = model.embed(train_texts)
            if cache_path is not None:
                layout.save_arrays(Path(cache_path), {"emb": emb.astype(np.float32)})

        # L2-normalise once so cosine similarity is a plain dot product.
        self.train_emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

    def explain_batch(self, texts: list[str], pred_ids: list[int]) -> list[Explanation]:
        """Explain many samples at once; embedding is the dominant cost."""
        out: list[Explanation] = []
        q = self.model.embed(texts)
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        sims = q @ self.train_emb.T  # (n_query, n_train)

        for row, pred_id in zip(sims, pred_ids):
            exp = Explanation(method=self.name, kind="neighbours")
            with timed(exp):
                idx = np.argpartition(-row, self.k)[: self.k]
                idx = idx[np.argsort(-row[idx])]
                exp.neighbours = [
                    {
                        "rank": r + 1,
                        "similarity": round(float(row[i]), 6),
                        "label": self.model.labels[int(self.train_labels[i])],
                        "label_id": int(self.train_labels[i]),
                        "agrees_with_prediction": int(self.train_labels[i]) == pred_id,
                        "text": self.train_texts[i][:400],
                    }
                    for r, i in enumerate(idx)
                ]
                agree = sum(n["agrees_with_prediction"] for n in exp.neighbours)
                exp.stats = {
                    "k": self.k,
                    "n_agreeing": agree,
                    # Fraction of retrieved neighbours carrying the predicted
                    # label. Low values flag predictions the training data
                    # does not obviously support.
                    "neighbour_agreement": round(agree / max(len(exp.neighbours), 1), 4),
                    "mean_similarity": round(
                        float(np.mean([n["similarity"] for n in exp.neighbours])), 6
                    ),
                }
            out.append(exp)
        return out


__all__ = ["LimeText", "ShapText", "PrototypeText"]
