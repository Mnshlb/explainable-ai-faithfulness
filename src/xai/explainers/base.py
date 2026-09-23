"""Common schema for explanations.

Every explainer in this package, across both modalities, returns the same
record type. That is what makes it possible to write one runner, one renderer
and one report generator instead of three of each, and it is what lets the
report state that *every* prediction carries *every* applicable explanation.

A token attribution is a ``(token, weight)`` pair where the weight is signed
with respect to the **predicted** class: positive means the token pushed the
model toward its prediction, negative means it pushed away.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator


@dataclass
class TokenAttribution:
    token: str
    weight: float

    def as_tuple(self) -> tuple[str, float]:
        return (self.token, round(float(self.weight), 6))


@dataclass
class Explanation:
    """One explanation of one prediction by one method."""

    method: str
    kind: str  # "tokens" | "neighbours" | "saliency"
    runtime_s: float = 0.0
    # kind == "tokens"
    tokens: list[tuple[str, float]] = field(default_factory=list)
    base_value: float | None = None
    # kind == "neighbours"
    neighbours: list[dict[str, Any]] = field(default_factory=list)
    # kind == "saliency" -- arrays live on disk, this holds pointers + summary
    arrays: dict[str, str] = field(default_factory=dict)
    render: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise, dropping only absent fields.

        Empty containers and ``None`` are omitted to keep records compact, but
        numeric zeros are kept: a base value or a Shapley sum of exactly 0.0 is
        a result, not a missing field.
        """
        d = asdict(self)
        return {
            k: v
            for k, v in d.items()
            if v is not None and not (isinstance(v, (list, dict)) and len(v) == 0)
        }


@dataclass
class Prediction:
    """A model output plus every explanation computed for it."""

    task: str
    sample_id: str
    split: str | None
    true_label: str
    true_label_id: int
    pred_label: str
    pred_label_id: int
    confidence: float
    probs: dict[str, float]
    correct: bool
    text: str | None = None
    image_path: str | None = None
    explanations: dict[str, Explanation] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "task": self.task,
            "sample_id": self.sample_id,
            "split": self.split,
            "true_label": self.true_label,
            "true_label_id": self.true_label_id,
            "pred_label": self.pred_label,
            "pred_label_id": self.pred_label_id,
            "confidence": round(self.confidence, 6),
            "probs": {k: round(v, 6) for k, v in self.probs.items()},
            "correct": self.correct,
            "explanations": {k: v.to_dict() for k, v in self.explanations.items()},
        }
        if self.text is not None:
            d["text"] = self.text
        if self.image_path is not None:
            d["image_path"] = self.image_path
        return d


@contextmanager
def timed(exp: Explanation) -> Iterator[Explanation]:
    """Record wall-clock runtime on an explanation, even if it fails.

    Runtime is part of the comparison between methods, so it is measured
    rather than estimated.
    """
    t0 = time.perf_counter()
    try:
        yield exp
    finally:
        exp.runtime_s = round(time.perf_counter() - t0, 4)


def top_k(tokens: list[tuple[str, float]], k: int) -> list[tuple[str, float]]:
    """The k attributions of largest magnitude, kept in signed form."""
    return sorted(tokens, key=lambda tw: -abs(tw[1]))[:k]


__all__ = ["Explanation", "Prediction", "TokenAttribution", "timed", "top_k"]
