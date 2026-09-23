"""Dataset loading.

Every task is normalised onto the same record shape so that downstream
explanation code never branches on which dataset it is looking at:

    sample_id : str   stable identifier, used for resumable runs
    text/path : str   the model input
    label_id  : int   ground-truth class index
    label     : str   ground-truth class name
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT, TaskConfig


@dataclass
class TextSample:
    sample_id: str
    text: str
    label_id: int
    label: str


@dataclass
class ImageSample:
    sample_id: str
    path: Path
    label_id: int
    label: str


# --------------------------------------------------------------------------
# Text tasks
# --------------------------------------------------------------------------


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing dataset file: {path}")
    return pd.read_csv(path)


def load_text_split(task: TaskConfig, split: str) -> list[TextSample]:
    """Load one split of a text task as normalised samples.

    ``split`` is one of train/val/test; a task that has no such split raises.
    """
    key = f"{split}_csv"
    if key not in task.raw:
        raise KeyError(f"task '{task.key}' has no '{split}' split")

    df = _read_csv(task.path(key))
    text_col, label_col = task.text_col, task.label_col
    labels: list[str] = task.labels

    df = df.dropna(subset=[text_col, label_col]).reset_index(drop=True)

    # Labels arrive either as integer indices (movie) or as strings (tweet).
    if pd.api.types.is_numeric_dtype(df[label_col]):
        label_ids = df[label_col].astype(int)
    else:
        lookup = {name: i for i, name in enumerate(labels)}
        normalised = df[label_col].astype(str).str.lower().str.strip()
        unknown = set(normalised) - set(lookup)
        if unknown:
            raise ValueError(f"unexpected labels in {task.key}/{split}: {unknown}")
        label_ids = normalised.map(lookup)

    # A stable id lets a long explanation run resume exactly where it stopped.
    if "textID" in df.columns:
        ids = df["textID"].astype(str)
    else:
        ids = pd.Series([f"{task.key}_{split}_{i:05d}" for i in range(len(df))])

    char_limit = task.get("char_limit")
    texts = df[text_col].astype(str).str.strip()
    if char_limit:
        texts = texts.str.slice(0, char_limit)

    return [
        TextSample(sample_id=str(i), text=t, label_id=int(l), label=labels[int(l)])
        for i, t, l in zip(ids, texts, label_ids)
    ]


# --------------------------------------------------------------------------
# Image task
# --------------------------------------------------------------------------

# ImageFolder assigns class indices alphabetically: neg -> 0, pos -> 1.
_GLASSES_DIRS = {"neg": 0, "pos": 1}


def load_image_samples(task: TaskConfig) -> list[ImageSample]:
    """Load every glasses image, ordered deterministically by class then name."""
    root = task.path("image_dir")
    labels: list[str] = task.labels
    samples: list[ImageSample] = []

    for dirname, label_id in sorted(_GLASSES_DIRS.items(), key=lambda kv: kv[1]):
        folder = root / dirname
        if not folder.is_dir():
            raise FileNotFoundError(f"missing image folder: {folder}")
        for img in sorted(folder.glob("*.jpg")):
            samples.append(
                ImageSample(
                    sample_id=f"{dirname}_{img.stem}",
                    path=img,
                    label_id=label_id,
                    label=labels[label_id],
                )
            )
    if not samples:
        raise RuntimeError(f"no images found under {root}")
    return samples


# --------------------------------------------------------------------------
# Convenience
# --------------------------------------------------------------------------


def describe(task: TaskConfig) -> dict:
    """Summary statistics used by the report generator."""
    if task.kind == "image":
        samples = load_image_samples(task)
        counts: dict[str, int] = {}
        for s in samples:
            counts[s.label] = counts.get(s.label, 0) + 1
        return {"n_total": len(samples), "class_counts": counts}

    out: dict = {}
    for split in ("train", "val", "test"):
        if f"{split}_csv" not in task.raw:
            continue
        samples = load_text_split(task, split)
        counts: dict[str, int] = {}
        for s in samples:
            counts[s.label] = counts.get(s.label, 0) + 1
        lengths = [len(s.text) for s in samples]
        out[split] = {
            "n": len(samples),
            "class_counts": counts,
            "chars_mean": round(sum(lengths) / len(lengths), 1),
            "chars_max": max(lengths),
        }
    return out


__all__ = [
    "TextSample",
    "ImageSample",
    "load_text_split",
    "load_image_samples",
    "describe",
    "PROJECT_ROOT",
]
