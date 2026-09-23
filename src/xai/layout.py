"""Where every per-prediction output goes, and in what form.

Two properties this module exists to guarantee:

**The output tree mirrors the input tree.** An image at
``Dataset/Glasses/image/neg/00012.jpg`` produces
``artifacts/glasses/explanations/neg/00012.html`` and
``artifacts/glasses/saliency/neg/00012/``. A text record from row 42 of the test
CSV produces ``artifacts/tweet/explanations/0042_<id>.html``, so listing the
directory reproduces the order of the input file rather than a hash order.

**Nothing is stored in an archive.** Saliency maps are written as one ``.npy``
per method inside a directory named for the sample, not bundled into a ``.npz``
— which is a zip archive and cannot be inspected, diffed or partially read
without unpacking it. One array per file costs a little more disk and makes the
output browsable with ordinary tools.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Methods whose arrays are attributions; anything else (LIME's segmentation)
# keeps its own dtype because rounding a superpixel id would corrupt it.
_FLOAT_MAPS = ("gradcam", "shap", "lime")


def text_stem(index: int, sample_id: str, total: int) -> str:
    """``0042_5420d58af3`` — ordered by input row, still unique by id.

    The index prefix is what makes a directory listing match the order of the
    input file; the id keeps the name traceable back to the record.
    """
    width = max(len(str(total)), 4)
    return f"{index:0{width}d}_{sample_id}"


def image_parts(sample_id: str) -> tuple[str, str]:
    """``neg_00012`` -> ``("neg", "00012")``, mirroring the input tree."""
    group, _, stem = sample_id.partition("_")
    return (group, stem) if stem else ("", sample_id)


def image_stem(sample_id: str) -> Path:
    group, stem = image_parts(sample_id)
    return Path(group) / stem if group else Path(stem)


def saliency_dir(out_dir: Path, sample_id: str) -> Path:
    """Directory holding one prediction's maps, one ``.npy`` per method."""
    return out_dir / "saliency" / image_stem(sample_id)


def save_saliency(dir_path: Path, maps: dict) -> Path:
    """Saliency maps, narrowed to float16 — see `save_arrays`."""
    return save_arrays(dir_path, maps, float_dtype=np.float16)


def save_arrays(dir_path: Path, maps: dict, float_dtype=None) -> Path:
    """Write one ``.npy`` per array.

    `float_dtype` narrows floating-point arrays on the way out. Saliency maps
    pass ``float16``, which is ample for values only ever read back for
    visualisation and ranking; callers storing anything the maths depends on —
    embeddings feeding a cosine similarity, say — leave it alone. Integer arrays
    are never narrowed, since rounding a superpixel id would corrupt it.
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    for name, arr in maps.items():
        a = arr
        if float_dtype is not None and not np.issubdtype(arr.dtype, np.integer):
            a = arr.astype(float_dtype)
        np.save(dir_path / f"{name}.npy", a)
    return dir_path


def load_arrays(dir_path: Path, names: tuple[str, ...] | None = None) -> dict:
    """Read back whichever maps are present. Missing files are simply absent."""
    if not dir_path.is_dir():
        return {}
    out = {}
    for p in sorted(dir_path.glob("*.npy")):
        if names is None or p.stem in names:
            out[p.stem] = np.load(p)
    return out


def has_arrays(dir_path: Path) -> bool:
    return dir_path.is_dir() and any(dir_path.glob("*.npy"))
