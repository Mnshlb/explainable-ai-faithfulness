"""Faithfulness — does removing what an explanation called important change the prediction?

Plausibility (evaluate.py) asks whether an explanation points where a human points,
and needs human annotation, which only two of the three tasks have. Faithfulness asks
something the model itself can answer, so it covers every task:

  comprehensiveness  remove the top-k features. How far does the predicted-class
                     probability fall? Higher is better — the explanation named
                     features the model actually relied on.

  sufficiency        keep ONLY the top-k features. How far does it fall? Lower is
                     better — what the explanation named was enough on its own.

Both are reported against a random-ranking baseline computed over the same inputs at
the same budgets. Without it the numbers are uninterpretable: deleting any 20% of an
input moves a classifier somewhat, and only the margin over random is evidence that
the *ranking* carried information.

Prototype retrieval produces no per-feature attribution and is not scored.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from . import config as cfg_mod
from . import layout
from . import models as models_mod

FRACTIONS = (0.01, 0.05, 0.10, 0.20, 0.50)
_WORD = re.compile(r"\S+")


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------

def _rank_positions_text(text: str, entry: dict, method: str, rng) -> list[int]:
    """Rank whitespace-token positions of `text` by the method's attribution.

    SHAP stores its tokens in document order, so positions map directly.
    LIME stores them in importance order with the position information gone, so
    each ranked token is matched to its next unconsumed occurrence in the text.
    """
    words = _WORD.findall(text)
    toks = entry.get("tokens") or []
    if method == "random":
        idx = list(range(len(words)))
        rng.shuffle(idx)
        return idx

    if method == "shap":
        # Document order: fold wordpieces onto whitespace tokens by walking both.
        order = sorted(range(len(toks)), key=lambda i: -abs(toks[i][1]))
        # Map each shap token index to a word index by cumulative text length.
        spans, cur = [], 0
        for t, _w in toks:
            spans.append(cur)
            cur += len(t)
        starts = [m.start() for m in _WORD.finditer(text)]
        out, seen = [], set()
        for i in order:
            pos = spans[i]
            # nearest word whose start is <= this token's offset
            wi = max([j for j, s in enumerate(starts) if s <= pos], default=0)
            if wi not in seen:
                seen.add(wi)
                out.append(wi)
        return out

    # LIME: match each ranked token string to an unconsumed word position.
    lowered = [w.lower().strip(".,!?;:'\"()") for w in words]
    used, out = set(), []
    for t, _w in sorted(toks, key=lambda x: -abs(x[1])):
        key = str(t).lower().strip(".,!?;:'\"()")
        for j, w in enumerate(lowered):
            if j not in used and w == key:
                used.add(j)
                out.append(j)
                break
    for j in range(len(words)):
        if j not in used:
            out.append(j)
    return out


def _variants_text(text: str, order: list[int], frac: float) -> tuple[str, str]:
    """(comprehensiveness input, sufficiency input) at this budget."""
    words = _WORD.findall(text)
    if not words:
        return text, text
    k = max(1, int(round(frac * len(words))))
    top = set(order[:k])
    removed = " ".join(w for i, w in enumerate(words) if i not in top)
    kept = " ".join(w for i, w in enumerate(words) if i in top)
    return removed, kept


def text_faithfulness(task_key: str, cfg, artifacts: Path, limit: int | None) -> dict:
    task = cfg.task(task_key)
    device = cfg_mod.get_device(cfg.raw["device"])
    model = models_mod.load_model(task, device)

    records = []
    with open(artifacts / task_key / "predictions.jsonl") as fh:
        for line in fh:
            records.append(json.loads(line))
    if limit:
        records = records[:limit]

    methods = ["lime", "shap", "random"]
    rng = np.random.default_rng(42)
    acc = {m: {"comp": [], "suff": []} for m in methods}

    BATCH = 128
    pending: list[tuple] = []

    def flush():
        if not pending:
            return
        probs = model.predict_proba([p[0] for p in pending])
        for (txt, meth, kind, pid, base), pr in zip(pending, probs):
            acc[meth][kind].append(base - float(pr[pid]))
        pending.clear()

    for n, r in enumerate(records, 1):
        text = r.get("text") or ""
        if not text.strip():
            continue
        pid = r["pred_label_id"]
        base = float(r["probs"][r["pred_label"]])
        for meth in methods:
            entry = r["explanations"].get(meth if meth != "random" else "lime")
            if meth != "random" and (not entry or entry.get("error") or not entry.get("tokens")):
                continue
            order = _rank_positions_text(text, entry or {}, meth, rng)
            for frac in FRACTIONS:
                removed, kept = _variants_text(text, order, frac)
                pending.append((removed, meth, "comp", pid, base))
                pending.append((kept, meth, "suff", pid, base))
                if len(pending) >= BATCH:
                    flush()
        if n % 200 == 0:
            flush()
            print(f"  [{task_key}] {n}/{len(records)}", flush=True)
    flush()

    return {
        "n_records": len(records),
        "fractions": list(FRACTIONS),
        "methods": {
            m: {
                "comprehensiveness": round(float(np.mean(d["comp"])), 4),
                "sufficiency": round(float(np.mean(d["suff"])), 4),
                "n": len(d["comp"]),
            }
            for m, d in acc.items() if d["comp"]
        },
    }


# --------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------

def image_faithfulness(cfg, artifacts: Path, limit: int | None) -> dict:
    task = cfg.task("glasses")
    device = cfg_mod.get_device(cfg.raw["device"])
    model = models_mod.load_model(task, device)
    root = cfg_mod.PROJECT_ROOT

    records = []
    with open(artifacts / "glasses" / "predictions.jsonl") as fh:
        for line in fh:
            records.append(json.loads(line))
    if limit:
        records = records[:limit]

    methods = ["gradcam", "shap", "lime", "random"]
    rng = np.random.default_rng(42)
    acc = {m: {"comp": [], "suff": []} for m in methods}
    sal_dir = artifacts / "glasses" / "saliency"

    BATCH = 64
    pending: list[tuple] = []

    def flush():
        if not pending:
            return
        arr = np.stack([p[0] for p in pending])
        probs = model.predict_proba(arr)
        for (_img, meth, kind, pid, base), pr in zip(pending, probs):
            acc[meth][kind].append(base - float(pr[pid]))
        pending.clear()

    for n, r in enumerate(records, 1):
        sdir = layout.saliency_dir(artifacts / "glasses", r["sample_id"])
        if not layout.has_arrays(sdir):
            continue
        x = model.load_tensor(root / r["image_path"])       # NCHW, pixel space
        img = x.squeeze(0).permute(1, 2, 0).cpu().numpy()   # HWC in [0,1]
        fill = img.reshape(-1, img.shape[-1]).mean(0)       # per-image mean colour
        pid = r["pred_label_id"]
        base = float(r["probs"][r["pred_label"]])

        z = layout.load_arrays(sdir)
        for meth in methods:
            if meth == "random":
                score = rng.random(img.shape[:2])
            elif meth in z:
                score = np.abs(z[meth].astype(np.float32))
            else:
                continue
            flat = score.ravel()
            order = np.argsort(-flat)
            npix = flat.size
            for frac in FRACTIONS:
                k = max(1, int(round(frac * npix)))
                top = order[:k]
                m = np.zeros(npix, dtype=bool)
                m[top] = True
                m2 = m.reshape(img.shape[:2])
                removed = np.where(m2[..., None], fill, img).astype(np.float32)
                kept = np.where(m2[..., None], img, fill).astype(np.float32)
                pending.append((removed, meth, "comp", pid, base))
                pending.append((kept, meth, "suff", pid, base))
                if len(pending) >= BATCH:
                    flush()
        if n % 200 == 0:
            flush()
            print(f"  [glasses] {n}/{len(records)}", flush=True)
    flush()

    return {
        "n_records": len(records),
        "fractions": list(FRACTIONS),
        "methods": {
            m: {
                "comprehensiveness": round(float(np.mean(d["comp"])), 4),
                "sufficiency": round(float(np.mean(d["suff"])), 4),
                "n": len(d["comp"]),
            }
            for m, d in acc.items() if d["comp"]
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="all")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = cfg_mod.load()
    artifacts = cfg_mod.PROJECT_ROOT / "artifacts"
    dest = artifacts / "faithfulness.json"
    out = json.loads(dest.read_text()) if dest.exists() else {}

    for key in ("tweet", "movie", "glasses"):
        if args.task not in ("all", key):
            continue
        print(f"=== {key} ===", flush=True)
        fn = image_faithfulness if key == "glasses" else None
        out[key] = (image_faithfulness(cfg, artifacts, args.limit) if fn
                    else text_faithfulness(key, cfg, artifacts, args.limit))
        print(json.dumps(out[key], indent=2), flush=True)
        dest.write_text(json.dumps(out, indent=2))

    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
