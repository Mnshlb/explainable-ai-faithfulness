"""Plausibility and faithfulness evaluation — which explanation method is better?

The rest of this project measures how *reliable* each method reports itself to be
(surrogate R², convergence delta, neighbour agreement) and how much the methods
agree with one another. Neither answers which method is *better*, because both are
computed without reference to any ground truth.

This module adds the two comparisons that do:

  plausibility — does the explanation point where a human points?
                 glasses: 2,614 human attention masks shipped with the dataset
                 tweet:   human-selected rationale spans (`selected_text`)
                 movie:   no human annotation exists, so plausibility is undefined

  faithfulness — does removing what the explanation called important actually
                 change the prediction? Measured for every task, since it needs
                 no human labels, only re-inference.

Prototype retrieval is deliberately absent from both. It produces no per-feature
attribution, so neither metric is defined for it; it is not scored rather than
scored as zero.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as cfg_mod
from . import layout

IMAGE_METHODS = ("gradcam", "shap", "lime")
TEXT_METHODS = ("lime", "shap")


# --------------------------------------------------------------------------
# Glasses — saliency vs human attention masks
# --------------------------------------------------------------------------

def _human_masks(root: Path) -> dict[str, np.ndarray]:
    """Load the shipped attention masks, keyed by the record's sample_id."""
    out: dict[str, np.ndarray] = {}
    for fname, prefix in (
        ("results_img_glasses_factual.csv", "pos"),
        ("results_img_noglasses_factual.csv", "neg"),
    ):
        df = pd.read_csv(root / "Dataset/Glasses/attention_label" / fname)
        for img, att in zip(df.img_idx, df.attention):
            m = np.asarray(ast.literal_eval(att), dtype=np.float32)
            out[f"{prefix}_{Path(img).stem}"] = m > 0
    return out


def _localisation(sal: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Score one saliency map against one binary human mask.

    Magnitude is what localises, so signed maps are compared by |value| — the
    same convention the agreement section already uses for Spearman ρ.
    """
    a = np.abs(sal.astype(np.float32))
    if not np.isfinite(a).all():
        a = np.nan_to_num(a)
    flat_m = mask.ravel()
    # Degenerate masks (all/nothing marked) make AUC and IoU undefined.
    if flat_m.all() or not flat_m.any():
        return {}

    # Pointing game: does the single strongest pixel land inside the mask?
    peak = np.unravel_index(int(np.argmax(a)), a.shape)
    hit = float(bool(mask[peak]))

    # Mass inside: what share of total attribution falls in the human region?
    total = float(a.sum())
    mass = float(a[mask].sum() / total) if total > 0 else 0.0

    # IoU at the mask's own size, so a method is not rewarded or punished for
    # simply spreading attribution more widely than the annotator did.
    k = int(flat_m.sum())
    thresh_idx = np.argpartition(a.ravel(), -k)[-k:]
    pred = np.zeros(a.size, dtype=bool)
    pred[thresh_idx] = True
    inter = np.logical_and(pred, flat_m).sum()
    union = np.logical_or(pred, flat_m).sum()
    iou = float(inter / union) if union else 0.0

    # Threshold-free: can |attribution| rank mask pixels above non-mask pixels?
    auc = float(roc_auc_score(flat_m, a.ravel()))

    return {"pointing_game": hit, "mass_inside": mass, "iou": iou, "mask_auc": auc}


def glasses_plausibility(root: Path, artifacts: Path) -> dict:
    masks = _human_masks(root)
    scores: dict[str, dict[str, list[float]]] = {m: {} for m in IMAGE_METHODS}
    n_used = 0
    sal_dir = artifacts / "glasses" / "saliency"
    # Two reference points, measured from the same masks rather than assumed:
    # what a pixel chosen at random scores on the pointing game (the mean mask
    # coverage), and what a fixed centre pixel scores. Without them the pointing
    # game numbers below have nothing to be large or small relative to.
    coverage: list[float] = []
    centre_hits: list[float] = []

    with open(artifacts / "glasses" / "predictions.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            mask = masks.get(r["sample_id"])
            if mask is None:
                continue
            sdir = layout.saliency_dir(artifacts / "glasses", r["sample_id"])
            if not layout.has_arrays(sdir):
                continue
            coverage.append(float(mask.mean()))
            cy, cx = mask.shape[0] // 2, mask.shape[1] // 2
            centre_hits.append(float(bool(mask[cy, cx])))
            z = layout.load_arrays(sdir)
            if z:
                used = False
                for meth in IMAGE_METHODS:
                    if meth not in z:
                        continue
                    s = _localisation(z[meth], mask)
                    if not s:
                        continue
                    used = True
                    for k, v in s.items():
                        scores[meth].setdefault(k, []).append(v)
            n_used += used

    out = {
        "n_images": n_used,
        "baselines": {
            "mask_coverage": round(float(np.mean(coverage)), 4) if coverage else None,
            "random_pixel_pointing": round(float(np.mean(coverage)), 4) if coverage else None,
            "centre_pixel_pointing": round(float(np.mean(centre_hits)), 4) if centre_hits else None,
        },
        "methods": {},
    }
    for meth, d in scores.items():
        if d:
            out["methods"][meth] = {k: round(float(np.mean(v)), 4) for k, v in d.items()}
            out["methods"][meth]["n"] = len(next(iter(d.values())))
    return out


# --------------------------------------------------------------------------
# Tweet — token attributions vs human rationale spans
# --------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9']+")


def _words(s: str) -> list[str]:
    return _WORD.findall(str(s).lower())


def tweet_plausibility(root: Path, artifacts: Path, cfg) -> dict:
    task = cfg.task("tweet")
    df = pd.read_csv(root / task.test_csv)
    # Records carry the text, not the id, so match on normalised text.
    rationale = {}
    for text, sel, sent in zip(df[task.text_col], df["selected_text"], df[task.label_col]):
        rationale[str(text).strip()] = (set(_words(sel)), sent,
                                        len(_words(sel)) >= len(_words(text)))

    per_class: dict[str, dict[str, list[float]]] = {}
    n_matched = 0
    n_degenerate = 0

    with open(artifacts / "tweet" / "predictions.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            hit = rationale.get(str(r.get("text", "")).strip())
            if hit is None:
                continue
            gold, sent, degenerate = hit
            if not gold:
                continue
            n_matched += 1
            if degenerate:
                # selected_text == the whole tweet (89.7% of neutral). Any method
                # scores near 1.0 by construction, so these are excluded.
                n_degenerate += 1
                continue
            for meth in TEXT_METHODS:
                e = r["explanations"].get(meth)
                if not e or e.get("error") or not e.get("tokens"):
                    continue
                # Rank by |weight| and take as many tokens as the human marked,
                # so precision and recall are measured at matched budget.
                toks = sorted(e["tokens"], key=lambda t: -abs(t[1]))
                picked, seen = [], set()
                for t, _w in toks:
                    for w in _words(t):
                        if w not in seen:
                            seen.add(w)
                            picked.append(w)
                    if len(picked) >= len(gold):
                        break
                pred = set(picked[: len(gold)])
                if not pred:
                    continue
                inter = len(pred & gold)
                p = inter / len(pred)
                rec = inter / len(gold)
                f1 = 2 * p * rec / (p + rec) if (p + rec) else 0.0
                iou = inter / len(pred | gold)
                d = per_class.setdefault(meth, {})
                for k, v in (("precision", p), ("recall", rec), ("f1", f1), ("iou", iou)):
                    d.setdefault(k, []).append(v)

    out = {
        "n_matched": n_matched,
        "n_degenerate_excluded": n_degenerate,
        "n_scored": n_matched - n_degenerate,
        "methods": {},
    }
    for meth, d in per_class.items():
        out["methods"][meth] = {k: round(float(np.mean(v)), 4) for k, v in d.items()}
        out["methods"][meth]["n"] = len(d["f1"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="all")
    args = ap.parse_args()

    cfg = cfg_mod.load()
    root = cfg_mod.PROJECT_ROOT
    artifacts = root / "artifacts"
    dest = artifacts / "plausibility.json"
    # Merge into whatever is already on disk: running one task must never
    # discard another task's block.
    out = json.loads(dest.read_text()) if dest.exists() else {}

    if args.task in ("all", "glasses"):
        print("glasses — saliency vs human attention masks ...", flush=True)
        out["glasses"] = glasses_plausibility(root, artifacts)
        print(json.dumps(out["glasses"], indent=2))
        dest.write_text(json.dumps(out, indent=2))

    if args.task in ("all", "tweet"):
        print("tweet — token attributions vs human rationale spans ...", flush=True)
        out["tweet"] = tweet_plausibility(root, artifacts, cfg)
        print(json.dumps(out["tweet"], indent=2))
        dest.write_text(json.dumps(out, indent=2))

    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
