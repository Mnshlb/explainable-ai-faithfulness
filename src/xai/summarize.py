"""Aggregate per-prediction explanations into task-level statistics.

This turns thousands of individual explanation records into the numbers the
report is built from: how the model performed, what each method cost, how
self-consistent each method was, and how far the methods agreed with each
other.

Nothing here compares explanations against a human ground truth -- these are
properties of the methods and the model alone, which is what makes them
computable for every prediction rather than for a labelled subset.

Usage:
    python -m xai.summarize --task tweet
    python -m xai.summarize --all
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import config as cfgmod
from . import layout


def load_records(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _mean(xs: list[float]) -> float | None:
    return round(float(np.mean(xs)), 4) if xs else None


# --------------------------------------------------------------------------
# Model performance
# --------------------------------------------------------------------------


def performance(records: list[dict], labels: list[str], split: str | None = None) -> dict:
    rs = [r for r in records if split is None or r.get("split") == split]
    if not rs:
        return {}
    n = len(rs)
    correct = sum(r["correct"] for r in rs)
    idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for r in rs:
        cm[idx[r["true_label"]], idx[r["pred_label"]]] += 1

    per_class = {}
    for i, l in enumerate(labels):
        tp, fp, fn = cm[i, i], cm[:, i].sum() - cm[i, i], cm[i, :].sum() - cm[i, i]
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per_class[l] = {
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0, 4),
            "support": int(cm[i, :].sum()),
        }

    conf_correct = [r["confidence"] for r in rs if r["correct"]]
    conf_wrong = [r["confidence"] for r in rs if not r["correct"]]
    return {
        "n": n,
        "accuracy": round(correct / n, 4),
        "macro_f1": round(float(np.mean([v["f1"] for v in per_class.values()])), 4),
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "per_class": per_class,
        "mean_confidence_correct": _mean(conf_correct),
        "mean_confidence_wrong": _mean(conf_wrong),
    }


# --------------------------------------------------------------------------
# Method-level statistics
# --------------------------------------------------------------------------


def method_stats(records: list[dict]) -> dict:
    """Cost, failure rate and self-consistency diagnostics per method."""
    acc: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    failures: Counter = Counter()
    counts: Counter = Counter()

    for r in records:
        for m, e in r["explanations"].items():
            counts[m] += 1
            if e.get("error"):
                failures[m] += 1
                continue
            acc[m]["runtime"].append(e.get("runtime_s", 0.0))
            st = e.get("stats", {})
            for key in ("surrogate_r2", "convergence_delta", "top5pct_mass",
                        "neighbour_agreement", "sum_values", "n_tokens"):
                if key in st:
                    acc[m][key].append(float(st[key]))

    out = {}
    for m in counts:
        d = acc[m]
        entry = {
            "n_explained": counts[m],
            "n_failed": failures[m],
            "failure_rate": round(failures[m] / counts[m], 4),
            # Falls back to 0.0 when every explanation of this method failed;
            # failure_rate above is what tells you that happened.
            "runtime_s_mean": _mean(d["runtime"]) or 0.0,
            "runtime_s_total": round(float(np.sum(d["runtime"])), 1) if d["runtime"] else 0.0,
        }
        # LIME reports how well its local linear model fits the classifier.
        # A low mean R^2 means the explanations are weakly grounded.
        if d["surrogate_r2"]:
            r2 = np.array(d["surrogate_r2"])
            entry["surrogate_r2_mean"] = round(float(r2.mean()), 4)
            entry["surrogate_r2_below_0.3"] = round(float((r2 < 0.3).mean()), 4)
        # GradientSHAP's own convergence error.
        if d["convergence_delta"]:
            entry["convergence_delta_mean"] = _mean(d["convergence_delta"])
        # How concentrated the saliency maps are.
        if d["top5pct_mass"]:
            entry["top5pct_mass_mean"] = _mean(d["top5pct_mass"])
        if d["neighbour_agreement"]:
            na = np.array(d["neighbour_agreement"])
            entry["neighbour_agreement_mean"] = round(float(na.mean()), 4)
            entry["all_neighbours_disagree"] = round(float((na == 0).mean()), 4)
        out[m] = entry
    return out



def quality_by_correctness(records: list[dict]) -> dict:
    """Does explanation quality hold up on the predictions the model gets wrong?

    The hope for post-hoc explanation is that it helps most where the model is
    least reliable. These numbers test that directly, by splitting each
    method's own reliability diagnostic on whether the prediction was correct.
    """
    out: dict[str, dict] = {}
    groups = {"correct": [r for r in records if r["correct"]],
              "wrong": [r for r in records if not r["correct"]]}

    for method, key in (("lime", "surrogate_r2"), ("prototype", "neighbour_agreement")):
        entry = {}
        for gname, rs in groups.items():
            vals = [
                r["explanations"][method]["stats"][key]
                for r in rs
                if method in r["explanations"]
                and not r["explanations"][method].get("error")
                and key in r["explanations"][method].get("stats", {})
            ]
            if vals:
                entry[gname] = {"n": len(vals), "mean": round(float(np.mean(vals)), 4)}
        if len(entry) == 2:
            entry["delta"] = round(entry["correct"]["mean"] - entry["wrong"]["mean"], 4)
            out[f"{method}.{key}"] = entry

    # SHAP has no fit statistic, but the magnitude of its attributions is
    # informative: near-zero totals mean "no token moved the model".
    for gname, rs in groups.items():
        vals = [
            abs(r["explanations"]["shap"]["stats"]["sum_values"])
            for r in rs
            if "shap" in r["explanations"]
            and not r["explanations"]["shap"].get("error")
            and "sum_values" in r["explanations"]["shap"].get("stats", {})
        ]
        if vals:
            out.setdefault("shap.abs_sum_values", {})[gname] = {
                "n": len(vals), "mean": round(float(np.mean(vals)), 4)
            }
    e = out.get("shap.abs_sum_values", {})
    if "correct" in e and "wrong" in e:
        e["delta"] = round(e["correct"]["mean"] - e["wrong"]["mean"], 4)

    # Mean confidence is the obvious baseline to compare all of this against.
    for gname, rs in groups.items():
        if rs:
            out.setdefault("confidence", {})[gname] = {
                "n": len(rs),
                "mean": round(float(np.mean([r["confidence"] for r in rs])), 4),
            }
    return out


# --------------------------------------------------------------------------
# Cross-method agreement
# --------------------------------------------------------------------------


def _norm_token(t: str) -> str:
    return t.strip().lower()


def token_agreement(records: list[dict], a: str = "lime", b: str = "shap",
                    k: int = 5) -> dict:
    """Do the two token-level methods pick out the same words?

    Measured as Jaccard overlap of the top-k tokens by absolute weight, and
    as the rate at which they agree on the single most important token.
    """
    jac, top1, signs = [], [], []
    for r in records:
        ea, eb = r["explanations"].get(a), r["explanations"].get(b)
        if not ea or not eb or ea.get("error") or eb.get("error"):
            continue
        ta = [(_norm_token(t), w) for t, w in ea.get("tokens", [])]
        tb = [(_norm_token(t), w) for t, w in eb.get("tokens", [])]
        if not ta or not tb:
            continue
        sa = {t for t, _ in sorted(ta, key=lambda x: -abs(x[1]))[:k]}
        sb = {t for t, _ in sorted(tb, key=lambda x: -abs(x[1]))[:k]}
        if sa | sb:
            jac.append(len(sa & sb) / len(sa | sb))
        fa = max(ta, key=lambda x: abs(x[1]))
        fb = max(tb, key=lambda x: abs(x[1]))
        top1.append(fa[0] == fb[0])
        # Where both methods scored the same token, did they agree on sign?
        da, db = dict(ta), dict(tb)
        shared = set(da) & set(db)
        if shared:
            signs.append(
                float(np.mean([np.sign(da[t]) == np.sign(db[t]) for t in shared]))
            )
    return {
        "pair": f"{a}_vs_{b}",
        "n_compared": len(jac),
        "top_k": k,
        "jaccard_topk_mean": _mean(jac),
        "top1_token_agreement": _mean([float(x) for x in top1]),
        "sign_agreement_shared_tokens": _mean(signs),
    }


def saliency_agreement(records: list[dict], artifact_dir: Path,
                       pairs=(("gradcam", "shap"), ("gradcam", "lime"), ("shap", "lime")),
                       limit: int | None = None) -> list[dict]:
    """Spearman correlation between pairs of saliency maps over pixels."""
    from scipy.stats import spearmanr

    acc: dict[str, list[float]] = defaultdict(list)
    rs = records[:limit] if limit else records
    for r in rs:
        rel = None
        for e in r["explanations"].values():
            rel = (e.get("arrays") or {}).get("dir")
            if rel:
                break
        if not rel:
            continue
        maps = {k: v.astype(np.float32).ravel()
                for k, v in layout.load_arrays(artifact_dir / rel).items()}
        if not maps:
            continue
        for a, b in pairs:
            if a not in maps or b not in maps:
                continue
            xa, xb = np.abs(maps[a]), np.abs(maps[b])
            # A failed explainer leaves an all-zero map, for which rank
            # correlation is undefined; skip rather than emit a NaN.
            if xa.std() == 0 or xb.std() == 0:
                continue
            # Absolute value: the question is whether the methods agree on
            # *where* the evidence is, not on its polarity (Grad-CAM is
            # unsigned by construction).
            rho = spearmanr(xa, xb).statistic
            if np.isfinite(rho):
                acc[f"{a}_vs_{b}"].append(float(rho))
    return [
        {"pair": k, "n_compared": len(v), "spearman_mean": _mean(v)}
        for k, v in acc.items()
    ]


# --------------------------------------------------------------------------


def summarize_task(cfg: cfgmod.Config, key: str) -> dict:
    task = cfg.task(key)
    art = cfg.artifact_dir(key)
    jsonl = art / "predictions.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(f"no predictions for '{key}'; run xai.run first")

    records = load_records(jsonl)
    labels = task.labels
    summary: dict = {
        "task": key,
        "name": task.name,
        "kind": task.kind,
        "n_records": len(records),
        "performance": performance(records, labels),
        "methods": method_stats(records),
        "quality_by_correctness": quality_by_correctness(records),
    }

    # For the retrained image model, also report the clean held-out figure.
    splits = {r.get("split") for r in records} - {None}
    if splits:
        summary["performance_by_split"] = {
            s: performance(records, labels, split=s) for s in sorted(splits)
        }

    if task.kind == "text":
        summary["agreement"] = [token_agreement(records)]
    else:
        summary["agreement"] = saliency_agreement(records, art)

    # Predictions where the methods disagree most are the interesting ones.
    summary["lowest_confidence"] = sorted(
        ({"sample_id": r["sample_id"], "confidence": r["confidence"],
          "correct": r["correct"], "pred": r["pred_label"], "true": r["true_label"]}
         for r in records),
        key=lambda d: d["confidence"],
    )[:15]
    summary["errors"] = [
        {"sample_id": r["sample_id"], "confidence": r["confidence"],
         "pred": r["pred_label"], "true": r["true_label"]}
        for r in records if not r["correct"]
    ][:50]
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    args = ap.parse_args()

    cfg = cfgmod.load(args.config)
    keys = cfg.task_keys if args.all else [args.task]
    if not args.task and not args.all:
        ap.error("pass --task or --all")

    combined = {}
    for k in keys:
        try:
            s = summarize_task(cfg, k)
        except FileNotFoundError as err:
            print(f"[{k}] skipped: {err}")
            continue
        combined[k] = s
        out = cfg.artifact_dir(k) / "summary.json"
        out.write_text(json.dumps(s, indent=2))
        perf = s["performance"]
        print(f"\n=== {k} ({s['n_records']} records) ===")
        print(f"accuracy {perf.get('accuracy')}  macro-F1 {perf.get('macro_f1')}")
        for m, v in s["methods"].items():
            rt = v["runtime_s_mean"] or 0.0
            print(f"  {m:10s} {rt:>7.2f}s/sample  failures {v['failure_rate']:.1%}"
                  + (f"  R²={v['surrogate_r2_mean']:.3f}" if "surrogate_r2_mean" in v else ""))
        for a in s["agreement"]:
            print(f"  agreement {a}")
        print(f"  -> {out}")

    if combined:
        (cfg.artifacts / "summary_all.json").write_text(json.dumps(combined, indent=2))
        print(f"\nwrote {cfg.artifacts / 'summary_all.json'}")


if __name__ == "__main__":
    main()
