"""Behavioural probes — not how well a method scores, but what it actually names.

`evaluate.py` asks whether an explanation points where a human points, and
`faithfulness.py` asks whether the model uses what the explanation names. Neither
describes the *content* of an explanation. These probes do, and they are what
make a comparison say something more useful than a ranking:

  token_composition   what kind of feature does each method put at the top —
                      content words, function words, punctuation, fragments?
  counter_evidence    when a strongly polar word sits in a text the model did NOT
                      classify with that polarity, does the method report it as
                      evidence the model overcame, or drop it to zero?
  contrast_probe      in "X but Y", sentiment follows Y. Does the method weight
                      the clause after the marker?
  negation_probe      does either method surface the negator at all?
  image_shape_probe   is the evidence one coherent region or scattered pixels?
  error_detection     can a method's own diagnostic flag the model's mistakes,
                      and do the methods flag the *same* mistakes?
  budget_per_feature  the cross-dataset view: how much does explanation quality
                      depend on evaluations per input feature?

Every probe takes records and returns a plain dict; file I/O happens only in
`main()`. Word lists are hand-built and written out in full below so they can be
audited — they are a limitation of these probes, not a hidden detail.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import string
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from . import config as cfg_mod
from . import layout
from .summarize import load_records

# --------------------------------------------------------------------------
# Hand-built word lists. Small and deliberately transparent: no NLTK or VADER
# dependency, and every membership decision is inspectable in this file.
# --------------------------------------------------------------------------

FUNCTION_WORDS = frozenset("""
the a an and or but if of to in on at for with is am are was were be been being
i you he she it we they me him her them my your his its our their this that
these those as by from so just now do does did have has had will would can
could get got up out not no
""".split())

POSITIVE_WORDS = frozenset("""
love loved loves great awesome good best happy thanks thank nice amazing
wonderful perfect fun cool excited glad beautiful sweet lucky win wins winning
enjoy enjoyed enjoying congrats fantastic brilliant lovely excellent yay
""".split())

NEGATIVE_WORDS = frozenset("""
hate hated bad worst sad sucks suck terrible awful sorry miss missing sick hurt
tired annoying boring fail failed poor lost lose crap stupid angry upset cry
crying damn ugh
""".split())

SENTIMENT_WORDS = POSITIVE_WORDS | NEGATIVE_WORDS

CONTRAST_MARKERS = frozenset("but although though however yet despite".split())

NEGATORS = re.compile(
    r"^(not|no|never|none|nothing|cannot|n't|dont|didnt|doesnt|isnt|wasnt|arent|wont|cant)$",
    re.I,
)

# Separates the highest failing point (movie LIME, 5.9 evals/feature -> 1.29x
# random) from the lowest passing one (glasses LIME, 7.4 -> 2.95x). Declared,
# not fitted: five points do not support a fitted threshold.
THRESHOLD_EVALS_PER_FEATURE = 7.0

_WORD = re.compile(r"[a-z0-9']+")
_PUNCT = str.maketrans("", "", string.punctuation)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def top_tokens(entry: dict, k: int) -> list[tuple[str, float]]:
    """The k highest-magnitude attributions, whatever order they were stored in.

    LIME returns its tokens already sorted by |weight|; SHAP stores them in
    document order. Slicing without sorting therefore silently gives SHAP's
    first k *words* rather than its k strongest, which is why every caller —
    including the report's worked examples — must come through here.
    """
    toks = entry.get("tokens") or []
    return sorted(toks, key=lambda t: -abs(float(t[1])))[:k]


def _norm(tok: str) -> str:
    return str(tok).strip().translate(_PUNCT).lower().strip()


def _words(s: str) -> list[str]:
    return _WORD.findall(str(s).lower())


def _attr_map(record: dict, method: str) -> dict[str, float]:
    """Normalised token -> signed weight, keeping the largest magnitude per word."""
    e = record["explanations"].get(method)
    if not e or e.get("error") or not e.get("tokens"):
        return {}
    out: dict[str, float] = {}
    for t, w in e["tokens"]:
        key = _norm(t)
        if key and (key not in out or abs(float(w)) > abs(out[key])):
            out[key] = float(w)
    return out


def _boot_diff(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> dict:
    """Paired bootstrap of mean(x) - mean(y).

    The p-value is two-sided: 2 * min(P(d <= 0), P(d >= 0)), floored at
    1/n_boot, since a bootstrap cannot resolve a p below its own resolution.
    """
    rng = np.random.default_rng(seed)
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    if d.size == 0:
        return {}
    idx = rng.integers(0, d.size, (n_boot, d.size))
    bs = d[idx].mean(axis=1)
    p = 2.0 * min((bs <= 0).mean(), (bs >= 0).mean())
    return {
        "diff": round(float(d.mean()), 4),
        "ci_low": round(float(np.percentile(bs, 2.5)), 4),
        "ci_high": round(float(np.percentile(bs, 97.5)), 4),
        "p": round(max(float(p), 1.0 / n_boot), 5),
        "n": int(d.size),
    }


# --------------------------------------------------------------------------
# What kind of feature does each method name?
# --------------------------------------------------------------------------


def _classify(tok: str, text_words: set[str]) -> str:
    t = str(tok).strip()
    if not t:
        return "whitespace"
    if all(c in string.punctuation for c in t):
        return "punctuation"
    low = _norm(t)
    if not low:
        return "punctuation"
    if low in SENTIMENT_WORDS:
        return "sentiment word"
    if low in FUNCTION_WORDS:
        return "function word"
    if low.isdigit():
        return "number"
    # A wordpiece: the model's tokenizer split a word and this is only part of
    # it, so it matches no whitespace-delimited word of the input. LIME scores
    # 0% here by construction because it perturbs whole words — that contrast
    # is the point, not an artefact.
    if low not in text_words:
        return "sub-word fragment"
    return "content word"


def token_composition(records: list[dict], methods=("lime", "shap"), top_k: int = 5) -> dict:
    tally: dict[str, defaultdict] = {m: defaultdict(int) for m in methods}
    total = dict.fromkeys(methods, 0)
    for r in records:
        text_words = set(_words(r.get("text") or ""))
        for m in methods:
            e = r["explanations"].get(m)
            if not e or e.get("error") or not e.get("tokens"):
                continue
            for t, _w in top_tokens(e, top_k):
                tally[m][_classify(t, text_words)] += 1
                total[m] += 1
    cats = ["content word", "function word", "sentiment word",
            "sub-word fragment", "punctuation", "number"]
    return {
        "top_k": top_k,
        "categories": cats,
        "methods": {
            m: {"n_tokens": total[m],
                **{c: round(100.0 * tally[m][c] / total[m], 1) for c in cats}}
            for m in methods if total[m]
        },
    }


# --------------------------------------------------------------------------
# Counter-evidence: a polar word the model's prediction contradicts
# --------------------------------------------------------------------------


def counter_evidence(records: list[dict], methods=("lime", "shap"), min_n: int = 8) -> dict:
    """Mean weight on a polar word when the prediction is NOT of that polarity.

    LIME asks "does deleting this word move the prediction?" — for a word the
    model already discounted, the answer is no, so the coefficient is ~0. SHAP
    asks "what did this word contribute relative to a baseline?" and additivity
    forces it to report the word as evidence the model overcame. Both are
    internally correct; only one shows the reader what the model argued against.
    """
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        pred = r.get("pred_label", "")
        for word_set, polarity in ((POSITIVE_WORDS, "positive"), (NEGATIVE_WORDS, "negative")):
            if pred == polarity:
                continue  # the prediction agrees with the word — not counter-evidence
            maps = {m: _attr_map(r, m) for m in methods}
            if not all(maps.values()):
                continue
            common = set.intersection(*(set(v) for v in maps.values())) & word_set
            for w in common:
                for m in methods:
                    acc[w][m].append(maps[m][w])
    words = {}
    for w, per in acc.items():
        n = min(len(v) for v in per.values())
        if n < min_n:
            continue
        words[w] = {"n": n, **{m: round(float(np.mean(v)), 4) for m, v in per.items()}}
    return {"min_n": min_n,
            "words": dict(sorted(words.items(), key=lambda kv: -kv[1]["n"]))}


# --------------------------------------------------------------------------
# Contrast and negation
# --------------------------------------------------------------------------


def contrast_probe(records: list[dict], methods=("lime", "shap"),
                   n_boot: int = 10000, seed: int = 42) -> dict:
    """Share of attribution mass falling after a contrast marker ("X but Y")."""
    per: dict[str, list[float]] = {m: [] for m in methods}
    for r in records:
        ws = _words(r.get("text") or "")
        hits = [i for i, w in enumerate(ws) if w in CONTRAST_MARKERS]
        if not hits:
            continue
        cut = hits[0]
        if cut < 2 or cut > len(ws) - 3:
            continue  # marker too close to an edge for the split to mean anything
        shares = {}
        for m in methods:
            a = _attr_map(r, m)
            if not a:
                break
            pre = sum(abs(a.get(w, 0.0)) for w in ws[:cut])
            post = sum(abs(a.get(w, 0.0)) for w in ws[cut + 1:])
            if pre + post <= 0:
                break
            shares[m] = post / (pre + post)
        if len(shares) == len(methods):
            for m in methods:
                per[m].append(shares[m])
    if not per[methods[0]]:
        return {}
    out = {"methods": {m: round(float(np.mean(v)), 4) for m, v in per.items()},
           "n": len(per[methods[0]]), "indifferent": 0.5}
    if len(methods) == 2:
        a, b = methods
        out["test"] = {"comparison": f"{b} - {a}",
                       **_boot_diff(np.array(per[b]), np.array(per[a]), n_boot, seed)}
    return out


def negation_probe(records: list[dict], methods=("lime", "shap")) -> dict:
    """Where does the negator sit in each method's ranking, and how much mass?"""
    ranks: dict[str, list[float]] = {m: [] for m in methods}
    mass: dict[str, list[float]] = {m: [] for m in methods}
    counts = dict.fromkeys(methods, 0)
    for r in records:
        for m in methods:
            e = r["explanations"].get(m)
            if not e or e.get("error") or not e.get("tokens"):
                continue
            toks = [(_norm(t), float(w)) for t, w in e["tokens"]]
            toks = [(t, w) for t, w in toks if t]
            negs = [(t, w) for t, w in toks if NEGATORS.match(t)]
            if not negs:
                continue
            counts[m] += 1
            order = sorted(range(len(toks)), key=lambda i: -abs(toks[i][1]))
            pos = {toks[i][0]: rank for rank, i in enumerate(order)}
            best = min(pos[t] for t, _ in negs)
            ranks[m].append(best / max(len(toks) - 1, 1))
            tot = sum(abs(w) for _, w in toks) or 1.0
            mass[m].append(sum(abs(w) for _, w in negs) / tot)
    return {
        "methods": {
            m: {"n": counts[m],
                "normalised_rank": round(float(np.mean(ranks[m])), 4),
                "mass_share": round(float(np.mean(mass[m])), 4)}
            for m in methods if counts[m]
        },
        "note": "normalised rank: 0 = the method's single most important feature",
    }


# --------------------------------------------------------------------------
# Image: the shape of the evidence
# --------------------------------------------------------------------------


def image_shape_probe(records: list[dict], artifact_dir: Path,
                      methods=("gradcam", "shap", "lime"),
                      top_frac: float = 0.05, limit: int | None = None) -> dict:
    """Is the top-5% of the map one coherent region, or scattered pixels?

    This is the measured form of a claim the report previously made by looking
    at panels: a focus score alone ranks a salt-and-pepper map above a coherent
    one, which is the opposite of how a human reads it.
    """
    from scipy import ndimage

    acc: dict[str, dict[str, list[float]]] = {m: defaultdict(list) for m in methods}
    n = 0
    for r in records if limit is None else records[:limit]:
        sdir = layout.saliency_dir(artifact_dir, r["sample_id"])
        if not layout.has_arrays(sdir):
            continue
        z = layout.load_arrays(sdir)
        for m in methods:
            if m not in z:
                continue
            a = np.abs(z[m].astype(np.float32))
            k = max(int(top_frac * a.size), 1)
            thr = np.partition(a.ravel(), -k)[-k]
            mask = a >= thr
            lab, ncomp = ndimage.label(mask)
            if ncomp:
                sizes = ndimage.sum(mask, lab, range(1, ncomp + 1))
                largest = float(max(sizes)) / max(int(mask.sum()), 1)
            else:
                largest = 0.0
            ys, xs = np.nonzero(mask)
            acc[m]["n_components"].append(ncomp)
            acc[m]["mass_in_largest"].append(largest)
            acc[m]["centroid_row"].append(float(ys.mean()) if ys.size else 0.0)
        n += 1
    return {
        "n_images": n,
        "top_frac": top_frac,
        "methods": {
            m: {k: round(float(np.mean(v)), 2) for k, v in d.items()}
            for m, d in acc.items() if d
        },
    }


# --------------------------------------------------------------------------
# Error detection and complementarity
# --------------------------------------------------------------------------

_SIGNAL_LABEL = {
    "lime": "LIME surrogate R²",
    "shap": "SHAP |sum of values|",
    "prototype": "prototype agreement",
    "confidence": "model confidence",
}


def _error_scores(record: dict) -> dict[str, float | None]:
    """Every signal oriented so that HIGHER means more likely to be wrong."""
    e = record["explanations"]
    r2 = e.get("lime", {}).get("stats", {}).get("surrogate_r2")
    sv = e.get("shap", {}).get("stats", {}).get("sum_values")
    na = e.get("prototype", {}).get("stats", {}).get("neighbour_agreement")
    return {
        "lime": None if r2 is None else 1.0 - float(r2),
        "shap": None if sv is None else -abs(float(sv)),
        "prototype": None if na is None else 1.0 - float(na),
        "confidence": 1.0 - float(record["confidence"]),
    }


def error_detection(records: list[dict], far: float = 0.10) -> dict:
    """Can each signal pick out the model's errors, and do they pick the same ones?

    Thresholds are set per signal at the chosen false-alarm rate on CORRECT
    predictions, so recalls are comparable. `prototype` takes only four distinct
    values (k=3 neighbours), so its achieved false-alarm rate can overshoot the
    target badly — the AUC column is the fair comparison, and the realised rate
    is reported alongside every recall so the discrepancy is visible.
    """
    keys = ("lime", "shap", "prototype", "confidence")
    cols: dict[str, list[float]] = {k: [] for k in keys}
    y: list[int] = []
    for r in records:
        s = _error_scores(r)
        if any(s[k] is None for k in keys):
            continue
        for k in keys:
            cols[k].append(s[k])
        y.append(0 if r["correct"] else 1)
    y_arr = np.asarray(y)
    n_err = int(y_arr.sum())
    if n_err == 0 or n_err == y_arr.size:
        return {}

    signals, flags = {}, {}
    for k in keys:
        v = np.asarray(cols[k])
        thr = float(np.percentile(v[y_arr == 0], 100 * (1 - far)))
        flagged = v >= thr
        flags[k] = set(np.nonzero(flagged & (y_arr == 1))[0].tolist())
        signals[k] = {
            "label": _SIGNAL_LABEL[k],
            "roc_auc": round(float(roc_auc_score(y_arr, v)), 4),
            "distinct_values": int(np.unique(v).size),
            "recall_at_far": round(len(flags[k]) / n_err, 4),
            "realised_far": round(float(flagged[y_arr == 0].mean()), 4),
        }

    expl = ("lime", "shap", "prototype")
    overlap = {}
    for i, a in enumerate(expl):
        for b in expl[i + 1:]:
            A, B = flags[a], flags[b]
            overlap[f"{a}|{b}"] = {
                "jaccard": round(len(A & B) / max(len(A | B), 1), 4),
                "both": len(A & B), f"only_{a}": len(A - B), f"only_{b}": len(B - A),
            }
    union = set().union(*(flags[m] for m in expl))
    best = max(len(flags[m]) for m in expl)
    return {
        "n_predictions": int(y_arr.size), "n_errors": n_err, "target_far": far,
        "signals": signals, "overlap": overlap,
        "union_recall": round(len(union) / n_err, 4),
        "best_single_recall": round(best / n_err, 4),
        "prototype_only": len(flags["prototype"] - flags["lime"] - flags["shap"]),
    }


# --------------------------------------------------------------------------
# Cross-dataset: evaluations per feature against measured faithfulness
# --------------------------------------------------------------------------

# Which stats key carries the sampling budget, and which the feature count.
_BUDGET_KEY = {"lime": ("num_samples", ("n_tokens", "n_superpixels")),
               "shap": (("max_evals", "n_samples"), ("n_tokens",))}


def _first(stats: dict, keys) -> float | None:
    for k in (keys,) if isinstance(keys, str) else keys:
        if k in stats:
            return float(stats[k])
    return None


def budget_per_feature(records_by_task: dict[str, list[dict]], faith: dict) -> dict:
    """How many model evaluations did each input feature actually receive?

    Perturbation methods estimate an attribution per feature by querying the
    model; the ratio of queries to features is what decides whether they are
    estimating or interpolating. Gradient methods read the network directly and
    have no such ratio — they are carried here with `budget_kind: "gradient"`
    so a figure can exclude them rather than invent an x-coordinate for them.
    """
    points = []
    for task, recs in records_by_task.items():
        fm = (faith.get(task) or {}).get("methods") or {}
        rand = (fm.get("random") or {}).get("comprehensiveness")
        for method, entry in fm.items():
            if method == "random":
                continue
            budgets, feats = [], []
            for r in recs:
                st = (r["explanations"].get(method) or {}).get("stats") or {}
                if not st:
                    continue
                bkeys, fkeys = _BUDGET_KEY.get(method, (None, None))
                b = _first(st, bkeys) if bkeys else None
                f = _first(st, fkeys) if fkeys else None
                if b is not None:
                    budgets.append(b)
                if f is not None:
                    feats.append(f)
            comp = entry.get("comprehensiveness")
            # Image SHAP is GradientSHAP and Grad-CAM is a single backward pass:
            # neither perturbs per feature, so neither gets an evals/feature.
            is_grad = method == "gradcam" or (method == "shap" and not feats)
            n_feat = float(np.mean(feats)) if feats else None
            budget = float(np.mean(budgets)) if budgets else None
            points.append({
                "task": task, "method": method,
                "budget": round(budget, 1) if budget is not None else None,
                "n_features_mean": round(n_feat, 1) if n_feat else None,
                "evals_per_feature": (None if is_grad or not n_feat or budget is None
                                      else round(budget / n_feat, 2)),
                "comprehensiveness": comp,
                "ratio_vs_random": (round(comp / rand, 2) if comp and rand else None),
                "budget_kind": "gradient" if is_grad else "perturbation",
            })
    points.sort(key=lambda p: (p["evals_per_feature"] is None,
                               p["evals_per_feature"] or 0))
    return {
        "threshold_evals_per_feature": THRESHOLD_EVALS_PER_FEATURE,
        "threshold_note": (
            "Declared, not fitted: it separates the highest failing point from the "
            "lowest passing one. Five points do not support a fitted threshold."
        ),
        "points": points,
    }


# --------------------------------------------------------------------------


def probe_task(cfg, key: str, records: list[dict], n_boot: int) -> dict:
    kind = cfg.task(key).kind
    out: dict = {"kind": kind, "n_records": len(records)}
    if kind == "text":
        out["token_composition"] = token_composition(records)
        out["counter_evidence"] = counter_evidence(records)
        out["contrast"] = contrast_probe(records, n_boot=n_boot)
        out["negation"] = negation_probe(records)
        out["error_detection"] = error_detection(records)
    else:
        out["shape"] = image_shape_probe(records, cfg.artifact_dir(key))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--config", default=str(cfg_mod.DEFAULT_CONFIG))
    args = ap.parse_args()

    cfg = cfg_mod.load(args.config)
    dest = cfg.artifacts / "behaviour.json"
    out = json.loads(dest.read_text()) if dest.exists() else {}

    keys = [k for k in cfg.task_keys
            if (cfg.artifact_dir(k) / "predictions.jsonl").exists()
            and args.task in ("all", k)]
    loaded: dict[str, list[dict]] = {}
    for k in keys:
        recs = load_records(cfg.artifact_dir(k) / "predictions.jsonl")
        if args.limit:
            recs = recs[:args.limit]
        loaded[k] = recs
        print(f"=== {k} ({len(recs)} records) ===", flush=True)
        out[k] = probe_task(cfg, k, recs, args.bootstrap)
        print(json.dumps(out[k], indent=2)[:1200], flush=True)
        dest.write_text(json.dumps(out, indent=2))

    # Cross-task blocks need every task, so only refresh them on a full run.
    if args.task == "all":
        faith = json.loads((cfg.artifacts / "faithfulness.json").read_text()) \
            if (cfg.artifacts / "faithfulness.json").exists() else {}
        cross = {"budget": budget_per_feature(loaded, faith)}
        out["cross"] = cross
        print("=== cross ===", flush=True)
        print(json.dumps(cross, indent=2)[:1500], flush=True)

    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
