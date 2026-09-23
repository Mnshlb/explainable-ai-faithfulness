"""Inline SVG figures for the report.

SVG rather than raster so the figures stay sharp when the report is printed to
PDF, and inline so the document remains a single self-contained file.

Palette: categorical slots 1-3 of the reference data-visualisation palette,
validated for colour-vision deficiency on a white surface (worst adjacent CVD
Delta E 9.2, normal-vision 27.6). The aqua slot sits below 3:1 contrast against
white, so every chart here carries direct value labels and every figure is
accompanied by the same numbers in a table -- the relief the contrast warning
requires.
"""

from __future__ import annotations

import math
from html import escape

# Categorical slots, assigned in fixed order and never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK = "#1a1a1a"
INK_2 = "#6b6660"
RULE = "#ddd8d0"
SURFACE = "#ffffff"

# Colour follows the entity, never its position in a list. Assigning by
# encounter order cycles once a fourth method appears and silently paints
# Grad-CAM the same blue as LIME.
METHOD_COLOUR = {"lime": SERIES[0], "shap": SERIES[1],
                 "gradcam": SERIES[2], "prototype": SERIES[2]}
TASK_COLOUR = {"tweet": SERIES[0], "movie": SERIES[1], "glasses": SERIES[2]}

# Three categorical slots is the honest limit of this palette. A fourth entity
# is therefore never a fourth hue: `random` and `model confidence` are
# baselines, drawn in INK_2 as reference marks, which is also what they are.
BASELINE = INK_2

FONT = ('font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,sans-serif"')


def _txt(x, y, s, size=10, fill=INK_2, anchor="start", weight="400") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" {FONT} font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}">{escape(str(s))}</text>'
    )


def _legend(x: float, y: float, items: list[tuple[str, str]]) -> str:
    """Swatch + label pairs. Identity is never carried by colour alone."""
    out, cx = [], x
    for colour, label in items:
        out.append(
            f'<rect x="{cx:.1f}" y="{y - 7:.1f}" width="9" height="9" rx="2" fill="{colour}"/>'
        )
        out.append(_txt(cx + 14, y + 1, label, size=10, fill=INK_2))
        cx += 16 + len(label) * 5.9
    return "".join(out)


def _frame(w: int, h: int, body: str, title: str, subtitle: str = "") -> str:
    head = _txt(0, 13, title, size=12.5, fill=INK, weight="600")
    sub = _txt(0, 28, subtitle, size=10, fill=INK_2) if subtitle else ""
    return (
        f'<figure class="fig"><svg viewBox="0 0 {w} {h}" width="100%" '
        f'role="img" aria-label="{escape(title)}" style="max-width:{w}px">'
        f"{head}{sub}{body}</svg></figure>"
    )


# --------------------------------------------------------------------------


def training_curve(history: list[dict]) -> str:
    """Train vs validation accuracy per epoch.

    A line chart with a truncated y-axis: the whole story lives in the top few
    percent, and a zero baseline would flatten both series into one line. The
    axis is labelled with its real bounds so the truncation is visible.
    """
    if not history:
        return ""
    w, h = 700, 276
    # top clears the header band (title / subtitle / legend) so nothing collides
    left, right, top, bottom = 44, 92, 66, 34
    pw, ph = w - left - right, h - top - bottom

    epochs = [d["epoch"] for d in history]
    tr = [d["train_acc"] for d in history]
    va = [d["val_acc"] for d in history]

    lo = min(min(tr), min(va))
    lo = max(0.0, (int(lo * 100) - 1) / 100)
    hi = 1.0
    span = max(hi - lo, 1e-9)

    def X(e):
        return left + (e - epochs[0]) / max(len(epochs) - 1, 1) * pw

    def Y(v):
        return top + (hi - v) / span * ph

    parts = []
    # Recessive hairline gridlines, solid, never dashed.
    steps = 5
    for i in range(steps + 1):
        v = lo + span * i / steps
        y = Y(v)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + pw}" y2="{y:.1f}" '
            f'stroke="{RULE}" stroke-width="1"/>'
        )
        parts.append(_txt(left - 8, y + 3.5, f"{v * 100:.0f}%", size=9.5, anchor="end"))
    for e in epochs:
        parts.append(_txt(X(e), top + ph + 17, e, size=9.5, anchor="middle"))
    parts.append(_txt(left + pw / 2, h - 4, "epoch", size=10, anchor="middle"))

    for series, colour, label in ((tr, SERIES[0], "train"), (va, SERIES[1], "validation")):
        pts = " ".join(f"{X(e):.1f},{Y(v):.1f}" for e, v in zip(epochs, series))
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{colour}" '
            f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for e, v in zip(epochs, series):
            # 2px surface ring keeps markers legible where the lines cross.
            parts.append(
                f'<circle cx="{X(e):.1f}" cy="{Y(v):.1f}" r="4" fill="{colour}" '
                f'stroke="{SURFACE}" stroke-width="2"/>'
            )
        # Direct label at the endpoint only; the axis carries the rest.
        parts.append(
            _txt(X(epochs[-1]) + 10, Y(series[-1]) + 3.5,
                 f"{label} {series[-1] * 100:.2f}%", size=10, fill=INK_2)
        )

    parts.append(_legend(left, 46, [(SERIES[0], "train"), (SERIES[1], "validation")]))
    return _frame(
        w, h, "".join(parts),
        "Glasses detector — accuracy per epoch",
        f"y-axis starts at {lo * 100:.0f}%, not zero, so the two series stay separable",
    )


def cost_small_multiples(summaries: dict) -> str:
    """Mean explanation runtime, faceted by task.

    Faceted rather than combined because the costs span three orders of
    magnitude; one shared linear axis would render Grad-CAM invisible, and a
    log axis would break the proportionality a bar length promises.
    """
    tasks = [(k, s) for k, s in summaries.items() if s.get("methods")]
    if not tasks:
        return ""

    row_h, bar_h = 26, 18
    facet_gap = 30
    left, right = 96, 74
    w = 700
    body, y = [], 68   # clears the header band

    for key, s in tasks:
        rows = [(m, (v.get("runtime_s_mean") or 0.0), v["failure_rate"])
                for m, v in s["methods"].items()]
        vmax = max((r[1] for r in rows), default=0.0) or 1.0
        body.append(_txt(0, y, s["name"].split("(")[0].strip(), size=10.5,
                         fill=INK, weight="600"))
        y += 10
        for m, val, fail in rows:
            bw = (val / vmax) * (w - left - right)
            cy = y + (row_h - bar_h) / 2
            body.append(_txt(left - 10, y + row_h / 2 + 3.5, m, size=10, anchor="end"))
            if bw > 0.5:
                # 4px rounded data-end, square at the baseline.
                r = min(4, bw / 2)
                body.append(
                    f'<path d="M{left} {cy} H{left + bw - r} a{r} {r} 0 0 1 {r} {r} '
                    f'V{cy + bar_h - r} a{r} {r} 0 0 1 {-r} {r} H{left} Z" '
                    f'fill="{SERIES[0]}"/>'
                )
            label = f"{val:.2f}s" if val >= 0.01 else "<0.01s"
            if fail >= 1.0:
                label = "all failed"
            body.append(_txt(left + bw + 8, y + row_h / 2 + 3.5, label,
                             size=10, fill=INK_2))
            y += row_h
        y += facet_gap - 10

    # One series, every row named on the left axis: colour would carry no
    # information here, and a legend for a single series is noise.
    return _frame(
        w, int(y + 6), "".join(body),
        "Cost of one explanation",
        "mean wall-clock seconds per prediction · each task on its own scale",
    )


def quality_by_correctness(summaries: dict) -> str:
    """LIME surrogate fit on correct predictions versus on errors.

    The question this figure exists to answer: does the explanation hold up
    where the model does not?
    """
    rows = []
    for key, s in summaries.items():
        q = (s.get("quality_by_correctness") or {}).get("lime.surrogate_r2")
        if q and "correct" in q and "wrong" in q:
            rows.append((s["name"].split("(")[0].strip(),
                         q["correct"]["mean"], q["wrong"]["mean"],
                         q["correct"]["n"], q["wrong"]["n"]))
    if not rows:
        return ""

    w = 700
    left, right = 150, 96
    bar_h, gap, group_gap = 15, 2, 22   # 2px surface gap between adjacent bars
    pw = w - left - right
    body, y = [], 68   # clears the header band

    for name, c, wr, nc, nw in rows:
        body.append(_txt(left - 12, y + bar_h + 4, name, size=10.5, anchor="end",
                         fill=INK, weight="600"))
        for val, colour, lbl, n in ((c, SERIES[0], "correct", nc),
                                    (wr, SERIES[1], "wrong", nw)):
            bw = max(val, 0) * pw
            r = min(4, max(bw, 1) / 2)
            if bw > 0.5:
                body.append(
                    f'<path d="M{left} {y} H{left + bw - r} a{r} {r} 0 0 1 {r} {r} '
                    f'V{y + bar_h - r} a{r} {r} 0 0 1 {-r} {r} H{left} Z" '
                    f'fill="{colour}"/>'
                )
            body.append(_txt(left + bw + 8, y + bar_h - 3,
                             f"{val:.3f}  (n={n:,})", size=9.5, fill=INK_2))
            y += bar_h + gap
        y += group_gap

    # Zero baseline: these are bars, so length must stay proportional.
    body.append(
        f'<line x1="{left}" y1="62" x2="{left}" y2="{y - group_gap + 2}" '
        f'stroke="{RULE}" stroke-width="1"/>'
    )
    body.insert(0, _legend(left, 46, [(SERIES[0], "when correct"),
                                      (SERIES[1], "when wrong")]))
    return _frame(
        w, int(y + 4), "".join(body),
        "Does the explanation hold up on the model's errors?",
        "mean LIME surrogate R² — how well the local linear model fits the classifier",
    )


# --------------------------------------------------------------------------
# Faithfulness — the primary criterion
# --------------------------------------------------------------------------

_HIGHER_IS_BETTER = {"comprehensiveness": True, "sufficiency": False}
_METRIC_SUB = {
    "comprehensiveness": "delete the top-ranked features — how far the predicted-class "
                         "probability falls · higher is better",
    "sufficiency": "keep only the top-ranked features — how far it still falls · "
                   "lower is better",
}
_TASK_NAME = {"tweet": "Tweet Sentiment", "movie": "Movie Review Sentiment",
              "glasses": "Glasses Detection"}


def faithfulness_bars(faith: dict, metric: str = "comprehensiveness") -> str:
    """Measured faithfulness per method, faceted by task.

    `random` is a baseline rather than a fourth series, so it is drawn as a
    labelled reference line instead of a bar: the margin over random ranking
    *is* the evidence, and a bar would invite reading it as a competitor.
    """
    tasks = [(k, v) for k, v in faith.items() if (v.get("methods") or {})]
    if not tasks:
        return ""

    vmax = max(
        (m.get(metric) or 0.0)
        for _, v in tasks for m in v["methods"].values()
    ) or 1.0

    w = 700
    left, right = 118, 132
    row_h, bar_h, facet_gap = 24, 17, 26
    pw = w - left - right
    body, y = [], 68

    for key, v in tasks:
        methods = v["methods"]
        rand = (methods.get("random") or {}).get(metric)
        body.append(_txt(0, y, _TASK_NAME.get(key, key), size=10.5, fill=INK, weight="600"))
        y += 10
        top = y
        for m, entry in methods.items():
            if m == "random":
                continue
            val = entry.get(metric) or 0.0
            bw = (val / vmax) * pw
            cy = y + (row_h - bar_h) / 2
            body.append(_txt(left - 10, y + row_h / 2 + 3.5, m, size=10, anchor="end"))
            if bw > 0.5:
                r = min(4, bw / 2)
                body.append(
                    f'<path d="M{left} {cy} H{left + bw - r} a{r} {r} 0 0 1 {r} {r} '
                    f'V{cy + bar_h - r} a{r} {r} 0 0 1 {-r} {r} H{left} Z" '
                    f'fill="{METHOD_COLOUR.get(m, SERIES[0])}"/>'
                )
            lbl = f"{val:.3f}"
            if rand:
                lbl += f"  ·  {val / rand:.2f}× random"
            body.append(_txt(left + bw + 8, y + row_h / 2 + 3.5, lbl, size=9.5, fill=INK_2))
            y += row_h
        # Baseline: where a random ranking of the same features lands.
        if rand:
            rx = left + (rand / vmax) * pw
            body.append(
                f'<line x1="{rx:.1f}" y1="{top - 2}" x2="{rx:.1f}" y2="{y - 2}" '
                f'stroke="{BASELINE}" stroke-width="1.25" stroke-dasharray="3 3"/>'
            )
            body.append(_txt(rx, top - 6, f"random {rand:.3f}", size=8.5,
                             fill=BASELINE, anchor="middle"))
        y += facet_gap

    body.append(
        f'<line x1="{left}" y1="62" x2="{left}" y2="{y - facet_gap + 2}" '
        f'stroke="{RULE}" stroke-width="1"/>'
    )
    return _frame(
        w, int(y + 4), "".join(body),
        f"Faithfulness — {metric}",
        _METRIC_SUB.get(metric, ""),
    )


# --------------------------------------------------------------------------
# The cross-dataset finding
# --------------------------------------------------------------------------


def budget_vs_reliability(budget: dict) -> str:
    """Faithfulness against how many model evaluations each feature received.

    The first scatter in this module, so it is also the first figure needing a
    two-sided axis: `training_curve` maps only Y because its X is categorical.
    Gradient methods are excluded rather than given an invented x — they have no
    per-feature sampling budget — and appear instead as horizontal references.
    """
    pts = [p for p in budget.get("points", [])
           if p.get("budget_kind") == "perturbation" and p.get("evals_per_feature")
           and p.get("ratio_vs_random")]
    if len(pts) < 2:
        return ""
    grads = [p for p in budget.get("points", [])
             if p.get("budget_kind") == "gradient" and p.get("ratio_vs_random")]

    w, h = 700, 340
    left, right, top, bottom = 58, 96, 78, 54
    pw, ph = w - left - right, h - top - bottom

    lx0, lx1 = 0.0, 2.0                      # 1 .. 100 evaluations per feature
    ylo, yhi = 1.0, 3.8

    def X(v: float) -> float:
        return left + (math.log10(v) - lx0) / (lx1 - lx0) * pw

    def Y(v: float) -> float:
        return top + (yhi - v) / (yhi - ylo) * ph

    body = []
    # Y gridlines and ticks.
    for gv in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
        gy = Y(gv)
        body.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{left + pw}" y2="{gy:.1f}" '
                    f'stroke="{RULE}" stroke-width="1"/>')
        body.append(_txt(left - 8, gy + 3.5, f"{gv:g}×", size=9.5, anchor="end"))
    # X ticks on a log scale.
    for tv in (1, 2, 5, 10, 20, 50, 100):
        tx = X(tv)
        body.append(f'<line x1="{tx:.1f}" y1="{top}" x2="{tx:.1f}" y2="{top + ph}" '
                    f'stroke="{RULE}" stroke-width="1"/>')
        body.append(_txt(tx, top + ph + 16, tv, size=9.5, anchor="middle"))
    body.append(_txt(left + pw / 2, h - 6, "model evaluations per input feature (log scale)",
                     size=10, anchor="middle"))

    # Gradient methods first, so points and their labels sit on top of them.
    for p in grads:
        gy = Y(p["ratio_vs_random"])
        body.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{left + pw}" y2="{gy:.1f}" '
                    f'stroke="{TASK_COLOUR.get(p["task"], SERIES[2])}" stroke-width="1.25" '
                    f'stroke-dasharray="1 3"/>')
        body.append(_txt(left + pw + 6, gy + 3.5,
                         f'{p["method"]} {p["ratio_vs_random"]:.2f}×', size=9,
                         fill=TASK_COLOUR.get(p["task"], INK_2), weight="600"))
        body.append(_txt(left + pw + 6, gy + 14, "gradient — no budget", size=8, fill=INK_2))

    # Reference marks: the random floor, and the declared threshold.
    body.append(f'<line x1="{left}" y1="{Y(1.0):.1f}" x2="{left + pw}" y2="{Y(1.0):.1f}" '
                f'stroke="{BASELINE}" stroke-width="1.5"/>')
    body.append(_txt(left + pw + 6, Y(1.0) + 3.5, "random", size=9, fill=BASELINE))
    thr = budget.get("threshold_evals_per_feature")
    if thr:
        tx = X(thr)
        body.append(f'<line x1="{tx:.1f}" y1="{top}" x2="{tx:.1f}" y2="{top + ph}" '
                    f'stroke="{BASELINE}" stroke-width="1.25" stroke-dasharray="5 4"/>')
        body.append(_txt(tx + 5, top + 11, f"≈{thr:g} evals per feature",
                         size=9, fill=BASELINE))

    # Points. The label carries the method and the ratio; the task is carried by
    # colour and named once in the legend, which keeps labels short enough that
    # the two close movie points do not collide.
    placed: list[float] = []
    for p in sorted(pts, key=lambda q: Y(q["ratio_vs_random"])):
        px, py = X(p["evals_per_feature"]), Y(p["ratio_vs_random"])
        colour = TASK_COLOUR.get(p["task"], SERIES[0])
        body.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{colour}" '
                    f'stroke="{SURFACE}" stroke-width="2"/>')
        ly = py + 3.5
        for prev in placed:
            if abs(ly - prev) < 13:
                ly = prev + 13
        placed.append(ly)
        flip = px > left + pw * 0.80
        body.append(_txt(px - 9 if flip else px + 9, ly,
                         f'{p["method"]} {p["ratio_vs_random"]:.2f}×',
                         size=9.5, fill=INK, weight="600",
                         anchor="end" if flip else "start"))

    body.insert(0, _legend(0, 46, [(TASK_COLOUR[t], t) for t in ("tweet", "movie", "glasses")]))
    return _frame(w, h, "".join(body),
                  "How much budget does an explanation need?",
                  "faithfulness against evaluations per feature · perturbation methods only")


# --------------------------------------------------------------------------
# What each method names
# --------------------------------------------------------------------------


def token_composition(behaviour: dict) -> str:
    """Composition of each method's top-ranked tokens, faceted by task.

    Grouped rather than stacked: six categories exceed the palette's three
    honest slots, and the question a reader brings here is within-category
    ("does LIME name more content words than SHAP?"), which grouping answers.
    """
    tasks = [(k, v["token_composition"]) for k, v in behaviour.items()
             if isinstance(v, dict) and v.get("token_composition", {}).get("methods")]
    if not tasks:
        return ""

    cats = tasks[0][1]["categories"]
    w = 700
    left, right = 132, 62
    bar_h, gap, group_gap, facet_gap = 11, 2, 12, 24
    pw = w - left - right
    body, y = [], 68

    for key, comp in tasks:
        methods = [m for m in ("lime", "shap") if m in comp["methods"]]
        body.append(_txt(0, y, _TASK_NAME.get(key, key), size=10.5, fill=INK, weight="600"))
        y += 12
        top = y
        for cat in cats:
            body.append(_txt(left - 10, y + bar_h + 2, cat, size=9.5, anchor="end"))
            for m in methods:
                val = comp["methods"][m].get(cat, 0.0)
                bw = (val / 100.0) * pw
                if bw > 0.5:
                    r = min(3, max(bw, 1) / 2)
                    body.append(
                        f'<path d="M{left} {y} H{left + bw - r} a{r} {r} 0 0 1 {r} {r} '
                        f'V{y + bar_h - r} a{r} {r} 0 0 1 {-r} {r} H{left} Z" '
                        f'fill="{METHOD_COLOUR[m]}"/>'
                    )
                body.append(_txt(left + bw + 6, y + bar_h - 1.5, f"{val:.1f}%",
                                 size=8.5, fill=INK_2))
                y += bar_h + gap
            y += group_gap - gap
        body.append(f'<line x1="{left}" y1="{top - 4}" x2="{left}" y2="{y - group_gap + 2}" '
                    f'stroke="{RULE}" stroke-width="1"/>')
        y += facet_gap

    body.insert(0, _legend(left, 46, [(METHOD_COLOUR[m], m.upper())
                                      for m in ("lime", "shap")]))
    return _frame(w, int(y + 4), "".join(body),
                  "What kind of feature does each method name?",
                  "share of each method's top-5 tokens, by category")


# --------------------------------------------------------------------------
# Error detection
# --------------------------------------------------------------------------


def error_detection(behaviour: dict) -> str:
    """How well each signal separates the model's errors, faceted by task.

    Drawn as dots on a stem from the chance line rather than bars: AUC has a
    meaningful floor at 0.5, so bars from zero would compress the entire range
    of interest into the last fifth of the axis. Model confidence is a baseline,
    not a fourth method, and is coloured as one.
    """
    tasks = [(k, v["error_detection"]) for k, v in behaviour.items()
             if isinstance(v, dict) and (v.get("error_detection") or {}).get("signals")]
    if not tasks:
        return ""

    w = 700
    left, right = 150, 104
    row_h, facet_gap = 22, 30
    pw = w - left - right
    lo, hi = 0.5, 0.85

    def X(v: float) -> float:
        return left + (min(max(v, lo), hi) - lo) / (hi - lo) * pw

    body, y = [], 68
    for key, ed in tasks:
        body.append(_txt(0, y, f'{_TASK_NAME.get(key, key)} · {ed["n_errors"]} errors',
                         size=10.5, fill=INK, weight="600"))
        y += 18
        top = y
        for m, sig in ed["signals"].items():
            auc = sig["roc_auc"]
            cx, cy = X(auc), y + row_h / 2
            colour = BASELINE if m == "confidence" else METHOD_COLOUR.get(m, SERIES[0])
            body.append(_txt(left - 10, cy + 3.5, sig["label"], size=9.5, anchor="end"))
            body.append(f'<line x1="{left}" y1="{cy:.1f}" x2="{cx:.1f}" y2="{cy:.1f}" '
                        f'stroke="{colour}" stroke-width="2"/>')
            body.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{colour}" '
                        f'stroke="{SURFACE}" stroke-width="2"/>')
            body.append(_txt(cx + 10, cy + 3.5, f"{auc:.3f}", size=9.5, fill=INK_2))
            y += row_h
        # Chance line is the anchor the stems grow from, so it must be visible.
        body.append(f'<line x1="{left}" y1="{top - 2}" x2="{left}" y2="{y - 2}" '
                    f'stroke="{BASELINE}" stroke-width="1.5"/>')
        body.append(_txt(left, top - 6, "0.50 — chance", size=8.5, fill=BASELINE,
                         anchor="middle"))
        y += facet_gap

    for tv in (0.6, 0.7, 0.8):
        body.append(_txt(X(tv), y - facet_gap + 14, f"{tv:g}", size=9, anchor="middle"))
    body.append(_txt(left + pw / 2, y - facet_gap + 30,
                     "ROC AUC — separating the model's errors from its correct predictions",
                     size=9.5, anchor="middle"))
    return _frame(w, int(y - facet_gap + 40), "".join(body),
                  "Can an explanation flag the model's own mistakes?",
                  "higher is better · 0.5 is chance")


FIG_CSS = """
NOTE: dead. The .fig rules live in report.py's _HEAD block, which is the single
home for this document's CSS. Add new figure styles there, not here.
"""

__all__ = [
    "training_curve", "cost_small_multiples", "quality_by_correctness",
    "faithfulness_bars", "budget_vs_reliability", "token_composition",
    "error_detection",
]
