"""Rendering of individual explanations to human-viewable files.

Text predictions become a self-contained HTML card; image predictions become
a single multi-panel PNG placing the original beside every saliency map.
Both are deliberately one file per prediction, so any single decision the
model made can be pulled up and inspected on its own.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

from . import layout
from .explainers.base import Prediction

# Diverging colours: teal supports the prediction, amber opposes it. Chosen to
# stay distinguishable in greyscale and for the most common colour-vision
# deficiencies, unlike the red/green pairing the original notebooks used.
_POS = (13, 115, 119)
_NEG = (178, 94, 12)


def _chip(token: str, weight: float, scale: float) -> str:
    """One token rendered with opacity proportional to its attribution."""
    a = min(abs(weight) / scale, 1.0) if scale > 0 else 0.0
    r, g, b = _POS if weight >= 0 else _NEG
    style = (
        f"background:rgba({r},{g},{b},{a * 0.75:.3f});"
        f"padding:1px 3px;border-radius:3px;"
    )
    if a > 0.55:
        style += "color:#fff;"
    return f'<span style="{style}" title="{weight:+.4f}">{html.escape(token)}</span>'


def _token_block(tokens: list, title: str, note: str = "", top_k: int = 12) -> str:
    """Render the strongest ``top_k`` attributions.

    Records store the full attribution vector; only the display is truncated.
    """
    if not tokens:
        return f"<h3>{title}</h3><p class='muted'>no attributions</p>"
    n_total = len(tokens)
    tokens = sorted(tokens, key=lambda tw: -abs(tw[1]))[:top_k]
    if n_total > len(tokens):
        note = (note + " &middot; " if note else "") + \
               f"showing {len(tokens)} of {n_total} tokens"
    scale = max(abs(w) for _, w in tokens) or 1.0
    chips = " ".join(_chip(t, w, scale) for t, w in tokens)
    rows = "".join(
        f"<tr><td>{html.escape(str(t))}</td>"
        f"<td class='num' style='color:rgb{_POS if w >= 0 else _NEG}'>{w:+.4f}</td></tr>"
        for t, w in tokens
    )
    return (
        f"<h3>{title}</h3>{f'<p class=muted>{note}</p>' if note else ''}"
        f"<p class='chips'>{chips}</p>"
        f"<table class='attrs'><tr><th>token</th><th>weight</th></tr>{rows}</table>"
    )


_CSS = """
:root{color-scheme:light dark}
body{font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
 max-width:860px;margin:0 auto;padding:24px 18px;background:#fbfaf8;color:#1a1a1a}
@media(prefers-color-scheme:dark){body{background:#16181a;color:#e8e6e3}
 .card{background:#1f2225!important;border-color:#33383d!important}
 th{border-color:#33383d!important}}
h1{font-size:19px;margin:0 0 4px}h3{font-size:14px;margin:20px 0 6px}
.muted{color:#77716b;font-size:12px;margin:2px 0}
.card{background:#fff;border:1px solid #e6e2dc;border-radius:8px;padding:14px 16px;margin:14px 0}
.verdict{display:inline-block;padding:2px 9px;border-radius:99px;font-weight:600;font-size:12px}
.ok{background:#d7efe4;color:#12603f}.bad{background:#fadfd5;color:#8a2f10}
.chips{line-height:2.1}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin-top:8px}
th,td{text-align:left;padding:3px 8px;border-bottom:1px solid #eeebe6}
th{font-weight:600;color:#77716b;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.num{text-align:right;font-variant-numeric:tabular-nums}
.bar{height:5px;border-radius:3px;background:#0d7377;display:inline-block;vertical-align:middle}
blockquote{margin:6px 0;padding:6px 12px;border-left:3px solid #d8d3cb;color:#555;font-size:13px}
"""


def render_text_prediction(pred: Prediction, out_dir: Path, stem: str | None = None) -> Path:
    """Write one self-contained HTML card for a text prediction.

    `stem` carries the input row index when the caller knows it, so that a
    directory listing reproduces the order of the input file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem or pred.sample_id}.html"

    verdict = "ok" if pred.correct else "bad"
    vtext = "correct" if pred.correct else f"wrong (true: {pred.true_label})"
    probs = "".join(
        f"<tr><td>{html.escape(k)}</td><td class='num'>{v:.4f}</td>"
        f"<td><span class='bar' style='width:{v * 160:.0f}px'></span></td></tr>"
        for k, v in sorted(pred.probs.items(), key=lambda kv: -kv[1])
    )

    parts = [
        f"<h1>{html.escape(pred.task)} &mdash; {html.escape(pred.sample_id)}</h1>",
        f"<p class='muted'>predicted <b>{html.escape(pred.pred_label)}</b> "
        f"at {pred.confidence:.1%} confidence &middot; "
        f"<span class='verdict {verdict}'>{html.escape(vtext)}</span></p>",
        f"<div class='card'><h3>input</h3><blockquote>{html.escape(pred.text or '')}</blockquote>"
        f"<table>{probs}</table></div>",
    ]

    for method, label, note in (
        ("lime", "LIME", "local linear surrogate over word perturbations"),
        ("shap", "SHAP", "Shapley values under a tokenizer-aware partition masker"),
    ):
        e = pred.explanations.get(method)
        if e is None:
            continue
        if e.error:
            parts.append(f"<div class='card'><h3>{label}</h3><p class='muted'>failed: {html.escape(e.error)}</p></div>")
            continue
        extra = f"{note} &middot; {e.runtime_s:.2f}s"
        if "surrogate_r2" in e.stats:
            extra += f" &middot; surrogate R²={e.stats['surrogate_r2']:.3f}"
        parts.append(f"<div class='card'>{_token_block(e.tokens, label, extra)}</div>")

    proto = pred.explanations.get("prototype")
    if proto is not None and not proto.error:
        rows = "".join(
            f"<div><p class='muted'>#{n['rank']} &middot; similarity {n['similarity']:.4f} "
            f"&middot; label <b>{html.escape(n['label'])}</b>"
            f"{' ✓' if n['agrees_with_prediction'] else ' ✗'}</p>"
            f"<blockquote>{html.escape(n['text'])}</blockquote></div>"
            for n in proto.neighbours
        )
        agree = proto.stats.get("neighbour_agreement", 0)
        parts.append(
            f"<div class='card'><h3>Prototypes</h3>"
            f"<p class='muted'>nearest training examples in [CLS] space &middot; "
            f"{agree:.0%} of neighbours carry the predicted label</p>{rows}</div>"
        )

    doc = f"<!doctype html><meta charset=utf-8><title>{html.escape(pred.sample_id)}</title><style>{_CSS}</style>" + "".join(parts)
    path.write_text(doc, encoding="utf-8")
    return path


def token_preview(tokens: list, k: int = 10) -> list:
    """The k strongest attributions, for callers that want a short list."""
    return sorted(tokens, key=lambda tw: -abs(tw[1]))[:k]


def render_image_prediction(
    pred: Prediction,
    image: np.ndarray,
    saliencies: dict[str, np.ndarray],
    out_dir: Path,
    dpi: int = 70,
) -> Path:
    """Write one multi-panel PNG: original image beside each saliency map."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Mirror the input tree: Dataset/Glasses/image/neg/00012.jpg is explained
    # at explanations/neg/00012.{png,html}.
    rel = layout.image_stem(pred.sample_id)
    (out_dir / rel.parent).mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{rel}.png"

    # Only the attribution maps get a panel; auxiliary arrays such as
    # lime_segments are stored but not plotted.
    order = [m for m in ("gradcam", "shap", "lime") if m in saliencies]
    fig, axes = plt.subplots(1, len(order) + 1, figsize=(2.5 * (len(order) + 1), 2.8))
    axes = np.atleast_1d(axes)

    axes[0].imshow(image)
    axes[0].set_title(
        f"{pred.sample_id}\ntrue: {pred.true_label}", fontsize=8.5, loc="left"
    )
    axes[0].axis("off")

    for ax, method in zip(axes[1:], order):
        sal = saliencies[method]
        ax.imshow(image)
        if method == "gradcam":
            # Already normalised to [0,1]; a sequential map reads as intensity.
            ax.imshow(sal, cmap="inferno", alpha=0.5)
        else:
            # Signed attributions: symmetric limits so zero stays neutral.
            lim = np.percentile(np.abs(sal), 99) or 1e-9
            ax.imshow(sal, cmap="coolwarm", alpha=0.55, vmin=-lim, vmax=lim)
        e = pred.explanations.get(method)
        rt = f"{e.runtime_s:.2f}s" if e else ""
        ax.set_title(f"{method}  {rt}", fontsize=8.5, loc="left")
        ax.axis("off")

    verdict = "correct" if pred.correct else f"WRONG (true {pred.true_label})"
    fig.suptitle(
        f"pred: {pred.pred_label}  ({pred.confidence:.1%})   —   {verdict}",
        fontsize=9.5, y=0.02, x=0.5,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pil_kwargs={"optimize": True})
    plt.close(fig)
    render_image_card(pred, path)
    return path


def render_image_card(pred: Prediction, panel: Path) -> Path:
    """Write the HTML card that accompanies an image prediction's panel.

    Text predictions have always been browsable as HTML; images were only a PNG,
    so the per-method numbers behind the picture — runtime, focus, peak location,
    convergence delta — were visible only inside the JSONL. This puts every
    prediction, in either modality, one click from a readable page.
    """
    path = panel.with_suffix(".html")
    verdict = "ok" if pred.correct else "bad"
    vtext = "correct" if pred.correct else f"wrong (true: {pred.true_label})"
    probs = "".join(
        f"<tr><td>{html.escape(k)}</td><td class='num'>{v:.4f}</td>"
        f"<td><span class='bar' style='width:{v * 160:.0f}px'></span></td></tr>"
        for k, v in sorted(pred.probs.items(), key=lambda kv: -kv[1])
    )

    rows = []
    for method in ("gradcam", "shap", "lime"):
        e = pred.explanations.get(method)
        if e is None:
            continue
        if e.error:
            rows.append(f"<tr><td>{method}</td><td colspan='4' class='muted'>"
                        f"failed: {html.escape(e.error)}</td></tr>")
            continue
        s = e.stats or {}
        peak = s.get("peak_yx")
        extras = []
        if "surrogate_r2" in s:
            extras.append(f"R²={s['surrogate_r2']:.3f}")
        if "convergence_delta" in s:
            extras.append(f"Δ={s['convergence_delta']:.3f}")
        if "n_superpixels" in s:
            extras.append(f"{s['n_superpixels']} superpixels")
        rows.append(
            f"<tr><td><b>{method}</b></td>"
            f"<td class='num'>{e.runtime_s:.2f}s</td>"
            f"<td class='num'>{s.get('top5pct_mass', float('nan')):.3f}</td>"
            f"<td class='num'>{f'{peak[0]}, {peak[1]}' if peak else '—'}</td>"
            f"<td class='muted'>{html.escape(' · '.join(extras))}</td></tr>"
        )

    doc = (
        f"<!doctype html><meta charset=utf-8>"
        f"<title>{html.escape(pred.sample_id)}</title><style>{_CSS}</style>"
        f"<h1>{html.escape(pred.task)} &mdash; {html.escape(pred.sample_id)}</h1>"
        f"<p class='muted'>predicted <b>{html.escape(pred.pred_label)}</b> at "
        f"{pred.confidence:.1%} confidence &middot; "
        f"<span class='verdict {verdict}'>{html.escape(vtext)}</span></p>"
        f"<div class='card'><h3>saliency</h3>"
        f"<img src='{html.escape(panel.name)}' alt='saliency panels for "
        f"{html.escape(pred.sample_id)}' style='width:100%;border-radius:6px'>"
        f"<p class='muted'>Grad-CAM is unsigned intensity; SHAP and LIME are signed and "
        f"use a diverging scale centred on zero. Full-resolution arrays for every method "
        f"are in <code>saliency/{html.escape(str(layout.image_stem(pred.sample_id)))}/</code>, "
        f"one <code>.npy</code> per method.</p></div>"
        f"<div class='card'><h3>per-method</h3><table>"
        f"<tr><th>method</th><th>runtime</th><th>top-5% mass</th><th>peak (y,x)</th>"
        f"<th></th></tr>{''.join(rows)}</table></div>"
        f"<div class='card'><h3>model output</h3><table>{probs}</table></div>"
    )
    path.write_text(doc, encoding="utf-8")
    return path


def build_index(task_key: str, records: list[dict], out_dir: Path) -> Path:
    """One page listing every input, in input order, with each method's outcome.

    The per-prediction cards answer "what did the methods say about this input?".
    This answers the complementary question — "is every input actually covered,
    by every method?" — which is otherwise only checkable by counting files.
    Rows are emitted in the order the records were written, which is the order of
    the input file.
    """
    methods = sorted({m for r in records for m in r.get("explanations", {})})
    n_wrong = sum(1 for r in records if not r["correct"])

    rows = []
    for i, r in enumerate(records, 1):
        card = None
        for e in r.get("explanations", {}).values():
            if e.get("render"):
                card = e["render"]
                break
        sid = html.escape(r["sample_id"])
        label = (f"<a href='{html.escape(card)}'>{sid}</a>" if card else sid)
        cells = []
        for m in methods:
            e = r.get("explanations", {}).get(m)
            if e is None:
                cells.append("<td class='miss'>absent</td>")
            elif e.get("error"):
                cells.append(f"<td class='miss' title='{html.escape(str(e['error']))}'>error</td>")
            else:
                cells.append(f"<td class='ok'>{e.get('runtime_s', 0):.2f}s</td>")
        verdict = "ok" if r["correct"] else "bad"
        vtext = "correct" if r["correct"] else f"wrong ({html.escape(r['true_label'])})"
        rows.append(
            f"<tr><td class='num'>{i}</td><td>{label}</td>"
            f"<td>{html.escape(r['pred_label'])}</td>"
            f"<td class='num'>{r['confidence']:.3f}</td>"
            f"<td><span class='verdict {verdict}'>{vtext}</span></td>"
            + "".join(cells) + "</tr>"
        )

    head = "".join(f"<th>{html.escape(m)}</th>" for m in methods)
    doc = (
        f"<!doctype html><meta charset=utf-8><title>{html.escape(task_key)} — all predictions</title>"
        f"<style>{_CSS}"
        "table{width:100%;border-collapse:collapse;font-size:13px}"
        "th{position:sticky;top:0;background:Canvas;text-align:left;padding:6px 8px;"
        "border-bottom:2px solid currentColor;font-size:11px;text-transform:uppercase;"
        "letter-spacing:.05em;opacity:.65}"
        "td{padding:4px 8px;border-bottom:1px solid rgba(128,128,128,.2)}"
        "td.num{text-align:right;font-variant-numeric:tabular-nums;opacity:.6}"
        "td.ok{font-variant-numeric:tabular-nums;opacity:.75}"
        "td.miss{color:#b25e0c;font-weight:600}"
        "</style>"
        f"<h1>{html.escape(task_key)} — every prediction</h1>"
        f"<p class='muted'>{len(records):,} inputs, listed in the order of the input file. "
        f"Each carries {len(methods)} explanation{'s' if len(methods) != 1 else ''} "
        f"({html.escape(', '.join(methods))}); {n_wrong:,} predictions are wrong. "
        f"Click a sample id for its card.</p>"
        f"<table><tr><th>#</th><th>sample</th><th>predicted</th><th>conf.</th>"
        f"<th>verdict</th>{head}</tr>{''.join(rows)}</table>"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "index.html"
    path.write_text(doc, encoding="utf-8")
    return path


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


__all__ = ["render_text_prediction", "render_image_prediction", "write_jsonl"]
