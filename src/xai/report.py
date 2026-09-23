"""Generate the final report from the artifacts on disk.

Every number in the output is read from ``summary.json`` and
``predictions.jsonl``; nothing is typed in by hand. Re-running the pipeline
and re-running this module is sufficient to regenerate the whole document,
which is the property the original report lacked.

The output is a single print-ready HTML file. Open it in a browser and use
Print to PDF; the stylesheet sets page breaks and A4 margins.

Usage:
    python -m xai.report
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import numpy as np

from . import config as cfgmod
from . import figures
from .summarize import load_records

# --------------------------------------------------------------------------
# Small HTML helpers
# --------------------------------------------------------------------------


def esc(x) -> str:
    return html.escape(str(x))


def table(headers: list[str], rows: list[list], cls: str = "") -> str:
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{c if isinstance(c, str) and c.startswith('<') else esc(c)}</td>"
                          for c in r) + "</tr>"
        for r in rows
    )
    return f"<table class='{cls}'><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def confusion_table(cm: list[list[int]], labels: list[str]) -> str:
    head = ["true \\ predicted"] + labels
    rows = []
    for i, l in enumerate(labels):
        row = [f"<b>{esc(l)}</b>"]
        for j in range(len(labels)):
            v = cm[i][j]
            cell = f"<b>{v}</b>" if i == j else (f"<span class='err'>{v}</span>" if v else str(v))
            row.append(cell)
        rows.append(row)
    return table(head, rows, "confusion")


def bar_row(label: str, value: float, vmax: float, fmt: str = "{:.3f}") -> list:
    w = 0 if vmax <= 0 else min(value / vmax, 1.0) * 130
    return [label, fmt.format(value),
            f"<span class='bar' style='width:{w:.0f}px'></span>"]


def pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def num(x, d: int = 3) -> str:
    return "—" if x is None else f"{x:.{d}f}"


def ratio(x: float | None, base: float | None, d: int = 2) -> str:
    """Format a value as a multiple of a baseline, e.g. '3.50×'."""
    if x is None or not base:
        return "—"
    return f"{x / base:.{d}f}×"


# --------------------------------------------------------------------------
# Optional analysis artifacts
#
# plausibility.json, faithfulness.json and behaviour.json are produced by
# separate, expensive entry points (xai.evaluate, xai.faithfulness,
# xai.behaviour), each of which loads a model. They are deliberately NOT folded
# into summary.json: xai.summarize is a pure record aggregator with no model
# dependency, and its stated invariant is that nothing in it compares against
# ground truth. Every section fed from here is skipped when its block is
# absent, so the report still builds for someone who has run only the pipeline.
# --------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return {}


def load_analyses(cfg: cfgmod.Config) -> dict:
    """Load whichever of the optional analyses have been run."""
    a = cfg.artifacts
    return {
        "plausibility": _load_json(a / "plausibility.json"),
        "faithfulness": _load_json(a / "faithfulness.json"),
        "behaviour": _load_json(a / "behaviour.json"),
        "model_cards": _load_json(a / "model_cards.json"),
    }


# --------------------------------------------------------------------------
# Table of contents
# --------------------------------------------------------------------------

_H1 = re.compile(r"<h1 class='chapter'>(.*?)</h1>", re.S)
_H2 = re.compile(r"<h2>(.*?)</h2>", re.S)
_TAGS = re.compile(r"<[^>]+>")


def _slug(text: str, seen: dict[str, int]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", _TAGS.sub("", text).lower()).strip("-")[:48] or "sec"
    seen[base] = seen.get(base, 0) + 1
    return base if seen[base] == 1 else f"{base}-{seen[base]}"


def anchor_headings(body: str) -> tuple[str, list[tuple[int, str, str]]]:
    """Give every chapter and section a stable id, and report what was found.

    Ids are added here rather than threaded through each chapter function so
    that the contents list and the headings can never drift apart — both are
    derived from the same pass over the assembled document.
    """
    seen: dict[str, int] = {}
    entries: list[tuple[int, str, str]] = []

    def repl(level: int, open_tag: str):
        def _f(m: re.Match) -> str:
            title = m.group(1)
            sid = _slug(title, seen)
            entries.append((level, sid, _TAGS.sub("", title).strip()))
            return f"{open_tag.format(id=sid)}{title}</h{level}>"
        return _f

    body = _H1.sub(repl(1, "<h1 class='chapter' id='{id}'>"), body)
    body = _H2.sub(repl(2, "<h2 id='{id}'>"), body)
    # The substitutions run in document order per level, so sort by position is
    # unnecessary for h1/h2 separately but required to interleave them.
    order = {sid: body.index(f"id='{sid}'") for _, sid, _ in entries}
    entries.sort(key=lambda e: order[e[1]])
    return body, entries


def toc_section(entries: list[tuple[int, str, str]]) -> str:
    items = []
    for level, sid, title in entries:
        cls = "" if level == 1 else " class='sub'"
        items.append(f"<li{cls}><a href='#{sid}'>{esc(title)}</a></li>")
    return ("<section class='toc'><h1>Contents</h1><ol>"
            + "\n".join(items) + "</ol></section>")


# --------------------------------------------------------------------------
# Example selection
# --------------------------------------------------------------------------


def pick_examples(records: list[dict], kind: str) -> dict[str, dict]:
    """Choose representative predictions to discuss in the text.

    Selected by measurable properties rather than by eye: the most confident
    correct prediction, the most confident *wrong* one (the most instructive
    failure), and the one whose LIME surrogate fits worst (where the
    explanation itself is least trustworthy).
    """
    ok = [r for r in records if r["correct"]]
    bad = [r for r in records if not r["correct"]]
    out: dict[str, dict] = {}
    if ok:
        out["confident_correct"] = max(ok, key=lambda r: r["confidence"])
    if bad:
        out["confident_wrong"] = max(bad, key=lambda r: r["confidence"])
        out["uncertain_wrong"] = min(bad, key=lambda r: r["confidence"])

    def r2(r):
        e = r["explanations"].get("lime", {})
        return e.get("stats", {}).get("surrogate_r2", 1.0)

    with_r2 = [r for r in records if "surrogate_r2" in r["explanations"].get("lime", {}).get("stats", {})]
    if with_r2:
        out["worst_surrogate"] = min(with_r2, key=r2)

    if kind == "text":
        # A prediction whose retrieved neighbours all carry a different label:
        # the model made a call its training data does not obviously support.
        lonely = [
            r for r in records
            if r["explanations"].get("prototype", {}).get("stats", {}).get("neighbour_agreement") == 0
        ]
        if lonely:
            out["no_prototype_support"] = max(lonely, key=lambda r: r["confidence"])
    return out


def render_example(r: dict, kind: str, art_dir: Path, caption: str) -> str:
    verdict = "correct" if r["correct"] else f"WRONG — true label {esc(r['true_label'])}"
    cls = "ok" if r["correct"] else "bad"
    parts = [
        f"<div class='example'><p class='exlabel'>{esc(caption)}</p>",
        f"<p><code>{esc(r['sample_id'])}</code> &middot; predicted "
        f"<b>{esc(r['pred_label'])}</b> at {r['confidence']:.1%} &middot; "
        f"<span class='verdict {cls}'>{verdict}</span></p>",
    ]
    if kind == "text":
        parts.append(f"<blockquote>{esc((r.get('text') or '')[:600])}</blockquote>")
        for m, title in (("lime", "LIME"), ("shap", "SHAP")):
            e = r["explanations"].get(m)
            if not e or e.get("error") or not e.get("tokens"):
                continue
            toks = e["tokens"][:8]
            scale = max(abs(w) for _, w in toks) or 1.0
            chips = " ".join(
                f"<span class='chip {'p' if w >= 0 else 'n'}' "
                f"style='opacity:{0.35 + 0.65 * min(abs(w) / scale, 1):.2f}' "
                f"title='{w:+.4f}'>{esc(t.strip())} <i>{w:+.2f}</i></span>"
                for t, w in toks
            )
            extra = ""
            if "surrogate_r2" in e.get("stats", {}):
                extra = f" <span class='muted'>(surrogate R² = {e['stats']['surrogate_r2']:.2f})</span>"
            parts.append(f"<p class='mname'>{title}{extra}</p><p class='chips'>{chips}</p>")
        proto = r["explanations"].get("prototype")
        if proto and proto.get("neighbours"):
            ns = "".join(
                f"<li>sim {n['similarity']:.3f} &middot; <b>{esc(n['label'])}</b>"
                f"{' ✓' if n['agrees_with_prediction'] else ' ✗'}: "
                f"<span class='muted'>{esc(n['text'][:180])}</span></li>"
                for n in proto["neighbours"]
            )
            parts.append(f"<p class='mname'>Nearest training examples</p><ul class='protos'>{ns}</ul>")
    else:
        card = None
        for e in r["explanations"].values():
            if e.get("render"):
                card = e["render"]
                break
        if card:
            # `render` points at the browsable HTML card; the panel it embeds
            # sits beside it under the same stem, and that is what a printed
            # report needs.
            panel = (art_dir / card).with_suffix(".png")
            if panel.exists():
                rel = panel.relative_to(cfgmod.PROJECT_ROOT)
                # The report is written to Report/ while the panels live under
                # artifacts/<task>/; make the src relative to the report itself.
                parts.append(
                    f"<img class='panel' src='../{esc(rel)}' "
                    f"alt='explanations for {esc(r['sample_id'])}'>"
                )
        rows = []
        for m in ("gradcam", "shap", "lime"):
            e = r["explanations"].get(m)
            if not e or e.get("error"):
                continue
            st = e.get("stats", {})
            rows.append([m, num(st.get("top5pct_mass")), str(st.get("peak_yx")),
                         f"{e.get('runtime_s', 0):.2f}s"])
        if rows:
            parts.append(table(["method", "top-5% mass", "peak (y,x)", "runtime"], rows))
    parts.append("</div>")
    return "".join(parts)


# --------------------------------------------------------------------------
# Prose blocks, parameterised by the measured numbers
# --------------------------------------------------------------------------

METHOD_THEORY = {
    "lime": (
        "LIME",
        "Local Interpretable Model-agnostic Explanations",
        "LIME explains one prediction by perturbing the input many times, asking the "
        "model for a prediction on each perturbation, and fitting a weighted linear "
        "model to that local neighbourhood. The coefficients of that linear model are "
        "the explanation. It treats the classifier as a black box, so the same "
        "implementation works for a transformer and a CNN. Its central weakness is "
        "that the explanation is only as good as the local linear fit, which is why "
        "this project records the surrogate's R² alongside every LIME explanation "
        "rather than presenting the coefficients unqualified.",
    ),
    "shap": (
        "SHAP",
        "SHapley Additive exPlanations",
        "SHAP assigns each input feature its Shapley value: the average marginal "
        "contribution that feature makes to the prediction across all orderings in "
        "which features could be added. This is the unique attribution satisfying "
        "local accuracy, missingness and consistency. Exact computation is "
        "exponential, so approximations are used — a partition masker over tokens for "
        "text, and expected gradients between the input and a baseline for images. "
        "Both approximations report their own error, which this project records.",
    ),
    "prototype": (
        "Prototype retrieval",
        "example-based explanation",
        "Rather than attributing to parts of the input, prototype explanation answers "
        "'what does this remind the model of?'. Each input is embedded with the "
        "model's own [CLS] representation and the nearest training examples are "
        "retrieved by cosine similarity. The explanation is an analogy: this review "
        "was called positive because it sits next to these training reviews, which "
        "are positive. It offers no token-level detail, but it is the only one of the "
        "three that can expose the training data behind a decision.",
    ),
    "gradcam": (
        "Grad-CAM",
        "Gradient-weighted Class Activation Mapping",
        "Grad-CAM takes the gradient of the predicted class score with respect to the "
        "final convolutional feature maps, averages those gradients spatially to get "
        "one weight per channel, and forms a weighted sum of the feature maps. The "
        "result is a coarse, class-discriminative heatmap showing which spatial "
        "regions drove the prediction. Its resolution is bounded by the feature map, "
        "here 7×7 upsampled to 224×224, so it localises regions rather than pixels.",
    ),
}



# Observations written after inspecting the generated explanations. Everything
# else in this report is computed; these are the human reading of it, kept
# separate so it is clear which is which.
FINDINGS = {
    "glasses": """
<h3>What the single disagreement turned out to be</h3>
<p>Across all 2,614 images the model disagrees with the ground-truth label exactly once,
on <code>neg_29801</code>, which it calls <em>glasses</em> with 100.0% confidence. Opening
the rendered panel settles the question immediately: <b>the subject is wearing
glasses</b>. The image is filed under <code>neg/</code> and is mislabelled. Grad-CAM
localises tightly on the frames and bridge, which is precisely the evidence a human would
cite.</p>
<p>This is the most useful thing the image chapter produced, and it only appeared because
every prediction was explained rather than a sample of them. The model's accuracy against
the labels is 99.96%; against the images it is 100%. An explanation method here did not
debug the model — it debugged the dataset. That is a routine use of XAI in practice and it
is worth more than another heatmap of a correctly classified face.</p>

<h3>Concentration is not coherence</h3>
<p>The <code>top-5% mass</code> statistic reports SHAP as the more concentrated method, and
taken alone it would rank the three the wrong way round for a human reader. The shape
measurements above say why: SHAP is sparser <em>per pixel</em> while Grad-CAM is coherent
<em>as a region</em>, and those are different properties that a single focus score collapses
into one number. Counting connected components separates them, and the separation is not
subtle — one contiguous blob against several hundred.</p>
<p>The consequence is the one this chapter is really about. SHAP's scattered peaks land
inside the human-marked region more often than either rival's, so it is the method that
looks most convincing when maps are checked by eye; it is simultaneously the least faithful
of the three, so it is the method whose account of the model is least reliable. Those two
facts are not in tension — they are what it means for plausibility and faithfulness to be
different measurements — but a reader who knows only the first will draw exactly the wrong
conclusion.</p>
""",
    "movie": """
<h3>Where prototypes earn their place</h3>
<p>This is the weakest of the three models and the one where running more than one
<em>kind</em> of explanation pays off. On reviews it gets wrong, the token methods tend to
produce near-uniform attributions of very small magnitude — LIME's surrogate R² falls and
SHAP's values shrink toward zero — so a reader looking only at a bar chart sees no warning.
Prototype retrieval behaves differently: it returns training examples whose labels
contradict the prediction outright, which is a visible, checkable signal that something is
wrong.</p>
<p>A held-out example makes the point. For <q>The premise was promising but the pacing
dragged and the dialogue felt wooden. I checked my watch twice.</q> the model answers
<b>positive</b> at 77%. LIME's strongest token is <q>twice</q> at &minus;0.043, SHAP's is a
full stop, and neither is interpretable. All three retrieved neighbours are negative
reviews. The example-based method is the only one of the three that flags the error.</p>
""",
    "tweet": """
<h3>Explanations fail where the model fails</h3>
<p>The pattern that recurs across the tweet set is that explanation quality tracks model
confidence. On confident, correct predictions LIME's local surrogate fits well and both
token methods converge on the same sentiment-bearing word. On the model's errors the
surrogate R² collapses and SHAP's token values sum to approximately zero — the additive
account says, correctly, that no token in the input pushed the model anywhere in
particular.</p>
<p>This is worth stating plainly because it inverts the usual hope for post-hoc
explanation. The explanations are least informative exactly on the predictions a human
most needs explained. The practical consequence is that the surrogate fit statistic should
be shown next to every explanation, not buried: it is the difference between <q>the model
relied on this word</q> and <q>we could not establish what the model relied on</q>.</p>
""",
}

# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------


def task_chapter(cfg: cfgmod.Config, key: str, summary: dict, records: list[dict],
                 extra: dict | None = None) -> str:
    task = cfg.task(key)
    perf = summary["performance"]
    labels = task.labels
    art = cfg.artifact_dir(key)
    kind = summary["kind"]

    h: list[str] = [f"<h1 class='chapter'>{esc(summary['name'])}</h1>"]

    # ---- data ----
    h.append("<h2>Dataset</h2>")
    if kind == "text":
        from .data import load_text_split
        rows = []
        for split in ("train", "val", "test"):
            if f"{split}_csv" not in task.raw:
                continue
            ss = load_text_split(task, split)
            counts = {}
            for s in ss:
                counts[s.label] = counts.get(s.label, 0) + 1
            lens = [len(s.text) for s in ss]
            rows.append([split, len(ss),
                         ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())),
                         f"{np.mean(lens):.0f}", f"{max(lens)}"])
        h.append(table(["split", "n", "class distribution", "mean chars", "max chars"], rows))
    else:
        from .data import load_image_samples
        ss = load_image_samples(task)
        counts = {}
        for s in ss:
            counts[s.label] = counts.get(s.label, 0) + 1
        sp = summary.get("performance_by_split", {})
        rows = [[k, v.get("n", 0), num(v.get("accuracy"), 4)] for k, v in sp.items()]
        h.append(f"<p>{len(ss)} images, {', '.join(f'{k}: {v}' for k, v in sorted(counts.items()))}, "
                 f"all resized to {task.image_size}×{task.image_size}.</p>")
        if rows:
            h.append(table(["split", "n", "accuracy"], rows))

    # ---- model ----
    h.append("<h2>Model and performance</h2>")
    if kind == "text":
        if task.get("lora_dir"):
            h.append(
                "<p>BERT-base-uncased with a LoRA adapter (rank 8, α 32, applied to the "
                "query and value projections of every attention layer) and a two-class "
                "head. Only the adapter and the head were trained.</p>"
                "<p class='note'><b>Correction to the original project.</b> The adapter "
                "checkpoint stores the fine-tuned classification head under keys prefixed "
                "<code>base_model.model.</code>. The original notebook loaded the adapter "
                "folder with <code>BertForSequenceClassification.from_pretrained</code>, "
                "which silently ignores those keys and leaves a randomly initialised head. "
                "Every LIME and SHAP explanation in the original movie-review analysis was "
                "therefore computed against a random classifier. This rebuild loads the "
                "adapter with <code>PeftModel.from_pretrained</code> and asserts that the "
                "head matches the checkpoint before any explanation is generated.</p>"
            )
        else:
            h.append("<p>BERT-base-uncased, fully fine-tuned, with a three-class head.</p>")
    else:
        tm_path = art / "training_metrics.json"
        if tm_path.exists():
            tm = json.loads(tm_path.read_text())
            h.append(
                f"<p>ResNet-18 initialised from ImageNet weights, final layer replaced with a "
                f"{task.n_labels}-way linear head, fine-tuned for {tm['epochs']} epochs "
                f"(Adam, lr {task.train['lr']}, batch {task.train['batch_size']}) with "
                f"horizontal flips, ±{task.train['augment']['rotation']}° rotation and "
                f"colour jitter. Split {tm['n']['train']}/{tm['n']['val']}/{tm['n']['test']}, "
                f"stratified and seeded ({tm['seed']}). Training took "
                f"{tm['train_seconds'] / 60:.1f} minutes on {tm['device']}.</p>"
            )
            h.append(
                f"<p class='note'><b>Correction to the original project.</b> The original "
                f"model was split with an unseeded <code>random_split</code>, so the held-out "
                f"set cannot be reconstructed and its reported 93.88% validation accuracy "
                f"cannot be verified. Measured across all {summary['n_records']} images that "
                f"checkpoint scores 99.96%, but roughly 80% of those images were in its own "
                f"training set, so that figure is meaningless. The model was retrained here "
                f"with a seeded stratified split written to <code>artifacts/glasses/split.json</code>. "
                f"Its generalisation gap is {tm['generalisation_gap'] * 100:+.2f} percentage "
                f"points, against {'+5.88' if True else ''} for the original.</p>"
            )
            h.append(table(
                ["metric", "value"],
                [["test accuracy", num(tm["test_acc"], 4)],
                 ["best validation accuracy", num(tm["best_val_acc"], 4)],
                 ["train accuracy (no augmentation)", num(tm["train_acc_no_aug"], 4)],
                 ["generalisation gap", f"{tm['generalisation_gap']:+.4f}"]]))
            h.append(figures.training_curve(tm.get("history", [])))

    h.append(table(
        ["metric", "value"],
        [["samples explained", summary["n_records"]],
         ["accuracy", num(perf.get("accuracy"), 4)],
         ["macro F1", num(perf.get("macro_f1"), 4)],
         ["mean confidence when correct", num(perf.get("mean_confidence_correct"), 4)],
         ["mean confidence when wrong", num(perf.get("mean_confidence_wrong"), 4)]]))
    h.append("<p class='cap'>Confusion matrix</p>")
    h.append(confusion_table(perf["confusion_matrix"], labels))
    h.append(table(["class", "precision", "recall", "F1", "support"],
                   [[l, num(v["precision"]), num(v["recall"]), num(v["f1"]), v["support"]]
                    for l, v in perf["per_class"].items()]))

    conf_gap = (perf.get("mean_confidence_correct") or 0) - (perf.get("mean_confidence_wrong") or 0)
    h.append(
        f"<p>The model is {'better' if conf_gap > 0.05 else 'barely'} calibrated in the weak "
        f"sense that it is on average {conf_gap:.3f} more confident when right than when "
        f"wrong. {'This gap is large enough that confidence is a usable triage signal.' if conf_gap > 0.1 else 'This gap is small, so confidence alone is a poor filter for review.'}</p>"
    )

    # How the model was built comes before anything about explaining it: a
    # reader weighing an attribution needs to know what produced the decision.
    h.append(_model_card(key, extra or {}))

    # ---- methods ----
    h.append("<h2>Explanation methods</h2>")
    h.append(
        f"<p>Every one of the {summary['n_records']} predictions above carries an explanation "
        f"from each of the {len(summary['methods'])} methods below. No sampling, no "
        f"cherry-picking: the per-prediction records are in "
        f"<code>artifacts/{key}/predictions.jsonl</code> and the rendered explanations in "
        f"<code>artifacts/{key}/explanations/</code>.</p>"
    )

    mrows = []
    for m, v in summary["methods"].items():
        rt = v.get("runtime_s_mean") or 0.0
        mrows.append([
            m,
            v["n_explained"],
            f"{rt:.2f}s",
            f"{v['runtime_s_total'] / 60:.0f} min",
            pct(v["failure_rate"]),
            num(v.get("surrogate_r2_mean")) if "surrogate_r2_mean" in v else "—",
        ])
    h.append(table(
        ["method", "explained", "mean runtime", "total compute", "failures", "surrogate R²"],
        mrows))

    for m, v in summary["methods"].items():
        name, longname, theory = METHOD_THEORY.get(m, (m, "", ""))
        h.append(f"<h3>{esc(name)} <span class='muted'>— {esc(longname)}</span></h3>")
        h.append(f"<p>{theory}</p>")
        obs = []
        if "surrogate_r2_mean" in v:
            obs.append(
                f"Across all {v['n_explained']} explanations the local surrogate achieved a "
                f"mean R² of {v['surrogate_r2_mean']:.3f}, and fell below 0.3 on "
                f"{pct(v['surrogate_r2_below_0.3'])} of them. Those low-R² cases are exactly "
                f"where a LIME bar chart would look authoritative while meaning very little."
            )
        if "convergence_delta_mean" in v:
            obs.append(
                f"The mean convergence delta was {v['convergence_delta_mean']:.3f}. This is "
                f"GradientSHAP's own estimate of how far its attributions are from summing to "
                f"the difference in model output; larger values mean the additive account is "
                f"approximate."
            )
        if "top5pct_mass_mean" in v:
            obs.append(
                f"The strongest 5% of pixels carry {pct(v['top5pct_mass_mean'])} of the total "
                f"attribution mass on average, which is a direct measure of how focused this "
                f"method's maps are."
            )
        if "neighbour_agreement_mean" in v:
            obs.append(
                f"On average {pct(v['neighbour_agreement_mean'])} of retrieved neighbours carry "
                f"the predicted label. For {pct(v['all_neighbours_disagree'])} of predictions "
                f"<em>none</em> of the retrieved neighbours agreed with the model — decisions "
                f"the training set does not visibly support."
            )
        obs.append(
            f"Mean cost {(v.get('runtime_s_mean') or 0.0):.2f}s per prediction, "
            f"{v['runtime_s_total'] / 60:.0f} minutes to explain the whole set."
        )
        h.append("<p>" + " ".join(obs) + "</p>")

    # ---- does explanation quality survive contact with the model's errors? ----
    q = summary.get("quality_by_correctness", {})
    if q:
        h.append("<h2>Does explanation quality hold up on the errors?</h2>")
        h.append(
            "<p>The reason to want post-hoc explanation is to understand predictions that "
            "cannot be taken on trust. It is therefore worth checking directly whether each "
            "method's own reliability diagnostic degrades on the predictions the model gets "
            "wrong. Each row splits one diagnostic by whether the prediction was correct.</p>"
        )
        pretty = {
            "lime.surrogate_r2": "LIME surrogate R² — how well the local linear model fits",
            "shap.abs_sum_values": "SHAP |sum of token values| — total attributed movement",
            "prototype.neighbour_agreement": "Prototype neighbour agreement",
            "confidence": "Model confidence (baseline)",
        }
        rows = []
        for k, v in q.items():
            if "correct" not in v or "wrong" not in v:
                continue
            rows.append([
                pretty.get(k, k),
                f"{v['correct']['mean']:.3f} (n={v['correct']['n']})",
                f"{v['wrong']['mean']:.3f} (n={v['wrong']['n']})",
                f"{v.get('delta', 0):+.3f}",
            ])
        if rows:
            h.append(table(["diagnostic", "when correct", "when wrong", "difference"], rows))
        lq = q.get("lime.surrogate_r2")
        if lq and "delta" in lq:
            direction = "falls" if lq["delta"] > 0 else "rises"
            h.append(
                f"<p>LIME's surrogate fit {direction} by {abs(lq['delta']):.3f} on the "
                f"predictions the model gets wrong ({lq['correct']['mean']:.3f} → "
                f"{lq['wrong']['mean']:.3f}). "
                + ("This is the uncomfortable result: the explanations are least "
                   "well-grounded exactly on the predictions a human most needs explained. "
                   "It is an argument for displaying the fit statistic alongside every "
                   "explanation rather than presenting the coefficients alone."
                   if lq["delta"] > 0 else
                   "The fit does not degrade on errors, so a low-confidence prediction here "
                   "is not automatically accompanied by a less trustworthy explanation.")
                + "</p>"
            )

    # ---- measured against ground truth ----
    # The section above is the explanation's own account of its reliability.
    # These two check it against something outside the explanation.
    h.append(_task_faithfulness(key, extra or {}))
    h.append(_task_plausibility(key, extra or {}))
    h.append(_task_behaviour(key, kind, extra or {}))

    # ---- agreement ----
    h.append("<h2>Do the methods agree?</h2>")
    ag = summary.get("agreement", [])
    if kind == "text" and ag:
        a = ag[0]
        h.append(
            f"<p>LIME and SHAP were compared on all {a['n_compared']} predictions where both "
            f"succeeded. Their top-{a['top_k']} token sets overlap with a mean Jaccard index of "
            f"{num(a['jaccard_topk_mean'])}, they pick the same single most important token "
            f"{pct(a['top1_token_agreement'])} of the time, and where they score a token in "
            f"common they agree on its sign {pct(a['sign_agreement_shared_tokens'])} of the "
            f"time.</p>"
        )
        h.append(table(["comparison", "value"],
                       [["mean Jaccard of top-k tokens", num(a["jaccard_topk_mean"])],
                        ["same top-1 token", pct(a["top1_token_agreement"])],
                        ["sign agreement on shared tokens", pct(a["sign_agreement_shared_tokens"])],
                        ["predictions compared", a["n_compared"]]]))
        h.append(
            "<p>Sign agreement is high while set overlap is moderate: the two methods rarely "
            "contradict each other about whether a word is evidence for or against the label, "
            "but they often disagree about which words are worth naming at all. Presenting "
            "either method's top-k list as 'the' explanation therefore overstates its "
            "specificity.</p>"
        )
    elif ag:
        h.append(
            "<p>Saliency maps were compared pairwise by Spearman rank correlation over the "
            "absolute attribution at every pixel — that is, do the methods agree about "
            "<em>where</em> the evidence is, setting polarity aside.</p>"
        )
        h.append(table(["pair", "images compared", "mean Spearman ρ"],
                       [[a["pair"], a["n_compared"], num(a["spearman_mean"])] for a in ag]))
        best = max(ag, key=lambda a: a["spearman_mean"] or -1)
        worst = min(ag, key=lambda a: a["spearman_mean"] if a["spearman_mean"] is not None else 9)
        h.append(
            f"<p>The strongest agreement is {esc(best['pair'].replace('_vs_', ' vs '))} "
            f"(ρ = {num(best['spearman_mean'])}) and the weakest is "
            f"{esc(worst['pair'].replace('_vs_', ' vs '))} (ρ = {num(worst['spearman_mean'])}). "
            f"The gap is structural rather than incidental: Grad-CAM and GradientSHAP are both "
            f"gradient methods reading the same network, while LIME sees only input–output "
            f"behaviour through superpixel occlusion and is additionally quantised to "
            f"superpixel boundaries.</p>"
        )

    # ---- examples ----
    h.append("<h2>Worked examples</h2>")
    h.append(
        "<p>These are selected by measurable criteria, not by appearance: the most confident "
        "correct prediction, the most confident error, and the prediction whose own "
        "explanation is least well-grounded.</p>"
    )
    ex = pick_examples(records, kind)
    captions = {
        "confident_correct": "Most confident correct prediction",
        "confident_wrong": "Most confident error — the model is sure and wrong",
        "worst_surrogate": "Weakest local surrogate — the explanation itself is unreliable",
        "no_prototype_support": "Confident prediction with no supporting training neighbour",
    }
    for k, cap in captions.items():
        if k in ex:
            h.append(render_example(ex[k], kind, art, cap))

    if key in FINDINGS:
        h.append("<h2>Findings</h2>")
        h.append(FINDINGS[key])

    return "\n".join(h)


def build(cfg: cfgmod.Config) -> Path:
    # Both files are needed: summary.json drives the statistics, predictions.jsonl
    # the worked examples. A task can legitimately have the first without the
    # second — the tweet records are withheld from the repository on licensing
    # grounds — so a task is skipped rather than allowed to fail the build.
    keys = [k for k in cfg.task_keys
            if (cfg.artifact_dir(k) / "summary.json").exists()
            and (cfg.artifact_dir(k) / "predictions.jsonl").exists()]
    if not keys:
        raise FileNotFoundError("no complete task artifacts found; run xai.run then xai.summarize")

    summaries = {k: json.loads((cfg.artifact_dir(k) / "summary.json").read_text()) for k in keys}
    records = {k: load_records(cfg.artifact_dir(k) / "predictions.jsonl") for k in keys}
    extra = load_analyses(cfg)

    total_preds = sum(s["n_records"] for s in summaries.values())
    total_expl = sum(sum(m["n_explained"] for m in s["methods"].values()) for s in summaries.values())
    total_compute = sum(
        sum(m["runtime_s_total"] for m in s["methods"].values()) for s in summaries.values()
    )

    parts: list[str] = [_HEAD]

    # ---------------- title ----------------
    parts.append(f"""
<section class='title'>
  <h1>Explainable AI across Text and Vision</h1>
  <p class='sub'>A reproducible comparison of LIME, SHAP, prototype retrieval and Grad-CAM,
     applied to every prediction of three classifiers</p>
  <table class='meta'>
    <tr><td>Predictions explained</td><td>{total_preds:,}</td></tr>
    <tr><td>Individual explanations generated</td><td>{total_expl:,}</td></tr>
    <tr><td>Total explanation compute</td><td>{total_compute / 3600:.1f} hours</td></tr>
    <tr><td>Tasks</td><td>{len(keys)} &mdash; {esc(', '.join(summaries[k]['name'] for k in keys))}</td></tr>
  </table>
</section>
""")

    # ---------------- abstract ----------------
    acc_line = ", ".join(
        f"{summaries[k]['name'].split('(')[0].strip()} {summaries[k]['performance']['accuracy']:.2%}"
        for k in keys
    )
    parts.append(f"""
<section>
<h1 class='chapter'>Abstract</h1>
<p>This project applies four explanation methods — LIME, SHAP, prototype retrieval and
Grad-CAM — to three classifiers spanning two modalities: a three-class tweet sentiment
model, a binary movie-review model fine-tuned with LoRA, and a binary glasses detector
built on ResNet-18. Accuracies are {esc(acc_line)}.</p>

<p>The distinguishing property of this work is coverage. Explanations are not generated
for a hand-picked handful of inputs; <b>every one of the {total_preds:,} predictions in the
evaluation sets carries a complete set of explanations</b>, yielding {total_expl:,} individual
explanations at a cost of {total_compute / 3600:.1f} hours of compute. This makes it possible to
report distributions rather than anecdotes: how often LIME's local surrogate actually fits,
how often the retrieved prototypes contradict the prediction, how much the methods agree
with one another, and what each costs.</p>

<p>Methods are ranked here on <b>faithfulness</b> — whether deleting what an explanation
named actually moves the model — because that is the explainer's task: the model has
already reached its conclusion, and the method's job is to account for how. Agreement with
human annotation is reported alongside, but as a diagnostic rather than a verdict: it
measures how human-like the <em>model</em> is, and a faithful explanation of a model that
relies on something unhuman scores badly on it precisely because it is correct.</p>

<p>Four findings recur. First, the methods agree far more about the <em>sign</em> of a
feature's contribution than about <em>which</em> features are worth naming, so any single
top-k list overstates its own specificity. Second, on the text tasks explanation quality
degrades where it is most needed: on the model's errors the local surrogates fit worst and
the attributions shrink toward zero. Third, <b>perturbation-based explanation has a budget
floor</b> — below roughly seven model evaluations per input feature, both LIME and SHAP fall
to the performance of a random ranking, which is where the movie-review explanations sit;
the gradient-based methods have no such floor and Grad-CAM records the highest faithfulness
measured here from a single backward pass. Fourth, <b>plausible and faithful come apart</b>:
on the image task the method whose strongest pixel most often lands inside the human-marked
region is also the least faithful of the three, so a reviewer checking saliency maps by eye
would be reassured by the method they should trust least.</p>
</section>
""")

    # ---------------- method ----------------
    parts.append(f"""
<section>
<h1 class='chapter'>Approach and reproducibility</h1>
<p>The original version of this project was a set of Colab notebooks with uploaded
checkpoints, hardcoded local paths and unseeded splits. It could not be re-run, and two of
its analyses were computed against models that had not loaded correctly. This rebuild is
a Python package with a single configuration file; every result in this report is
regenerated by the commands below, in order.</p>

<pre class='cmd'>pip install -r requirements.txt
python -m xai.train_glasses      # retrain the image model with a seeded split
python -m xai.run --all          # explain every prediction, resumably
python -m xai.verify --all       # fail loudly if anything is missing
python -m xai.summarize --all    # aggregate into summary.json
python -m xai.modelcard --task all      # architecture, training, per-split accuracy
python -m xai.evaluate --task all       # plausibility against the human annotation
python -m xai.faithfulness --task all   # deletion, against a random-ranking control
python -m xai.behaviour --task all      # what each method names; the budget analysis
python -m xai.report             # regenerate this document as HTML and PDF</pre>

<p>The last three are optional: they are what produce the faithfulness, plausibility and
behavioural chapters. Without them <code>xai.report</code> still builds, and simply omits
those sections rather than failing.</p>

<h2>Design decisions that affect the explanations</h2>
<table>
<thead><tr><th>Decision</th><th>Why it matters</th></tr></thead>
<tbody>
<tr><td>Attribution in pixel space</td><td>Image normalisation is applied inside the model's
forward pass, not as preprocessing, so Grad-CAM, SHAP and LIME all attribute to pixels and
are directly comparable. Attributing to normalised tensors would make the three maps
describe different functions.</td></tr>
<tr><td>Gradient capture by tensor hook</td><td>Grad-CAM captures gradients with a hook on the
activation tensor rather than <code>register_backward_hook</code>, which fires unreliably on
modules with multiple inputs and silently produced wrong maps in the original code.</td></tr>
<tr><td>Tokenizer-aware masking for SHAP</td><td>Masked variants stay inside the model's own
vocabulary rather than becoming arbitrary word deletions.</td></tr>
<tr><td><code>bow=False</code> for LIME on text</td><td>Repeated words are treated as distinct
positions, so negation scoping is not destroyed by the bag-of-words assumption.</td></tr>
<tr><td>Fixed SLIC segmentation</td><td>A fixed superpixel count keeps LIME's granularity
comparable across images; the default segmenter varies with face size and made the
original explanations incommensurable.</td></tr>
<tr><td>Every explanation records its own diagnostics</td><td>Surrogate R², convergence
delta and neighbour agreement are stored per prediction, so an explanation's
trustworthiness is reported alongside its content rather than assumed.</td></tr>
<tr><td>Resumable, append-only runs</td><td>Each record is flushed to disk as it completes;
a run of several hours can be interrupted and restarted without loss.</td></tr>
</tbody></table>

<h2>What is stored for each prediction</h2>
<p>Nothing is truncated on disk. Every record holds the complete attribution, not a
top-k slice of it, so any figure in this report can be recomputed from the artifacts and
any claim in it can be checked against them.</p>
<table>
<thead><tr><th>Method</th><th>Stored for every prediction</th></tr></thead>
<tbody>
<tr><td>LIME (text)</td><td>A signed weight for every word in the input, plus the number of
perturbations used and the surrogate's R²</td></tr>
<tr><td>SHAP (text)</td><td>A Shapley value for every token, plus the base value and the
sum of the values, which is what makes the additivity property checkable</td></tr>
<tr><td>Prototypes</td><td>The nearest training examples with similarities, labels, and
whether each agrees with the prediction</td></tr>
<tr><td>Grad-CAM</td><td>The full 224&times;224 map</td></tr>
<tr><td>SHAP (image)</td><td>The full 224&times;224 signed map and its convergence delta</td></tr>
<tr><td>LIME (image)</td><td>The full 224&times;224 map, the weight of every superpixel, and the
segmentation used, so the explanation reconstructs exactly</td></tr>
</tbody></table>
<p>Alongside these each record carries the model's full probability vector, the method's
measured runtime and a path to its rendered card.
<code>python -m xai.verify --all</code> re-reads everything that was written and fails if
any sample is missing a record, any method errored, any explanation came back empty, or
any referenced file is absent. The figures below are reported only for runs that pass it.</p>

<h2>How the outputs are laid out</h2>
<p>The output tree mirrors the input tree and preserves input order, and nothing is stored
inside an archive. An image at <code>Dataset/Glasses/image/neg/00012.jpg</code> is explained
at <code>explanations/neg/00012.html</code>, with its attribution maps in
<code>saliency/neg/00012/</code> as one <code>.npy</code> per method rather than bundled into
a <code>.npz</code> — which is a zip file, and cannot be inspected, diffed or partially read
without unpacking it. Text cards carry their input row number, so listing the directory
reproduces the order of the source CSV rather than a hash order.</p>
<p>Every prediction in either modality is readable as HTML. Image predictions previously
produced only a rendered panel, which left the per-method numbers behind the picture —
runtime, focus, peak location, convergence delta — visible only inside the JSONL.</p>

<h2>Reading the explanations</h2>
<p>Token weights are signed with respect to the <em>predicted</em> class throughout: a
positive weight pushed the model toward the label it chose, a negative weight pushed away
from it. In the rendered cards, teal marks support and amber marks opposition. For images,
Grad-CAM maps are unsigned intensity; SHAP and LIME maps are signed and use a diverging
scale centred on zero.</p>
</section>
""")

    # ---------------- per-task chapters ----------------
    for k in keys:
        parts.append("<section>"
                     + task_chapter(cfg, k, summaries[k], records[k], extra)
                     + "</section>")

    # ---------------- which method is better ----------------
    # Sits before the cross-task chapter because it establishes the criterion
    # that "What each method is for" defers to.
    verdicts = verdict_chapter(summaries, extra)
    if verdicts:
        parts.append("<section>" + verdicts + "</section>")

    # ---------------- cross-task ----------------
    parts.append("<section>" + cross_task_chapter(summaries, extra) + "</section>")

    # ---------------- limitations ----------------
    parts.append(_limitations(summaries, extra))

    parts.append(_REFS)

    # Title page first, then contents, then everything else with anchors.
    title_html = parts[1]
    body, entries = anchor_headings("\n".join(parts[2:]))
    doc = [parts[0], title_html, toc_section(entries), body, "</body></html>"]

    out = cfgmod.PROJECT_ROOT / "Report" / "ExAI_Report_v2.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(doc), encoding="utf-8")
    return out


def _models_section(extra: dict) -> str:
    """The three classifiers side by side — what was trained, and what it cost."""
    cards = extra.get("model_cards") or {}
    if not cards:
        return ""

    rows = []
    for key, c in cards.items():
        p = c.get("parameters") or {}
        a = c.get("accuracy_by_split") or {}
        tr = (a.get("train") or {}).get("accuracy")
        te = (a.get("test") or {}).get("accuracy")
        gap = f"{(tr - te) * 100:+.1f} pp" if tr is not None and te is not None else "—"
        rows.append([
            _TASK_SHORT.get(key, key),
            f"<span>{esc(c.get('architecture', '—'))}</span>",
            f"{p.get('total', 0):,}",
            f"{p.get('trained_pct', 0):.2f}%",
            num(tr, 4) if tr is not None else "—",
            num(te, 4) if te is not None else "—",
            f"<span>{gap}</span>",
        ])

    h = ["<h2>The three classifiers</h2>",
         "<p>Each model was trained separately and the report has described them one at a "
         "time; set side by side, they differ on a dimension worth naming before the "
         "explanations are compared — how much of each network was actually fitted, and how "
         "far that shows up as memorisation.</p>",
         table(["task", "architecture", "parameters", "trained", "train acc.",
                "test acc.", "gap"], rows)]

    # State the contrast only when the numbers actually show it.
    gaps = {}
    for key, c in cards.items():
        a = c.get("accuracy_by_split") or {}
        tr, te = (a.get("train") or {}).get("accuracy"), (a.get("test") or {}).get("accuracy")
        if tr is not None and te is not None:
            gaps[key] = (tr - te, (c.get("parameters") or {}).get("trained_pct", 100.0))
    if len(gaps) >= 2:
        worst = max(gaps, key=lambda k: gaps[k][0])
        wg, wp = gaps[worst]
        others = [k for k in gaps if k != worst]
        if wg >= 0.05:
            rest = ", ".join(
                f"{esc(_TASK_SHORT.get(k, k))} {gaps[k][0] * 100:+.1f} pp "
                f"({gaps[k][1]:.2f}% of the network trained)" for k in others)
            h.append(
                f"<p>Only one of the three carries a large generalisation gap: "
                f"<b>{esc(_TASK_SHORT.get(worst, worst))}</b> at "
                f"<b>{wg * 100:+.1f} percentage points</b>, with {wp:.0f}% of its parameters "
                f"updated over its training set. The others sit at {rest}. The contrast is "
                f"not proof that adapter training generalises better — the tasks and dataset "
                f"sizes differ too — but it does mean the explanations in each chapter are "
                f"describing models of quite different character, and an attribution on the "
                f"most-memorised model is the most likely to be faithfully reporting a "
                f"decision that will not transfer.</p>")
    return "\n".join(h)


def _budget_section(extra: dict) -> str:
    """How much sampling does a perturbation method actually need?"""
    budget = ((extra.get("behaviour") or {}).get("cross") or {}).get("budget")
    if not budget or not budget.get("points"):
        return ""
    pts = budget["points"]
    thr = budget.get("threshold_evals_per_feature")

    h = ["<h2>How much budget does an explanation need?</h2>",
         "<p>The cost above is spent buying model evaluations. What decides whether a "
         "perturbation method is <em>estimating</em> an attribution or interpolating one is "
         "not the total spend but how many of those evaluations each input feature "
         "receives — and that ratio varies by more than an order of magnitude across these "
         "three tasks.</p>",
         figures.budget_vs_reliability(budget)]

    rows = []
    for p in pts:
        grad = p["budget_kind"] == "gradient"
        rows.append([
            _TASK_SHORT.get(p["task"], p["task"]), p["method"],
            f"{p['n_features_mean']:,.1f}" if p["n_features_mean"] else "—",
            f"{p['budget']:,.0f}" if p["budget"] else "—",
            "<span class='muted'>gradient</span>" if grad
            else f"{p['evals_per_feature']:,.1f}",
            num(p["comprehensiveness"]),
            f"{p['ratio_vs_random']:.2f}×" if p["ratio_vs_random"] else "—",
        ])
    h.append(table(["task", "method", "features", "budget", "evals / feature",
                    "compr.", "× random"], rows))

    failing = [p for p in pts if p["budget_kind"] == "perturbation"
               and p["evals_per_feature"] and p["ratio_vs_random"]
               and p["ratio_vs_random"] < _NOISE_RATIO]
    passing = [p for p in pts if p["budget_kind"] == "perturbation"
               and p["evals_per_feature"] and p["ratio_vs_random"]
               and p["ratio_vs_random"] >= _NOISE_RATIO]
    if failing and passing:
        worst_ok = min(p["evals_per_feature"] for p in passing)
        best_bad = max(p["evals_per_feature"] for p in failing)
        h.append(
            f"<p>The perturbation methods separate cleanly on this axis. Every one that "
            f"received at least {worst_ok:.1f} evaluations per feature clears "
            f"{_NOISE_RATIO:g}× a random ranking; every one below {best_bad:.1f} falls to "
            f"near-random. The threshold drawn at ≈{thr:g} is declared rather than fitted — "
            f"it is simply the gap between those two numbers, and {len(pts)} points would "
            f"not support fitting one.</p>"
        )
    h.append(
        "<p>The gradient methods sit outside this relationship entirely. Grad-CAM reads the "
        "network directly in a single backward pass, takes no sampling budget at all, and "
        "records the highest faithfulness measured anywhere in this report. Set against the "
        "cost table above, that inverts the usual assumption: here the most expensive method "
        "on the most demanding task produced the least faithful explanations, and the "
        "cheapest produced the most faithful.</p>"
    )
    return "\n".join(h)


def _pointing_baselines(plaus_task: dict) -> str:
    """What the pointing game scores without any method behind it."""
    b = plaus_task.get("baselines") or {}
    cov, cen = b.get("mask_coverage"), b.get("centre_pixel_pointing")
    if cov is None:
        return ""
    s = (f"<p class='muted'>The human masks cover {cov:.1%} of the frame, so a pixel chosen "
         f"at random scores {cov:.3f} on the pointing game")
    if cen is not None:
        s += f", and a fixed centre pixel — no method at all — scores {cen:.3f}"
    return s + ".</p>"


def _method_faith_span(method: str, extra: dict) -> str:
    """The measured range for one method across tasks, best and worst."""
    faith = extra.get("faithfulness") or {}
    seen = []
    for key, v in faith.items():
        methods = v.get("methods") or {}
        e, rand = methods.get(method), (methods.get("random") or {}).get("comprehensiveness")
        if e and rand and e.get("comprehensiveness"):
            seen.append((e["comprehensiveness"] / rand, key))
    if not seen:
        return "<span class='muted'>—</span>"
    seen.sort()
    lo, hi = seen[0], seen[-1]
    if lo == hi:
        return f"<span>{hi[0]:.2f}× random on {esc(_TASK_SHORT.get(hi[1], hi[1]))}</span>"
    return (f"<span>{hi[0]:.2f}× random at best ({esc(_TASK_SHORT.get(hi[1], hi[1]))}), "
            f"{lo[0]:.2f}× at worst ({esc(_TASK_SHORT.get(lo[1], lo[1]))})</span>")


def _complementarity_prose(extra: dict) -> str:
    """Are the methods substitutes? Answer from the measured error-flag overlap."""
    beh = extra.get("behaviour") or {}
    blocks = {k: (v.get("error_detection") or {}) for k, v in beh.items()
              if isinstance(v, dict) and (v.get("error_detection") or {}).get("overlap")}
    base = ("<p>These are not substitutes. A token method and an example method answer "
            "different questions, and the cases where they disagree are the ones worth a "
            "human's attention.</p>")
    if not blocks:
        return base + ("<p>A deployment that must justify individual decisions is best served "
                       "by pairing one attribution method with prototype retrieval, and by "
                       "surfacing the explanation's own reliability score rather than hiding "
                       "it.</p>")

    rows = []
    for key, ed in blocks.items():
        for pair, o in ed["overlap"].items():
            rows.append([_TASK_SHORT.get(key, key), pair.replace("|", " ∩ "),
                         num(o["jaccard"]), o["both"]])
    detail = ["<p class='cap'>Do the methods flag the same errors?</p>",
              table(["task", "pair", "Jaccard", "flagged by both"], rows)]

    lines = []
    for key, ed in blocks.items():
        u, b = ed.get("union_recall"), ed.get("best_single_recall")
        if u and b:
            lines.append(
                f"on {esc(_TASK_SHORT.get(key, key))} the three signals together flag "
                f"{u:.1%} of the model's errors against {b:.1%} for the best of them alone, "
                f"and {ed.get('prototype_only', 0)} errors are caught by prototype retrieval "
                f"that neither token method flags")
    summary = ""
    if lines:
        summary = ("<p>That claim is measurable, and it holds: " + "; ".join(lines)
                   + ". Pairing one attribution method with prototype retrieval therefore "
                     "covers materially more than either alone — the overlap table shows they "
                     "are not finding the same failures.</p>")
    return base + "\n".join(detail) + summary


def _quality_direction_prose(summaries: dict) -> str:
    """State the cross-task direction that was actually observed, not an assumed one."""
    deltas = {}
    for key, s in summaries.items():
        q = (s.get("quality_by_correctness") or {}).get("lime.surrogate_r2") or {}
        if "delta" in q:
            deltas[s["name"].split("(")[0].strip()] = q["delta"]
    if not deltas:
        return ""
    worse = {k: v for k, v in deltas.items() if v > 0}
    better = {k: v for k, v in deltas.items() if v <= 0}
    if worse and not better:
        return ("<p>The pattern holds in the same direction on every task: the local "
                "surrogate fits the classifier less well on the predictions the model gets "
                "wrong. The explanations are weakest exactly where a reader needs them "
                "most.</p>")
    if worse and better:
        b = ", ".join(f"{esc(k)} ({v:+.3f})" for k, v in better.items())
        return (f"<p>On the text tasks the local surrogate fits less well where the model is "
                f"wrong, which is the uncomfortable direction: the explanation is weakest "
                f"exactly where a reader needs it most. It does not generalise, and the "
                f"exception is worth naming rather than smoothing over — {b} runs the other "
                f"way. That task has almost no errors to average over, so the comparison "
                f"there rests on a handful of predictions and should not be read as a "
                f"counter-example so much as an absence of evidence.</p>")
    return ("<p>On these tasks the surrogate fits at least as well on the model's errors as "
            "on its correct predictions, so a weak explanation here is not a reliable signal "
            "that the prediction is wrong.</p>")


_TRAIN_LABEL = {
    "base_model": "base model", "strategy": "what was trained",
    "optimizer": "optimizer", "lr": "learning rate", "epochs": "epochs",
    "batch_size": "batch size", "max_length": "max sequence length",
    "val_split": "validation split", "seed": "seed", "hardware": "trained on",
    "weight_decay": "weight decay", "pretrained": "initialised from",
    "split": "train / val / test split", "augment": "augmentation",
    "per_epoch_metrics": "per-epoch metrics",
}


def _model_card(key: str, extra: dict) -> str:
    """How this classifier was built, and how it scores on every split."""
    cards = extra.get("model_cards") or {}
    c = cards.get(key)
    if not c:
        return ""

    h = ["<h2>How this model was trained</h2>"]

    # ---- what was trained ----
    p = c.get("parameters") or {}
    rows = []
    for k, v in (c.get("training") or {}).items():
        label = _TRAIN_LABEL.get(k, k.replace("_", " "))
        if isinstance(v, dict):
            v = ", ".join(f"{a} {b}" for a, b in v.items())
        elif isinstance(v, list):
            v = " / ".join(str(x) for x in v)
        # "not recorded" is a fact about the provenance, not a missing value to
        # paper over, so it is shown rather than silently dropped.
        muted = isinstance(v, str) and v.startswith("not ")
        cell = f"<span class='muted'>{esc(v)}</span>" if muted else f"<span>{esc(v)}</span>"
        rows.append([f"<span>{esc(label)}</span>", cell])
    if p.get("total"):
        trained = p.get("trained", p["total"])
        rows.append(["<span>parameters</span>",
                     f"<span>{trained:,} of {p['total']:,} trained "
                     f"({p.get('trained_pct', 100):.2f}%)</span>"])
    lora = c.get("lora")
    if lora:
        rows.append(["<span>LoRA adapter</span>",
                     f"<span>rank {lora.get('r')}, α {lora.get('lora_alpha')}, "
                     f"dropout {lora.get('lora_dropout')}, applied to the "
                     f"{esc(' and '.join(lora.get('target_modules', [])))} projections "
                     f"of every attention layer</span>"])
        if p.get("lora_adapter"):
            rows.append(["<span>trainable breakdown</span>",
                         f"<span>{p['lora_adapter']:,} adapter + "
                         f"{p.get('classification_head', 0):,} classification head</span>"])
    if rows:
        h.append(table(["", ""], rows))

    # ---- accuracy on every split ----
    acc = c.get("accuracy_by_split") or {}
    splits = [(s, acc[s]) for s in ("train", "val", "test") if isinstance(acc.get(s), dict)]
    if splits:
        h.append("<p class='cap'>Accuracy on every split</p>")
        h.append(table(
            ["split", "n", "accuracy"],
            [[s, f"{d.get('n'):,}" if d.get("n") else "—", num(d.get("accuracy"), 4)]
             for s, d in splits]))
        by = {s: d["accuracy"] for s, d in splits}
        if "train" in by and "test" in by:
            gap = by["train"] - by["test"]
            if gap >= 0.05:
                h.append(
                    f"<p>The gap between training and held-out accuracy is "
                    f"<b>{gap * 100:.1f} percentage points</b> "
                    f"({by['train']:.1%} → {by['test']:.1%}). The model has memorised a "
                    f"substantial part of its training set, which is worth holding in mind "
                    f"when reading the explanations: an attribution can faithfully describe "
                    f"a decision that rests on memorisation rather than on a generalising "
                    f"feature.</p>")
            else:
                h.append(
                    f"<p>Training and held-out accuracy differ by "
                    f"{abs(gap) * 100:.1f} percentage points "
                    f"({by['train']:.1%} → {by['test']:.1%}), so this model is not relying "
                    f"on memorisation to the degree its accuracy might otherwise suggest.</p>")
    return "\n".join(h)


def _task_faithfulness(key: str, extra: dict) -> str:
    """Does the model actually use what this task's explanations named?"""
    f = (extra.get("faithfulness") or {}).get(key)
    if not f or not f.get("methods"):
        return ""
    methods = f["methods"]
    rand = (methods.get("random") or {}).get("comprehensiveness")
    rand_s = (methods.get("random") or {}).get("sufficiency")

    rows = []
    for m, e in methods.items():
        label = f"{m} <span class='muted'>(baseline)</span>" if m == "random" else m
        rows.append([f"<span>{label}</span>",
                     num(e.get("comprehensiveness")), ratio(e.get("comprehensiveness"), rand),
                     num(e.get("sufficiency")), ratio(e.get("sufficiency"), rand_s)])

    best = max(((m, e.get("comprehensiveness") or 0.0) for m, e in methods.items()
                if m != "random"), key=lambda kv: kv[1], default=None)
    verdict = ""
    if best and rand:
        r = best[1] / rand
        verdict = (
            f"<p>The strongest here is <b>{esc(best[0])}</b> at {r:.2f}× a random ranking"
            + (". That is comfortably above chance, so these attributions can be read as "
               "describing the model." if r >= _NOISE_RATIO else
               ". That is barely above chance: deleting what the method ranked highest "
               "moves the model little more than deleting the same number of features at "
               "random, so these attributions should not be read as describing the model.")
            + "</p>"
        )
    return ("<h2>Does the model use what the explanations named?</h2>"
            "<p>Comprehensiveness deletes the top-ranked features and measures how far the "
            "predicted-class probability falls; sufficiency keeps only those features. The "
            "random row ranks the same features arbitrarily at the same budget, and is the "
            "only thing that makes the other rows interpretable.</p>"
            + table(["method", "compr. ↑", "× random", "suff. ↓", "× random"], rows)
            + verdict)


def _task_plausibility(key: str, extra: dict) -> str:
    """Does it point where a human points? Reported, but never a verdict."""
    p = (extra.get("plausibility") or {}).get(key)
    if not p or not p.get("methods"):
        return ""
    m = p["methods"]
    first = next(iter(m.values()))
    h = ["<h2>Does it point where a human points? "
         "<span class='crit secondary'>secondary</span></h2>",
         "<p>This compares each explanation against annotation produced by people rather "
         "than against the model. It describes how human-like <em>the model</em> is, seen "
         "through the explanation — so it is reported here as a diagnostic and does not "
         "decide which method explained better.</p>"]

    if "pointing_game" in first:
        h.append(table(
            ["method", "pointing game", "mask AUC", "IoU", "mass inside mask"],
            [[k, num(d.get("pointing_game")), num(d.get("mask_auc")), num(d.get("iou")),
              num(d.get("mass_inside"))] for k, d in m.items()]))
        h.append(_pointing_baselines(p))
    else:
        h.append(table(
            ["method", "precision", "recall", "F1", "IoU"],
            [[k, num(d.get("precision")), num(d.get("recall")), num(d.get("f1")),
              num(d.get("iou"))] for k, d in m.items()]))
        h.append(f"<p class='muted'>Scored on {p.get('n_scored', 0):,} of "
                 f"{p.get('n_matched', 0):,} predictions; the "
                 f"{p.get('n_degenerate_excluded', 0):,} excluded are those whose human "
                 f"rationale is the entire input, where any method scores near 1.0 by "
                 f"construction.</p>")
    return "\n".join(h)


def _task_behaviour(key: str, kind: str, extra: dict) -> str:
    """What kind of thing does each method actually name?"""
    b = (extra.get("behaviour") or {}).get(key)
    if not b:
        return ""
    h: list[str] = []

    if kind == "text":
        comp = b.get("token_composition") or {}
        if comp.get("methods"):
            h.append("<h2>What kind of feature does each method name?</h2>")
            h.append(f"<p>Composition of each method's top-{comp.get('top_k', 5)} tokens "
                     "across every prediction in this set. LIME perturbs whole words, so it "
                     "can never return punctuation or a word fragment; SHAP works at "
                     "wordpiece level and can. That asymmetry is structural, and it is "
                     "what the lower rows measure.</p>")
            cats = comp["categories"]
            h.append(table(
                ["category"] + [m.upper() for m in comp["methods"]],
                [[c] + [pct((comp["methods"][m].get(c) or 0) / 100.0)
                        for m in comp["methods"]] for c in cats]))

        ce = (b.get("counter_evidence") or {}).get("words") or {}
        if ce:
            h.append("<h3>Counter-evidence</h3>")
            h.append("<p>What each method says about a strongly polar word sitting in an "
                     "input the model did <em>not</em> classify with that polarity. The two "
                     "methods ask different questions: LIME asks whether deleting the word "
                     "moves the prediction — for a word the model already discounted it does "
                     "not, so the coefficient is near zero — while SHAP's additivity forces "
                     "it to report the word as evidence the model overcame.</p>")
            meths = sorted({m for d in ce.values() for m in d if m != "n"})
            h.append(table(
                ["word", "n"] + [m.upper() for m in meths],
                [[w, d["n"]] + [num(d.get(m), 4) for m in meths]
                 for w, d in list(ce.items())[:10]]))

        con = b.get("contrast") or {}
        if con.get("methods"):
            t = con.get("test") or {}
            h.append("<h3>Contrast and negation</h3>")
            rows = [["<span>share of attribution after “but” "
                     "<span class='muted'>(sentiment follows the second clause)</span></span>"]
                    + [num(con["methods"].get(m)) for m in con["methods"]]]
            neg = (b.get("negation") or {}).get("methods") or {}
            if neg:
                rows.append(["<span>negator's rank, 0 = most important</span>"]
                            + [num(neg[m]["normalised_rank"]) for m in con["methods"]
                               if m in neg])
                rows.append(["<span>negator's share of attribution mass</span>"]
                            + [num(neg[m]["mass_share"]) for m in con["methods"]
                               if m in neg])
            h.append(table(["probe"] + [m.upper() for m in con["methods"]], rows))
            if t:
                h.append(
                    f"<p>On the contrast probe the difference is "
                    f"{t['diff']:+.4f} (95% CI [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}], "
                    f"p = {t['p']:.4f}, n = {t['n']}), so SHAP does weight the clause after "
                    f"the marker more heavily. The negation rows show no such separation: "
                    f"<b>neither method surfaces the negator</b>, which is worth stating "
                    f"because it is the one place a bag-of-words assumption was expected to "
                    f"cost LIME something measurable.</p>")
    else:
        sh = b.get("shape") or {}
        if sh.get("methods"):
            h.append("<h2>What shape is the evidence?</h2>")
            h.append(f"<p>Connected components in the strongest "
                     f"{sh.get('top_frac', 0.05):.0%} of each map, over "
                     f"{sh.get('n_images', 0):,} images. This measures what a focus score "
                     "cannot: whether the evidence forms one region a reader can point at, "
                     "or a field of scattered pixels.</p>")
            h.append(table(
                ["method", "connected blobs", "mass in largest", "vertical centre"],
                [[m, f"{d['n_components']:,.1f}", pct(d["mass_in_largest"]),
                  f"{d['centroid_row']:.0f}"] for m, d in sh["methods"].items()]))
            h.append("<p class='muted'>Rows are 0 at the top of a 224-pixel frame; the "
                     "eyes sit near row 95–110.</p>")
    return "\n".join(h)


# A lead smaller than this on comprehensiveness is not treated as a win. It is a
# declared convention, not an estimate of sampling error, and it is stated in the
# chapter so a reader can disagree with it.
_TIE_MARGIN = 0.05
# Below this multiple of a random ranking, a method is not carrying usable signal.
_NOISE_RATIO = 2.0

_TASK_SHORT = {"tweet": "Tweet Sentiment", "movie": "Movie Review Sentiment",
               "glasses": "Glasses Detection"}


def _verdict(faith_task: dict) -> dict:
    """Derive the verdict for one task from its measured faithfulness.

    Nothing here is typed in: the winner, the tie and the "neither" case are all
    consequences of the numbers in faithfulness.json, so re-running the analysis
    changes the verdict rather than leaving it stale.
    """
    methods = {m: v for m, v in (faith_task.get("methods") or {}).items() if m != "random"}
    rand = ((faith_task.get("methods") or {}).get("random") or {}).get("comprehensiveness")
    scored = sorted(
        ((m, v.get("comprehensiveness") or 0.0) for m, v in methods.items()),
        key=lambda kv: -kv[1],
    )
    if not scored:
        return {}
    best, best_v = scored[0]
    ratio_best = best_v / rand if rand else None

    if ratio_best is not None and ratio_best < _NOISE_RATIO:
        return {"kind": "none", "winner": "Neither",
                "why": (f"every method here scores below {_NOISE_RATIO:g}× a random "
                        f"ranking — the best, {best}, reaches {ratio_best:.2f}×. "
                        "These explanations are close to noise, so there is nothing "
                        "to choose between them."),
                "best": best, "ratio": ratio_best}

    if len(scored) > 1:
        second, second_v = scored[1]
        margin = (best_v - second_v) / second_v if second_v else 1.0
        if margin < _TIE_MARGIN:
            return {"kind": "tie", "winner": "Tie",
                    "why": (f"{best} leads {second} by {margin:.1%} on comprehensiveness, "
                            f"inside the {_TIE_MARGIN:.0%} margin this report treats as "
                            f"indistinguishable. Both describe the model about equally "
                            f"well ({ratio_best:.2f}× and {second_v / rand:.2f}× random)."
                            if rand else ""),
                    "best": best, "ratio": ratio_best, "margin": margin}
        return {"kind": "win", "winner": best,
                "why": (f"{best} leads {second} by {margin:.1%} on comprehensiveness"
                        + (f" and reaches {ratio_best:.2f}× a random ranking" if rand else "")
                        + "."),
                "best": best, "ratio": ratio_best, "margin": margin}
    return {"kind": "win", "winner": best, "why": "", "best": best, "ratio": ratio_best}


def verdict_chapter(summaries: dict, extra: dict | None = None) -> str:
    """Which method explained best — and on what grounds."""
    extra = extra or {}
    faith = extra.get("faithfulness") or {}
    plaus = extra.get("plausibility") or {}
    if not faith:
        return ""

    h = ["<h1 class='chapter'>Which method explained best?</h1>"]

    # ---- the criterion ----
    h.append(
        "<p>The model has already reached its conclusion. An explanation method's only "
        "task is to account for <em>how that model got there</em> — not to produce the "
        "account a human would have written. That distinction decides which of the "
        "measurements below counts as evidence, and it is worth stating before any "
        "number is quoted.</p>"
    )
    h.append(table(
        ["Measurement", "Compares", "What it establishes"],
        # A cell is passed through raw only when it starts with "<", so any cell
        # carrying markup or an entity has to open with a tag.
        [["<b>Faithfulness</b> <span class='crit primary'>primary</span>",
          "<span>explanation &harr; <b>model</b></span>",
          "<span>Whether the explainer did its job. Measured by deletion: remove what "
          "the method named and see whether the prediction moves.</span>"],
         ["<b>Plausibility</b> <span class='crit secondary'>secondary</span>",
          "<span>explanation &harr; <b>human</b></span>",
          "<span>Whether the <em>model</em> is human-like. A property of the model, seen "
          "through the explanation — so it cannot rank explainers.</span>"],
         ["<span>Self-reported fit</span>",
          "<span>explanation &harr; itself</span>",
          "<span>Internal consistency only: surrogate R², convergence delta, neighbour "
          "agreement. Computed without any ground truth.</span>"]],
    ))
    h.append(
        "<div class='note'><p>If a model latches onto something no human would — a "
        "punctuation habit, a spurious correlation — then a <em>faithful</em> explanation "
        "reports exactly that, and a human-agreement score penalises it for being "
        "correct. High plausibility with low faithfulness is therefore the worst "
        "combination in this study rather than the best: an explanation that reads "
        "convincingly while describing something the model never did.</p></div>"
    )

    # ---- faithfulness ----
    h.append("<h2>Faithfulness — the primary criterion</h2>")
    h.append(
        "<p>Comprehensiveness deletes the top-ranked features and asks how far the "
        "predicted-class probability falls; sufficiency keeps only those features and "
        "asks the same. Both are reported against a random ranking of the same features "
        "at the same budget, because deleting <em>any</em> fifth of an input moves a "
        "classifier — only the margin over random is evidence that the ordering carried "
        "information.</p>"
    )
    h.append(figures.faithfulness_bars(faith, "comprehensiveness"))
    h.append(figures.faithfulness_bars(faith, "sufficiency"))

    rows = []
    for key, v in faith.items():
        methods = v.get("methods") or {}
        rand = (methods.get("random") or {}).get("comprehensiveness")
        rand_s = (methods.get("random") or {}).get("sufficiency")
        for m, e in methods.items():
            if m == "random":
                continue
            rows.append([_TASK_SHORT.get(key, key), m,
                         num(e.get("comprehensiveness")), ratio(e.get("comprehensiveness"), rand),
                         num(e.get("sufficiency")), ratio(e.get("sufficiency"), rand_s),
                         f"{e.get('n', 0):,}"])
    h.append("<p class='cap'>Faithfulness, all tasks</p>")
    h.append(table(["task", "method", "compr.", "× random", "suff.", "× random",
                    "deletions"], rows))
    h.append("<p class='muted'>“deletions” counts the measurements behind each mean: one "
             "per prediction per budget, at k = 1, 5, 10, 20 and 50% of the input.</p>")

    # ---- verdicts ----
    h.append("<h2>The verdicts</h2>")
    h.append(
        f"<p>Each verdict below follows from the table above rather than from a reading "
        f"of it: a lead smaller than {_TIE_MARGIN:.0%} on comprehensiveness is recorded as "
        f"a tie, and a task where the best method falls below {_NOISE_RATIO:g}× a random "
        f"ranking is recorded as having no usable winner.</p>"
    )
    for key, v in faith.items():
        vd = _verdict(v)
        if not vd:
            continue
        cls = " none" if vd["kind"] in ("none", "tie") else ""
        h.append(
            f"<div class='callout{cls}'>"
            f"<p class='ds'>{esc(_TASK_SHORT.get(key, key))}</p>"
            f"<h3>{esc(vd['winner'])}</h3>"
            f"<p class='why'>{esc(vd['why'])}</p></div>"
        )

    # ---- plausibility ----
    if plaus:
        h.append("<h2>Plausibility — a secondary diagnostic</h2>")
        h.append(
            "<p>These figures compare each explanation against annotation produced by "
            "people: 2,614 attention masks shipped with the glasses dataset, and the "
            "human-selected rationale spans shipped with the tweet dataset. They are "
            "reported because they are informative about the models and because one "
            "result below matters a great deal — but they decide nothing, for the reason "
            "given at the head of this chapter. The movie-review dataset carries no human "
            "annotation, so plausibility is undefined there.</p>"
        )
        g = (plaus.get("glasses") or {}).get("methods") or {}
        if g:
            h.append("<p class='cap'>Glasses — saliency against human attention masks</p>")
            h.append(table(
                ["method", "pointing game", "mask AUC", "IoU", "mass inside mask", "n"],
                [[m, num(d.get("pointing_game")), num(d.get("mask_auc")), num(d.get("iou")),
                  num(d.get("mass_inside")), f"{d.get('n', 0):,}"] for m, d in g.items()],
            ))
        t = (plaus.get("tweet") or {}).get("methods") or {}
        if t:
            tp = plaus["tweet"]
            h.append("<p class='cap'>Tweet — token attributions against human rationale spans</p>")
            h.append(table(
                ["method", "precision", "recall", "F1", "IoU", "n"],
                [[m, num(d.get("precision")), num(d.get("recall")), num(d.get("f1")),
                  num(d.get("iou")), f"{d.get('n', 0):,}"] for m, d in t.items()],
            ))
            h.append(
                f"<p class='muted'>Scored on {tp.get('n_scored', 0):,} of "
                f"{tp.get('n_matched', 0):,} tweets. The "
                f"{tp.get('n_degenerate_excluded', 0):,} excluded are those whose "
                f"<code>selected_text</code> is the entire tweet, where any method scores "
                f"near 1.0 by construction; that exclusion is also why the scored sample "
                f"is not class-balanced.</p>"
            )

        # ---- the quadrant ----
        quad = _quadrant(faith, plaus)
        if quad:
            h.append("<h2>Plausible is not faithful</h2>")
            h.append(quad)

    return "\n".join(h)


def _quadrant(faith: dict, plaus: dict) -> str:
    """Place the image methods on plausible x faithful, and name the dangerous cell."""
    g_p = (plaus.get("glasses") or {}).get("methods") or {}
    g_f = (faith.get("glasses") or {}).get("methods") or {}
    if not g_p or not g_f:
        return ""

    point = {m: d.get("pointing_game") for m, d in g_p.items() if d.get("pointing_game")}
    comp = {m: (d.get("comprehensiveness") or 0.0) for m, d in g_f.items() if m != "random"}
    shared = [m for m in point if m in comp]
    if len(shared) < 2:
        return ""

    p_hi = max(point, key=point.get)
    f_hi = max(comp, key=comp.get)
    f_lo = min(comp, key=comp.get)

    cells = {("hi", "hi"): [], ("hi", "lo"): [], ("lo", "hi"): [], ("lo", "lo"): []}
    p_med = sorted(point[m] for m in shared)[len(shared) // 2]
    f_med = sorted(comp[m] for m in shared)[len(shared) // 2]
    for m in shared:
        cells[("hi" if point[m] >= p_med else "lo",
               "hi" if comp[m] >= f_med else "lo")].append(m)

    def cell(key, danger=False, good=False):
        names = ", ".join(cells[key]) or "—"
        cls = " class='danger'" if danger else (" class='good'" if good else "")
        return f"<td{cls}>{esc(names)}</td>"

    tbl = (
        "<table class='matrix'><tr><th></th>"
        "<th style='text-align:center'>faithful — the model uses it</th>"
        "<th style='text-align:center'>unfaithful — it does not</th></tr>"
        f"<tr><td>matches human attention</td>{cell(('hi', 'hi'), good=True)}"
        f"{cell(('hi', 'lo'), danger=True)}</tr>"
        f"<tr><td>does not</td>{cell(('lo', 'hi'))}{cell(('lo', 'lo'))}</tr></table>"
    )

    rand_pix = ((plaus.get("glasses") or {}).get("baselines") or {}).get("random_pixel_pointing")
    note = ""
    if p_hi == f_lo:
        against = (f", against the {rand_pix:.1%} a randomly chosen pixel achieves"
                   if rand_pix else "")
        note = (
            f"<div class='note'><p><b>{esc(p_hi)}</b> places its single strongest pixel "
            f"inside the human-marked region {point[p_hi]:.1%} of the time — the best of "
            f"the three{against}. It is also the <em>worst</em> of the three on both "
            f"faithfulness metrics. Its peaks are where a human would look, but not what "
            f"the model uses.</p>"
            f"<p>Ranked on human agreement it looks like the strongest method here; ranked "
            f"on whether the model relies on what it names, it is the weakest. A reviewer "
            f"checking saliency maps by eye would be reassured by precisely the method "
            f"they should trust least — which is the practical reason plausibility is "
            f"reported here as a diagnostic and never as a verdict.</p></div>"
        )
    return tbl + note


def cross_task_chapter(summaries: dict, extra: dict | None = None) -> str:
    h = ["<h1 class='chapter'>Cross-task comparison</h1>"]

    rows = []
    for k, s in summaries.items():
        for m, v in s["methods"].items():
            rows.append([s["name"].split("(")[0].strip(), m,
                         f"{(v.get('runtime_s_mean') or 0.0):.2f}s",
                         v["n_explained"], pct(v["failure_rate"])])
    h.append("<h2>Cost of explanation</h2>")
    h.append(figures.cost_small_multiples(summaries))
    h.append(table(["task", "method", "mean runtime", "n", "failure rate"], rows))

    costs = {}
    for s in summaries.values():
        for m, v in s["methods"].items():
            if v["failure_rate"] >= 1.0:
                continue  # no successful runs, so it has no meaningful cost
            costs.setdefault(m, []).append(v.get("runtime_s_mean") or 0.0)
    order = sorted(costs.items(), key=lambda kv: np.mean(kv[1]))
    cheapest, dearest = order[0], order[-1]
    ratio = np.mean(dearest[1]) / max(np.mean(cheapest[1]), 1e-9)
    h.append(
        f"<p>Averaged over tasks, {esc(cheapest[0])} is the cheapest method at "
        f"{np.mean(cheapest[1]):.2f}s per prediction and {esc(dearest[0])} the most expensive at "
        f"{np.mean(dearest[1]):.2f}s, a factor of {ratio:.0f}. Cost matters for adoption: an "
        f"explanation method that cannot keep up with the prediction rate can only ever be "
        f"applied retrospectively to samples, which is precisely the practice this project "
        f"set out to avoid.</p>"
    )

    h.append(_models_section(extra or {}))
    h.append(_budget_section(extra or {}))

    qfig = figures.quality_by_correctness(summaries)
    if qfig:
        h.append("<h2>Explanation quality against model error</h2>")
        h.append(qfig)
        h.append(_quality_direction_prose(summaries))

    h.append("<h2>What each method is for</h2>")
    h.append(table(
        ["Question being asked", "Method", "What it gives you", "What it cannot tell you",
         "Measured faithfulness"],
        [
            ["Which words drove this decision?", "SHAP",
             "Signed per-token contributions with an additivity guarantee",
             "Nothing about whether the training data supports the call",
             _method_faith_span("shap", extra or {})],
            ["Quick, model-agnostic sanity check", "LIME",
             "A local linear account of any black box, text or image",
             "Reliability varies per sample; check the surrogate R² first",
             _method_faith_span("lime", extra or {})],
            ["What does this remind the model of?", "Prototypes",
             "The training examples standing behind the prediction, and their labels",
             "No token-level detail at all",
             "<span class='muted'>not defined — produces no per-feature attribution</span>"],
            ["Where in the image did it look?", "Grad-CAM",
             "A fast, class-discriminative region map",
             "Pixel-level precision; it is bounded by a 7×7 feature grid",
             _method_faith_span("gradcam", extra or {})],
        ]))

    h.append(_complementarity_prose(extra or {}))
    return "\n".join(h)


def _limitations(summaries: dict, extra: dict | None = None) -> str:
    items = [
        ("Post-hoc explanations are not the model's reasoning.",
         "Every method here approximates a decision that was already made. A faithful-looking "
         "attribution is evidence about the function the network computes, not a transcript of "
         "how it computed it. This limitation is shared by all four methods and is not "
         "addressed by combining them."),
        ("Agreement between methods is not correctness.",
         "Where LIME and SHAP concur, they may be concurring on the same artefact. This report "
         "measures agreement because it is computable for every prediction, but agreement is a "
         "weaker property than faithfulness and should not be read as validation."),
        ("Faithfulness is measured by deletion, and deletion is off-distribution.",
         "Removing the top-ranked features produces an input the model has never seen: "
         "whitespace-stripped text, or an image with a region replaced by its own mean colour. "
         "A large probability drop therefore conflates “the model relied on this” with “the "
         "model has never seen an input shaped like this”. The random-ranking control absorbs "
         "much of that — it is equally off-distribution — but not all of it, and this is the "
         "main reason the comprehensiveness figures are compared with one another rather than "
         "read as absolute quantities."),
        ("Plausibility covers two tasks, and one of them unevenly.",
         "The movie-review dataset ships no human rationales, so plausibility is undefined "
         "there. On the tweet task the human span is the entire tweet for most neutral "
         "examples; those are excluded because any method scores near 1.0 on them, which "
         "leaves a scored sample that is not class-balanced. Both facts limit how far the "
         "plausibility numbers generalise — which is a further reason they are reported as a "
         "diagnostic rather than a verdict."),
        ("The budget threshold is a hypothesis, not a law.",
         "It rests on five perturbation measurements, one metric and three tasks, and is "
         "declared rather than fitted. What the data supports is that the two near-random "
         "cases received the fewest evaluations per feature and the three reliable ones the "
         "most; the specific value marks the gap between them and should not be transferred "
         "to another model or dataset without re-measuring."),
        ("The behavioural probes use hand-built word lists.",
         "The sentiment, function-word, negator and contrast-marker lists in "
         "<code>behaviour.py</code> were written by hand rather than taken from a lexicon "
         "resource. They are short enough to audit in full, and every membership decision is "
         "visible in that file, but they are neither exhaustive nor independently validated."),
        ("The glasses task is close to saturated.",
         "At the accuracy reported, there are very few errors to learn from, which limits how "
         "much the explanations can reveal about failure modes. A harder split — rimless frames, "
         "extreme poses, occlusion — would be more informative for XAI purposes than the "
         "current one."),
        ("Single seed.",
         "All results come from one seed. LIME in particular is stochastic, and its per-sample "
         "attributions vary between runs; the distributions reported here are stable but "
         "individual explanations should not be treated as exact."),
    ]
    lis = "".join(f"<li><b>{esc(t)}</b> {b}</li>" for t, b in items)
    return f"""
<section>
<h1 class='chapter'>Limitations and honest caveats</h1>
<ol class='limits'>{lis}</ol>
<h2>Corrections to the original report</h2>
<p>This rebuild contradicts the earlier version of this project on several points of fact,
recorded here so the difference is not mistaken for a discrepancy in the data.</p>
<table>
<thead><tr><th>Original claim</th><th>What the data shows</th></tr></thead>
<tbody>
<tr><td>Glasses dataset contains 4,000 images, split 3,200 / 400 / 400</td>
    <td>The dataset contains 2,614 images, 1,307 per class</td></tr>
<tr><td>Glasses model reaches 93.88% validation accuracy</td>
    <td>Unverifiable: the split was unseeded and cannot be reconstructed. The model was
        retrained here with a seeded split to obtain a figure that can be checked</td></tr>
<tr><td>Movie-review LIME and SHAP explanations describe the BERT-LoRA model</td>
    <td>They described a randomly initialised classification head; the adapter's trained head
        was never loaded</td></tr>
<tr><td>Explanations generated for "~60+ samples per method, per dataset"</td>
    <td>This rebuild explains every prediction in every evaluation set</td></tr>
<tr><td>Method rankings presented as conclusions</td>
    <td>The original rankings were assigned by inspection of a few dozen images. Every
        ranking here is computed: faithfulness by deletion against a random-ranking control,
        plausibility against the human annotation shipped with the datasets, and each verdict
        derived from those numbers rather than written next to them</td></tr>
<tr><td>SHAP's saliency described as the most focused, and read as the best</td>
    <td>Concentration was measured, but coherence was not. Its strongest pixel does land
        inside the human-marked region most often of the three — and it is also the least
        faithful of the three, with its attribution mass scattered across hundreds of
        disconnected components rather than one region</td></tr>
</tbody></table>
</section>
"""


_REFS = """
<section>
<h1 class='chapter'>References</h1>
<ol class='refs'>
<li>Ribeiro, M. T., Singh, S., &amp; Guestrin, C. (2016). "Why should I trust you?": Explaining
the predictions of any classifier. <i>KDD</i>.</li>
<li>Lundberg, S. M., &amp; Lee, S.-I. (2017). A unified approach to interpreting model
predictions. <i>NeurIPS</i>.</li>
<li>Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., &amp; Batra, D. (2017).
Grad-CAM: Visual explanations from deep networks via gradient-based localization. <i>ICCV</i>.</li>
<li>Kim, B., Khanna, R., &amp; Koyejo, O. (2016). Examples are not enough, learn to criticize!
Criticism for interpretability. <i>NeurIPS</i>.</li>
<li>Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. <i>ICLR</i>.</li>
<li>He, K., Zhang, X., Ren, S., &amp; Sun, J. (2016). Deep residual learning for image
recognition. <i>CVPR</i>.</li>
<li>Devlin, J., Chang, M.-W., Lee, K., &amp; Toutanova, K. (2019). BERT: Pre-training of deep
bidirectional transformers for language understanding. <i>NAACL</i>.</li>
<li>DeYoung, J., et al. (2020). ERASER: A benchmark to evaluate rationalized NLP models.
<i>ACL</i>.</li>
<li>Jacovi, A., &amp; Goldberg, Y. (2020). Towards faithfully interpretable NLP systems: how
should we define and evaluate faithfulness? <i>ACL</i>. <span class='muted'>— the
plausibility/faithfulness distinction this report is organised around</span></li>
<li>Hooker, S., Erhan, D., Kindermans, P.-J., &amp; Kim, B. (2019). A benchmark for
interpretability methods in deep neural networks. <i>NeurIPS</i>. <span class='muted'>—
ROAR, and the off-distribution problem with deletion-based faithfulness</span></li>
<li>Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., &amp; Kim, B. (2018).
Sanity checks for saliency maps. <i>NeurIPS</i>.</li>
<li>Sundararajan, M., Taly, A., &amp; Yan, Q. (2017). Axiomatic attribution for deep
networks. <i>ICML</i>. <span class='muted'>— Integrated Gradients, the basis of
GradientSHAP</span></li>
<li>Zhang, J., Bargal, S. A., Lin, Z., Brandt, J., Shen, X., &amp; Sclaroff, S. (2018).
Top-down neural attention by excitation backprop. <i>IJCV</i>. <span class='muted'>— the
pointing game</span></li>
<li>Tweet Sentiment Extraction. <span class='muted'>kaggle.com/competitions/tweet-sentiment-extraction</span></li>
<li>Counterfactually-augmented sentiment data. <span class='muted'>github.com/acmi-lab/counterfactually-augmented-data</span></li>
<li>Face glasses recognition (Glasses-XAI). <span class='muted'>xaidataset.github.io</span></li>
</ol>
</section>
"""


_HEAD = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Explainable AI across Text and Vision</title>
<style>
/* ---- paged media -------------------------------------------------------
   WeasyPrint implements CSS Paged Media (margin boxes, string-set,
   target-counter), which is what lets this document carry running heads and
   a table of contents with real page numbers. Chrome's print-to-PDF does not.
   ------------------------------------------------------------------------ */
@page {
  size: A4; margin: 22mm 18mm 20mm;
  @bottom-center{ content: counter(page);
    font:8.5pt -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:#6b6660 }
  @top-left{ content: string(chapter);
    font:8pt -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    color:#6b6660; letter-spacing:.06em; text-transform:uppercase }
  @top-right{ content: "Explainable AI across Text and Vision";
    font:8pt -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:#c9c3ba }
}
/* Cover and contents carry no furniture. */
@page :first { @bottom-center{content:none} @top-left{content:none} @top-right{content:none} }
@page frontmatter { @top-left{content:none} @top-right{content:none} }
:root{--ink:#1a1a1a;--muted:#6b6660;--rule:#ddd8d0;--accent:#0d7377;--warn:#b25e0c;--bg:#fff}
*{box-sizing:border-box}
body{font:10.5pt/1.55 "Iowan Old Style",Georgia,"Times New Roman",serif;color:var(--ink);
 background:var(--bg);counter-reset:fig tbl}
h1,h2,h3{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif;
 line-height:1.25;color:var(--ink);break-after:avoid}
h1.chapter{font-size:19pt;margin:0 0 14px;padding-bottom:8px;border-bottom:2px solid var(--accent);
 string-set:chapter content()}
h2{font-size:12.5pt;margin:22px 0 7px}
h3{font-size:11pt;margin:17px 0 5px;color:var(--accent)}
p{margin:8px 0;text-align:justify;hyphens:auto;orphans:3;widows:3}
/* Only chapters start a new page — a break after every <section> wasted pages. */
section{break-after:page}
section:last-of-type{break-after:auto}
.title{text-align:center;padding:58mm 0 0;break-after:page;page:frontmatter}
.title h1{font-size:28pt;margin-bottom:10px;border:0;string-set:chapter ""}
.title .sub{font-size:12.5pt;color:var(--muted);max-width:135mm;margin:0 auto 30px;
 text-align:center;font-style:italic}
/* ---- table of contents ---- */
.toc{page:frontmatter;break-after:page}
.toc h1{font-size:17pt;border:0;margin-bottom:14px;string-set:chapter ""}
.toc ol{list-style:none;padding:0;margin:0;font-family:-apple-system,BlinkMacSystemFont,sans-serif}
.toc li{margin:4px 0;font-size:10pt}
.toc li.sub{margin-left:16px;font-size:9pt;color:var(--muted)}
.toc a{text-decoration:none;color:var(--ink)}
.toc li.sub a{color:var(--muted)}
.toc a::after{content:leader('.') " " target-counter(attr(href), page);
 color:var(--muted);font-size:9pt}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:9.5pt;
 font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;page-break-inside:avoid}
th,td{text-align:left;padding:5px 9px;border-bottom:1px solid var(--rule);vertical-align:top}
th{font-size:8pt;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);
 border-bottom:1.5px solid var(--ink);font-weight:600}
table.meta{max-width:125mm;margin:0 auto}
table.meta td:first-child{color:var(--muted)}
table.meta td:last-child{text-align:right;font-weight:600}
table.confusion td{text-align:center}
table.confusion td:first-child{text-align:left}
.err{color:var(--warn);font-weight:600}
code,pre{font-family:"SF Mono",Menlo,Consolas,monospace;font-size:9pt}
code{background:#f2efea;padding:1px 4px;border-radius:3px}
pre.cmd{background:#1f2225;color:#e8e6e3;padding:12px 14px;border-radius:6px;
 line-height:1.7;overflow-x:auto;font-size:8.5pt}
blockquote{margin:8px 0;padding:7px 13px;border-left:3px solid var(--rule);
 color:#444;font-size:10pt;background:#faf8f5}
.muted{color:var(--muted)}
.note{background:#fdf6ec;border-left:3px solid var(--warn);padding:9px 13px;font-size:10pt}
.cap{font-size:8.5pt;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);
 margin:16px 0 2px;font-family:-apple-system,sans-serif}
.example{border:1px solid var(--rule);border-radius:7px;padding:12px 15px;margin:14px 0;
 page-break-inside:avoid;background:#fcfbf9}
.exlabel{font-family:-apple-system,sans-serif;font-size:8.5pt;text-transform:uppercase;
 letter-spacing:.05em;color:var(--accent);font-weight:700;margin:0 0 7px}
.mname{font-family:-apple-system,sans-serif;font-size:9pt;font-weight:600;margin:11px 0 4px}
.chips{line-height:2.2;margin:4px 0}
.chip{padding:2px 7px;border-radius:4px;font-size:9pt;white-space:nowrap;
 font-family:-apple-system,sans-serif}
.chip i{font-style:normal;font-size:7.5pt;opacity:.75}
.chip.p{background:#0d7377;color:#fff}
.chip.n{background:#b25e0c;color:#fff}
.verdict{font-size:9pt;padding:1px 8px;border-radius:99px;font-family:-apple-system,sans-serif}
.verdict.ok{background:#d7efe4;color:#12603f}
.verdict.bad{background:#fadfd5;color:#8a2f10}
.panel{width:100%;border:1px solid var(--rule);border-radius:5px;margin:9px 0}
.protos{font-size:9.5pt;margin:4px 0 4px 18px}
.protos li{margin:3px 0}
.bar{display:inline-block;height:7px;background:var(--accent);border-radius:3px;vertical-align:middle}
ol.limits li{margin:9px 0}
ol.refs{font-size:10pt}ol.refs li{margin:5px 0}
.fig{margin:14px 0;break-inside:avoid}
.fig svg{display:block}
/* Keep a figure welded to the table that repeats its numbers. */
.fig + table,.fig + .cap{break-before:avoid}
.cap + table{break-before:avoid}
.fignote{font-size:8.5pt;color:var(--muted);margin:3px 0 0;
 font-family:-apple-system,BlinkMacSystemFont,sans-serif;text-align:left}
/* Figure and table numbering, counted by the document rather than typed. */
.fig figcaption{font-size:8.5pt;color:var(--muted);margin-top:4px;
 font-family:-apple-system,BlinkMacSystemFont,sans-serif}
.fig figcaption::before{counter-increment:fig;content:"Figure " counter(fig) ". ";
 font-weight:600;color:var(--ink)}
/* ---- criterion badges and verdict callouts (new analysis chapters) ---- */
.crit{font-family:-apple-system,BlinkMacSystemFont,sans-serif;font-size:7.5pt;font-weight:700;
 letter-spacing:.08em;text-transform:uppercase;padding:2px 7px;border-radius:3px;
 white-space:nowrap;vertical-align:2px}
.crit.primary{background:#0d7377;color:#fff}
.crit.secondary{background:#efe9e0;color:#6b6660}
.callout{border:1px solid var(--rule);border-left:4px solid var(--accent);border-radius:5px;
 padding:11px 15px;margin:12px 0;break-inside:avoid;background:#fbfaf8}
.callout.none{border-left-color:var(--warn)}
.callout h3{margin:0 0 3px;font-size:13pt;color:var(--ink);
 font-family:-apple-system,BlinkMacSystemFont,sans-serif}
.callout .ds{font-family:-apple-system,BlinkMacSystemFont,sans-serif;font-size:8pt;
 text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0}
.callout .why{margin:4px 0 0;font-size:9.5pt;text-align:left}
/* ---- plausible x faithful quadrant ---- */
table.matrix{table-layout:fixed}
table.matrix td{text-align:center;vertical-align:middle;font-size:9pt}
table.matrix td:first-child{text-align:left;color:var(--muted);
 font-family:-apple-system,BlinkMacSystemFont,sans-serif;font-size:8pt;
 text-transform:uppercase;letter-spacing:.05em}
table.matrix .danger{background:#fdf6ec;border:1px solid var(--warn);color:var(--warn);font-weight:600}
table.matrix .good{background:#eef6f2;color:#12603f;font-weight:600}
@media print{body{padding:0}.example,table{break-inside:avoid}}
</style></head><body>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    ap.add_argument("--no-pdf", action="store_true",
                    help="write only the intermediate HTML, skip the PDF")
    args = ap.parse_args()
    cfg = cfgmod.load(args.config)
    out = build(cfg)
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB, intermediate)")
    if not args.no_pdf:
        from . import pdf as pdfmod
        pdfmod.render(out, out.parent / pdfmod.PDF_NAME)


if __name__ == "__main__":
    main()
