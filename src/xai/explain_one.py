"""Explain a single new input, interactively.

Everything else in this package explains a fixed evaluation set. This is the
demo path: hand it a sentence or an image it has never seen and it returns the
prediction together with every applicable explanation, printed to the terminal
and optionally written as a rendered card.

Usage:
    python -m xai.explain_one --task tweet --text "this film was a slog"
    python -m xai.explain_one --task movie --file review.txt
    python -m xai.explain_one --task glasses --image path/to/face.jpg
    python -m xai.explain_one --task tweet --text "..." --save out/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import config as cfgmod
from .data import load_text_split
from .explainers.base import Prediction
from .explainers.image import GradCAM, LimeImage, ShapImage
from .explainers.text import LimeText, PrototypeText, ShapText
from .models import load_model
from .render import render_image_prediction, render_text_prediction

# ANSI colours, disabled when the output is not a terminal.
_TTY = sys.stdout.isatty()


def _c(s: str, code: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s


def _bar(weight: float, scale: float, width: int = 24) -> str:
    """A signed horizontal bar, drawn from a centre line."""
    n = 0 if scale <= 0 else int(round(abs(weight) / scale * width))
    if weight >= 0:
        return " " * width + "│" + _c("█" * n, "36") + " " * (width - n)
    return " " * (width - n) + _c("█" * n, "33") + "│" + " " * width


def _print_tokens(title: str, tokens: list, note: str = "") -> None:
    print(f"\n  {_c(title, '1')}  {_c(note, '2')}")
    if not tokens:
        print("    (none)")
        return
    scale = max(abs(w) for _, w in tokens) or 1.0
    for t, w in tokens:
        label = t.strip()[:22].ljust(22)
        print(f"    {label} {_bar(w, scale)} {w:+.4f}")


def explain_text(cfg: cfgmod.Config, task_key: str, text: str,
                 save: Path | None) -> Prediction:
    task = cfg.task(task_key)
    device = cfgmod.get_device(cfg.raw["device"])
    model = load_model(task, device)

    limit = task.get("char_limit")
    if limit:
        text = text[:limit]

    probs = model.predict_proba([text])[0]
    pid = int(probs.argmax())

    pred = Prediction(
        task=task_key, sample_id="interactive", split=None,
        true_label="(unknown)", true_label_id=-1,
        pred_label=model.labels[pid], pred_label_id=pid,
        confidence=float(probs[pid]),
        probs={l: float(p) for l, p in zip(model.labels, probs)},
        correct=False, text=text,
    )

    print(f"\n{_c('input', '1;4')}\n  {text[:400]}{'...' if len(text) > 400 else ''}")
    print(f"\n{_c('prediction', '1;4')}")
    for l, p in sorted(pred.probs.items(), key=lambda kv: -kv[1]):
        mark = _c("<<", "1;36") if l == pred.pred_label else "  "
        print(f"  {l:<12} {p:.4f}  {'█' * int(p * 30)} {mark}")

    k = cfg.explain["top_k_tokens"]
    lime = LimeText(model, task, k).explain(text, pid)
    _print_tokens("LIME", lime.tokens,
                  f"{lime.runtime_s:.2f}s · surrogate R²="
                  f"{lime.stats.get('surrogate_r2', float('nan')):.3f}")
    pred.explanations["lime"] = lime

    shp = ShapText(model, task, k).explain(text, pid)
    _print_tokens("SHAP", shp.tokens,
                  f"{shp.runtime_s:.2f}s · {shp.stats.get('n_tokens', 0)} tokens")
    pred.explanations["shap"] = shp

    train = load_text_split(task, "train")
    proto = PrototypeText(
        model, task, [s.text for s in train], [s.label_id for s in train],
        k=cfg.explain["n_prototypes"],
        cache_path=cfg.artifact_dir(task_key) / "train_embeddings",
    ).explain_batch([text], [pid])[0]
    pred.explanations["prototype"] = proto

    agreement = proto.stats["neighbour_agreement"]
    print(f"\n  {_c('Nearest training examples', '1')}  "
          f"{_c(f'{agreement:.0%} agree with the prediction', '2')}")
    for n in proto.neighbours:
        tick = _c("✓", "32") if n["agrees_with_prediction"] else _c("✗", "33")
        print(f"    {tick} sim {n['similarity']:.3f}  [{n['label']}]  {n['text'][:110]}")

    if save:
        p = render_text_prediction(pred, save)
        print(f"\n  written to {p}")
    return pred


def explain_image(cfg: cfgmod.Config, task_key: str, path: Path,
                  save: Path | None) -> Prediction:
    task = cfg.task(task_key)
    device = cfgmod.get_device(cfg.raw["device"])
    model = load_model(task, device)

    x = model.load_tensor(path)
    probs = model.predict_proba_tensor(x.unsqueeze(0))[0]
    pid = int(probs.argmax())

    pred = Prediction(
        task=task_key, sample_id=path.stem, split=None,
        true_label="(unknown)", true_label_id=-1,
        pred_label=model.labels[pid], pred_label_id=pid,
        confidence=float(probs[pid]),
        probs={l: float(p) for l, p in zip(model.labels, probs)},
        correct=False, image_path=str(path),
    )

    print(f"\n{_c('input', '1;4')}\n  {path}")
    print(f"\n{_c('prediction', '1;4')}")
    for l, p in sorted(pred.probs.items(), key=lambda kv: -kv[1]):
        mark = _c("<<", "1;36") if l == pred.pred_label else "  "
        print(f"  {l:<12} {p:.4f}  {'█' * int(p * 30)} {mark}")

    sal: dict[str, np.ndarray] = {}
    print(f"\n{_c('explanations', '1;4')}")
    for name, ex in (("gradcam", GradCAM(model, task)),
                     ("shap", ShapImage(model, task)),
                     ("lime", LimeImage(model, task))):
        e, m = ex.explain(x, pid)
        pred.explanations[name] = e
        sal[name] = m
        if e.error:
            print(f"  {name:<9} failed: {e.error}")
            continue
        st = e.stats
        extra = ""
        if "surrogate_r2" in st:
            extra = f"surrogate R²={st['surrogate_r2']:.3f}"
        elif "convergence_delta" in st:
            extra = f"convergence Δ={st['convergence_delta']:.3f}"
        print(f"  {name:<9} {e.runtime_s:6.2f}s  "
              f"focus(top-5% mass)={st.get('top5pct_mass', 0):.3f}  "
              f"peak={st.get('peak_yx')}  {extra}")

    if save:
        img = x.permute(1, 2, 0).cpu().numpy()
        p = render_image_prediction(pred, img, sal, save)
        print(f"\n  written to {p}")
    return pred


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, help="tweet | movie | glasses")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="text to classify and explain")
    src.add_argument("--file", help="read the text from this file")
    src.add_argument("--image", help="image to classify and explain")
    ap.add_argument("--save", help="directory to write the rendered card into")
    ap.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    args = ap.parse_args()

    cfg = cfgmod.load(args.config)
    save = Path(args.save) if args.save else None

    if args.image:
        explain_image(cfg, args.task, Path(args.image), save)
    else:
        text = Path(args.file).read_text() if args.file else args.text
        explain_text(cfg, args.task, text, save)
    print()


if __name__ == "__main__":
    main()
