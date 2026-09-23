"""Explain every prediction a model makes.

The runner is resumable by design. Explaining thousands of predictions with
LIME and SHAP takes hours, so each record is appended to disk the moment it
is finished and a restart skips whatever is already there. Killing the
process is always safe.

Usage:
    python -m xai.run --task tweet
    python -m xai.run --task glasses --limit 20
    python -m xai.run --all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import config as cfgmod
from . import layout
from .data import load_image_samples, load_text_split
from .explainers.base import Explanation, Prediction
from .explainers.image import GradCAM, LimeImage, ShapImage
from .explainers.text import LimeText, PrototypeText, ShapText
from .models import load_model
from .render import build_index, render_image_prediction, render_text_prediction


def _done_ids(path: Path) -> set[str]:
    """Sample ids already present in a partially written JSONL."""
    if not path.exists():
        return set()
    done = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["sample_id"])
            except (json.JSONDecodeError, KeyError):
                continue  # tolerate a torn final line from a hard kill
    return done


def _eta(done: int, total: int, elapsed: float) -> str:
    if done == 0:
        return "?"
    remaining = (total - done) * elapsed / done
    h, rem = divmod(int(remaining), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"




# PyTorch's MPS caching allocator grows as it sees new tensor shapes, and LIME
# feeds it a differently-shaped perturbation batch for every single sample. Over
# thousands of samples the cache becomes a significant, never-released claim on
# unified memory. Releasing it periodically keeps the footprint flat; the call
# costs a synchronise, so it is amortised over a block of samples rather than
# run every time.
_CACHE_CLEAR_EVERY = 50


def _release_cache(i: int, device: torch.device) -> None:
    if i % _CACHE_CLEAR_EVERY:
        return
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def _mem_note(device: torch.device) -> str:
    """Current allocator footprint, so degradation is visible in the log."""
    try:
        if device.type == "mps":
            return f"  mps {torch.mps.driver_allocated_memory() / 2**30:.2f}GiB"
        if device.type == "cuda":
            return f"  cuda {torch.cuda.memory_reserved() / 2**30:.2f}GiB"
    except Exception:  # noqa: BLE001 - diagnostics must never break a run
        pass
    return ""


# Saliency persistence lives in layout.py: one .npy per method inside a
# directory named for the sample, so nothing is stored inside an archive.


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------


def run_text(cfg: cfgmod.Config, task_key: str, methods: list[str], limit: int | None,
             render: bool, split: str = "test") -> Path:
    task = cfg.task(task_key)
    device = cfgmod.get_device(cfg.raw["device"])
    print(f"[{task_key}] loading model on {device}")
    model = load_model(task, device)

    samples = load_text_split(task, split)
    if limit:
        samples = samples[:limit]
    print(f"[{task_key}] {len(samples)} samples in split '{split}'")

    out_dir = cfg.artifact_dir(task_key)
    jsonl = out_dir / "predictions.jsonl"
    done = _done_ids(jsonl)
    if done:
        print(f"[{task_key}] resuming, {len(done)} already explained")
    # Position in the input file, kept so that a resumed run names its cards the
    # same way a single-pass run would, and a directory listing reproduces the
    # order of the source CSV rather than a hash order.
    row_of = {s.sample_id: i for i, s in enumerate(samples, 1)}
    n_total = len(samples)
    todo = [s for s in samples if s.sample_id not in done]
    if not todo:
        print(f"[{task_key}] nothing to do")
        return jsonl

    # --- predictions for everything we still need -------------------------
    t0 = time.time()
    probs = model.predict_proba([s.text for s in todo])
    pred_ids = probs.argmax(1)
    print(f"[{task_key}] predicted {len(todo)} samples in {time.time() - t0:.1f}s")

    # --- explainers -------------------------------------------------------
    k_tokens = cfg.explain["top_k_tokens"]
    lime = LimeText(model, task, k_tokens) if "lime" in methods else None
    shap_x = ShapText(model, task, k_tokens) if "shap" in methods else None

    proto = None
    if "prototype" in methods:
        print(f"[{task_key}] embedding training set for prototype retrieval...")
        t1 = time.time()
        train = load_text_split(task, "train")
        proto = PrototypeText(
            model, task,
            [s.text for s in train], [s.label_id for s in train],
            k=cfg.explain["n_prototypes"],
            cache_path=out_dir / "train_embeddings",
        )
        print(f"[{task_key}] embedded {len(train)} training texts in {time.time() - t1:.1f}s")

    render_dir = out_dir / "explanations"
    t0 = time.time()
    with open(jsonl, "a", encoding="utf-8") as fh:
        for i, (s, pid) in enumerate(zip(todo, pred_ids), 1):
            pid = int(pid)
            pred = Prediction(
                task=task_key, sample_id=s.sample_id, split=split,
                true_label=s.label, true_label_id=s.label_id,
                pred_label=model.labels[pid], pred_label_id=pid,
                confidence=float(probs[i - 1][pid]),
                probs={l: float(p) for l, p in zip(model.labels, probs[i - 1])},
                correct=(pid == s.label_id), text=s.text,
            )
            if lime:
                pred.explanations["lime"] = lime.explain(s.text, pid)
            if shap_x:
                pred.explanations["shap"] = shap_x.explain(s.text, pid)
            if proto:
                pred.explanations["prototype"] = proto.explain_batch([s.text], [pid])[0]
            if render:
                stem = layout.text_stem(row_of[s.sample_id], s.sample_id, n_total)
                card = render_text_prediction(pred, render_dir, stem)
                rel = str(card.relative_to(out_dir))
                for e in pred.explanations.values():
                    e.render = rel

            fh.write(json.dumps(pred.to_dict(), ensure_ascii=False) + "\n")
            fh.flush()  # a kill at any point loses at most the current sample
            _release_cache(i, device)

            if i % 25 == 0 or i == len(todo):
                el = time.time() - t0
                print(
                    f"[{task_key}] {i}/{len(todo)}  "
                    f"{el / i:.2f}s/sample  eta {_eta(i, len(todo), el)}"
                    f"{_mem_note(device)}",
                    flush=True,
                )
    print(f"[{task_key}] done -> {jsonl}")
    return jsonl


# --------------------------------------------------------------------------
# Image
# --------------------------------------------------------------------------


def _redo_image(cfg, task_key, task, model, samples, methods, out_dir, jsonl, render) -> Path:
    """Recompute selected image methods in place over an existing JSONL."""
    if not jsonl.exists():
        raise FileNotFoundError(f"--redo needs existing records at {jsonl}")

    records = {}
    order = []
    with open(jsonl, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            records[r["sample_id"]] = r
            order.append(r["sample_id"])
    by_id = {s.sample_id: s for s in samples}
    print(f"[{task_key}] redoing {methods} over {len(records)} existing records")

    explainers = {
        "gradcam": GradCAM(model, task) if "gradcam" in methods else None,
        "shap": ShapImage(model, task) if "shap" in methods else None,
        "lime": LimeImage(model, task) if "lime" in methods else None,
    }
    array_dir = out_dir / "saliency"
    render_dir = out_dir / "explanations"
    t0 = time.time()

    for i, sid in enumerate(order, 1):
        r = records[sid]
        s = by_id.get(sid)
        if s is None:
            continue
        x = model.load_tensor(s.path)
        pid = r["pred_label_id"]

        sdir = layout.saliency_dir(out_dir, sid)
        # Keep the maps we are not recomputing.
        sal: dict[str, np.ndarray] = {k: v.astype(np.float32)
                                      for k, v in layout.load_arrays(sdir).items()}

        for name, ex in explainers.items():
            if ex is None:
                continue
            e, maps = ex.explain(x, pid)
            e.arrays = {"dir": str(sdir.relative_to(out_dir))}
            r["explanations"][name] = e.to_dict()
            sal.update(maps)

        layout.save_saliency(sdir, sal)

        if render:
            pred = Prediction(
                task=task_key, sample_id=sid, split=r.get("split"),
                true_label=r["true_label"], true_label_id=r["true_label_id"],
                pred_label=r["pred_label"], pred_label_id=pid,
                confidence=r["confidence"], probs=r["probs"], correct=r["correct"],
                image_path=r.get("image_path"),
            )
            for name, d in r["explanations"].items():
                pred.explanations[name] = Explanation(
                    method=name, kind="saliency",
                    runtime_s=d.get("runtime_s", 0.0), stats=d.get("stats", {}),
                    error=d.get("error"),
                )
            img = x.permute(1, 2, 0).cpu().numpy()
            png = render_image_prediction(pred, img, sal, render_dir)
            for name in r["explanations"]:
                r["explanations"][name]["render"] = str(png.relative_to(out_dir))

        _release_cache(i, model.device)
        if i % 10 == 0 or i == len(order):
            el = time.time() - t0
            print(f"[{task_key}] redo {i}/{len(order)}  {el / i:.2f}s/image  "
                  f"eta {_eta(i, len(order), el)}{_mem_note(model.device)}", flush=True)

    tmp = jsonl.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for sid in order:
            fh.write(json.dumps(records[sid], ensure_ascii=False) + "\n")
    tmp.replace(jsonl)  # atomic swap, so an interrupted rewrite cannot truncate
    print(f"[{task_key}] redo done -> {jsonl}")
    return jsonl


def run_image(cfg: cfgmod.Config, task_key: str, methods: list[str], limit: int | None,
              render: bool, redo: bool = False) -> Path:
    task = cfg.task(task_key)
    device = cfgmod.get_device(cfg.raw["device"])
    print(f"[{task_key}] loading model on {device}")
    model = load_model(task, device)

    samples = load_image_samples(task)

    # Tag each image with the split it came from so the report can quote
    # test-set-only figures even though every image gets explained.
    split_of: dict[str, str] = {}
    sf = cfgmod.PROJECT_ROOT / task.get("split_file", "")
    if sf.is_file():
        data = json.loads(sf.read_text())["sample_ids"]
        for name, ids in data.items():
            for i in ids:
                split_of[i] = name
        print(f"[{task_key}] split map loaded: "
              + ", ".join(f"{k}={len(v)}" for k, v in data.items()))

    if limit:
        samples = samples[:limit]
    print(f"[{task_key}] {len(samples)} images")

    out_dir = cfg.artifact_dir(task_key)
    jsonl = out_dir / "predictions.jsonl"

    if redo:
        # Recompute only the requested methods over records that already exist,
        # keeping every other method's result. Used when one explainer is fixed
        # and rerunning the rest would be pure waste.
        return _redo_image(cfg, task_key, task, model, samples, methods, out_dir,
                           jsonl, render)

    done = _done_ids(jsonl)
    if done:
        print(f"[{task_key}] resuming, {len(done)} already explained")
    todo = [s for s in samples if s.sample_id not in done]
    if not todo:
        print(f"[{task_key}] nothing to do")
        return jsonl

    gradcam = GradCAM(model, task) if "gradcam" in methods else None
    shap_x = ShapImage(model, task) if "shap" in methods else None
    lime = LimeImage(model, task) if "lime" in methods else None

    render_dir = out_dir / "explanations"
    array_dir = out_dir / "saliency"
    array_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with open(jsonl, "a", encoding="utf-8") as fh:
        for i, s in enumerate(todo, 1):
            x = model.load_tensor(s.path)
            p = model.predict_proba_tensor(x.unsqueeze(0))[0]
            pid = int(p.argmax())

            pred = Prediction(
                task=task_key, sample_id=s.sample_id, split=split_of.get(s.sample_id),
                true_label=s.label, true_label_id=s.label_id,
                pred_label=model.labels[pid], pred_label_id=pid,
                confidence=float(p[pid]),
                probs={l: float(v) for l, v in zip(model.labels, p)},
                correct=(pid == s.label_id), image_path=str(s.path.relative_to(cfgmod.PROJECT_ROOT)),
            )

            sal: dict[str, np.ndarray] = {}
            for name, ex in (("gradcam", gradcam), ("shap", shap_x), ("lime", lime)):
                if ex is None:
                    continue
                e, maps = ex.explain(x, pid)
                pred.explanations[name] = e
                sal.update(maps)  # a method may emit more than one array

            sdir = layout.save_saliency(layout.saliency_dir(out_dir, s.sample_id), sal)
            for name in pred.explanations:
                pred.explanations[name].arrays = {"dir": str(sdir.relative_to(out_dir))}

            if render:
                img = x.permute(1, 2, 0).cpu().numpy()
                png = render_image_prediction(pred, img, sal, render_dir)
                for name in pred.explanations:
                    pred.explanations[name].render = str(png.relative_to(out_dir))

            fh.write(json.dumps(pred.to_dict(), ensure_ascii=False) + "\n")
            fh.flush()
            _release_cache(i, device)

            if i % 10 == 0 or i == len(todo):
                el = time.time() - t0
                print(
                    f"[{task_key}] {i}/{len(todo)}  "
                    f"{el / i:.2f}s/image  eta {_eta(i, len(todo), el)}"
                    f"{_mem_note(device)}",
                    flush=True,
                )
    print(f"[{task_key}] done -> {jsonl}")
    return jsonl


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", help="tweet | movie | glasses")
    ap.add_argument("--all", action="store_true", help="run every task in turn")
    ap.add_argument("--methods", help="comma-separated subset of methods")
    ap.add_argument("--limit", type=int, help="only the first N samples (for smoke tests)")
    ap.add_argument("--split", default="test", help="text split to explain")
    ap.add_argument("--no-render", action="store_true", help="skip HTML/PNG rendering")
    ap.add_argument("--redo", action="store_true",
                    help="recompute --methods over existing records instead of skipping them")
    ap.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    args = ap.parse_args()

    cfg = cfgmod.load(args.config)
    if not args.task and not args.all:
        ap.error("pass --task or --all")
    tasks = cfg.task_keys if args.all else [args.task]

    for key in tasks:
        task = cfg.task(key)
        default = (
            cfg.explain["methods_image"] if task.kind == "image"
            else cfg.explain["methods_text"]
        )
        methods = args.methods.split(",") if args.methods else default
        print(f"\n=== {key}: {task.name} ===")
        print(f"methods: {', '.join(methods)}")
        runner = run_image if task.kind == "image" else run_text
        kwargs = {"redo": args.redo} if task.kind == "image" else {"split": args.split}
        runner(cfg, key, methods, args.limit, not args.no_render, **kwargs)


if __name__ == "__main__":
    sys.exit(main())
