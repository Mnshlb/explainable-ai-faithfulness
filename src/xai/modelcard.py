"""What each classifier is, how it was trained, and how well it does on every split.

The report documented the image model's training in full — it is retrained here by
`train_glasses.py`, which records its own metrics — while the two text models,
trained separately by the author, were described only by architecture. That is
the wrong way round: a reader has more reason to ask how a model they cannot
retrain was produced than one they can.

This module reads the facts back out of the checkpoints rather than restating
them from memory: parameter counts and the LoRA breakdown come from the loaded
network, the adapter configuration from `adapter_config.json`, and accuracy is
measured on every split the task defines rather than the evaluation split alone.
Training hyper-parameters are declarative, in `config.yaml`'s per-task `train:`
block, so they live beside everything else describing a run.

Writes `artifacts/model_cards.json`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import config as cfgmod
from .models import load_model


def _params(net) -> dict:
    total = sum(p.numel() for p in net.parameters())
    out = {"total": int(total)}
    lora = sum(p.numel() for n, p in net.named_parameters() if "lora_" in n)
    head = sum(p.numel() for n, p in net.named_parameters()
               if "classifier" in n or n.startswith("fc."))
    if lora:
        trained = lora + head
        out |= {"lora_adapter": int(lora), "classification_head": int(head),
                "trained": int(trained),
                "trained_pct": round(100.0 * trained / total, 3)}
    else:
        out |= {"trained": int(total), "trained_pct": 100.0}
    return out


def _text_accuracy(cfg, key: str, model, splits: tuple[str, ...]) -> dict:
    """Accuracy on each split the task defines, not only the evaluated one."""
    from .data import load_text_split

    task = cfg.task(key)
    out = {}
    for split in splits:
        try:
            samples = load_text_split(task, split)
        except (FileNotFoundError, KeyError):
            continue
        if not samples:
            continue
        t0 = time.time()
        probs = model.predict_proba([s.text for s in samples])
        pred = probs.argmax(1)
        true = np.asarray([s.label_id for s in samples])
        out[split] = {"n": len(samples),
                      "accuracy": round(float((pred == true).mean()), 4),
                      "seconds": round(time.time() - t0, 1)}
        print(f"    {split:6s} n={len(samples):6,d}  acc={out[split]['accuracy']:.4f}",
              flush=True)
    return out


def _image_accuracy(cfg, key: str, model) -> dict:
    """Per-split accuracy read from the seeded split file written at training time."""
    art = cfg.artifact_dir(key)
    tm = art / "training_metrics.json"
    if not tm.exists():
        return {}
    d = json.loads(tm.read_text())
    n = d.get("n") or {}
    out = {}
    for split, acc_key in (("train", "train_acc_no_aug"),
                           ("val", "best_val_acc"),
                           ("test", "test_acc")):
        if acc_key in d:
            out[split] = {"n": n.get(split), "accuracy": round(float(d[acc_key]), 4)}
    if "generalisation_gap" in d:
        out["generalisation_gap"] = d["generalisation_gap"]
    if "train_seconds" in d:
        out["train_seconds"] = d["train_seconds"]
    return out


def card(cfg, key: str, device: torch.device, measure: bool = True) -> dict:
    task = cfg.task(key)
    print(f"[{key}] loading model", flush=True)
    model = load_model(task, device)
    net = model.model

    out: dict = {
        "task": key,
        "name": task.name,
        "kind": task.kind,
        "labels": list(task.labels),
        "architecture": type(net).__name__,
        "parameters": _params(net),
        # Declared in config.yaml so it travels with the rest of the run
        # description rather than living only in a notebook.
        "training": dict(task.raw.get("train") or {}),
    }

    if task.kind == "text":
        adapter = task.raw.get("lora_dir")
        if adapter:
            p = cfgmod.PROJECT_ROOT / adapter / "adapter_config.json"
            if p.exists():
                a = json.loads(p.read_text())
                out["lora"] = {k: a[k] for k in
                               ("r", "lora_alpha", "lora_dropout", "target_modules",
                                "bias", "task_type", "base_model_name_or_path")
                               if k in a}
        out["max_length"] = task.raw.get("max_length")
        if measure:
            out["accuracy_by_split"] = _text_accuracy(
                cfg, key, model, ("train", "val", "test"))
    else:
        out["image_size"] = task.raw.get("image_size")
        out["accuracy_by_split"] = _image_accuracy(cfg, key, model)

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    ap.add_argument("--task", default="all")
    ap.add_argument("--no-measure", action="store_true",
                    help="skip re-scoring the splits; record architecture only")
    args = ap.parse_args()

    cfg = cfgmod.load(args.config)
    device = cfgmod.get_device(cfg.raw["device"])
    dest = cfg.artifacts / "model_cards.json"
    out = json.loads(dest.read_text()) if dest.exists() else {}

    for key in cfg.task_keys:
        if args.task not in ("all", key):
            continue
        out[key] = card(cfg, key, device, measure=not args.no_measure)
        dest.write_text(json.dumps(out, indent=2))

    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
