"""Retrain the glasses ResNet-18 with a reproducible, stratified split.

The original model was trained with an unseeded ``random_split``, so which
images were held out cannot be recovered and no honest test accuracy can be
quoted for it. This script fixes that: the split is stratified, seeded and
written to disk alongside the checkpoint, so every number in the report can
be regenerated and every explained image can be tagged with the split it
came from.

Usage:
    python -m xai.train_glasses [--epochs N] [--config path]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from . import config as cfgmod
from .data import ImageSample, load_image_samples


class GlassesDataset(Dataset):
    """Images loaded from disk with an optional training-time augmentation."""

    def __init__(self, samples: list[ImageSample], size: int, augment: dict | None):
        from torchvision import transforms

        self.samples = samples
        steps: list = [transforms.Resize((size, size))]
        if augment:
            if augment.get("hflip"):
                steps.append(transforms.RandomHorizontalFlip(augment["hflip"]))
            if augment.get("rotation"):
                steps.append(transforms.RandomRotation(augment["rotation"]))
            if augment.get("color_jitter"):
                j = augment["color_jitter"]
                steps.append(transforms.ColorJitter(brightness=j, contrast=j))
        steps.append(transforms.ToTensor())
        # Normalisation is applied inside the model wrapper so that explanations
        # stay in pixel space; training must therefore match that convention.
        steps.append(
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )
        self.tf = transforms.Compose(steps)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        from PIL import Image

        s = self.samples[i]
        with Image.open(s.path) as im:
            x = self.tf(im.convert("RGB"))
        return x, s.label_id


def stratified_split(
    samples: list[ImageSample], fractions: list[float], seed: int
) -> dict[str, list[int]]:
    """Split indices per class so every subset keeps the original class balance."""
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = {}
    for i, s in enumerate(samples):
        by_class.setdefault(s.label_id, []).append(i)

    out: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    f_train, f_val, _ = fractions
    for label_id in sorted(by_class):
        idx = np.array(by_class[label_id])
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(f_train * n))
        n_val = int(round(f_val * n))
        out["train"] += idx[:n_train].tolist()
        out["val"] += idx[n_train : n_train + n_val].tolist()
        out["test"] += idx[n_train + n_val :].tolist()

    for k in out:
        out[k] = sorted(out[k])
    return out


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    preds, ys = [], []
    for x, y in loader:
        logits = model(x.to(device))
        preds.append(logits.argmax(-1).cpu().numpy())
        ys.append(y.numpy())
    p, y = np.concatenate(preds), np.concatenate(ys)
    return float((p == y).mean()), p, y


def confusion(p: np.ndarray, y: np.ndarray, n: int) -> list[list[int]]:
    m = np.zeros((n, n), dtype=int)
    for t, q in zip(y, p):
        m[t, q] += 1
    return m.tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = cfgmod.load(args.config)
    task = cfg.task("glasses")
    tcfg = task.train
    device = cfgmod.get_device(cfg.raw["device"])
    epochs = args.epochs or tcfg["epochs"]

    print(f"device={device}  seed={cfg.seed}  epochs={epochs}")

    samples = load_image_samples(task)
    print(f"images: {len(samples)}  classes: {Counter(s.label for s in samples)}")

    split = stratified_split(samples, tcfg["split"], cfg.seed)
    split_path = cfg.artifact_dir("glasses") / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "seed": cfg.seed,
                "fractions": tcfg["split"],
                "sample_ids": {
                    k: [samples[i].sample_id for i in v] for k, v in split.items()
                },
            },
            indent=2,
        )
    )
    print(
        "split -> "
        + "  ".join(f"{k}={len(v)}" for k, v in split.items())
        + f"  (written to {split_path.relative_to(cfgmod.PROJECT_ROOT)})"
    )

    size, bs = task.image_size, tcfg["batch_size"]
    sets = {
        "train": GlassesDataset([samples[i] for i in split["train"]], size, tcfg["augment"]),
        "val": GlassesDataset([samples[i] for i in split["val"]], size, None),
        "test": GlassesDataset([samples[i] for i in split["test"]], size, None),
    }
    loaders = {
        k: DataLoader(v, batch_size=bs, shuffle=(k == "train"), num_workers=4)
        for k, v in sets.items()
    }

    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT if tcfg["pretrained"] else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, task.n_labels)
    model = model.to(device)

    opt = torch.optim.Adam(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )
    lossf = nn.CrossEntropyLoss()

    history, best_val, best_state = [], -1.0, None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot, correct, loss_sum = 0, 0, 0.0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = lossf(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(-1) == y).sum().item()
            tot += y.size(0)
        tr_acc, tr_loss = correct / tot, loss_sum / tot
        val_acc, _, _ = evaluate(model, loaders["val"], device)
        history.append(
            {"epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc, "val_acc": val_acc}
        )
        flag = ""
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = "  <- best"
        print(
            f"epoch {ep:2d}/{epochs}  loss {tr_loss:.4f}  "
            f"train {tr_acc:.4f}  val {val_acc:.4f}{flag}"
        )

    # Report the model actually selected, not the last epoch's.
    model.load_state_dict(best_state)
    test_acc, p, y = evaluate(model, loaders["test"], device)
    cm = confusion(p, y, task.n_labels)
    train_acc_clean, _, _ = evaluate(model, loaders["train"], device)
    elapsed = time.time() - t0

    print(f"\nselected by val accuracy = {best_val:.4f}")
    print(f"TEST accuracy = {test_acc:.4f}  (n={len(y)})")
    print(f"train accuracy (no augmentation) = {train_acc_clean:.4f}")
    print(f"generalisation gap = {train_acc_clean - test_acc:+.4f}")
    print(f"confusion matrix (rows=true {task.labels}): {cm}")

    out = cfgmod.PROJECT_ROOT / "models" / "resnet18_glasses_v2.pt"
    torch.save(
        {
            "state_dict": best_state,
            "arch": "resnet18",
            "labels": task.labels,
            "normalize": True,
            "image_size": size,
            "seed": cfg.seed,
        },
        out,
    )

    metrics = {
        "seed": cfg.seed,
        "epochs": epochs,
        "device": str(device),
        "train_seconds": round(elapsed, 1),
        "n": {k: len(v) for k, v in split.items()},
        "best_val_acc": best_val,
        "test_acc": test_acc,
        "train_acc_no_aug": train_acc_clean,
        "generalisation_gap": train_acc_clean - test_acc,
        "confusion_matrix": cm,
        "labels": task.labels,
        "history": history,
    }
    mpath = cfg.artifact_dir("glasses") / "training_metrics.json"
    mpath.write_text(json.dumps(metrics, indent=2))
    print(f"\nsaved {out.relative_to(cfgmod.PROJECT_ROOT)}")
    print(f"saved {mpath.relative_to(cfgmod.PROJECT_ROOT)}")
    print(f"total time {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
