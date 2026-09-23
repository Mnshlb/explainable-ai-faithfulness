"""Migrate existing artifacts to the current output layout.

The layout changed in three ways, described in `layout.py`: saliency maps moved
out of `.npz` archives into per-sample directories of `.npy` files, image
explanations gained an HTML card alongside the panel, and both kinds of output
moved into a tree that mirrors the input tree and preserves input order.

Re-running `xai.run` would produce the new layout, but it would also spend hours
recomputing explanations that are already correct. This converts what is on disk
instead: it moves and renames files, unpacks the archives, generates the missing
HTML cards from the stored records, and rewrites the paths inside
`predictions.jsonl`. It is idempotent — running it twice is harmless — and it
never recomputes an explanation.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np

from . import config as cfgmod
from . import layout
from .explainers.base import Explanation, Prediction
from .render import build_index, render_image_card


def _rewrite_jsonl(path: Path, records: list[dict]) -> None:
    """Replace the file only once every record has been rewritten."""
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def migrate_text(out_dir: Path, dry_run: bool = False) -> dict:
    """Rename text cards to `<row>_<id>.html`, so listing matches input order."""
    jsonl = out_dir / "predictions.jsonl"
    if not jsonl.exists():
        return {}
    records = [json.loads(l) for l in open(jsonl)]
    exp = out_dir / "explanations"
    moved = skipped = 0

    for i, r in enumerate(records, 1):
        stem = layout.text_stem(i, r["sample_id"], len(records))
        dest = exp / f"{stem}.html"
        rel = str(dest.relative_to(out_dir))
        # Where the card currently is, old name first.
        src = exp / f"{r['sample_id']}.html"
        if dest.exists():
            skipped += 1
        elif src.exists():
            if not dry_run:
                src.rename(dest)
            moved += 1
        else:
            continue
        for e in r.get("explanations", {}).values():
            if e.get("render"):
                e["render"] = rel

    if not dry_run:
        _rewrite_jsonl(jsonl, records)
    return {"records": len(records), "renamed": moved, "already_done": skipped}


def migrate_image(out_dir: Path, dry_run: bool = False) -> dict:
    """Unpack .npz archives into directories, and add the missing HTML cards."""
    jsonl = out_dir / "predictions.jsonl"
    if not jsonl.exists():
        return {}
    records = [json.loads(l) for l in open(jsonl)]
    exp, sal = out_dir / "explanations", out_dir / "saliency"
    unpacked = kept = panels = cards = 0

    for r in records:
        sid = r["sample_id"]
        rel_stem = layout.image_stem(sid)

        # --- saliency: archive -> directory of .npy -----------------------
        sdir = layout.saliency_dir(out_dir, sid)
        old_npz = sal / f"{sid}.npz"
        if layout.has_arrays(sdir):
            kept += 1
        elif old_npz.exists():
            if not dry_run:
                with np.load(old_npz) as z:
                    layout.save_saliency(sdir, {k: z[k] for k in z.files})
                old_npz.unlink()
            unpacked += 1
        sal_rel = str(sdir.relative_to(out_dir))

        # --- panel: flat -> mirrored tree ---------------------------------
        new_png = exp / f"{rel_stem}.png"
        old_png = exp / f"{sid}.png"
        if not new_png.exists() and old_png.exists():
            if not dry_run:
                new_png.parent.mkdir(parents=True, exist_ok=True)
                old_png.rename(new_png)
            panels += 1

        # --- the HTML card images never had ------------------------------
        card = new_png.with_suffix(".html")
        if not card.exists() and not dry_run and new_png.exists():
            pred = Prediction(
                task=r["task"], sample_id=sid, split=r.get("split"),
                true_label=r["true_label"], true_label_id=r["true_label_id"],
                pred_label=r["pred_label"], pred_label_id=r["pred_label_id"],
                confidence=r["confidence"], probs=r["probs"], correct=r["correct"],
                image_path=r.get("image_path"),
            )
            for name, d in r.get("explanations", {}).items():
                pred.explanations[name] = Explanation(
                    method=name, kind="saliency", runtime_s=d.get("runtime_s", 0.0),
                    stats=d.get("stats", {}), error=d.get("error"),
                )
            render_image_card(pred, new_png)
            cards += 1

        # The record points at the readable card; the panel sits beside it.
        card_rel = str(card.relative_to(out_dir))
        for e in r.get("explanations", {}).values():
            e["arrays"] = {"dir": sal_rel}
            if e.get("render"):
                e["render"] = card_rel

    if not dry_run:
        _rewrite_jsonl(jsonl, records)
        # Remove the now-empty flat saliency directory if nothing is left in it.
        for d in (sal,):
            if d.exists() and not any(d.iterdir()):
                d.rmdir()
    return {"records": len(records), "archives_unpacked": unpacked,
            "already_unpacked": kept, "panels_moved": panels, "cards_written": cards}


def migrate_embedding_cache(out_dir: Path, dry_run: bool = False) -> dict:
    """`train_embeddings.npz` -> `train_embeddings/emb.npy`.

    The cache is written at full float32 precision: it feeds a cosine
    similarity, so unlike a saliency map it is not safe to narrow.
    """
    old = out_dir / "train_embeddings.npz"
    new = out_dir / "train_embeddings"
    if layout.has_arrays(new):
        return {"embedding_cache": "already converted"}
    if not old.exists():
        return {}
    if not dry_run:
        with np.load(old) as z:
            layout.save_arrays(new, {k: z[k] for k in z.files})
        old.unlink()
    return {"embedding_cache": "converted"}


def unpack_legacy_outputs(root: Path, dry_run: bool = False) -> dict:
    """Extract the notebook-era `outputs/*.zip` into folders of the same name.

    These predate the rebuild and nothing regenerates them, so they are
    extracted rather than discarded — they are the record of what the original
    analysis produced, and the report's corrections table refers to them.
    """
    outputs = root / "outputs"
    if not outputs.is_dir():
        return {}
    done = skipped = 0
    for z in sorted(outputs.rglob("*.zip")):
        # "lime_explanations_html (2).zip" -> "lime_explanations_html"
        name = re.sub(r"\s*\(\d+\)$", "", z.stem).strip()
        dest = z.parent / name
        if dest.is_dir() and any(dest.iterdir()):
            skipped += 1
            continue
        if not dry_run:
            dest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(z) as zf:
                for info in zf.infolist():
                    # macOS resource forks carry no content and clutter the tree.
                    if info.filename.startswith("__MACOSX/") or info.is_dir():
                        continue
                    target = dest / Path(info.filename).name
                    with zf.open(info) as src, open(target, "wb") as fh:
                        shutil.copyfileobj(src, fh)
            z.unlink()
        done += 1
    return {"legacy_zips_extracted": done, "already_extracted": skipped}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    ap.add_argument("--task", default="all")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without touching anything")
    args = ap.parse_args()

    cfg = cfgmod.load(args.config)
    tag = "(dry run) " if args.dry_run else ""
    for key in cfg.task_keys:
        if args.task not in ("all", key):
            continue
        out_dir = cfg.artifact_dir(key)
        if not (out_dir / "predictions.jsonl").exists():
            continue
        kind = cfg.task(key).kind
        fn = migrate_text if kind == "text" else migrate_image
        stats = {**fn(out_dir, args.dry_run),
                 **migrate_embedding_cache(out_dir, args.dry_run)}
        if stats:
            print(f"[{key}] {tag}" + "  ".join(f"{k}={v}" for k, v in stats.items()))

    # An index per task, listing every input in input order with each method's
    # outcome — so coverage is visible rather than something to count files for.
    for key in cfg.task_keys:
        if args.task not in ("all", key):
            continue
        out_dir = cfg.artifact_dir(key)
        jsonl = out_dir / "predictions.jsonl"
        if not jsonl.exists() or args.dry_run:
            continue
        records = [json.loads(l) for l in open(jsonl)]
        idx = build_index(key, records, out_dir / "explanations")
        print(f"[{key}] index -> {idx.relative_to(cfgmod.PROJECT_ROOT)}  ({len(records):,} rows)")

    if args.task == "all":
        legacy = unpack_legacy_outputs(cfgmod.PROJECT_ROOT, args.dry_run)
        if legacy:
            print(f"[outputs] {tag}" + "  ".join(f"{k}={v}" for k, v in legacy.items()))


if __name__ == "__main__":
    main()
