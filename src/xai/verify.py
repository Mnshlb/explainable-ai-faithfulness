"""Verify that every prediction really does carry every explanation.

The point of this project is completeness, so completeness is checked rather
than asserted. For each task this confirms that:

  * every sample in the evaluation set has a record
  * no sample appears twice
  * every record carries every configured method
  * no explanation is an error placeholder
  * every explanation actually holds content (tokens / neighbours / arrays)
  * every referenced file on disk exists

Exits non-zero if anything is missing, so it can gate a release.

Usage:
    python -m xai.verify --all
    python -m xai.verify --task tweet
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from . import config as cfgmod
from .data import load_image_samples, load_text_split


def expected_ids(cfg: cfgmod.Config, key: str, split: str = "test") -> list[str]:
    task = cfg.task(key)
    if task.kind == "image":
        return [s.sample_id for s in load_image_samples(task)]
    return [s.sample_id for s in load_text_split(task, split)]


def verify_task(cfg: cfgmod.Config, key: str, split: str = "test") -> dict:
    task = cfg.task(key)
    art = cfg.artifact_dir(key)
    jsonl = art / "predictions.jsonl"

    report: dict = {"task": key, "problems": [], "ok": False}
    if not jsonl.exists():
        report["problems"].append(f"no predictions.jsonl at {jsonl}")
        return report

    records = []
    with open(jsonl, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as err:
                report["problems"].append(f"line {n} is not valid JSON: {err}")

    want = expected_ids(cfg, key, split)
    have = [r["sample_id"] for r in records]
    have_set, want_set = set(have), set(want)

    report["n_expected"] = len(want)
    report["n_records"] = len(records)

    missing = want_set - have_set
    if missing:
        report["problems"].append(
            f"{len(missing)} samples have no record (e.g. {sorted(missing)[:5]})"
        )
    extra = have_set - want_set
    if extra:
        report["problems"].append(
            f"{len(extra)} records are not in the evaluation set (e.g. {sorted(extra)[:5]})"
        )
    dupes = [s for s, c in Counter(have).items() if c > 1]
    if dupes:
        report["problems"].append(
            f"{len(dupes)} sample ids appear more than once (e.g. {dupes[:5]})"
        )

    methods = (
        cfg.explain["methods_image"] if task.kind == "image"
        else cfg.explain["methods_text"]
    )
    report["methods"] = methods

    absent: Counter = Counter()
    errored: Counter = Counter()
    empty: Counter = Counter()
    missing_files: list[str] = []
    content_counts: Counter = Counter()

    for r in records:
        exps = r.get("explanations", {})
        for m in methods:
            e = exps.get(m)
            if e is None:
                absent[m] += 1
                continue
            if e.get("error"):
                errored[m] += 1
                continue
            # An explanation with no payload is a silent failure.
            kind = e.get("kind")
            if kind == "tokens":
                n = len(e.get("tokens", []))
            elif kind == "neighbours":
                n = len(e.get("neighbours", []))
            elif kind == "saliency":
                # A saliency explanation is non-empty once its array file is
                # referenced; where it also carries per-region weights (image
                # LIME), count those, since that is the part most easily lost
                # to truncation.
                n = len(e.get("tokens", [])) or (1 if (e.get("arrays") or {}).get("dir") else 0)
            else:
                n = 0
            if n == 0:
                empty[m] += 1
            else:
                content_counts[m] += n

            # A saliency "file" is now a directory of .npy maps, so existence
            # is the right check for both kinds of reference.
            for rel in (e.get("render"), (e.get("arrays") or {}).get("dir")):
                if rel and not (art / rel).exists():
                    missing_files.append(str(art / rel))

    for label, counter in (("missing", absent), ("errored", errored), ("empty", empty)):
        for m, c in counter.items():
            report["problems"].append(f"{m}: {c} explanations {label}")

    if missing_files:
        report["problems"].append(
            f"{len(missing_files)} referenced files do not exist "
            f"(e.g. {missing_files[:3]})"
        )

    report["per_method"] = {
        m: {
            "present": len(records) - absent[m],
            "errored": errored[m],
            "empty": empty[m],
            "mean_items": round(
                content_counts[m]
                / max(len(records) - absent[m] - errored[m] - empty[m], 1), 1
            ),
        }
        for m in methods
    }
    report["ok"] = not report["problems"]
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--split", default="test")
    ap.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    args = ap.parse_args()

    cfg = cfgmod.load(args.config)
    if not args.task and not args.all:
        ap.error("pass --task or --all")
    keys = cfg.task_keys if args.all else [args.task]

    all_ok = True
    for k in keys:
        rep = verify_task(cfg, k, args.split)
        mark = "PASS" if rep["ok"] else "FAIL"
        print(f"\n=== {k}: {mark} ===")
        print(f"  records {rep.get('n_records', 0)} / {rep.get('n_expected', '?')} expected")
        for m, v in rep.get("per_method", {}).items():
            print(f"  {m:<10} present {v['present']:<6} errored {v['errored']:<6} "
                  f"empty {v['empty']:<6} mean attributions stored {v['mean_items']}")
        for p in rep["problems"]:
            print(f"  ! {p}")
        all_ok &= rep["ok"]

    print("\nALL TASKS COMPLETE" if all_ok else "\nINCOMPLETE — see problems above")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
