#!/usr/bin/env python3
"""Fetch the three datasets from their original sources.

None of them are redistributed with this project. Two are permissively licensed
and could have been; the tweet corpus could not, and treating all three the same
way keeps the rule simple and the repository honest about what it contains.

    python scripts/get_data.py            # everything that is missing
    python scripts/get_data.py --task tweet
    python scripts/get_data.py --check    # report what is present, download nothing

The Kaggle download needs the official client and an API token:

    pip install kaggle
    # https://www.kaggle.com/settings -> API -> Create New Token
    mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json

You must also accept the competition rules once, on the competition page, before
the API will serve the files.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Dataset"

# Each entry: where it lands, how to get it, and what it is licensed under.
SOURCES = {
    "tweet": {
        "dest": DATA / "Tweet sentiment analysis",
        "marker": DATA / "Tweet sentiment analysis/Processed/test.csv",
        "kind": "kaggle-competition",
        "ref": "tweet-sentiment-extraction",
        "url": "https://www.kaggle.com/competitions/tweet-sentiment-extraction",
        "licence": ("Kaggle competition rules: academic/non-commercial use; "
                    "redistribution of the data is not permitted."),
        "note": ("After download, the notebooks under Notebooks/Tweet sentiment/ "
                 "produce the Processed/{train,test}.csv this project reads."),
    },
    "movie": {
        "dest": DATA / "Sentiment analysis-Movies Review",
        "marker": DATA / "Sentiment analysis-Movies Review/test.csv",
        "kind": "git",
        "ref": "https://github.com/acmi-lab/counterfactually-augmented-data",
        "url": "https://github.com/acmi-lab/counterfactually-augmented-data",
        "licence": "Apache 2.0. Cite Kaushik, Hovy & Lipton (ICLR 2020).",
        "note": "The sentiment/combined/paired split provides train/dev/test.",
    },
    "glasses": {
        "dest": DATA / "Glasses",
        "marker": DATA / "Glasses/image",
        "kind": "manual",
        "ref": "https://xaidataset.github.io/",
        "url": "https://xaidataset.github.io/",
        "licence": ("No licence stated by the authors. Obtain directly from them "
                    "and observe whatever terms they attach."),
        "note": ("Needs image/{neg,pos}/*.jpg and attention_label/*.csv. "
                 "Cite Zhang et al., arXiv:2310.08537."),
    },
}


def present(entry: dict) -> bool:
    return entry["marker"].exists()


def report() -> None:
    print(f"{'task':10s}{'present':10s}source")
    for name, e in SOURCES.items():
        print(f"  {name:8s}{'yes' if present(e) else 'MISSING':10s}{e['url']}")
    print("\nLicence terms:")
    for name, e in SOURCES.items():
        print(f"  {name:8s}{e['licence']}")


def fetch_kaggle(entry: dict) -> None:
    if shutil.which("kaggle") is None:
        raise SystemExit(
            "The kaggle client is not installed.\n"
            "    pip install kaggle\n"
            "then create an API token at https://www.kaggle.com/settings")
    entry["dest"].mkdir(parents=True, exist_ok=True)
    print(f"  downloading {entry['ref']} ...", flush=True)
    r = subprocess.run(
        ["kaggle", "competitions", "download", "-c", entry["ref"],
         "-p", str(entry["dest"])],
        capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(
            f"kaggle download failed:\n{r.stderr.strip()}\n\n"
            f"The usual cause is not having accepted the rules. Open\n"
            f"  {entry['url']}/rules\n"
            f"accept them once, then re-run this script.")
    for z in entry["dest"].glob("*.zip"):
        print(f"  extracting {z.name} ...")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(entry["dest"])
        z.unlink()


def fetch_git(entry: dict) -> None:
    if shutil.which("git") is None:
        raise SystemExit("git is not available")
    entry["dest"].mkdir(parents=True, exist_ok=True)
    print(f"  cloning {entry['ref']} ...", flush=True)
    r = subprocess.run(["git", "clone", "--depth", "1", entry["ref"],
                        str(entry["dest"] / "_src")], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"clone failed:\n{r.stderr.strip()}")
    print(f"  cloned into {entry['dest'] / '_src'}")
    print(f"  {entry['note']}")


def fetch_manual(entry: dict) -> None:
    raise SystemExit(
        f"This dataset has no automated download.\n"
        f"  Source : {entry['url']}\n"
        f"  Licence: {entry['licence']}\n"
        f"  Place it at: {entry['dest']}\n"
        f"  {entry['note']}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="all", choices=["all", *SOURCES])
    ap.add_argument("--check", action="store_true",
                    help="report what is present and download nothing")
    ap.add_argument("--force", action="store_true",
                    help="download even if the data already appears to be present")
    args = ap.parse_args()

    if args.check:
        report()
        return

    for name, entry in SOURCES.items():
        if args.task not in ("all", name):
            continue
        print(f"[{name}]")
        if present(entry) and not args.force:
            print(f"  already present at {entry['dest'].relative_to(ROOT)}")
            continue
        print(f"  licence: {entry['licence']}")
        {"kaggle-competition": fetch_kaggle,
         "git": fetch_git,
         "manual": fetch_manual}[entry["kind"]](entry)

    print("\nDone. Verify with: python scripts/get_data.py --check")


if __name__ == "__main__":
    sys.exit(main())
