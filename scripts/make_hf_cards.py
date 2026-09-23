#!/usr/bin/env python3
"""Write a Hugging Face model card for each checkpoint, then the upload commands.

Every figure in a card is read from artifacts/model_cards.json rather than typed,
so a card cannot quietly disagree with the report it accompanies. Re-run this
after xai.modelcard and the cards follow.

    python scripts/make_hf_cards.py --user <your-hf-username>

Writes Report/hf/<repo>/README.md. Uploading is left to you: it needs your own
credentials, and pushing weights is not something a script should do behind your
back.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Which local files make up each publishable checkpoint.
CHECKPOINTS = {
    "tweet": {
        "repo": "bert-tweet-sentiment-3class",
        "files": ["models/bert_sentiment_model"],
        "base": "google-bert/bert-base-uncased",
        "licence": "apache-2.0",
        "pipeline": "text-classification",
        "data_note": (
            "Trained on the Kaggle Tweet Sentiment Extraction corpus, whose rules "
            "permit academic and non-commercial use and prohibit redistributing the "
            "data itself. These are model weights, not that corpus; the corpus is not "
            "included here and must be obtained from Kaggle."),
        "labels": "negative / neutral / positive",
    },
    "movie": {
        "repo": "bert-lora-movie-sentiment",
        "files": ["models/lora-bert-model"],
        "base": "google-bert/bert-base-uncased",
        "licence": "apache-2.0",
        "pipeline": "text-classification",
        "data_note": (
            "Trained on the counterfactually-augmented IMDb data of Kaushik, Hovy & "
            "Lipton (ICLR 2020), Apache 2.0."),
        "labels": "negative / positive",
    },
    "glasses": {
        "repo": "resnet18-glasses-detection",
        "files": ["models/resnet18_glasses_v2.pt"],
        "base": "torchvision resnet18 (ResNet18_Weights.DEFAULT)",
        "licence": "bsd-3-clause",
        "pipeline": "image-classification",
        "data_note": (
            "Trained on the Saliency-Bench glasses images (Zhang et al., "
            "arXiv:2310.08537). No licence is stated by that dataset's authors; the "
            "images are not included here."),
        "labels": "no_glasses / glasses",
    },
}

USAGE = {
    "tweet": '''```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForSequenceClassification.from_pretrained("{repo_id}")
```''',
    "movie": '''```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("{repo_id}")
base = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2)
model = PeftModel.from_pretrained(base, "{repo_id}")
```

The adapter carries its own fine-tuned classification head under keys prefixed
`base_model.model.classifier.*`. Loading the folder with
`BertForSequenceClassification.from_pretrained` silently ignores those keys and
leaves a randomly initialised head — load it through `PeftModel`, and assert the
head matches the checkpoint before trusting the output.''',
    "glasses": '''```python
import torch
from torchvision.models import resnet18

ckpt = torch.load("resnet18_glasses_v2.pt", map_location="cpu")
model = resnet18()
model.fc = torch.nn.Linear(512, 2)
model.load_state_dict(ckpt["state_dict"])
```''',
}


def card(key: str, spec: dict, c: dict, user: str) -> str:
    repo_id = f"{user}/{spec['repo']}"
    p = c.get("parameters", {})
    t = c.get("training", {})
    acc = c.get("accuracy_by_split", {})

    rows = []
    for split in ("train", "val", "test"):
        d = acc.get(split)
        if isinstance(d, dict) and d.get("accuracy") is not None:
            n = f"{d['n']:,}" if d.get("n") else "—"
            rows.append(f"| {split} | {n} | {d['accuracy']:.4f} |")
    acc_table = ("| split | n | accuracy |\n|---|---|---|\n" + "\n".join(rows)) if rows else ""

    gap = ""
    tr = (acc.get("train") or {}).get("accuracy")
    te = (acc.get("test") or {}).get("accuracy")
    if tr is not None and te is not None:
        d = (tr - te) * 100
        gap = (f"\nGeneralisation gap: **{d:+.1f} percentage points** "
               f"({tr:.1%} train, {te:.1%} test)."
               + ("  This model has memorised a substantial part of its training "
                  "set; weigh that when interpreting attributions on it." if d >= 5
                  else ""))

    tr_rows = []
    for k, v in t.items():
        if isinstance(v, (dict, list)):
            v = ", ".join(map(str, v.values() if isinstance(v, dict) else v))
        tr_rows.append(f"| {k.replace('_', ' ')} | {v} |")
    tr_rows.append(f"| parameters | {p.get('trained', 0):,} of {p.get('total', 0):,} "
                   f"trained ({p.get('trained_pct', 0):.2f}%) |")
    if c.get("lora"):
        lo = c["lora"]
        tr_rows.append(f"| LoRA | rank {lo.get('r')}, alpha {lo.get('lora_alpha')}, "
                       f"dropout {lo.get('lora_dropout')}, on "
                       f"{' and '.join(lo.get('target_modules', []))} |")

    return f"""---
license: {spec['licence']}
base_model: {spec['base']}
pipeline_tag: {spec['pipeline']}
tags:
  - explainable-ai
  - interpretability
library_name: {'peft' if key == 'movie' else 'transformers'}
---

# {spec['repo']}

{c.get('name', key)} — {spec['labels']}.

Trained for a study comparing LIME, SHAP, prototype retrieval and Grad-CAM
across every prediction of three classifiers, ranking them on faithfulness
rather than on agreement with human annotation.

## Results

{acc_table}
{gap}

## Training

| | |
|---|---|
{chr(10).join(tr_rows)}

## Usage

{USAGE[key].format(repo_id=repo_id)}

## Data

{spec['data_note']}

## Licence and attribution

Derived from `{spec['base']}`, licensed {spec['licence']}. The modifications made
are described in the Training table above.

## Citation

If you use this model, please cite the datasets it was trained on, listed in the
project's NOTICE file.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--user", required=True, help="your Hugging Face username")
    ap.add_argument("--out", default="Report/hf")
    args = ap.parse_args()

    src = ROOT / "artifacts" / "model_cards.json"
    if not src.exists():
        raise SystemExit("artifacts/model_cards.json missing — run: python -m xai.modelcard")
    cards = json.loads(src.read_text())

    out_root = ROOT / args.out
    cmds = []
    for key, spec in CHECKPOINTS.items():
        c = cards.get(key)
        if not c:
            print(f"  [{key}] no model card recorded, skipping")
            continue
        d = out_root / spec["repo"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "README.md").write_text(card(key, spec, c, args.user), encoding="utf-8")
        print(f"  wrote {(d / 'README.md').relative_to(ROOT)}")
        files = " ".join(spec["files"])
        cmds.append(
            f"# {key}\n"
            f"huggingface-cli repo create {spec['repo']} --type model -y\n"
            f"huggingface-cli upload {args.user}/{spec['repo']} {files} . \\\n"
            f"    --commit-message 'Add {key} checkpoint'\n"
            f"huggingface-cli upload {args.user}/{spec['repo']} "
            f"{d.relative_to(ROOT)}/README.md README.md")

    script = out_root / "upload.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "# Upload the checkpoints to Hugging Face.\n"
        "#   pip install -U 'huggingface_hub[cli]'\n"
        "#   huggingface-cli login\n"
        "set -euo pipefail\n"
        "cd \"$(dirname \"$0\")/../..\"\n\n" + "\n\n".join(cmds) + "\n",
        encoding="utf-8")
    script.chmod(0o755)
    print(f"  wrote {script.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
