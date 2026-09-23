---
license: apache-2.0
base_model: google-bert/bert-base-uncased
pipeline_tag: text-classification
tags:
  - explainable-ai
  - interpretability
library_name: peft
---

# bert-lora-movie-sentiment

Movie Review Sentiment (binary, counterfactually-augmented) — negative / positive.

Trained for a study comparing LIME, SHAP, prototype retrieval and Grad-CAM
across every prediction of three classifiers, ranking them on faithfulness
rather than on agreement with human annotation.

## Results

| split | n | accuracy |
|---|---|---|
| train | 1,600 | 0.7662 |
| val | 200 | 0.7150 |
| test | 199 | 0.7588 |

Generalisation gap: **+0.7 percentage points** (76.6% train, 75.9% test).

## Training

| | |
|---|---|
| base model | bert-base-uncased |
| strategy | LoRA — adapter and classification head trained, base frozen |
| hardware | Google Colab GPU |
| optimizer | not recorded |
| lr | not recorded |
| epochs | not recorded |
| batch size | not recorded |
| per epoch metrics | not captured |
| parameters | 297,988 of 109,780,228 trained (0.27%) |
| LoRA | rank 8, alpha 32, dropout 0.1, on value and query |

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("YOUR-HF-USERNAME/bert-lora-movie-sentiment")
base = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2)
model = PeftModel.from_pretrained(base, "YOUR-HF-USERNAME/bert-lora-movie-sentiment")
```

The adapter carries its own fine-tuned classification head under keys prefixed
`base_model.model.classifier.*`. Loading the folder with
`BertForSequenceClassification.from_pretrained` silently ignores those keys and
leaves a randomly initialised head — load it through `PeftModel`, and assert the
head matches the checkpoint before trusting the output.

## Data

Trained on the counterfactually-augmented IMDb data of Kaushik, Hovy & Lipton (ICLR 2020), Apache 2.0.

## Licence and attribution

Derived from `google-bert/bert-base-uncased`, licensed apache-2.0. The modifications made
are described in the Training table above.

## Citation

If you use this model, please cite the datasets it was trained on, listed in the
project's NOTICE file.
