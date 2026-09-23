---
license: apache-2.0
base_model: google-bert/bert-base-uncased
pipeline_tag: text-classification
tags:
  - explainable-ai
  - interpretability
library_name: transformers
---

# bert-tweet-sentiment-3class

Tweet Sentiment (3-class) — negative / neutral / positive.

Trained for a study comparing LIME, SHAP, prototype retrieval and Grad-CAM
across every prediction of three classifiers, ranking them on faithfulness
rather than on agreement with human annotation.

## Results

| split | n | accuracy |
|---|---|---|
| train | 24,732 | 0.9715 |
| test | 2,748 | 0.7820 |

Generalisation gap: **+18.9 percentage points** (97.2% train, 78.2% test).  This model has memorised a substantial part of its training set; weigh that when interpreting attributions on it.

## Training

| | |
|---|---|
| base model | bert-base-uncased |
| strategy | full fine-tune — every parameter updated |
| optimizer | AdamW |
| lr | 5e-05 |
| epochs | 3 |
| batch size | 16 |
| max length | 128 |
| val split | 0.1 |
| seed | 42 |
| hardware | Google Colab GPU |
| per epoch metrics | not captured |
| parameters | 109,484,547 of 109,484,547 trained (100.00%) |

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("YOUR-HF-USERNAME/bert-tweet-sentiment-3class")
model = AutoModelForSequenceClassification.from_pretrained("YOUR-HF-USERNAME/bert-tweet-sentiment-3class")
```

## Data

Trained on the Kaggle Tweet Sentiment Extraction corpus, whose rules permit academic and non-commercial use and prohibit redistributing the data itself. These are model weights, not that corpus; the corpus is not included here and must be obtained from Kaggle.

## Licence and attribution

Derived from `google-bert/bert-base-uncased`, licensed apache-2.0. The modifications made
are described in the Training table above.

## Citation

If you use this model, please cite the datasets it was trained on, listed in the
project's NOTICE file.
