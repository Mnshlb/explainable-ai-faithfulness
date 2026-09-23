---
license: bsd-3-clause
base_model: torchvision resnet18 (ResNet18_Weights.DEFAULT)
pipeline_tag: image-classification
tags:
  - explainable-ai
  - interpretability
library_name: transformers
---

# resnet18-glasses-detection

Glasses Detection (binary) — no_glasses / glasses.

Trained for a study comparing LIME, SHAP, prototype retrieval and Grad-CAM
across every prediction of three classifiers, ranking them on faithfulness
rather than on agreement with human annotation.

## Results

| split | n | accuracy |
|---|---|---|
| train | 1,830 | 1.0000 |
| val | 392 | 0.9974 |
| test | 392 | 1.0000 |

Generalisation gap: **+0.0 percentage points** (100.0% train, 100.0% test).

## Training

| | |
|---|---|
| split | 0.7, 0.15, 0.15 |
| pretrained | True |
| epochs | 10 |
| batch size | 32 |
| lr | 0.0001 |
| weight decay | 0.0001 |
| augment | 0.5, 10, 0.15 |
| parameters | 11,177,538 of 11,177,538 trained (100.00%) |

## Usage

```python
import torch
from torchvision.models import resnet18

ckpt = torch.load("resnet18_glasses_v2.pt", map_location="cpu")
model = resnet18()
model.fc = torch.nn.Linear(512, 2)
model.load_state_dict(ckpt["state_dict"])
```

## Data

Trained on the Saliency-Bench glasses images (Zhang et al., arXiv:2310.08537). No licence is stated by that dataset's authors; the images are not included here.

## Licence and attribution

Derived from `torchvision resnet18 (ResNet18_Weights.DEFAULT)`, licensed bsd-3-clause. The modifications made
are described in the Training table above.

## Citation

If you use this model, please cite the datasets it was trained on, listed in the
project's NOTICE file.
