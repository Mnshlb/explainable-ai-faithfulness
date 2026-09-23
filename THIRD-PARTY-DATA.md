# Third-party data

This project studies explanation methods applied to three datasets. **None of
them is mine, and none is redistributed here.** `scripts/get_data.py` obtains
each from its original source.

This file records what was used, whose it is, and why some material that the
work depends on is deliberately absent.

---

## Glasses detection — CelebA-HQ, via Saliency-Bench

**Used. Not redistributed — neither the images nor any saliency visualisation
computed on them.**

The images are **CelebA-HQ** — 1024×1024 photographs of public figures,
identifiable by their dimensions and by filename numbering across 0–29,999. They
reach this project through a chain in which **no party holds the photographs'
copyright**:

| Layer | Who | Contribution |
|---|---|---|
| The photographs | photographers and agencies | the copyright |
| **CelebA** | MMLAB, The Chinese University of Hong Kong | aggregation, attribute labelling |
| **CelebA-HQ** | Karras et al. (2018) | the high-resolution derivation |
| **Saliency-Bench** | Zhang, Song, Gu, Jiang, Pan, Bai & Zhao (KDD '25) | the glasses split, human attention annotations |

CelebA's terms of use state:

> "The CelebA dataset is available for non-commercial research purposes only."
>
> "You agree not to further copy, publish or distribute any portion of the
> CelebA dataset. Except, for internal use at a single site within the same
> organization it is allowed to make copies of the dataset."
>
> "All images of the CelebA dataset are obtained from the Internet which are not
> property of MMLAB, The Chinese University of Hong Kong."

Two things follow, and together they decided the matter:

1. **No one in that chain can grant rights to the photographs**, because no one
   in it owns them. Crediting the curator obtains nothing, and asking would not
   have helped.
2. **The terms permit internal copies, not publication.** A public repository is
   neither internal nor a single site.

Separately from copyright, these are identifiable faces — personal data, which a
licence would not resolve in any case.

So the glasses chapter is published as **numbers, not pictures**: the
per-prediction records carry file paths and attributions only, and the aggregate
statistics, figures and findings all remain. The report reproduces three images
as worked examples, which is the ordinary basis on which a paper illustrates its
findings — one of them is the mislabelled image the report identifies as a
defect in the dataset, and that finding needs the picture to make sense.

## Movie reviews — counterfactually-augmented IMDb

**Used. Derived records published in full**, because the licence permits it.

Kaushik, Hovy & Lipton (ICLR 2020), released under the **Apache License 2.0**.
`artifacts/movie/predictions.jsonl` therefore carries the review text alongside
every attribution — the one corpus here a reader can check end to end.

## Tweet sentiment — Kaggle competition data

**Used. Not redistributed.**

The Tweet Sentiment Extraction rules permit use for "academic research and
education, and other non-commercial purposes" and state:

> "You agree not to transmit, duplicate, publish, redistribute or otherwise
> provide or make available the Data to any party not participating in the
> Competition."

No file here reproduces that corpus — which ruled out the per-prediction records
for that task, since they would have embedded 2,748 test tweets verbatim plus
5,569 training tweets as prototype neighbours. Aggregate statistics derived from
the corpus remain, and twelve tweets appear in the report as four worked
examples with their retrieved neighbours.

---

## Model weights

Derived from `bert-base-uncased` (**Apache 2.0**) and torchvision's ImageNet
ResNet-18 (**BSD-3-Clause**), both of which permit derivative works. `NOTICE`
records the attribution and the modifications those licences require. The
weights are published on Hugging Face rather than here.

## What *is* mine

Everything written for this project — `src/`, `scripts/`, `config.yaml`, the
report and its figures — is under the **MIT License** in `LICENSE`.

The distinction is the point of this file: the code is offered without
restriction; the data it operates on was never mine to offer.

## If you hold rights in any of this

If you are a photographer, an agency, someone depicted, or an author of any
dataset above, and you would like attribution corrected or anything removed,
please open an issue.

---

## Citation

Cite the datasets rather than this repository:

```bibtex
@inproceedings{zhang2025saliencybench,
  title     = {Saliency-Bench: A Comprehensive Benchmark for Evaluating Visual Explanations},
  author    = {Zhang, Yifei and Song, James and Gu, Siyi and Jiang, Tianxu and
               Pan, Bo and Bai, Guangji and Zhao, Liang},
  booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery
               and Data Mining},
  year      = {2025}
}

@inproceedings{liu2015celeba,
  title     = {Deep Learning Face Attributes in the Wild},
  author    = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision},
  year      = {2015}
}

@inproceedings{karras2018progressive,
  title     = {Progressive Growing of GANs for Improved Quality, Stability, and Variation},
  author    = {Karras, Tero and Aila, Timo and Laine, Samuli and Lehtinen, Jaakko},
  booktitle = {International Conference on Learning Representations},
  year      = {2018}
}

@inproceedings{kaushik2020learning,
  title     = {Learning the Difference that Makes a Difference with
               Counterfactually-Augmented Data},
  author    = {Kaushik, Divyansh and Hovy, Eduard and Lipton, Zachary C.},
  booktitle = {International Conference on Learning Representations},
  year      = {2020}
}
```
