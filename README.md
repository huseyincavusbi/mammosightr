# MammosighTR — Case-Level BI-RADS / Density / Quadrant

Demo on the MammosighTR screening dataset: predict grouped
BI-RADS, breast density and lesion quadrant per case from four mammographic views.

Independent work using the public data release, not an official competition
submission, not clinically validated, and not a medical device.

## Data

| Split | Cases | Views |
| :--- | ---: | ---: |
| Train (EGITIM1 + EGITIM2) | 10,740 | 42,074 |
| Test (YARISMA) | 2,000 | 8,000 |

[Raw dataset](https://huggingface.co/datasets/huseyincavus/MammosighTR) ·
[preprocessed train copy](https://huggingface.co/datasets/gulluk/mammosightr-preprocessed) —
CC-BY-NC-4.0. Paper: Koç et al., *Radiology: AI* 2025;7(6):e240841.

## Results (official YARISMA, n = 2,000)

| Metric | 4-fold CV |
| :--- | ---: |
| BI-RADS macro-F1 (0 / 1-2 / 4-5) | **0.630 ± 0.012** |
| Density macro-F1 (A-D) | **0.731 ± 0.007** |
| Quadrant side accuracy | **0.675 ± 0.001** |
| **Gated score** | **0.641 ± 0.008** |

`score = 0.70·BI-RADS + 0.15·density + 0.15·quadrant` (aux voided on a wrong
BI-RADS call). Best single split: 0.6435 gated.

## Method

- Frozen ConvNeXt-small (MAM-CLIP, fine-tuned on this dataset); one 768-d feature per view.
- Hierarchical attention MIL: views -> breast side -> case.
- Three heads (BI-RADS, density, quadrant) trained with a gated 0.70/0.15/0.15 loss.
- Shipping model: logit soup of three seeded heads.
- YARISMA DICOMs preprocessed classically to 768x1024 (8,000 views, 0 failures).

## Training setup

| Split | Cases | Views | Role |
| :--- | ---: | ---: | :--- |
| Full dataset | 12,740 | 50,074 | — |
| Official train (EGITIM1 + EGITIM2) | 10,740 | 42,074 | feature extraction |
| ├ head training (folds 2-4) | 8,055 | 31,573 | 5 epochs, 3 seeds |
| └ validation (fold 1) | 2,685 | 10,501 | model/calibration selection |
| Official test (YARISMA) | 2,000 | 8,000 | evaluation only |

- **Features:** frozen ConvNeXt-small on all 12,740 cases (one 768-d vector per view).
- **Heads:** AdamW, lr 1e-3, **5 epochs**, gated 0.70/0.15/0.15 loss; batch = 1 case (soups: 256).
- **Shipping model:** 3 seeds × 5 epochs (logit soup); 4-fold CV repeats 5 epochs per fold.
- **Image-level baseline:** 10 epochs on the same per-view features (0.501 YARISMA).
- **Fast-iteration subset:** 1,200 balanced cases from folds 2-4 — same 5-epoch recipe.
- **Fine-tune ablation:** last encoder stage, **3 epochs** on those 1,200 cases (rejected).

## Findings

- Frozen in-domain features beat fine-tuning (0.641 vs 0.582 gated).
- Case-level MIL is more robust than image-level (0.630 vs 0.501 on YARISMA).
- Large domain gap: in-domain 0.825 vs 0.641 on the official test.


## Weights

Trained by this project (head weights only; the encoder is frozen and not redistributed):

- [`assets/heads/soup_ce_seed{0,1,2}.pt`](assets/heads) — shipping 3-seed soup (768-d heads)
- [`assets/checkpoints/`](assets/checkpoints) — 4-fold CV, full-data, ensemble (1536-d) and push seed heads

Backbone used: `convnext_small_ftune.pth` from MAM-CLIP —
[weights folder (Google Drive)](https://drive.google.com/drive/folders/1faHwdyjpDB5yQX0acgBlT8A3KgsVI2MM) ·
[code](https://github.com/igulluk/MAM-CLIP).

## Citation

> Koç U, Beşler MS, Sezer EA, Karakaş E, et al. **MammosighTR: Nationwide Breast Cancer
> Screening Mammogram Dataset with BI-RADS Annotations for Artificial Intelligence
> Applications.** *Radiology: Artificial Intelligence.* 2025;7(6):e240841.
> doi:10.1148/ryai.240841

> Gulluk HI, Gevaert O. **MAM-CLIP: Vision-Language Pretraining on Mammography Atlases
> for BI-RADS Classification.** arXiv:2605.19359, 2026.

---

*Dataset CC-BY-NC-4.0 · research use only, not a medical device*
