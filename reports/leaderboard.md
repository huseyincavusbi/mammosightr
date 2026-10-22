# YARISMA Leaderboard (official test, 2,000 cases)

Case-level predictions from 4 views (LCC/RCC/LMLO/RMLO). BI-RADS grouped {0},{1-2},{4-5}.
Gated score = 0.70·BI-RADS + 0.15·density + 0.15·quadrant (aux credited only when BI-RADS correct).

| Model | BI-RADS macro-F1 | density macro-F1 | quadrant side acc | gated score (95% CI) | notes |
|---|---|---|---|---|---|
| **4-fold CV (one model per fold)** | **0.6297 ± 0.0116** | **0.7309 ± 0.0072** | **0.6749 ± 0.0006** | **0.6413 ± 0.0075** (95% t-CI 0.6293–0.6533) | error bars; `results/cv4/` |
| **3-seed CE soup (Gulluk ConvNeXt-s frozen)** | 0.6286 | **0.7311** | 0.6752 | **0.6435 (0.6251–0.6607)** | best single-split; `results/improve/soup_ce_seed{0,1,2}.pt` |
| 5-seed soup + density weights + tuned rule | **0.6378** | 0.7272 | 0.6752 | 0.6414 (0.6221–0.6587) | best BI-RADS; tuning fit on val |
| 8-seed combined soup + tuned rule | 0.6277 | 0.7158 | 0.6752 | 0.6394 | val-selected, test slightly lower |
| Ensemble (Gulluk⊕DINOv3) + calibration | 0.6209 | 0.6894 | 0.6752 | 0.6272 (0.6085–0.6435) | full 8,055 train |
| Gulluk frozen + HieraMIL heads, full data | 0.6196 | **0.7245** | 0.6752 | 0.6246 (0.6045–0.6427) | 8,055 train cases, 5 epochs; head `results/full_heads/heads_trackA.pt` (demo) |
| Gulluk frozen + HieraMIL heads (Track A subset) | 0.6127 | 0.7150 | 0.6752 | 0.6230 (0.6040–0.6416) | 1,200 balanced train / 300 val |
| Gulluk frozen + HieraMIL, BI-RADS only | 0.6366 | – | – | – | 8,055 train, 3 epochs (ladder v6) |
| Partial fine-tune (last stage, 3 epochs) | 0.5382 | 0.6793 | 0.6643 | 0.5815 (0.5614–0.6023) | val 0.822→0.842 but test degraded (overfit, rx) |
| DINOv3 frozen + HieraMIL, BI-RADS only | 0.5069 | – | – | – | ladder v6 |

Reference points (not directly comparable — different protocols):
- Image-level (Gulluk protocol), our features + MLP: 0.781 in-domain / 0.501 on YARISMA (`results/image_level/`).
- Competition 2023 field: per-class F1 0.255 / 0.417 / 0.537 (BI-RADS only, case-level); macro ≈ 0.40.
- MAM-CLIP (Gulluk): 3-class macro-F1 0.706–0.711, image-level, internal 4-fold CV on train — no YARISMA evaluation.
- LoRSA: density macro-F1 0.515 on YARISMA, image-level external (ours: 0.72 case-level).
- CMES benchmark: zero-shot detection → BI-RADS12-vs-45 breast-level macro-F1 ~0.83–0.85 on YARISMA (different task/granularity).

Calibration (fit on fold-1 val): temperature birads 1.25 / density 1.15; bias birads [-0.5, 0.0, -0.3], density [0.6, 0.0, -0.5, -0.1].

Artifacts:
- `results/ensemble/` — ensemble heads + metrics (best gated)
- `results/full_heads/` — Gulluk-only full-data heads (used by the demo)
- `results/heads_trackA.pt` + `results/track_a_metrics.json` — subset model
- `results/finetune_rx.pt` + `results/finetune_rx_metrics.json` — fine-tune ablation (on rx)
- `src/predict.py` (4 PNGs → predictions), `demo/app.py` (Gradio UI, `python demo/app.py` → http://127.0.0.1:7860)
