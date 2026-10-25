| Model | CV PR-AUC (mean ± sd) | Test PR-AUC [95% CI] | Test ROC-AUC [95% CI] | F1 | Recall | Precision | MCC | Brier | Cost / claim (USD) |
|---|---|---|---|---|---|---|---|---|---|
| **CatBoost** | 0.707 ± 0.054 | 0.553 [0.463; 0.708] | 0.839 [0.769; 0.902] | 0.730 | 0.857 | 0.636 | 0.639 | 0.124 | 680 |
| Stacking (top-3 + logistic meta) | 0.704 ± 0.062 | 0.588 [0.482; 0.721] | 0.827 [0.747; 0.895] | 0.730 | 0.857 | 0.636 | 0.639 | 0.126 | 680 |
| Random forest | 0.701 ± 0.061 | 0.600 [0.489; 0.736] | 0.827 [0.748; 0.897] | 0.718 | 0.857 | 0.618 | 0.622 | 0.123 | 690 |
| Gradient boosting (sklearn) | 0.695 ± 0.057 | 0.592 [0.483; 0.726] | 0.837 [0.766; 0.899] | 0.730 | 0.857 | 0.636 | 0.639 | 0.120 | 680 |
| Logistic regression (elastic net) | 0.693 ± 0.054 | 0.524 [0.434; 0.669] | 0.818 [0.743; 0.884] | 0.609 | 0.857 | 0.472 | 0.472 | 0.152 | 795 |
| XGBoost | 0.683 ± 0.059 | 0.512 [0.432; 0.657] | 0.832 [0.763; 0.894] | 0.726 | 0.837 | 0.641 | 0.631 | 0.143 | 720 |
| MLP + embeddings (PyTorch) | 0.668 ± 0.062 | 0.537 [0.447; 0.682] | 0.840 [0.777; 0.896] | 0.621 | 0.837 | 0.494 | 0.488 | 0.146 | 815 |
| LightGBM | 0.659 ± 0.058 | 0.520 [0.439; 0.665] | 0.830 [0.753; 0.897] | 0.707 | 0.837 | 0.612 | 0.606 | 0.140 | 735 |
| Logistic regression (no penalty) | 0.629 ± 0.065 | 0.499 [0.402; 0.646] | 0.771 [0.686; 0.844] | 0.503 | 0.735 | 0.383 | 0.302 | 0.175 | 1,120 |

Trivial policies on the test set: investigate nothing = 2,450 USD/claim, investigate all = 1,000 USD/claim.

Reference model (best mean CV PR-AUC): CatBoost. Exported model: CatBoost.

| Model vs reference | Δ CV PR-AUC | corrected t-test p (Holm) | Δ test PR-AUC [95% CI] | bootstrap p (Holm) | McNemar p (Holm) |
|---|---|---|---|---|---|
| Stacking (top-3 + logistic meta) | -0.003 | 1.000 | +0.035 [-0.027; +0.078] | 1.000 | 1.000 |
| Random forest | -0.006 | 1.000 | +0.047 [-0.046; +0.113] | 1.000 | 1.000 |
| Gradient boosting (sklearn) | -0.012 | 1.000 | +0.039 [-0.041; +0.094] | 1.000 | 1.000 |
| Logistic regression (elastic net) | -0.014 | 1.000 | -0.029 [-0.110; +0.052] | 1.000 | 0.000 |
| XGBoost | -0.024 | 1.000 | -0.041 [-0.115; +0.022] | 1.000 | 1.000 |
| MLP + embeddings (PyTorch) | -0.039 | 1.000 | -0.016 [-0.092; +0.065] | 1.000 | 0.000 |
| LightGBM | -0.048 | 0.220 | -0.033 [-0.106; +0.041] | 1.000 | 1.000 |
| Logistic regression (no penalty) | -0.078 | 0.312 | -0.054 [-0.144; +0.032] | 1.000 | 0.000 |

| Calibration (test) | Brier | PR-AUC |
|---|---|---|
| CatBoost (uncalibrated) | 0.1237 | 0.553 |
| Logistic regression (no penalty) (uncalibrated) | 0.1748 | 0.499 |
| CatBoost (sigmoid) | 0.1242 | 0.576 |
| CatBoost (isotonic) | 0.1222 | 0.576 |
