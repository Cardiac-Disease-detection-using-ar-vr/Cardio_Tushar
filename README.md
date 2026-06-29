# Cardiovascular Risk Scoring — Dempster-Shafer Fusion Pipeline

**Author:** Tushar  
**Model Version:** v6_final  
**Architecture:** Constrained Elastic Net Logistic Scorer → Isotonic Calibration → Murphy-Dempster Evidence Fusion

---

## 📈 Key Metrics
All performance metrics are calculated on the holdout test set ($n = 1,026$) using the 3-class stratification (Low / Mild / High):

| Metric | Value |
|---|---|
| Cross-validated AUC | **0.9795** |
| Fused AUC (common_alpha) | **0.9868** |
| Overall Accuracy | **98.64%** |
| Cohen's Kappa | **0.9587** |
| ECE (calibration error) | **0.0041** |
| Stability Index | **0.9958** |
| Flagged for clinical review | **39** patients ($3.80\%$) |
| Dempster-rule usage | **100%** ($6837/6837$) |

---

## 🔍 Alpha Scores
| Score | Value |
|---|---|
| Mean $\alpha_1$ (Clinical Score) | **0.0998** |
| $\alpha_2$ (Model Performance) | **0.5019** (frozen scalar) |
| Mean Fused (common_alpha) | **0.1001** |

---

## 🧠 Deep Learning Model
**HybridModel — TCN + FT-Transformer Fusion**
- **TCN Encoder:** 4 residual blocks ($2 \to 64 \to 128 \to 256 \to 256$) with BiGRU.
- **Tab Encoder:** FT-Transformer (64-dim, 2 attention layers).
- **Fusion:** Gated MLP $\to$ 320-dim embedding.
- **Heads:** 3-class classification head (`prob_low_risk`, `prob_mild_risk`, `prob_high_risk`).
- **Parameters:** $1,222,219$.

---

## 📁 File Manifest
| File | Description |
|---|---|
| `clinical_scoring_complete_dataset_v6.json` | Source of truth — full patient dataset ($6,837$ patients). |
| `clinical_scoring_results_v6.json` | Per-patient scoring results ($\alpha_1$, $\alpha_2$, fused scores). |
| `clinical_scoring_results_v6.parquet` | Same results in Parquet format. |
| `predictions_v6.json` | Model predictions (class probs, risk score, ejection fraction). |
| `extended_score_v6.json` | Extended scoring breakdown per patient. |
| `model_config_v6.json` | Model configuration and hyperparameters. |
| `model.pt` | Trained PyTorch TCN+Transformer model weights. |
| `features.parquet` | Feature matrix ($6,837 \times 33$). |
| `isotonic_calibration_frozen.pkl` | Frozen isotonic calibration for VR runtime. |
| `performance_analysis_v6.png` | Performance visualization chart. |
| `complete_validation.py` | Validation suite ($45/45$ tests pass). |

---

## ⚙️ Validation
Run the validation suite to verify all formulas and data integrity:
```bash
python complete_validation.py
```

Expected output: **45/45 tests PASS** across 8 validation parts:
*   **Part A:** Alpha-1 Formula (7 tests)
*   **Part B:** Alpha-2 Formula (4 tests)
*   **Part C:** Common-Alpha Fusion (9 tests)
*   **Part D:** 12-Method Validation Suite (12 tests)
*   **Part E:** Metric Recomputation (3 tests)
*   **Part F:** Correlations (3 tests)
*   **Part G:** Cross-Validation Folds (2 tests)
*   **Part H:** Data Integrity (5 tests)

---

## 🚀 VR Integration
The `isotonic_calibration_frozen.pkl` can be loaded directly in the VR runtime:
```python
import pickle
with open('isotonic_calibration_frozen.pkl', 'rb') as f:
    calibrator = pickle.load(f)

calibrated_risk = calibrator.predict([raw_alpha_common])[0]
```

**Made by Tushar**
