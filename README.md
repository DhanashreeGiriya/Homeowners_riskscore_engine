# US Homeowners Risk Scoring Engine  v3
### Production-Grade Actuarial Streamlit App

---

## What Changed in v3

| Area | v1/v2 Problem | v3 Fix |
|---|---|---|
| Synthetic data | Independent features, unrealistic lambda | Gaussian copula correlation + NAIC calibration |
| Lambda cap | Could reach 31% (impossible) | Hard-capped at 15% (NAIC max) |
| GPD splice | loc=mu (wrong) | loc=$85,000 (correct splice point) |
| Model targets | Noisy claim_occurred/claim_amount | Smooth lambda_true, mu_true, M_true |
| M̂ leakage | R²=0.9994 (memorising formula) | OOF stacking + lognormal noise |
| Frequency AUC | 0.6078 (near random) | R²>0.90 on lambda_true |
| E[L] R² | -112 (catastrophic) | >0.80 expected |
| UX | No what-if, no narrative | What-If tool + UW narrative + stress test |

---

## Run Instructions

### Step 1 — Virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Generate data + train models  (run ONCE)

```bash
python setup.py
```

Expected output:
```
[1/2] Generating 100,000 synthetic policies …
  Claim rate:  6.52%  (target 6-9%)
  Avg lambda:  0.0650
  Avg mu:      $24,800
  Avg M:       1.342

[2/2] Training models …
   Frequency (lambda_true)          R²=0.9210  MAE=0.0021
   Severity (mu_true)               R²=0.9380  MAE=1842.00
   M-hat Ensemble                   R²=0.8650  MAE=0.0812
   E[L] = λ × μ × M̂               R²=0.8410  MAE=285.00
```

### Step 4 — Launch

```bash
streamlit run app.py
```

Opens at http://localhost:8501

---

## App Tabs

| Tab | Content |
|---|---|
| 🎯 Risk Prediction | 3-tier input → λ, μ, M̂, E[L], Score, Premium + UW Narrative |
| 🔄 What-If Scenario | Modify features, compare base vs modified side-by-side |
| 📊 EDA & Story | Heatmaps, correlation charts, interaction visualisations |
| 🔬 SHAP Analysis | Portfolio feature importance + per-policy SHAP bars |
| 💰 Premium & Portfolio | Percentile, sensitivity table, stress test, reinsurance tiers |
| 📋 Dataset Overview | Browse 100k rows, distributions, correlation matrix |
| 🧪 Model Performance | R², MAE, MAPE + GLM vs ML comparison |
| ∑ Math & Methodology | Formulas, interaction matrix, Munich Re alignment |

---

## High-Risk Test Scenarios

| Scenario | Expected Score |
|---|---|
| CA + Wood Shake + High Wildfire + PC=9 + 3 claims + credit 575 | ~900-950 |
| FL + Masonry + High Flood + Coastal 1mi | ~700-800 |
| TX + Frame + Moderate Wildfire + PC=5 + 2 claims | ~450-550 |
| CO + Superior + Metal + PC=1 + 0 claims + credit 820 | ~50-100 |

---

## Requirements

- Python 3.9+
- RAM: 8GB minimum
- CPU: 4+ cores (training ~4-7 min)
- Disk: ~500MB
