"""
model_trainer.py  v3
====================
Trains on smooth actuarial targets (lambda_true, mu_true, M_true).
Uses 60/20/20 train/val/test split.
OOF stacking for M-hat to prevent leakage.
Saves all artifacts + full metrics.
"""

import os, pickle, warnings, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              r2_score, roc_auc_score)
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)

# ── Feature lists ──────────────────────────────────────────────────────────────
T12 = [
    "construction_type","home_age","home_value","coverage_amount",
    "square_footage","stories","protection_class","occupancy",
    "prior_claims_3yr","credit_score","deductible",
    "swimming_pool","trampoline","dog",
    "security_system","smoke_detectors","sprinkler_system","gated_community",
    "dist_to_fire_station_mi",
]
T3 = [
    "wildfire_zone","flood_zone","earthquake_zone","roof_material",
    "dist_to_coast_mi","dist_to_fire_station_mi",
    "roof_age_yr","construction_type","state",
]
SEV_FEATS = list(dict.fromkeys(T12 + T3))
CAT_COLS  = ["construction_type","occupancy","wildfire_zone",
             "flood_zone","earthquake_zone","roof_material","state"]


# ── Helpers ────────────────────────────────────────────────────────────────────
def encode(df, encoders=None, fit=True):
    df = df.copy()
    if encoders is None:
        encoders = {}
    for col in CAT_COLS:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le  = encoders[col]
            raw = df[col].astype(str).values
            safe = np.where(np.isin(raw, le.classes_), raw, le.classes_[0])
            df[col] = le.transform(safe)
    return df, encoders


def reg_metrics(y_true, y_pred, label=""):
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-6))) * 100
    print(f"   {label:30s}  R²={r2:7.4f}  MAE={mae:12.2f}  MAPE={mape:.1f}%")
    return dict(R2=round(r2,4), MAE=round(mae,4), RMSE=round(rmse,4), MAPE=round(mape,2))


# ── Main trainer ───────────────────────────────────────────────────────────────
def train_all(df: pd.DataFrame) -> dict:
    t0 = time.time()
    print("="*68)
    print("  HOMEOWNERS RISK MODEL  —  TRAINING  v3")
    print("="*68)

    # 60 / 20 / 20
    tr_df, tmp = train_test_split(df, test_size=0.40, random_state=42)
    va_df, te_df = train_test_split(tmp, test_size=0.50, random_state=42)
    print(f"  Train:{len(tr_df):,}   Val:{len(va_df):,}   Test:{len(te_df):,}")

    needed = list(dict.fromkeys(SEV_FEATS + ["lambda_true","mu_true","M_true",
                  "expected_loss_true","claim_occurred","risk_score_true"]))
    tr, enc = encode(tr_df[needed], fit=True)
    va, _   = encode(va_df[needed], encoders=enc, fit=False)
    te, _   = encode(te_df[needed], encoders=enc, fit=False)

    metrics = {}

    # ══════════════════════════════════════════════════════════════════════════
    # 1. FREQUENCY  →  predict lambda_true (smooth, R² > 0.90 expected)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[1/3] Frequency  —  XGBoost → lambda_true")
    Xf_tr, yf_tr = tr[T12].values, tr["lambda_true"].values
    Xf_va, yf_va = va[T12].values, va["lambda_true"].values
    Xf_te, yf_te = te[T12].values, te["lambda_true"].values

    freq_m = xgb.XGBRegressor(
        n_estimators=800, max_depth=5, learning_rate=0.02,
        subsample=0.80, colsample_bytree=0.80,
        min_child_weight=30, reg_alpha=0.05, reg_lambda=1.5,
        objective="reg:squarederror", random_state=42, n_jobs=-1, verbosity=0,
    )
    freq_m.fit(Xf_tr, yf_tr, eval_set=[(Xf_va, yf_va)], verbose=False)

    lam_pred = np.clip(freq_m.predict(Xf_te), 0.005, 0.15)
    metrics["frequency"] = reg_metrics(yf_te, lam_pred, "Frequency (lambda_true)")

    # AUC on binary claim_occurred for reference
    try:
        auc = roc_auc_score(te["claim_occurred"].values, lam_pred)
        metrics["frequency"]["AUC_binary"] = round(auc, 4)
        print(f"   {'AUC on claim_occurred':30s}  {auc:.4f}")
    except Exception:
        pass

    # ══════════════════════════════════════════════════════════════════════════
    # 2. SEVERITY  →  predict mu_true (smooth, R² > 0.92 expected)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[2/3] Severity  —  XGBoost Gamma → mu_true")
    Xs_tr, ys_tr = tr[SEV_FEATS].values, tr["mu_true"].values
    Xs_va, ys_va = va[SEV_FEATS].values, va["mu_true"].values
    Xs_te, ys_te = te[SEV_FEATS].values, te["mu_true"].values

    sev_m = xgb.XGBRegressor(
        n_estimators=800, max_depth=5, learning_rate=0.02,
        subsample=0.80, colsample_bytree=0.80,
        min_child_weight=25, reg_alpha=0.05, reg_lambda=1.5,
        objective="reg:gamma", random_state=42, n_jobs=-1, verbosity=0,
    )
    sev_m.fit(Xs_tr, ys_tr, eval_set=[(Xs_va, ys_va)], verbose=False)

    mu_pred = np.clip(sev_m.predict(Xs_te), 1_000, 500_000)
    metrics["severity"] = reg_metrics(ys_te, mu_pred, "Severity (mu_true)")

    # ══════════════════════════════════════════════════════════════════════════
    # 3. M-HAT  →  stacked OOF ensemble on M_true + noise (anti-leakage)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[3/3] M-hat  —  RF+XGB+LGB → Ridge+Isotonic  (OOF stacking)")

    RNG2    = np.random.default_rng(777)
    # Small lognormal noise so model must generalise, not memorise the formula
    ym_tr   = (tr["M_true"].values * np.exp(RNG2.normal(0, 0.07, len(tr)))).clip(1.0, 4.0)
    ym_va   = (va["M_true"].values * np.exp(RNG2.normal(0, 0.07, len(va)))).clip(1.0, 4.0)
    ym_te   = te["M_true"].values
    Xm_tr, Xm_va, Xm_te = tr[T3].values, va[T3].values, te[T3].values

    # Final base models (trained on full train set)
    rf_m = RandomForestRegressor(
        n_estimators=300, max_depth=7, min_samples_leaf=35,
        n_jobs=-1, random_state=42
    )
    rf_m.fit(Xm_tr, ym_tr)

    xgb_m = xgb.XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.03,
        subsample=0.80, colsample_bytree=0.80,
        min_child_weight=35, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    xgb_m.fit(Xm_tr, ym_tr, eval_set=[(Xm_va, ym_va)], verbose=False)

    lgb_m = lgb.LGBMRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.03,
        subsample=0.80, colsample_bytree=0.80,
        min_child_samples=35, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    lgb_m.fit(Xm_tr, ym_tr,
              eval_set=[(Xm_va, ym_va)],
              callbacks=[lgb.early_stopping(60, verbose=False),
                         lgb.log_evaluation(-1)])

    # OOF meta features (train+val together, 5-fold)
    Xm_all = np.vstack([Xm_tr, Xm_va])
    ym_all = np.concatenate([ym_tr, ym_va])
    oof = np.zeros((len(Xm_all), 3))
    kf  = KFold(n_splits=5, shuffle=True, random_state=42)

    for fi, (idx_tr, idx_va) in enumerate(kf.split(Xm_all)):
        _rf  = RandomForestRegressor(n_estimators=150, max_depth=7,
                                     min_samples_leaf=35, n_jobs=-1, random_state=fi)
        _rf.fit(Xm_all[idx_tr], ym_all[idx_tr])
        oof[idx_va, 0] = _rf.predict(Xm_all[idx_va])

        _xgb = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                  verbosity=0, random_state=fi, n_jobs=-1)
        _xgb.fit(Xm_all[idx_tr], ym_all[idx_tr])
        oof[idx_va, 1] = _xgb.predict(Xm_all[idx_va])

        _lgb = lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                   verbose=-1, random_state=fi, n_jobs=-1)
        _lgb.fit(Xm_all[idx_tr], ym_all[idx_tr])
        oof[idx_va, 2] = _lgb.predict(Xm_all[idx_va])

    ridge_m = Ridge(alpha=1.0)
    ridge_m.fit(oof, ym_all)
    iso_m   = IsotonicRegression(out_of_bounds="clip")
    iso_m.fit(ridge_m.predict(oof), ym_all)

    meta_te  = np.column_stack([rf_m.predict(Xm_te),
                                 xgb_m.predict(Xm_te),
                                 lgb_m.predict(Xm_te)])
    m_hat_te = np.clip(iso_m.predict(ridge_m.predict(meta_te)), 1.0, 4.0)
    metrics["m_hat"] = reg_metrics(ym_te, m_hat_te, "M-hat Ensemble")

    # ── E[L] pipeline validation ───────────────────────────────────────────────
    print("\n  E[L] pipeline:")
    el_pred  = lam_pred * mu_pred * m_hat_te
    el_true  = te["expected_loss_true"].values
    metrics["expected_loss"] = reg_metrics(el_true, el_pred, "E[L] = λ × μ × M̂")

    # Risk score validation
    el_min = float(tr_df["expected_loss_true"].min())
    el_max = float(tr_df["expected_loss_true"].max())
    sc_pred = np.clip(50 + 900*(el_pred - el_min)/(el_max - el_min), 50, 950)
    sc_true = te["risk_score_true"].values
    metrics["risk_score"] = reg_metrics(sc_true, sc_pred, "Risk Score (A1)")

    print(f"\n  Total time: {time.time()-t0:.1f}s")

    # ── Persist ────────────────────────────────────────────────────────────────
    arts = dict(
        freq_model=freq_m, sev_model=sev_m,
        rf_m=rf_m, xgb_m=xgb_m, lgb_m=lgb_m,
        ridge_meta=ridge_m, iso_meta=iso_m,
        encoders=enc, t12=T12, t3=T3, sev_feats=SEV_FEATS, cat_cols=CAT_COLS,
        metrics=metrics,
        el_min=el_min, el_max=el_max,
        n_train=len(tr_df), n_val=len(va_df), n_test=len(te_df),
    )
    with open("models/artifacts.pkl", "wb") as f:
        pickle.dump(arts, f)

    te_df.to_parquet("data/test_data.parquet",  index=False)
    tr_df.to_parquet("data/train_data.parquet", index=False)
    print("✓  models/artifacts.pkl  data/test_data.parquet  data/train_data.parquet")
    print("="*68)
    return arts


if __name__ == "__main__":
    df = pd.read_parquet("data/homeowners_data.parquet")
    train_all(df)
