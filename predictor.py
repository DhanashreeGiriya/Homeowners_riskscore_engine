"""
predictor.py  v3
================
Loads trained artifacts, runs full lambda x mu x M pipeline.
Includes input validation, what-if scenario support, SHAP.
"""

import pickle, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_arts = None


def load_arts():
    global _arts
    if _arts is None:
        with open("models/artifacts.pkl", "rb") as f:
            _arts = pickle.load(f)
    return _arts


def validate_inputs(d: dict) -> list:
    """Return list of warning strings (empty = all OK)."""
    warnings_out = []
    hv = d.get("home_value", 0)
    ca = d.get("coverage_amount", 0)
    yb = d.get("year_built", 2000)
    ded = d.get("deductible", 1000)
    pc  = d.get("protection_class", 5)

    if ca < hv * 0.70:
        warnings_out.append(f"⚠️ Coverage (${ca:,.0f}) is below 70% of home value — underinsurance risk")
    if ca > hv * 1.30:
        warnings_out.append(f"⚠️ Coverage (${ca:,.0f}) exceeds 130% of home value — over-insurance / moral hazard")
    if yb > 2024:
        warnings_out.append("⚠️ Year built is in the future")
    if yb < 1900:
        warnings_out.append("⚠️ Year built before 1900 — unusual for active policy")
    if ded > hv * 0.06:
        warnings_out.append(f"⚠️ Deductible (${ded:,}) exceeds 6% of home value — high self-retention")
    if pc == 10 and d.get("dist_to_fire_station_mi", 3) < 1:
        warnings_out.append("⚠️ PC=10 but fire station distance < 1mi — data inconsistency")
    return warnings_out


def _encode_row(d: dict) -> pd.DataFrame:
    arts    = load_arts()
    enc     = arts["encoders"]
    cat_cols= arts["cat_cols"]
    row     = pd.DataFrame([d])
    for col in cat_cols:
        if col not in row.columns:
            continue
        le  = enc[col]
        val = str(row[col].iloc[0])
        if val not in le.classes_:
            val = le.classes_[0]
        row[col] = le.transform([val])
    return row


def _risk_band(score):
    if score < 200: return "Very Low",  "#22c55e"
    if score < 400: return "Low",       "#84cc16"
    if score < 600: return "Moderate",  "#eab308"
    if score < 800: return "High",      "#f97316"
    return               "Very High",   "#ef4444"


def _uw_action(band):
    return {
        "Very Low":  ("✅ Accept — Best Terms",              "#22c55e"),
        "Low":       ("✅ Accept — Standard Terms",          "#84cc16"),
        "Moderate":  ("🔶 Accept with Conditions",          "#eab308"),
        "High":      ("🔴 Refer to Senior Underwriter",     "#f97316"),
        "Very High": ("🚫 Decline / Surplus Lines Market",  "#ef4444"),
    }[band]


def predict(inp: dict) -> dict:
    arts = load_arts()
    row  = _encode_row(inp)

    # 1. Frequency (lambda)
    Xf       = row[arts["t12"]].values
    lam_pred = float(np.clip(arts["freq_model"].predict(Xf)[0], 0.005, 0.15))

    # 2. Severity (mu)
    Xs      = row[arts["sev_feats"]].values
    mu_pred = float(np.clip(arts["sev_model"].predict(Xs)[0], 1_000, 500_000))

    # 3. M-hat
    Xm       = row[arts["t3"]].values
    meta_in  = np.column_stack([
        arts["rf_m"].predict(Xm),
        arts["xgb_m"].predict(Xm),
        arts["lgb_m"].predict(Xm),
    ])
    ridge_p  = arts["ridge_meta"].predict(meta_in)[0]
    m_hat    = float(np.clip(arts["iso_meta"].predict([ridge_p])[0], 1.0, 4.0))

    # 4. E[L]
    el = lam_pred * mu_pred * m_hat

    # 5. Risk Score A1 (portfolio normalised)
    el_min, el_max = arts["el_min"], arts["el_max"]
    score_a1 = float(np.clip(50 + 900*(el - el_min)/(el_max - el_min), 50, 950))

    # 6. Risk Score A2 (F+S composite)
    f_score = min(500.0, lam_pred / 0.15 * 500)
    s_score = min(500.0, (mu_pred * m_hat) / 600_000 * 500)
    score_a2 = float(np.clip(
        (0.45*(f_score+1)**0.8 + 0.55*(s_score+1)**0.8)**(1/0.8) - 1, 0, 1000
    ))

    # 7. Premium
    premium = el / 0.65 * 1.18

    band,  color  = _risk_band(score_a1)
    action, acol  = _uw_action(band)

    # 8. Interaction breakdown
    interactions = _get_interactions(inp)

    return dict(
        lambda_pred=round(lam_pred, 5),
        mu_pred=round(mu_pred, 2),
        m_hat=round(m_hat, 3),
        expected_loss=round(el, 2),
        risk_score_a1=round(score_a1, 1),
        risk_score_a2=round(score_a2, 1),
        f_score=round(f_score, 1),
        s_score=round(s_score, 1),
        premium=round(premium, 2),
        pure_premium=round(el / 0.65, 2),
        risk_band=band,
        risk_color=color,
        uw_action=action,
        uw_color=acol,
        interactions=interactions,
        warnings=validate_inputs(inp),
    )


def predict_whatif(base_inp: dict, changes: dict) -> dict:
    """Return prediction for base + modified scenario."""
    modified = {**base_inp, **changes}
    return predict(modified)


def _get_interactions(inp: dict) -> list:
    """Return list of (label, multiplier, color) for active interactions."""
    out  = []
    wood = inp.get("roof_material") == "Wood Shake"
    wf   = inp.get("wildfire_zone", "Low")
    fl   = inp.get("flood_zone", "Low")
    eq   = inp.get("earthquake_zone", "Low")
    coast= inp.get("dist_to_coast_mi", 99) < 5
    old_r= inp.get("roof_age_yr", 0) > 20
    frame= inp.get("construction_type") == "Frame"

    if   wood and wf == "High":    out.append(("Wood Shake × High Wildfire",   3.50, "#ef4444"))
    elif wood and wf == "Moderate":out.append(("Wood Shake × Mod Wildfire",    2.10, "#f97316"))
    elif wf == "High":             out.append(("Non-Wood × High Wildfire",     1.80, "#f97316"))
    elif wood:                     out.append(("Wood Shake (base fire risk)",  1.40, "#eab308"))
    if   fl == "High" and coast:   out.append(("High Flood × Coastal <5mi",   2.20, "#ef4444"))
    elif fl == "High":             out.append(("High Flood Zone",              1.60, "#f97316"))
    elif fl == "Moderate":         out.append(("Moderate Flood Zone",          1.20, "#eab308"))
    if   eq == "High":             out.append(("High Earthquake Zone",         1.50, "#f97316"))
    elif eq == "Moderate":         out.append(("Moderate Earthquake Zone",     1.15, "#eab308"))
    if   old_r and frame:          out.append(("Old Roof(>20yr) × Frame",      1.35, "#f97316"))
    elif old_r:                    out.append(("Aged Roof > 20 years",         1.15, "#eab308"))
    return out


def get_shap_values(inp: dict) -> dict:
    import shap
    arts = load_arts()
    row  = _encode_row(inp)

    out = {}
    for name, model, feats in [
        ("Frequency (λ)", arts["freq_model"], arts["t12"]),
        ("Severity (μ)",  arts["sev_model"],  arts["sev_feats"]),
        ("M-hat (M̂)",    arts["xgb_m"],      arts["t3"]),
    ]:
        expl = shap.TreeExplainer(model)
        sv   = expl.shap_values(row[feats].values)
        if isinstance(sv, list):
            sv = sv[1]
        base = expl.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = float(base[0])
        out[name] = dict(values=sv[0].tolist(), features=feats, base=float(base))
    return out
