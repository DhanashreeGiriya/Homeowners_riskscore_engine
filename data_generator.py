"""
data_generator.py  v3
=====================
Actuarially calibrated synthetic US Homeowners dataset.

Fixes from v1/v2:
  1. Gaussian copula for realistic feature correlations
  2. Lambda capped at 15% max, calibrated to 6.5% mean (NAIC 2023)
  3. GPD splice point fixed at u=$85,000 (not loc=mu)
  4. Compound loss model: NegBinom claim count x per-claim severity
  5. Feature-driven zero inflation (not flat constant)
  6. Home value calibrated to Census ACS state-level percentiles
  7. Roof material correlated with home age
  8. Model targets are smooth actuarial signals (lambda_true, mu_true, M_true)
     NOT noisy realized outcomes
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

RNG = np.random.default_rng(42)

# ── Industry calibration constants ─────────────────────────────────────────────
BASE_CLAIM_RATE   = 0.065   # NAIC 2023: 6.5% US homeowner annual claim rate
CAT_THRESHOLD     = 85_000  # P95 splice point for GPD
CAT_BASE_FRAC     = 0.055   # 5.5% of claims are CAT (AIR/RMS benchmark)
TREND_RATE        = 0.06    # BLS PPI construction inflation
TARGET_LR         = 0.65    # Industry median loss ratio
EXPENSE_LOAD      = 0.18    # Carrier expense loading

# ── State parameters (calibrated to ISO loss cost indices + Census ACS) ────────
STATES = {
    "CA": dict(wf=0.55, fl=0.12, eq=0.65, val_med=650_000, coastal=False),
    "FL": dict(wf=0.05, fl=0.65, eq=0.02, val_med=360_000, coastal=True),
    "TX": dict(wf=0.28, fl=0.32, eq=0.05, val_med=300_000, coastal=False),
    "LA": dict(wf=0.05, fl=0.68, eq=0.02, val_med=220_000, coastal=True),
    "OK": dict(wf=0.32, fl=0.22, eq=0.18, val_med=195_000, coastal=False),
    "CO": dict(wf=0.42, fl=0.08, eq=0.12, val_med=460_000, coastal=False),
    "NC": dict(wf=0.08, fl=0.28, eq=0.04, val_med=275_000, coastal=True),
    "GA": dict(wf=0.08, fl=0.22, eq=0.04, val_med=285_000, coastal=False),
    "AZ": dict(wf=0.38, fl=0.06, eq=0.22, val_med=335_000, coastal=False),
    "NV": dict(wf=0.32, fl=0.04, eq=0.38, val_med=360_000, coastal=False),
}
STATE_KEYS  = list(STATES.keys())
STATE_PROBS = [0.18, 0.15, 0.15, 0.07, 0.06, 0.09, 0.07, 0.07, 0.08, 0.08]

CONSTRUCTION = {
    "Frame":    dict(freq_m=1.28, sev_m=1.10, p=0.50),
    "Masonry":  dict(freq_m=0.86, sev_m=0.90, p=0.25),
    "Superior": dict(freq_m=0.72, sev_m=0.80, p=0.10),
    "Mixed":    dict(freq_m=1.08, sev_m=1.00, p=0.15),
}
ROOF = {
    "Asphalt Shingle": dict(fire_r=1.00, p=0.55),
    "Wood Shake":      dict(fire_r=1.80, p=0.08),
    "Metal":           dict(fire_r=0.70, p=0.14),
    "Tile":            dict(fire_r=0.80, p=0.18),
    "Flat/Built-Up":   dict(fire_r=1.20, p=0.05),
}
OCCUPANCY = {
    "Owner Occupied":  dict(freq_m=1.00, p=0.74),
    "Tenant Occupied": dict(freq_m=1.22, p=0.19),
    "Vacant":          dict(freq_m=1.65, p=0.07),
}
DED_VALS  = [500, 1000, 2500, 5000]
DED_PROBS = [0.12, 0.42, 0.32, 0.14]
DED_FREQ  = {500: 1.00, 1000: 0.90, 2500: 0.75, 5000: 0.62}


def generate_dataset(n: int = 100_000) -> pd.DataFrame:
    print(f"Generating {n:,} actuarially calibrated homeowners records...")

    # ── State assignment ───────────────────────────────────────────────────────
    state = RNG.choice(STATE_KEYS, size=n, p=STATE_PROBS)

    # ── Copula: correlated continuous features ─────────────────────────────────
    # Correlation matrix from insurance studies:
    # [home_age, home_value, credit_score, prior_claims_latent, protection_class]
    corr = np.array([
        [ 1.000, -0.150, -0.180,  0.220,  0.280],
        [-0.150,  1.000,  0.230, -0.120, -0.320],
        [-0.180,  0.230,  1.000, -0.420, -0.190],
        [ 0.220, -0.120, -0.420,  1.000,  0.150],
        [ 0.280, -0.320, -0.190,  0.150,  1.000],
    ])
    # Ensure PD
    ev = np.linalg.eigvalsh(corr)
    if ev.min() < 1e-8:
        corr += np.eye(5) * (abs(ev.min()) + 1e-6)

    L = np.linalg.cholesky(corr)
    Z = RNG.standard_normal((n, 5))
    U = norm.cdf(Z @ L.T)   # correlated uniforms in [0,1]

    val_meds = np.array([STATES[s]["val_med"] for s in state])

    home_age         = np.clip(np.round(stats.beta.ppf(U[:,0], 2, 3) * 70 + 3), 2, 74).astype(int)
    home_value       = np.clip(np.round(val_meds * stats.lognorm.ppf(
                           np.clip(U[:,1], 0.01, 0.99), s=0.26)), 80_000, 2_500_000).astype(int)
    credit_score     = np.clip(np.round(stats.beta.ppf(U[:,2], 6, 2.5) * 350 + 500), 500, 850).astype(int)
    latent_claims    = stats.expon.ppf(np.clip(U[:,3], 0.001, 0.999), scale=0.35)
    protection_class = np.clip(np.round(stats.beta.ppf(U[:,4], 2, 2) * 9 + 1), 1, 10).astype(int)

    year_built       = np.clip(2024 - home_age, 1950, 2022)
    coverage_ratio   = RNG.uniform(0.85, 1.15, n)
    coverage_amount  = np.clip((home_value * coverage_ratio).astype(int), 80_000, 2_500_000)
    square_footage   = np.clip(np.round(RNG.normal(2100, 700, n)), 500, 6000).astype(int)
    stories          = RNG.choice([1, 2, 3], size=n, p=[0.56, 0.37, 0.07])

    # ── Prior claims: zero-inflated NB driven by latent tendency ──────────────
    zero_p = np.clip(0.82 - 0.14 * latent_claims, 0.40, 0.92)
    is_zero = RNG.random(n) < zero_p
    raw_cnt = RNG.poisson(latent_claims * 0.85, n)
    prior_claims_3yr = np.where(is_zero, 0, np.clip(raw_cnt, 1, 5)).astype(int)

    # ── Categorical features ───────────────────────────────────────────────────
    construction_type = _pick(CONSTRUCTION, n)

    # Wood Shake more likely on older homes (age correlation)
    roof_keys = list(ROOF.keys())
    roof_base_p = np.array([ROOF[k]["p"] for k in roof_keys])
    roof_material = np.array([
        RNG.choice(roof_keys, p=_roof_probs(roof_base_p, home_age[i]))
        for i in range(n)
    ])

    occupancy  = _pick(OCCUPANCY, n)
    deductible = RNG.choice(DED_VALS, size=n, p=DED_PROBS)

    # ── Binary behavioral — correlated with credit/claims ─────────────────────
    cr_n = (credit_score - 500) / 350.0   # normalized 0-1
    cl_n = prior_claims_3yr / 5.0

    security_system  = (RNG.random(n) < np.clip(0.28 + 0.40 * cr_n, 0.12, 0.72)).astype(int)
    smoke_detectors  = (RNG.random(n) < np.clip(0.76 + 0.16 * cr_n, 0.62, 0.94)).astype(int)
    sprinkler_system = (RNG.random(n) < np.clip(0.07 + 0.13 * cr_n, 0.03, 0.22)).astype(int)
    gated_community  = (RNG.random(n) < np.clip(0.07 + 0.22 * cr_n, 0.03, 0.30)).astype(int)
    swimming_pool    = (RNG.random(n) < np.clip(0.24 - 0.04 * cl_n, 0.10, 0.30)).astype(int)
    trampoline       = (RNG.random(n) < np.clip(0.11 + 0.05 * cl_n, 0.05, 0.20)).astype(int)
    dog              = (RNG.random(n) < 0.38).astype(int)
    roof_age_yr      = np.clip(np.round(RNG.beta(2, 4, n) * 32).astype(int), 0, home_age)

    # ── Tier 3: Hazard zones ───────────────────────────────────────────────────
    wf_p = np.array([STATES[s]["wf"] for s in state])
    fl_p = np.array([STATES[s]["fl"] for s in state])
    eq_p = np.array([STATES[s]["eq"] for s in state])

    wildfire_zone   = _zone(wf_p, n)
    flood_zone      = _zone(fl_p, n)
    earthquake_zone = _zone(eq_p, n)

    coastal_st = {"FL", "NC", "GA", "LA"}
    dist_to_coast = np.where(
        np.isin(state, list(coastal_st)),
        np.clip(RNG.exponential(7, n),  0.2,  80),
        np.clip(RNG.exponential(60, n), 1.0, 500),
    ).round(1)
    dist_to_fire_station = np.clip(RNG.exponential(3.2, n), 0.2, 30).round(1)

    # ── Assemble ───────────────────────────────────────────────────────────────
    df = pd.DataFrame({
        "policy_id":               [f"POL{i+1:06d}" for i in range(n)],
        "state":                   state,
        "construction_type":       construction_type,
        "roof_material":           roof_material,
        "occupancy":               occupancy,
        "year_built":              year_built,
        "home_age":                home_age,
        "home_value":              home_value,
        "coverage_amount":         coverage_amount,
        "square_footage":          square_footage,
        "stories":                 stories,
        "protection_class":        protection_class,
        "prior_claims_3yr":        prior_claims_3yr,
        "credit_score":            credit_score,
        "deductible":              deductible,
        "swimming_pool":           swimming_pool,
        "trampoline":              trampoline,
        "dog":                     dog,
        "security_system":         security_system,
        "smoke_detectors":         smoke_detectors,
        "sprinkler_system":        sprinkler_system,
        "gated_community":         gated_community,
        "roof_age_yr":             roof_age_yr,
        "wildfire_zone":           wildfire_zone,
        "flood_zone":              flood_zone,
        "earthquake_zone":         earthquake_zone,
        "dist_to_coast_mi":        dist_to_coast,
        "dist_to_fire_station_mi": dist_to_fire_station,
    })

    df = _compute_targets(df)

    cr = df["claim_occurred"].mean() * 100
    print(f"  Rows: {len(df):,}  Cols: {df.shape[1]}")
    print(f"  Claim rate:  {cr:.2f}%  (target 6-9%)")
    print(f"  Avg lambda:  {df['lambda_true'].mean():.4f}")
    print(f"  Avg mu:      ${df['mu_true'].mean():,.0f}")
    print(f"  Avg M:       {df['M_true'].mean():.3f}")
    print(f"  Avg E[L]:    ${df['expected_loss_true'].mean():,.0f}")
    print(f"  Avg premium: ${df['annual_premium'].mean():,.0f}")
    print(f"  High/VHigh:  {df['risk_band'].isin(['High','Very High']).mean()*100:.1f}%")
    return df


# ── Actuarial target computation ───────────────────────────────────────────────
def _compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    # ── FREQUENCY lambda ──────────────────────────────────────────────────────
    lam = np.full(n, BASE_CLAIM_RATE)
    lam *= df["construction_type"].map({k: v["freq_m"] for k, v in CONSTRUCTION.items()}).values
    lam *= 1 + 0.0025 * np.maximum(df["home_age"].values - 10, 0)
    lam *= 0.82 + 0.036 * (df["protection_class"].values - 1)
    lam *= df["occupancy"].map({k: v["freq_m"] for k, v in OCCUPANCY.items()}).values
    lam *= np.power(1.32, df["prior_claims_3yr"].values)
    lam *= np.power(750 / df["credit_score"].values.clip(500, 850), 0.55)
    lam *= df["deductible"].map(DED_FREQ).values
    lam *= np.where(df["swimming_pool"].values,   1.10, 1.0)
    lam *= np.where(df["trampoline"].values,      1.14, 1.0)
    lam *= np.where(df["dog"].values,             1.08, 1.0)
    lam *= np.where(df["security_system"].values, 0.90, 1.0)
    lam *= np.where(df["smoke_detectors"].values, 0.93, 1.0)
    lam *= np.where(df["sprinkler_system"].values,0.82, 1.0)
    lam *= np.where(df["gated_community"].values, 0.91, 1.0)
    # Calibrate mean to BASE_CLAIM_RATE, then hard-cap
    lam  = lam / lam.mean() * BASE_CLAIM_RATE
    lam  = np.clip(lam, 0.008, 0.15)
    df["lambda_true"] = lam.round(5)

    # ── SEVERITY mu (smooth actuarial signal) ─────────────────────────────────
    mu = df["home_value"].values * 0.075
    mu *= 1 + (df["square_footage"].values - 2000) / 25_000
    mu *= 1 + 0.04 * (df["stories"].values - 1)
    mu *= 0.87 + 0.025 * (df["protection_class"].values - 1)
    mu *= 1 + 0.018 * df["dist_to_fire_station_mi"].values
    mu *= np.where(df["smoke_detectors"].values,  0.87, 1.0)
    mu *= np.where(df["sprinkler_system"].values, 0.62, 1.0)
    mu  = np.minimum(mu, df["coverage_amount"].values * 0.80)
    mu  = np.clip(mu, 2_000, 500_000)
    df["mu_true"] = mu.round(0)

    # ── INTERACTION M (Tier-3 multipliers) ────────────────────────────────────
    M     = np.ones(n)
    wood  = df["roof_material"].values == "Wood Shake"
    wf_h  = df["wildfire_zone"].values == "High"
    wf_m  = df["wildfire_zone"].values == "Moderate"
    fl_h  = df["flood_zone"].values    == "High"
    fl_m  = df["flood_zone"].values    == "Moderate"
    near  = df["dist_to_coast_mi"].values < 5
    eq_h  = df["earthquake_zone"].values == "High"
    eq_m  = df["earthquake_zone"].values == "Moderate"
    old_r = df["roof_age_yr"].values > 20
    frame = df["construction_type"].values == "Frame"

    M = np.where(wood & wf_h,        M * 3.50, M)
    M = np.where(wood & wf_m,        M * 2.10, M)
    M = np.where(~wood & wf_h,       M * 1.80, M)
    M = np.where(wood & ~wf_h & ~wf_m, M * 1.40, M)
    M = np.where(fl_h & near,        M * 2.20, M)
    M = np.where(fl_h & ~near,       M * 1.60, M)
    M = np.where(fl_m,               M * 1.20, M)
    M = np.where(eq_h,               M * 1.50, M)
    M = np.where(eq_m,               M * 1.15, M)
    M = np.where(old_r & frame,      M * 1.35, M)
    M = np.where(old_r & ~frame,     M * 1.15, M)
    M = np.clip(M, 1.0, 4.0)
    df["M_true"] = M.round(4)

    # ── Expected loss (smooth) ────────────────────────────────────────────────
    el = lam * mu * M
    df["expected_loss_true"] = el.round(2)

    # ── Claim occurrence & amount (compound NB model) ─────────────────────────
    nb_r   = 0.8
    nb_p   = nb_r / (nb_r + lam)
    cnt    = RNG.negative_binomial(nb_r, nb_p, n).clip(0, 5)
    df["claim_occurred"] = (cnt > 0).astype(int)
    df["claim_count"]    = cnt

    # Per-policy total loss via spliced Gamma + GPD
    total_loss = np.zeros(n)
    for i in np.where(cnt > 0)[0]:
        losses = []
        for _ in range(int(cnt[i])):
            cat_p = float(np.clip(CAT_BASE_FRAC * M[i], 0.02, 0.35))
            if RNG.random() < cat_p:
                # GPD tail — splice point = CAT_THRESHOLD
                x = stats.genpareto.rvs(c=0.25,
                                        scale=float(mu[i]) * 0.6,
                                        loc=CAT_THRESHOLD)
            else:
                # Gamma attritional
                shape = 2.5
                x = float(RNG.gamma(shape, float(mu[i]) / shape))
            losses.append(x)
        raw = sum(losses) * (1 + TREND_RATE) ** 2
        raw = min(raw, float(df["coverage_amount"].iloc[i]))
        total_loss[i] = max(raw, 100.0)

    # Cap at 99.5th percentile
    pos = total_loss[total_loss > 0]
    cap = float(np.percentile(pos, 99.5)) if len(pos) > 0 else 1e6
    total_loss = np.minimum(total_loss, cap)
    df["claim_amount"] = total_loss.round(2)

    # ── Risk score & premium ──────────────────────────────────────────────────
    el_arr = df["expected_loss_true"].values
    df["risk_score_true"] = np.clip(
        50 + 900 * (el_arr - el_arr.min()) / (el_arr.max() - el_arr.min()),
        50, 950
    ).round(1)

    def _band(s):
        if s < 200: return "Very Low"
        if s < 400: return "Low"
        if s < 600: return "Moderate"
        if s < 800: return "High"
        return "Very High"

    df["risk_band"]     = df["risk_score_true"].apply(_band)
    df["annual_premium"] = (el_arr / TARGET_LR * (1 + EXPENSE_LOAD)).round(2)
    return df


# ── Helpers ────────────────────────────────────────────────────────────────────
def _pick(d: dict, n: int) -> np.ndarray:
    keys = list(d.keys())
    p    = [d[k]["p"] for k in keys]
    return RNG.choice(keys, size=n, p=p)


def _roof_probs(base: np.ndarray, age: int) -> np.ndarray:
    p = base.copy()
    boost = float(np.clip((age - 20) / 80, 0, 0.08))
    p[1] += boost          # index 1 = Wood Shake
    return (p / p.sum())


def _zone(prob: np.ndarray, n: int) -> np.ndarray:
    r = RNG.random(n)
    return np.where(r < prob * 0.38, "High",
           np.where(r < prob * 0.78, "Moderate", "Low"))


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    df = generate_dataset(100_000)
    df.to_parquet("data/homeowners_data.parquet", index=False)
    print("\nSample:")
    print(df[["lambda_true","mu_true","M_true","expected_loss_true","risk_band"]].describe().round(2))
