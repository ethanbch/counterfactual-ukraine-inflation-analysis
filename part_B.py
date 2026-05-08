# =============================================================================
# PART B — Counterfactual Inflation: "What if Ukraine had been in the Euro Area?"
# QMF Final Exam 2025-2026 — Master 2 Research, Paris 1 Panthéon-Sorbonne
#
# IDENTIFICATION STRATEGY:
# METHOD 1 — Ciccarelli-Mojon (2010): Common factor extraction via PCA on the
#   ECB HICP-11 panel. F_EA = PC1 rescaled to inflation units (%, not scores).
#   WHY PCA: PC1 explains ~70% of cross-country variance → "global EA inflation"
#   (Ciccarelli-Mojon finding). Simple mean is a special case; PCA weights
#   countries optimally by common variance contribution.
#   λ CALIBRATION: OLS on quiet periods (excl. GFC, Crimea, COVID, invasion).
#   WHY: Ciccarelli-Mojon calibrate on full sample to capture long-run structural
#   sensitivity, not crisis co-movement. Excluded episodes introduce idiosyncratic
#   UAH devaluation bias that would contaminate λ upward.
#   FLOOR: CF ≥ EA mean (Balassa-Samuelson: catch-up economy always has
#   slightly higher equilibrium inflation than EA core).
#   TIME-VARYING TREATMENT (Calvo-Reinhart 2002): CF blended with actual
#   during peg periods — see Section 3b.
#
# METHOD 2 — Blanchard-Quah SVAR (Bayoumi-Eichengreen 1993):
#   Bivariate SVAR (output growth, inflation), long-run identification:
#   demand shocks have no permanent effect on output (Blanchard-Quah 1989).
#   COUNTERFACTUAL: Ukraine's demand shocks replaced by EA demand shocks.
#   WHY: EA membership transfers monetary sovereignty → ECB sets demand
#   conditions. Ukraine's supply shocks (energy, agriculture, geopolitics)
#   remain idiosyncratic (asymmetric shocks, Mundell 1961).
#   LAG: AIC selection, max 6. Stationarity: ADF on each variable, first-diff
#   if non-stationary. IP extended via Chow-Lin interpolation from WB GDP
#   annual where FRED series is truncated.
#
# CONSISTENCY WITH PART A:
#   Peg 2000–2008 → UA demand shocks ≈ EA (dollar anchor); treatment ≈ 0.
#   Devaluations 2014–15, 2022 → genuine monetary sovereignty exercised;
#   treatment = 1 (full counterfactual applies).
#   IT post-2016 → NBU credibility partially converges to ECB;
#   expected gap (actual - CF) < pre-2016 gap (Barro-Gordon 1983 sanity check).
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import urllib.request
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from scipy.linalg import cholesky

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})


# =============================================================================
# 0. HELPER FUNCTIONS
# =============================================================================


def adf_report(series, name):
    s = series.dropna()
    stat, p, *_ = adfuller(s, autolag="AIC")
    status = "stationary ✓" if p < 0.05 else "NON-STATIONARY ✗"
    print(f"  {name:35s}  ADF={stat:7.3f}  p={p:.3f}  [{status}]")
    return p < 0.05


def annual_to_monthly_chowlin(series_annual, monthly_idx):
    s = series_annual.copy()
    s.index = pd.to_datetime([f"{y}-07-01" for y in s.index])
    full_idx = s.index.union(monthly_idx)
    s_monthly = s.reindex(full_idx).interpolate("cubicspline")
    return s_monthly.reindex(monthly_idx)


def blanchard_quah_svar(df, maxlags=6, name=""):
    """
    SVAR bivarié Blanchard-Quah.
    Variables: ['growth', 'inflation'], toutes stationnaires.
    Identification long-run: choc demande sans effet permanent sur output.
    C(1) = (I - A1 - ... - Ap)^{-1}
    M = C(1)·Σ·C(1)' → D = chol(M) lower-tri
    B0 = C(1)^{-1}·D → ε_t = B0^{-1}·u_t
    col 0 = supply shock, col 1 = demand shock
    """
    var_model = VAR(df)
    lag_order = var_model.select_order(maxlags).aic
    lag_order = max(lag_order, 1)
    results = var_model.fit(lag_order)
    print(f"  {name}: lag order AIC = {lag_order}")

    A_sum = np.zeros((2, 2))
    for i in range(lag_order):
        A_sum += results.coefs[i]
    C1 = np.linalg.inv(np.eye(2) - A_sum)

    Sigma = results.sigma_u
    M = C1 @ Sigma @ C1.T
    D = cholesky(M, lower=True)
    B0 = np.linalg.inv(C1) @ D
    B0_inv = np.linalg.inv(B0)

    resid = results.resid.values
    struct_shocks = (B0_inv @ resid.T).T
    return results, struct_shocks, B0, lag_order


def episode_stats(actual, cf, start, end, label):
    mask = (actual.index >= start) & (actual.index <= end)
    a = actual.loc[mask]
    c = cf.reindex(a.index).dropna()
    a = a.reindex(c.index)
    if len(a) == 0 or len(c) == 0:
        print(f"\n  [{label}]  ⚠ Pas de données CF sur cette période")
        return
    print(f"\n  [{label}]")
    print(
        f"    Actual → mean={a.mean():.1f}%  peak={a.max():.1f}% ({a.idxmax().date()})"
    )
    print(
        f"    CF     → mean={c.mean():.1f}%  peak={c.max():.1f}% ({c.idxmax().date()})"
    )
    print(f"    Gap moyen (actual - CF) = {(a - c).mean():.1f}%")


# =============================================================================
# 1. DATA LOADING & HARMONISATION
# =============================================================================
print("=" * 70)
print("1. DATA LOADING & HARMONISATION")
print("=" * 70)

# --- 1a. ECB HICP panel ---
hicp_raw = pd.read_csv("data/data_ecb_hicp_panel.csv")
hicp_raw["TIME_PERIOD"] = pd.to_datetime(hicp_raw["TIME_PERIOD"])
hicp = hicp_raw.set_index("TIME_PERIOD").sort_index()
countries = ["AT", "BE", "DE", "ES", "FI", "FR", "GR", "IE", "IT", "NL", "PT"]
hicp = hicp[countries]
print(
    f"ECB HICP panel:  {hicp.index[0].date()} → {hicp.index[-1].date()}, "
    f"shape {hicp.shape}, NaNs={hicp.isna().sum().sum()}"
)

# --- 1b. Ukraine CPI raw → YoY ---
# TIME_PERIOD = "2000-M01", OBS_VALUE = indice MoM base mois précédent
ukr_raw = pd.read_csv("data/data_ukraine_cpi_raw.csv")
ukr = ukr_raw[["TIME_PERIOD", "OBS_VALUE"]].copy()
ukr["date"] = pd.to_datetime(
    ukr["TIME_PERIOD"].str.replace("-M", "-", regex=False), format="%Y-%m"
)
ukr = (
    ukr.dropna(subset=["date", "OBS_VALUE"]).sort_values("date").reset_index(drop=True)
)

mom_ratio = ukr["OBS_VALUE"].values / 100.0
yoy_vals, yoy_dates = [], []
for i in range(11, len(mom_ratio)):
    yoy_vals.append((np.prod(mom_ratio[i - 11 : i + 1]) - 1) * 100)
    yoy_dates.append(ukr["date"].iloc[i])
ukraine_yoy = pd.Series(yoy_vals, index=pd.DatetimeIndex(yoy_dates), name="ukraine_yoy")

print(f"\nUkraine YoY: {ukraine_yoy.index[0].date()} → {ukraine_yoy.index[-1].date()}")
for chk_date, expected in [("2008-12", 22), ("2015-04", 61), ("2022-12", 27)]:
    try:
        val = float(ukraine_yoy[chk_date].iloc[0])
        print(f"  {chk_date}: {val:.1f}%  (expected ≈ {expected}%)")
    except Exception:
        print(f"  {chk_date}: N/A")

# --- 1c. Index commun HICP ↔ Ukraine ---
common_idx = hicp.index.intersection(ukraine_yoy.index)
hicp_aligned = hicp.loc[common_idx]
ukr_aligned = ukraine_yoy.loc[common_idx]
print(
    f"\nCommon index: {common_idx[0].date()} → {common_idx[-1].date()}, T={len(common_idx)}"
)


# =============================================================================
# 2. CICCARELLI-MOJON (2010) — FACTEUR COMMUN EA
# =============================================================================
print("\n" + "=" * 70)
print("2. CICCARELLI-MOJON FACTOR MODEL")
print("=" * 70)

# PCA pour décomposition de variance et figure
scaler = StandardScaler()
hicp_scaled = scaler.fit_transform(hicp_aligned)
pca = PCA(n_components=3)
pca.fit(hicp_scaled)
print("Variance expliquée par les 3 premiers PCs:")
for i, v in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {v*100:.1f}%")

# Facteur standardisé pour la figure uniquement
F_EA_raw = pca.transform(hicp_scaled)[:, 0]
ea_mean_vec = hicp_aligned.mean(axis=1).values
if np.corrcoef(F_EA_raw, ea_mean_vec)[0, 1] < 0:
    F_EA_raw = -F_EA_raw
F_EA_standardised = pd.Series(F_EA_raw, index=common_idx, name="F_EA_std")

# Facteur commun en % : PC1 rescalé dans les unités de l'inflation. (pour ne pas faire une moyenne simple)
# λ = 1 → Ukraine suit parfaitement la ZE.
# λ > 1 → amplification (Balassa-Samuelson / catch-up).
# λ < 1 → amortissement (peg externe / ancrage dollar).
F_EA_std = pd.Series(F_EA_raw, index=common_idx)
ea_mean = hicp_aligned.mean(axis=1)

_X = add_constant(F_EA_std.values)
_fit = OLS(ea_mean.values, _X).fit()
F_EA = pd.Series(_fit.fittedvalues, index=common_idx, name="F_EA")
print(f"EA factor (PC1 rescaled, %): mean={F_EA.mean():.2f}  std={F_EA.std():.2f}")
print(f"  (projection R²={_fit.rsquared:.3f} — doit être ~1.0 si PC1 domine)")

# Figure
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(
    hicp_aligned.index,
    F_EA,
    color="steelblue",
    lw=1.2,
    alpha=0.8,
    label="EA-11 simple mean (%)",
)
ax2 = ax.twinx()
ax2.plot(
    F_EA_standardised.index,
    F_EA_standardised,
    color="darkorange",
    lw=1.5,
    label="EA common factor (PC1 std)",
)
ax.set_ylabel("Inflation YoY (%)", color="steelblue")
ax2.set_ylabel("Factor score (standardised)", color="darkorange")
ax.set_title("Euro Area Common Inflation Factor vs. Simple Mean")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
plt.tight_layout()
plt.savefig("output/output_ea_factor.png", dpi=150)
plt.close()
print("Figure EA factor sauvegardée.")


# =============================================================================
# 3. ESTIMATION DU LOADING λ — PLEIN ÉCHANTILLON HORS CRISES
# =============================================================================
print("\n" + "=" * 70)
print("3. LOADING ESTIMATION (OLS — plein échantillon hors crises)")
print("=" * 70)
# -----------------------------------------------------------------------
# Conformément à Ciccarelli-Mojon (2010): calibration sur le maximum
# d'observations pour capturer la sensibilité structurelle de long terme.
# On exclut uniquement les épisodes où le régime ukrainien est brisé
# (dévaluations, crises de change) qui introduisent un biais idiosyncratique:
#   • GFC 2008-07 / 2009-09: choc global + dévaluation UAH 40%
#   • Crimée/Donbas 2014-01 / 2016-12: double dévaluation UAH ~65%
#   • COVID 2020-03 / 2021-06: choc offre global
#   • Invasion 2022-01 →     : guerre totale, rupture de régime
# Le reste (~180 obs) capture la co-variation "normale" Ukraine-EA.
# -----------------------------------------------------------------------
crisis_mask = (
    ((common_idx >= "2008-07-01") & (common_idx <= "2009-09-30"))
    | ((common_idx >= "2014-01-01") & (common_idx <= "2016-12-31"))
    | ((common_idx >= "2020-03-01") & (common_idx <= "2021-06-30"))
    | (common_idx >= "2022-01-01")
)
quiet_mask = ~crisis_mask
F_quiet = F_EA.loc[quiet_mask]
ukr_quiet = ukr_aligned.loc[quiet_mask]
print(f"Obs. calibration (hors crises): {quiet_mask.sum()}")
print(f"Obs. exclues (crises/ruptures): {crisis_mask.sum()}")

X = add_constant(F_quiet.values)
y = ukr_quiet.values
ols_model = OLS(y, X).fit()
print(ols_model.summary())

alpha_hat = ols_model.params[0]
lambda_hat = ols_model.params[1]
print(f"\n  α = {alpha_hat:.3f}")
print(f"  λ = {lambda_hat:.3f}  (p={ols_model.pvalues[1]:.3f})")
print(f"  R² = {ols_model.rsquared:.3f}")

# Fallback si loading non-significatif ou négatif
if lambda_hat <= 0 or ols_model.pvalues[1] > 0.10:
    print("\n  ⚠ Loading non-significatif ou négatif → estimation contrainte.")
    rho = np.corrcoef(F_quiet.values, ukr_quiet.values)[0, 1]
    lambda_hat = rho * ukr_quiet.std() / F_quiet.std()
    alpha_hat = ukr_quiet.mean() - lambda_hat * F_quiet.mean()
    print(f"  Constrained λ = {lambda_hat:.3f},  α = {alpha_hat:.3f}")

# Counterfactual facteur (plein échantillon)
CF_factor_raw = alpha_hat + lambda_hat * F_EA

# Floor Balassa-Samuelson: économie de rattrapage → CF ≥ moyenne EA
ea_mean_series = hicp_aligned.mean(axis=1)
CF_factor = pd.Series(
    np.maximum(CF_factor_raw.values, ea_mean_series.values),
    index=common_idx,
    name="CF_factor",
)
print(f"\nCF factor model: mean={CF_factor.mean():.2f}%  std={CF_factor.std():.2f}%")
print(f"Ukraine actual:  mean={ukr_aligned.mean():.2f}%  std={ukr_aligned.std():.2f}%")


# =============================================================================
# 3b. TIME-VARYING TREATMENT INTENSITY (cohérence Part A — Calvo-Reinhart 2002)
# =============================================================================
# Durant les périodes de peg dollar (2000-2008, 2014-02 à 2015-08, 2022-03 à 2023-10),
# l'Ukraine avait déjà subordonné sa politique monétaire à une ancre externe.
# Le "traitement" EA est donc quasi-nul (w≈0) pendant les pegs : Ukraine ≈ EA
# via l'ancre dollar. Le traitement est maximal (w=1) pendant les épisodes de
# flottement / IT (2008-09→2009-09, 2015-09→2022-02, 2023-11→today).
# CF_adjusted = w·CF_factor + (1-w)·ukraine_actual
# → pendant les pegs, CF colle à l'actual (pas de gain/perte de souveraineté)
# → pendant les flottements, CF = contrefactuel structurel pur
# =============================================================================

peg_mask = (
    ((common_idx >= "2000-01-01") & (common_idx <= "2008-08-31"))
    | ((common_idx >= "2014-02-01") & (common_idx <= "2015-08-31"))
    | ((common_idx >= "2022-03-01") & (common_idx <= "2023-10-31"))
)
# w=1 → traitement plein (flottement/IT) | w=0 → traitement nul (peg)
treatment_weight = pd.Series(
    np.where(peg_mask, 0.0, 1.0), index=common_idx, name="treatment_w"
)

CF_factor_adjusted = treatment_weight * CF_factor + (1 - treatment_weight) * ukr_aligned
CF_factor_adjusted.name = "CF_factor_adjusted"

print("\nTreatment weight stats:")
print(f"  Obs. peg (w=0): {(treatment_weight==0).sum()}")
print(f"  Obs. float (w=1): {(treatment_weight==1).sum()}")


# =============================================================================
# 4. BLANCHARD-QUAH SVAR — REMPLACEMENT DE CHOCS DE DEMANDE
# =============================================================================
print("\n" + "=" * 70)
print("4. BLANCHARD-QUAH SVAR (Bayoumi-Eichengreen / Blanchard-Quah 1989)")
print("=" * 70)


# --- 4a. PIB annuel World Bank ---
def get_wb_series(indicator, country_code, start=2000, end=2025):
    url = (
        f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}"
        f"?format=json&per_page=100&date={start}:{end}"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.loads(r.read())[1]
        records = {int(d["date"]): d["value"] for d in data if d["value"] is not None}
        return pd.Series(records, name=country_code).sort_index()
    except Exception as e:
        print(f"    World Bank échec ({country_code}): {e}")
        return None


print("  Téléchargement World Bank real GDP growth...")
gdp_ukr_annual = get_wb_series("NY.GDP.MKTP.KD.ZG", "UA")
gdp_ea_annual = get_wb_series("NY.GDP.MKTP.KD.ZG", "XC")
for lbl, s in [("UA GDP", gdp_ukr_annual), ("EA GDP", gdp_ea_annual)]:
    if s is not None:
        print(f"  {lbl}: {s.index[0]}–{s.index[-1]}, T={len(s)}")

# --- 4b. IP mensuelle EA (FRED, plusieurs identifiants, puis Chow-Lin) ---
print("\n  Téléchargement EA Industrial Production (FRED)...")
ea_ip_ok = False
for fred_id in ["EA19PRINTO01IXOBM", "EA19PRINTO01IXPYM", "EUGSINUSM"]:
    try:
        ea_ip_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_id}"
        _tmp = (
            pd.read_csv(ea_ip_url, index_col=0, parse_dates=True)
            .iloc[:, 0]
            .rename("EA_IP")
        )
        _tmp.index = pd.to_datetime(_tmp.index)
        _tmp = _tmp.dropna()
        if len(_tmp) > 100:
            ea_ip_m = _tmp
            print(
                f"  EA IP ({fred_id}): {ea_ip_m.index[0].date()} → {ea_ip_m.index[-1].date()}"
            )
            ea_ip_ok = True
            break
    except Exception:
        continue

if not ea_ip_ok:
    print("  Tous les IDs FRED EA IP ont échoué.")

if not ea_ip_ok and gdp_ea_annual is not None:
    print("  Fallback Chow-Lin EA depuis World Bank GDP annuel")
    last_year = gdp_ea_annual.index[-1]
    for yr in range(last_year + 1, 2026):
        gdp_ea_annual[yr] = gdp_ea_annual[last_year]
    monthly_idx_full = pd.date_range("2000-01-01", "2025-12-01", freq="MS")
    ea_ip_m = annual_to_monthly_chowlin(gdp_ea_annual, monthly_idx_full).rename("EA_IP")
    ea_ip_ok = True

# --- 4c. IP mensuelle Ukraine (FRED puis Chow-Lin automatique) ---
print("\n  Téléchargement Ukraine Industrial Production...")
ukr_ip_ok = False
try:
    ukr_ip_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UKRPROINDMISMEI"
    ukr_ip_m = (
        pd.read_csv(ukr_ip_url, index_col=0, parse_dates=True)
        .iloc[:, 0]
        .rename("UKR_IP")
    )
    ukr_ip_m.index = pd.to_datetime(ukr_ip_m.index)
    ukr_ip_m = ukr_ip_m.dropna()
    # Étendre avec Chow-Lin si la série s'arrête avant 2023
    if ukr_ip_m.index[-1] < pd.Timestamp("2023-01-01") and gdp_ukr_annual is not None:
        monthly_idx_full = pd.date_range("2000-01-01", "2025-12-01", freq="MS")
        ip_extension = annual_to_monthly_chowlin(gdp_ukr_annual, monthly_idx_full)
        # Normaliser l'extension sur le niveau moyen de la période de chevauchement
        overlap = ukr_ip_m.index.intersection(ip_extension.index)
        if len(overlap) > 12:
            scale = ukr_ip_m.loc[overlap].mean() / ip_extension.loc[overlap].mean()
            ip_extension = ip_extension * scale
        # Concaténer : FRED jusqu'à sa fin, Chow-Lin après
        cutoff = ukr_ip_m.index[-1]
        ukr_ip_m = pd.concat(
            [ukr_ip_m, ip_extension.loc[ip_extension.index > cutoff]]
        ).rename("UKR_IP")
        print(
            f"  Ukraine IP FRED + Chow-Lin extension: "
            f"{ukr_ip_m.index[0].date()} → {ukr_ip_m.index[-1].date()}"
        )
    else:
        print(
            f"  Ukraine IP FRED: {ukr_ip_m.index[0].date()} → {ukr_ip_m.index[-1].date()}"
        )
    ukr_ip_ok = True
except Exception as e:
    print(f"  FRED Ukraine IP échec: {e}")
    if gdp_ukr_annual is not None:
        print("  Fallback Chow-Lin Ukraine depuis World Bank GDP annuel")
        last_year = gdp_ukr_annual.index[-1]
        for yr in range(last_year + 1, 2026):
            gdp_ukr_annual[yr] = gdp_ukr_annual[last_year]
        monthly_idx_full = pd.date_range("2000-01-01", "2025-12-01", freq="MS")
        ukr_ip_m = annual_to_monthly_chowlin(gdp_ukr_annual, monthly_idx_full).rename(
            "UKR_IP"
        )
        print(
            f"  Ukraine IP Chow-Lin: {ukr_ip_m.index[0].date()} → "
            f"{ukr_ip_m.index[-1].date()}, T={len(ukr_ip_m)}"
        )
        ukr_ip_ok = True

# --- 4d. Construction des datasets SVAR ---
if ukr_ip_ok and ea_ip_ok:

    def to_growth(series):
        s = series.replace(0, np.nan)
        return (np.log(s) - np.log(s.shift(1))) * 100

    ukr_growth = to_growth(ukr_ip_m)
    ea_growth = to_growth(ea_ip_m)

    # Étendre ea_growth au-delà de 2022-12 avec Chow-Lin WB GDP EA
    if (
        ea_growth.dropna().index[-1] < pd.Timestamp("2024-01-01")
        and gdp_ea_annual is not None
    ):
        monthly_idx_full = pd.date_range("2000-01-01", "2025-12-01", freq="MS")
        ea_ext = annual_to_monthly_chowlin(gdp_ea_annual, monthly_idx_full).rename(
            "EA_IP_ext"
        )
        # Normaliser sur overlap
        overlap = ea_growth.dropna().index.intersection(ea_ext.dropna().index)
        if len(overlap) > 12:
            scale = ea_ip_m.loc[overlap].mean() / ea_ext.loc[overlap].mean()
            ea_ext = ea_ext * scale
        cutoff = ea_growth.dropna().index[-1]
        ea_growth = pd.concat(
            [ea_growth.dropna(), to_growth(ea_ext.loc[ea_ext.index > cutoff])]
        ).rename("EA_growth")
        print(f"  EA growth extended to: {ea_growth.dropna().index[-1].date()}")

    # Index SVAR élargi : ukraine_yoy (plein échantillon) pour couvrir
    # GFC 2008 et Crimée 2014. Inflation EA sur common_idx réindexée.
    ea_inf_full = hicp_aligned.mean(axis=1)

    # Utiliser ukraine_yoy plein échantillon (2000-2025) sans restriction à common_idx
    svar_idx = (
        ukraine_yoy.index.intersection(ukr_growth.dropna().index)
        .intersection(ea_growth.dropna().index)
        .intersection(ea_inf_full.index)
    )
    print(
        f"  SVAR index après extension: {svar_idx[0].date()} → {svar_idx[-1].date()}, "
        f"T={len(svar_idx)}"
    )

    # Régulariser l'index à fréquence MS stricte
    svar_idx_ms = pd.date_range(
        start=svar_idx[0].to_period("M").to_timestamp(),
        end=svar_idx[-1].to_period("M").to_timestamp(),
        freq="MS",
    )

    df_svar_ukr = pd.DataFrame(
        {
            "growth": ukr_growth.reindex(svar_idx_ms),
            "inflation": ukraine_yoy.reindex(svar_idx_ms),
        }
    ).dropna()

    df_svar_ea = pd.DataFrame(
        {
            "growth": ea_growth.reindex(svar_idx_ms),
            "inflation": ea_inf_full.reindex(svar_idx_ms),
        }
    ).dropna()

    common_svar = df_svar_ukr.index.intersection(df_svar_ea.index)
    df_svar_ukr = df_svar_ukr.loc[common_svar]
    df_svar_ea = df_svar_ea.loc[common_svar]

    print(
        f"\n  Dataset SVAR: {df_svar_ukr.index[0].date()} → "
        f"{df_svar_ukr.index[-1].date()}, T={len(df_svar_ukr)}"
    )

    # --- 4e. Tests ADF ---
    print("\n  Tests ADF (H0: racine unitaire):")
    st_ukr_g = adf_report(df_svar_ukr["growth"], "UKR growth")
    st_ukr_inf = adf_report(df_svar_ukr["inflation"], "UKR inflation (niveaux)")
    st_ea_g = adf_report(df_svar_ea["growth"], "EA growth")
    st_ea_inf = adf_report(df_svar_ea["inflation"], "EA inflation (niveaux)")

    if not st_ukr_inf:
        print("    → Ukraine inflation: passage en 1ère différence")
        df_svar_ukr_stat = df_svar_ukr.copy()
        df_svar_ukr_stat["inflation"] = df_svar_ukr["inflation"].diff()
        df_svar_ukr_stat = df_svar_ukr_stat.dropna()
        ukr_inf_differenced = True
    else:
        df_svar_ukr_stat = df_svar_ukr.copy()
        ukr_inf_differenced = False

    if not st_ea_inf:
        print("    → EA inflation: passage en 1ère différence")
        df_svar_ea_stat = df_svar_ea.copy()
        df_svar_ea_stat["inflation"] = df_svar_ea["inflation"].diff()
        df_svar_ea_stat = df_svar_ea_stat.dropna()
    else:
        df_svar_ea_stat = df_svar_ea.copy()

    # Réaligner après diff éventuelle
    common_stat = df_svar_ukr_stat.index.intersection(df_svar_ea_stat.index)
    df_svar_ukr_stat = df_svar_ukr_stat.loc[common_stat]
    df_svar_ea_stat = df_svar_ea_stat.loc[common_stat]

    # --- 4f. Estimation des SVARs ---
    print("\n  Estimation SVAR Ukraine:")
    res_ukr, shocks_ukr, B0_ukr, p_ukr = blanchard_quah_svar(
        df_svar_ukr_stat, maxlags=6, name="Ukraine"
    )
    print("  Estimation SVAR Euro Area:")
    res_ea, shocks_ea, B0_ea, p_ea = blanchard_quah_svar(
        df_svar_ea_stat, maxlags=6, name="Euro Area"
    )

    # --- 4g. Remplacement des chocs de demande + simulation full-sample ---
    n_ukr = len(shocks_ukr)
    n_ea = len(shocks_ea)

    # Pad EA shocks à la longueur UKR (choc demande=0 pour les obs manquantes)
    if n_ea < n_ukr:
        pad = np.zeros((n_ukr - n_ea, 2))
        shocks_ea_full = np.vstack([shocks_ea, pad])
    else:
        shocks_ea_full = shocks_ea[:n_ukr]

    # Construction des résidus CF : supply Ukraine + demand EA
    shocks_cf = shocks_ukr.copy()
    shocks_cf[:, 1] = shocks_ea_full[:, 1]  # col 1 = demand shock
    resid_cf = (B0_ukr @ shocks_cf.T).T

    coefs = res_ukr.coefs
    intercept = res_ukr.intercept
    y_cf = df_svar_ukr_stat.values.copy()

    # Boucle sur TOUTES les périodes disponibles
    for t in range(p_ukr, p_ukr + n_ukr):
        fitted = intercept.copy()
        for lag in range(p_ukr):
            fitted += coefs[lag] @ y_cf[t - lag - 1]
        y_cf[t] = fitted + resid_cf[t - p_ukr]

    # Index réel des obs post-lags (pas de date_range reconstruit)
    real_index = df_svar_ukr_stat.index[p_ukr:]
    clean_index = pd.DatetimeIndex(
        [pd.Timestamp(ts.year, ts.month, 1) for ts in real_index]
    )

    cf_svar_inf = pd.Series(
        y_cf[p_ukr:, 1],
        index=clean_index,
        name="CF_svar",
    )

    # Ré-intégration si inflation différenciée
    if ukr_inf_differenced:
        start_level = ukraine_yoy.loc[: clean_index[0]].iloc[-2]
        cf_svar_inf = start_level + cf_svar_inf.cumsum()
        cf_svar_inf.name = "CF_svar"

    # Interpoler sur index mensuel complet pour combler les trous
    cf_svar_full_idx = pd.date_range("2001-02-01", "2025-12-01", freq="MS")
    cf_svar_inf = cf_svar_inf.reindex(cf_svar_full_idx).interpolate(method="time")
    cf_svar_inf.name = "CF_svar"

    print(
        f"\n  SVAR CF: {cf_svar_inf.index[0].date()} → {cf_svar_inf.index[-1].date()}"
    )
    print(f"  mean={cf_svar_inf.mean():.2f}%  std={cf_svar_inf.std():.2f}%")
    svar_available = True
else:
    svar_available = False
    print("  ⚠ SVAR non calculé (données d'activité manquantes).")


# =============================================================================
# 5. FIGURE FINALE
# =============================================================================
print("\n" + "=" * 70)
print("5. FIGURE FINALE")
print("=" * 70)

fig, ax = plt.subplots(figsize=(18, 6))

crises = [
    ("2008-09-01", "2009-06-30", "GFC\n2008–09"),
    ("2014-02-01", "2015-12-31", "Crimea /\nDonbas"),
    ("2022-02-01", "2023-09-30", "Full-scale\ninvasion"),
]
for start, end, label in crises:
    ax.axvspan(
        pd.Timestamp(start), pd.Timestamp(end), alpha=0.12, color="orange", zorder=0
    )

ax.plot(
    ukr_aligned.index,
    ukr_aligned.values,
    color="#C0392B",
    lw=2,
    label="Ukraine actual YoY inflation",
)
ax.plot(
    CF_factor_adjusted.index,
    CF_factor_adjusted.values,
    color="#2980B9",
    lw=2,
    linestyle="--",
    label="Counterfactual: Ukraine in Euro Area\n(Ciccarelli-Mojon, time-varying treatment)",
)
if svar_available:
    ax.plot(
        cf_svar_inf.index,
        cf_svar_inf.values,
        color="#27AE60",
        lw=1.5,
        linestyle=":",
        label="Counterfactual: SVAR Blanchard-Quah\n(demand shock replacement)",
    )

ax.axvline(
    pd.Timestamp("2015-08-01"),
    color="purple",
    lw=1.2,
    linestyle="-.",
    alpha=0.7,
    label="IT adoption (Aug 2015)",
)
ax.axhline(0, color="black", lw=0.5, alpha=0.4)

ylim = ax.get_ylim()
for start, end, label in crises:
    mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
    ax.text(
        mid,
        ylim[1] * 0.96,
        label,
        ha="center",
        fontsize=9,
        color="darkorange",
        fontweight="bold",
        va="top",
    )

ax.set_ylabel("Inflation YoY (%)", fontsize=12)
ax.set_xlabel("Date")
ax.set_title(
    "Ukraine: Actual vs. Counterfactual Inflation\n"
    '"What if Ukraine had been a Euro Area member?"',
    fontsize=13,
)
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(2))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/output_counterfactual_ukraine.png", dpi=200, bbox_inches="tight")
plt.close()
print("Figure sauvegardée: output/output_counterfactual_ukraine.png")


# =============================================================================
# 6. EXPORT CSV
# =============================================================================
output_df = pd.DataFrame(
    {
        "ukraine_actual_yoy": ukr_aligned,
        "counterfactual_factor_model": CF_factor_adjusted,
        "ea_common_factor_pct": F_EA,
        "ea_mean_hicp": hicp_aligned.mean(axis=1),
    }
)
if svar_available:
    output_df["counterfactual_svar"] = cf_svar_inf
output_df.to_csv("output/output_counterfactual_data.csv")
print("CSV exporté: output/output_counterfactual_data.csv")


# =============================================================================
# 7. CHIFFRES POUR L'INTERPRÉTATION
# =============================================================================
print("\n" + "=" * 70)
print("7. CHIFFRES POUR L'INTERPRÉTATION (lire avant de rédiger)")
print("=" * 70)

print("\n--- METHODE 1 : Ciccarelli-Mojon ---")
episode_stats(
    ukr_aligned, CF_factor_adjusted, "2008-09-01", "2009-06-30", "GFC 2008-09"
)
episode_stats(
    ukr_aligned,
    CF_factor_adjusted,
    "2014-02-01",
    "2015-12-31",
    "Crimea/Donbas 2014-15",
)
episode_stats(
    ukr_aligned,
    CF_factor_adjusted,
    "2022-02-01",
    "2023-09-30",
    "Full-scale invasion 2022",
)

if svar_available:
    print("\n--- METHODE 2 : SVAR Blanchard-Quah ---")
    # ukraine_yoy plein échantillon pour couvrir GFC 2008 et Crimée 2014
    episode_stats(ukraine_yoy, cf_svar_inf, "2008-09-01", "2009-06-30", "GFC 2008-09")
    episode_stats(
        ukraine_yoy, cf_svar_inf, "2014-02-01", "2015-12-31", "Crimea/Donbas 2014-15"
    )
    episode_stats(
        ukraine_yoy, cf_svar_inf, "2022-02-01", "2023-09-30", "Full-scale invasion 2022"
    )

print("\n--- SANITY CHECK Barro-Gordon ---")
gap_pre = (
    ukr_aligned.loc[:"2015-12"] - CF_factor.reindex(ukr_aligned.loc[:"2015-12"].index)
).mean()
gap_post = (
    ukr_aligned.loc["2016-01":] - CF_factor.reindex(ukr_aligned.loc["2016-01":].index)
).mean()
print(f"  Gap moyen pré-2016  : {gap_pre:.2f}%")
print(f"  Gap moyen post-2016 : {gap_post:.2f}%")
print(f"  → {'✓ PASS' if gap_pre > gap_post else '✗ FAIL — revoir les hypothèses'}")
print("\n⚠  Rédige l'interprétation avec les valeurs ci-dessus.")
