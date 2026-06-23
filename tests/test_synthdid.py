# =============================================================================
# test_synthdid.py — weighted SDID extension tests using the ACA application data
# Run from synthdid_weights/tests/ with: pytest test_synthdid.py -v
# or: python test_synthdid.py
# =============================================================================

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from synthdid.sdid import sdid, SDID
from synthdid.synthdid import Synthdid
from synthdid.vcov import jackknife_se, bootstrap_se, bootstrap_se_weighted
from synthdid.utils import panel_matrices

# =============================================================================
# Load and prepare data (mirrors analyze_application.R)
# =============================================================================
# Load data
panel = pd.read_csv("../data/analysis_data.csv")

panel = panel.rename(columns={
    "fips": "unit",
    "year": "time",
    "crude_rate": "y",
    "expansion": "treated_unit",
    "population": "pop"
})

panel = panel[["unit", "time", "y", "treated_unit", "pop", "state_fips"]]

panel["post"] = (panel["time"] >= 2014).astype(int)

# Summary stats (sanity check)
n_counties = panel["unit"].nunique()
n_treated = panel.loc[panel["treated_unit"] == 1, "unit"].nunique()
n_control = panel.loc[panel["treated_unit"] == 0, "unit"].nunique()
year_range = (panel["time"].min(), panel["time"].max())
n_years = panel["time"].nunique()

T0 = (np.sort(panel["time"].unique()) < 2014).sum()
T1 = n_years - T0

panel_sdid = (
    panel
    .sort_values(["unit", "time"])
    .assign(
        W=lambda df: ((df["treated_unit"] == 1) & (df["time"] >= 2014)).astype(int)
    )
)

data_ref = panel_sdid.rename(columns={
    "y": "outcome",
    "W": "treatment"
})

# Define treated indicator (unit-level)
data_ref["treated"] = data_ref.groupby("unit")["treatment"].transform("max")

# Define treatment year (tyear)
def get_tyear(df):
    treated_times = df.loc[df["treatment"] == 1, "time"]
    return treated_times.min() if len(treated_times) > 0 else 0

data_ref["tyear"] = data_ref.groupby("unit").apply(get_tyear).reindex(data_ref["unit"]).values

pop_2013 = (
    panel
    .query("treated_unit == 1 and time == 2013")
    [["unit", "pop"]]
)

treated_units = sorted(data_ref.loc[data_ref["treated"] == 1, "unit"].unique())

pop_map = dict(zip(pop_2013["unit"], pop_2013["pop"]))

treated_weights = np.array([pop_map[u] for u in treated_units])
treated_weights = treated_weights / treated_weights.sum()
uniform_weights = np.ones(len(treated_units)) / len(treated_units)
N1 = len(treated_units)

unit_state = panel[["unit", "state_fips"]].drop_duplicates()
cluster_map = dict(zip(unit_state["unit"], unit_state["state_fips"]))
data_ref["cluster"] = data_ref["unit"].map(cluster_map)

print(f"\nData loaded: {data_ref['unit'].nunique()} units, "
      f"{data_ref['time'].nunique()} periods, {N1} treated\n")

# =============================================================================
# 1. Unweighted baseline ATT is finite
# =============================================================================
result_sdid = sdid(data_ref, "unit", "time", "treatment", "outcome")
tau_sdid = result_sdid["att"]
assert np.isfinite(tau_sdid), "FAIL 1"
print(f"PASS  1: unweighted ATT finite: {tau_sdid:.3f}")

# =============================================================================
# 2. Weighted estimate is finite
# =============================================================================
result_w = sdid(data_ref, "unit", "time", "treatment", "outcome",
                treated_weights=treated_weights)
tau_w = result_w["att"]
assert np.isfinite(tau_w), "FAIL 2"
print(f"PASS  2: weighted ATT finite: {tau_w:.3f}")

# =============================================================================
# 3. Uniform treated weights recover unweighted estimate
# =============================================================================
result_unif = sdid(data_ref, "unit", "time", "treatment", "outcome",
                   treated_weights=uniform_weights)
tau_unif = result_unif["att"]
assert abs(tau_unif - tau_sdid) < 1e-4, f"FAIL 3: {tau_unif} vs {tau_sdid}"
print(f"PASS  3: uniform weighted == unweighted ({tau_unif:.3f} vs {tau_sdid:.3f})")

# =============================================================================
# 4. Population weights produce a different ATT than uniform weights
# =============================================================================
assert abs(tau_w - tau_sdid) > 1.0, f"FAIL 4: difference too small {abs(tau_w - tau_sdid):.3f}"
print(f"PASS  4: population weighted differs from unweighted by >1 "
      f"({tau_w:.3f} vs {tau_sdid:.3f}, diff={tau_w - tau_sdid:.3f})")

# =============================================================================
# 5. Omega and lambda sum to one
# =============================================================================
for i, (om, lm) in enumerate(zip(result_w["weights"]["omega"],
                                   result_w["weights"]["lambda"])):
    assert abs(np.sum(om) - 1.0) < 1e-6, f"FAIL 5: omega[{i}] sums to {np.sum(om)}"
    assert abs(np.sum(lm) - 1.0) < 1e-6, f"FAIL 5: lambda[{i}] sums to {np.sum(lm)}"
print("PASS  5: omega and lambda sum to one")

# =============================================================================
# 6. Synthdid class matches functional sdid()
# =============================================================================
m_unif = Synthdid(data_ref, "unit", "time", "treatment", "outcome").fit()
m_w    = Synthdid(data_ref, "unit", "time", "treatment", "outcome",
                  treated_weights=treated_weights).fit()
assert np.isfinite(m_unif.att), "FAIL 6a"
assert np.isfinite(m_w.att), "FAIL 6b"
assert abs(m_unif.att - tau_sdid) < 1e-4, f"FAIL 6c: {m_unif.att} vs {tau_sdid}"
assert abs(m_w.att - tau_w) < 1e-4, f"FAIL 6d: {m_w.att} vs {tau_w}"
print(f"PASS  6: Synthdid class matches functional sdid() "
      f"(unweighted={m_unif.att:.3f}, weighted={m_w.att:.3f})")

# =============================================================================
# 7. summary() runs without error
# =============================================================================
m_w.se = None
m_w.summary()
assert hasattr(m_w, 'summary2'), "FAIL 7"
print(f"PASS  7: summary() runs\n{m_w.summary2.to_string(index=False)}")

# =============================================================================
# 8. Jackknife SE is positive and finite
# =============================================================================
time_breaks = sorted(data_ref.loc[data_ref["tyear"] > 0, "tyear"].unique())
se_jk = jackknife_se(data_ref, 
                     time_breaks=time_breaks,
                     att = tau_sdid,
                     weights=result_sdid["weights"]
                     )

assert np.isfinite(se_jk) and se_jk > 0, f"FAIL 8: {se_jk}"
print(f"PASS  8: jackknife SE positive: {se_jk:.3f}")

# =============================================================================
# 9. Weighted jackknife SE is positive and finite
# =============================================================================
m_w.vcov(method="jackknife")
assert np.isfinite(m_w.se) and m_w.se > 0, f"FAIL 9: {m_w.se}"
print(f"PASS  9: weighted jackknife SE positive: {m_w.se:.3f}")

# =============================================================================
# 10. Weighted bootstrap SE is positive and finite
# =============================================================================
m_w.vcov(method="bootstrap", n_reps=200)
assert np.isfinite(m_w.se) and m_w.se > 0, f"FAIL 10: {m_w.se}"
print(f"PASS 10: weighted bootstrap SE positive: {m_w.se:.3f}")

# =============================================================================
# 11. Weighted placebo SE is positive and finite
# =============================================================================
m_w.vcov(method="placebo", n_reps=200)
assert np.isfinite(m_w.se) and m_w.se > 0, f"FAIL 11: {m_w.se}"
print(f"PASS 11: weighted placebo SE positive: {m_w.se:.3f}")

# =============================================================================
# 12. DID weighted estimate is finite
# =============================================================================
result_did_w = sdid(data_ref, "unit", "time", "treatment", "outcome",
                    treated_weights=treated_weights, did=True)
tau_did_w = result_did_w["att"]
assert np.isfinite(tau_did_w), "FAIL 12"
print(f"PASS 12: DID weighted ATT finite: {tau_did_w:.3f}")

# =============================================================================
# 13. SC weighted estimate is finite
# =============================================================================
result_sc_w = sdid(data_ref, "unit", "time", "treatment", "outcome",
                   treated_weights=treated_weights, synth=True)
tau_sc_w = result_sc_w["att"]
assert np.isfinite(tau_sc_w), "FAIL 13"
print(f"PASS 13: SC weighted ATT finite: {tau_sc_w:.3f}")

# =============================================================================
# 14. Effective N1 is less than raw N1
# =============================================================================
n1_eff = 1 / np.sum(treated_weights ** 2)
assert n1_eff < N1, f"FAIL 14: n1_eff {n1_eff:.0f} >= N1 {N1}"
print(f"PASS 14: N1_eff {n1_eff:.0f} < N1 {N1}")

# =============================================================================
# 15. R replication check: weighted ATT in expected range (~-17.5)
# =============================================================================
assert -25 < tau_w < -5, f"FAIL 15: weighted ATT {tau_w:.3f} outside expected range"
print(f"PASS 15: weighted ATT {tau_w:.3f} in expected range (-25, -5)")

print("\n=== All 15 tests passed ===\n")