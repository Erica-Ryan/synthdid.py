import pandas as pd
import numpy as np
import os
from synthdid.sdid import sdid
from synthdid.vcov import jackknife_se, bootstrap_se_weighted

# Load data
panel = pd.read_csv("data/analysis_data.csv")

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

unit_state = panel[["unit", "state_fips"]].drop_duplicates()

cluster_map = dict(zip(unit_state["unit"], unit_state["state_fips"]))

data_ref["cluster"] = data_ref["unit"].map(cluster_map)

n_states = data_ref["cluster"].nunique()

result_sdid = sdid(
    data_ref,
    unit="unit",
    time="time",
    treatment="treatment",
    outcome="outcome"
)
tau_sdid = result_sdid["att"]

# result_sdid_w = sdid(
#     data_ref,
#     unit="unit",
#     time="time",
#     treatment="treatment",
#     outcome="outcome",
#     treated_weights=treated_weights
# )
# tau_sdid_w = result_sdid_w["att"]

se_sdid = jackknife_se(
    data_ref,
    time_breaks=sorted(data_ref.loc[data_ref["tyear"] > 0, "tyear"].unique()),
    att=tau_sdid,
    weights=result_sdid["weights"]
)

#jackknife_se_weighted(data_ref,time_breaks,att,weights,treated_weights=None):

# se_sdid_w = bootstrap_se_weighted(
#     data_ref,
#     treated_weights=treated_weights,
#     cluster="cluster",
#     n_reps=200
# )

# results = pd.DataFrame({
#     "Estimator": ["SDID"],
#     "Equally weighted": [tau_sdid],
#     "SE (eq.)": [se_sdid],
#     "Population weighted": [tau_sdid_w],
#     "SE (wt.)": [se_sdid_w]
# })

# print(results.round(2))

# assert len(treated_weights) == data_ref.loc[data_ref["treated"] == 1, "unit"].nunique()
