import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def sum_normalize(x):
    x = np.asarray(x, dtype=float)
    s = x.sum()
    if s != 0:
        return x / s
    return np.ones_like(x) / len(x)

def _reindex_treated_weights(df, original_treated_units, treated_weights):
    """
    Align treated weights to the treated units present in df.
    This mirrors synthdid's implicit subsetting behavior.
    """
    if treated_weights is None:
        return None

    df_treated_units = df[df["treated"] == 1]["unit"].unique()

    unit_to_weight = dict(zip(original_treated_units, treated_weights))

    w = np.array([unit_to_weight[u] for u in df_treated_units])

    return sum_normalize(w)

def _cluster_groups(df, cluster_col):
    """
    Returns dict: cluster_id -> dataframe slice
    """
    return {c: df[df[cluster_col] == c] for c in df[cluster_col].unique()}


def _sdid(df, sdid_fn, treated_weights=None, original_treated_units=None):
    return sdid_fn(
        df,
        unit="unit",
        time="time",
        treatment="treatment",
        outcome="outcome",
        treated_weights=_reindex_treated_weights(
            df,
            original_treated_units,
            treated_weights
        )
    )["att"]

def _precompute_panel(df, unit_col="unit", time_col="time", y_col="outcome"):
    units = df[unit_col].unique()
    times = df[time_col].unique()

    u_index = {u: i for i, u in enumerate(units)}
    t_index = {t: j for j, t in enumerate(times)}

    Y = np.full((len(units), len(times)), np.nan)

    for r in df.itertuples(index=False):
        Y[u_index[getattr(r, unit_col)], t_index[getattr(r, time_col)]] = getattr(r, y_col)

    return Y, units, times, u_index



# ---------------------------------------------------------------------
# Bootstrap (Algorithm 2)
# ---------------------------------------------------------------------

def bootstrap_se(df, sdid_fn, B=200, treated_weights=None, seed=None):

    if seed is not None:
        np.random.seed(seed)

    units = df["unit"].unique()
    original_treated_units = df[df["treated"] == 1]["unit"].unique()

    estimates = []

    for _ in range(B):
        sampled_units = np.random.choice(units, size=len(units), replace=True)

        df_b = pd.concat(
            [df[df["unit"] == u].assign(__u=i)
             for i, u in enumerate(sampled_units)],
            ignore_index=True
        ).rename(columns={"__u": "unit"})

        if df_b["treatment"].nunique() < 2:
            continue

        estimates.append(
            _sdid(df_b, sdid_fn, treated_weights, original_treated_units)
        )

    estimates = np.asarray(estimates)

    if len(estimates) < 2:
        return np.nan

    return np.sqrt((len(estimates) - 1) / len(estimates)) * np.std(estimates, ddof=0)


def cluster_bootstrap_se(df, sdid_fn, cluster_col="cluster", B=200, seed=None):
    if seed is not None:
        np.random.seed(seed)

    clusters = df[cluster_col].unique()
    G = len(clusters)

    cluster_map = _cluster_groups(df, cluster_col)

    estimates = []

    for _ in range(B):
        sampled_clusters = np.random.choice(clusters, size=G, replace=True)

        df_b = pd.concat(
            [cluster_map[c].copy().assign(__g=i)
             for i, c in enumerate(sampled_clusters)],
            ignore_index=True
        )

        # reassign cluster id for SDID internal consistency
        df_b = df_b.rename(columns={"__g": cluster_col})

        if df_b["treatment"].nunique() < 2:
            continue

        estimates.append(_sdid(df_b, sdid_fn))

    estimates = np.asarray(estimates)

    if len(estimates) < 2:
        return np.nan

    return np.sqrt((len(estimates) - 1) / len(estimates)) * np.std(estimates, ddof=0)

# ---------------------------------------------------------------------
# Jackknife (Algorithm 3)
# ---------------------------------------------------------------------

def jackknife_se(df, sdid_fn, treated_weights=None):

    units = df["unit"].unique()
    original_treated_units = df[df["treated"] == 1]["unit"].unique()

    estimates = []

    for u in units:
        df_j = df[df["unit"] != u]

        if df_j["treatment"].nunique() < 2:
            continue

        estimates.append(
            _sdid(df_j, sdid_fn, treated_weights, original_treated_units)
        )

    estimates = np.asarray(estimates)

    if len(estimates) < 2:
        return np.nan

    theta_bar = estimates.mean()
    n = len(estimates)

    return np.sqrt(((n - 1) / n) * np.sum((estimates - theta_bar) ** 2))

import numpy as np

def jackknife_se_fast(df, sdid_fn, treated_weights=None):
    units = df["unit"].unique()
    treated_units = df[df["treated"] == 1]["unit"].unique()

    tpos = {u: i for i, u in enumerate(treated_units)}

    estimates = []

    for u in units:
        df_j = df[df["unit"] != u]

        if df_j["treatment"].nunique() < 2:
            continue

        tw_j = None
        if treated_weights is not None:
            kept = df_j[df_j["treated"] == 1]["unit"].unique()
            idx = [tpos[x] for x in kept if x in tpos]

            if len(idx) > 0:
                tw_j = np.array(treated_weights)[idx]
                tw_j = tw_j / tw_j.sum()

        estimates.append(
            sdid_fn(
                df_j,
                "unit",
                "time",
                "treatment",
                "outcome",
                treated_weights=tw_j
            )["att"]
        )

    estimates = np.asarray(estimates)

    if len(estimates) < 2:
        return np.nan

    n = len(estimates)
    mean = estimates.mean()

    return np.sqrt(((n - 1) / n) * np.sum((estimates - mean) ** 2))


# Ultrafast requires sdid_matrix_fn(Y, treated_mask), have to modify sdid if you want that.
# def build_panel(df):
#     units = df["unit"].unique()
#     times = df["time"].unique()

#     u = {x:i for i,x in enumerate(units)}
#     t = {x:i for i,x in enumerate(times)}

#     Y = np.full((len(units), len(times)), np.nan)

#     for r in df.itertuples(index=False):
#         Y[u[r.unit], t[r.time]] = r.outcome

#     treated_mask = np.array([
#         df[df["unit"] == x]["treated"].iloc[0]
#         for x in units
#     ])

#     return Y, units, treated_mask

# def jackknife_se_ultrafast(df, sdid_matrix_fn):
#     Y, units, treated_mask = build_panel(df)

#     estimates = []

#     for i in range(len(units)):
#         Y_j = np.delete(Y, i, axis=0)
#         treated_j = np.delete(treated_mask, i)

#         if treated_j.sum() == 0 or treated_j.sum() == len(treated_j):
#             continue

#         estimates.append(
#             sdid_matrix_fn(Y_j, treated_j)
#         )

#     estimates = np.asarray(estimates)

#     if len(estimates) < 2:
#         return np.nan

#     n = len(estimates)
#     return np.sqrt(((n - 1) / n) * np.var(estimates, ddof=0))

def cluster_jackknife_se(df, sdid_fn, cluster_col="cluster"):
    clusters = df[cluster_col].unique()

    estimates = []

    for c in clusters:
        df_j = df[df[cluster_col] != c]

        if df_j["treatment"].nunique() < 2:
            continue

        estimates.append(_sdid(df_j, sdid_fn))

    estimates = np.asarray(estimates)

    if len(estimates) < 2:
        return np.nan

    theta_bar = estimates.mean()
    G = len(estimates)

    return np.sqrt(((G - 1) / G) * np.sum((estimates - theta_bar) ** 2))


# ---------------------------------------------------------------------
# Placebo (Algorithm 4)
# ---------------------------------------------------------------------

def placebo_se(df, sdid_fn, B=200, treated_weights=None, seed=None):

    if seed is not None:
        np.random.seed(seed)

    control = df[df["treated"] == 0].copy()

    treated_units = df[df["treated"] == 1]["unit"].unique()
    treated_years = df.loc[df["treated"] == 1, "tyear"].unique()

    N1 = len(treated_years)
    control_units = control["unit"].unique()

    estimates = []

    for _ in range(B):
        sampled_units = np.random.choice(control_units, size=N1, replace=False)

        map_df = pd.DataFrame({
            "unit": sampled_units,
            "tyear": treated_years
        })

        df_pb = control.merge(map_df, on="unit", how="left")
        df_pb["tyear"] = df_pb["tyear"].fillna(0)

        df_pb["treatment"] = (
            (df_pb["tyear"] != 0) &
            (df_pb["time"] == df_pb["tyear"])
        ).astype(int)

        df_pb["treated"] = df_pb.groupby("unit")["treatment"].transform("max")

        estimates.append(
            _sdid(df_pb, sdid_fn, treated_weights, treated_units)
        )

    estimates = np.asarray(estimates)

    if len(estimates) < 2:
        return np.nan

    return np.sqrt((len(estimates) - 1) / len(estimates)) * np.std(estimates, ddof=0)

# ---------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------

class Variance:
    def __init__(self, data_ref, sdid_fn, treated_weights=None, cluster_col=None):
        self.data_ref = data_ref
        self.sdid_fn = sdid_fn
        self.treated_weights = treated_weights
        self.cluster_col = cluster_col
        self.se = None

    def vcov(self, method="bootstrap", B=200, seed=None):

        df = self.data_ref

        # -----------------------------
        # CLUSTERED PATH (R equivalent)
        # -----------------------------
        if self.cluster_col is not None:

            if method == "bootstrap":
                se = cluster_bootstrap_se(
                    df, self.sdid_fn,
                    cluster_col=self.cluster_col,
                    B=B,
                    seed=seed
                )

            elif method == "jackknife":
                se = cluster_jackknife_se(
                    df, self.sdid_fn,
                    cluster_col=self.cluster_col
                )

            elif method == "placebo":
                raise NotImplementedError(
                    "R does not implement cluster placebo SE"
                )

            else:
                raise ValueError(method)

        # -----------------------------
        # UNIT LEVEL PATH
        # -----------------------------
        else:
            if method == "bootstrap":
                se = bootstrap_se(df, self.sdid_fn, B=B, seed=seed)

            elif method == "jackknife":
                se = jackknife_se(df, self.sdid_fn)

            elif method == "placebo":
                se = placebo_se(df, self.sdid_fn, B=B, seed=seed)

            else:
                raise ValueError(method)

        self.se = se
        return self