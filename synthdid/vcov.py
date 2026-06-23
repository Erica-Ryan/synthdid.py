
import itertools, pandas as pd, numpy as np
from .sdid import sdid
from .utils import sum_normalize, varianza

def renormalize_weights(treated_weights, counts):
    """Bootstrap weight renormalization — eq:boot-w from paper."""
    w = np.array(treated_weights) * np.array(counts)
    total = w.sum()
    if total == 0:
        return np.full(len(treated_weights), 1/len(treated_weights))
    return w / total

def build_setup(data):
    """
    Creates R-like setup object:
    Y: (N units × T times) matrix
    unit_index maps dataframe units to matrix rows
    """
    units = np.unique(data["unit"])
    times = np.unique(data["time"])

    unit_to_row = {u: i for i, u in enumerate(units)}
    time_to_col = {t: j for j, t in enumerate(times)}

    Y = np.full((len(units), len(times)), np.nan)

    for _, r in data.iterrows():
        Y[unit_to_row[r["unit"]], time_to_col[r["time"]]] = r["outcome"]

    return {
        "Y": Y,
        "units": units,
        "times": times,
        "unit_to_row": unit_to_row,
        "time_to_col": time_to_col
    }

def bootstrap_se_weighted(data_ref, treated_weights, weights=None, cluster=None, n_reps=50):
    """
    Fixed-weight bootstrap SE (Algorithm 2, weighted).
    Mirrors R: build Y matrix once, resample row indices, compute ATT via matrix product.
    """
    if weights is None:
        return _bootstrap_se_reestimate_weighted(data_ref, treated_weights, n_reps=n_reps)

    omega = weights["omega"][0]
    lambda_pre = weights["lambda"][0]

    sorted_data = data_ref.sort_values(["treated", "unit", "time"])
    tyear_vals = sorted(sorted_data.loc[sorted_data["tyear"] > 0, "tyear"].unique())
    treatment_year = tyear_vals[0]

    tyear_data = sorted_data[sorted_data.tyear.isin([0, treatment_year])]
    Y = tyear_data.pivot_table(values="outcome", index="unit", columns="time", sort=False).to_numpy()
    N = Y.shape[0]
    N0 = len(omega)
    T0 = len(lambda_pre)
    T1 = Y.shape[1] - T0
    lmd = np.concatenate([-lambda_pre, np.full(T1, 1 / T1)])

    def theta(ind):
        control_ind = np.sort(ind[ind < N0])
        treated_ind = np.sort(ind[ind >= N0])
        if len(control_ind) == 0 or len(treated_ind) == 0:
            return np.nan

        omega_b = sum_normalize(omega[control_ind])
        tw_local = treated_weights[treated_ind - N0]
        tw_b = sum_normalize(tw_local)

        Y_boot = Y[np.concatenate([control_ind, treated_ind]), :]
        omg = np.concatenate([-omega_b, tw_b])
        return omg @ Y_boot @ lmd

    atts = []
    while len(atts) < n_reps:
        ind = np.random.choice(N, size=N, replace=True)
        val = theta(ind)
        if not np.isnan(val):
            atts.append(val)

    atts = np.array(atts)
    return np.sqrt((n_reps - 1) / n_reps) * np.std(atts, ddof=1)


def _bootstrap_se_reestimate_weighted(data_ref, treated_weights, n_reps=50):
    """Fallback: re-estimate weights each bootstrap rep (no fixed weights provided)."""
    treated_units_orig = list(np.unique(data_ref[data_ref.treated == 1].unit))
    tw_dict = {u: w for u, w in zip(treated_units_orig, treated_weights)}
    unique_units = np.unique(data_ref.unit)
    N = len(unique_units)

    def draw():
        sampled = np.random.choice(unique_units, replace=True, size=N)
        sample_df = pd.DataFrame({'unit': sampled, 'idx': np.arange(N)})
        merged = data_ref.merge(sample_df, on='unit')
        merged['unit'] = merged['unit'].astype(str) + '__' + merged['idx'].astype(str)
        merged = merged.drop(columns=['idx'])
        treated_ids = [u for u in sampled if u in set(treated_units_orig)]
        tw_b = sum_normalize(np.array([tw_dict[u] for u in treated_ids]))
        return merged, tw_b

    atts = []
    attempts = 0
    max_attempts = n_reps * 10
    while len(atts) < n_reps and attempts < max_attempts:
        attempts += 1
        try:
            sampled_df, tw_b = draw()
            if len(np.unique(sampled_df.treatment)) != 2:
                continue
            att = sdid(sampled_df, "unit", "time", "treatment", "outcome",
                      treated_weights=tw_b)["att"]
            atts.append(att)
        except Exception:
            continue

    atts = np.array(atts)
    if len(atts) == 0:
        return np.nan
    n = len(atts)
    return np.sqrt((n - 1) / n) * np.std(atts, ddof=1)


def bootstrap_se(data_ref, weights=None, n_reps=50):
    """
    Fixed-weight bootstrap SE (Algorithm 2, unweighted).
    Mirrors R: build Y matrix once, resample row indices, compute ATT via matrix product.
    """
    if weights is None:
        return _bootstrap_se_reestimate(data_ref, n_reps=n_reps)

    omega = weights["omega"][0]
    lambda_pre = weights["lambda"][0]

    sorted_data = data_ref.sort_values(["treated", "unit", "time"])
    tyear_vals = sorted(sorted_data.loc[sorted_data["tyear"] > 0, "tyear"].unique())
    treatment_year = tyear_vals[0]

    tyear_data = sorted_data[sorted_data.tyear.isin([0, treatment_year])]
    Y = tyear_data.pivot_table(values="outcome", index="unit", columns="time", sort=False).to_numpy()
    N = Y.shape[0]
    N0 = len(omega)
    N1 = N - N0
    T0 = len(lambda_pre)
    T1 = Y.shape[1] - T0
    lmd = np.concatenate([-lambda_pre, np.full(T1, 1 / T1)])

    def theta(ind):
        control_ind = np.sort(ind[ind < N0])
        treated_ind = np.sort(ind[ind >= N0])
        if len(control_ind) == 0 or len(treated_ind) == 0:
            return np.nan

        omega_b = sum_normalize(omega[control_ind])
        N1_b = len(treated_ind)
        tw_b = np.full(N1_b, 1 / N1_b)

        Y_boot = Y[np.concatenate([control_ind, treated_ind]), :]
        omg = np.concatenate([-omega_b, tw_b])
        return omg @ Y_boot @ lmd

    atts = []
    while len(atts) < n_reps:
        ind = np.random.choice(N, size=N, replace=True)
        val = theta(ind)
        if not np.isnan(val):
            atts.append(val)

    atts = np.array(atts)
    return np.sqrt((n_reps - 1) / n_reps) * np.std(atts, ddof=1)


def _bootstrap_se_reestimate(data_ref, n_reps=50):
    """Fallback: re-estimate weights each bootstrap rep (no fixed weights provided)."""
    unique_units = np.unique(data_ref.unit)
    N = len(unique_units)

    def draw():
        sampled = np.random.choice(unique_units, replace=True, size=N)
        sample_df = pd.DataFrame({'unit': sampled, 'idx': np.arange(N)})
        merged = data_ref.merge(sample_df, on='unit')
        merged['unit'] = merged['unit'].astype(str) + '__' + merged['idx'].astype(str)
        merged = merged.drop(columns=['idx'])
        return merged

    atts = []
    attempts = 0
    max_attempts = n_reps * 10
    while len(atts) < n_reps and attempts < max_attempts:
        attempts += 1
        try:
            sampled_df = draw()
            if len(np.unique(sampled_df.treatment)) != 2:
                continue
            att = sdid(sampled_df, "unit", "time", "treatment", "outcome")["att"]
            atts.append(att)
        except Exception:
            continue

    atts = np.array(atts)
    if len(atts) == 0:
        return np.nan
    n = len(atts)
    return np.sqrt((n - 1) / n) * np.std(atts, ddof=1)

def placebo_se_weighted(data_ref, treated_weights, n_reps=50, placebo_weights="uniform"):
    """
    Weighted placebo SE (Algorithm 4 adapted for treated weights)
    """
    tr_years = data_ref.query("time == tyear and tyear != 0").time
    N_tr = len(tr_years)

    df_co = data_ref.query("treated == 0")
    control_units = np.unique(df_co.unit)

    if len(control_units) <= N_tr:
        raise ValueError("Must have more control units than treated units")

    treated_weights = np.array(treated_weights)

    def draw_placebo_weights(sampled_units):
        if placebo_weights == "uniform":
            return np.full(N_tr, 1 / N_tr)

        elif placebo_weights == "permute":
            return np.random.permutation(treated_weights)[:N_tr]

        elif placebo_weights == "size_match":
            # weight by pre-treatment outcome means
            pre = data_ref[data_ref.time < data_ref.tyear]
            means = []
            for u in sampled_units:
                vals = pre[pre.unit == u].outcome
                means.append(np.abs(vals.mean()) if len(vals) > 0 else 0)
            w = np.array(means)
            if w.sum() == 0:
                return np.full(N_tr, 1 / N_tr)
            return w / w.sum()

        else:
            raise ValueError("Invalid placebo_weights option")

    def theta_pb():
        sampled_units = np.random.choice(control_units, size=N_tr, replace=False)

        placebo_years = pd.DataFrame({
            "unit": sampled_units,
            "tyear1": tr_years
        })

        aux_data = df_co.merge(placebo_years, on="unit", how='outer')
        aux_data = aux_data.assign(
            tyear=aux_data.tyear1.fillna(aux_data["tyear"])
        )

        aux_data = aux_data.assign(
            treatment=np.where(
                ((aux_data.tyear != 0) & (aux_data.time == aux_data.tyear)),
                1, 0
            )
        ).reset_index(drop=True)

        aux_data["treated"] = aux_data.groupby("unit")["treatment"].transform("max")

        tw = draw_placebo_weights(sampled_units)

        return sdid(
            aux_data,
            "unit",
            "time",
            "treatment",
            "outcome",
            treated_weights=tw
        )["att"]

    atts = []
    attempts = 0
    max_attempts = n_reps * 10

    while len(atts) < n_reps and attempts < max_attempts:
        attempts += 1
        try:
            atts.append(theta_pb())
        except Exception:
            continue

    if len(atts) < n_reps:
        print(f"Warning: only {len(atts)} successful placebo reps")

    atts = np.array(atts)
    return np.sqrt(np.var(atts, ddof=0))
    
def placebo_se(data_ref, n_reps=50):
    tr_years = data_ref.query("time == tyear and tyear != 0").time
    N_tr = len(tr_years)
    df_co = data_ref.query("treated == 0")
    units_df_co = np.unique(df_co.unit)
    N_co = len(units_df_co)
    N_aux = N_co - N_tr
    
    def theta_pb():
        plabeo_years = pd.DataFrame({
            "unit": np.random.choice(units_df_co, size=N_tr, replace=False),
            'tyear1': tr_years
        })
        aux_data = df_co.merge(plabeo_years, on="unit", how='outer').sort_values("tyear1")
        aux_data = aux_data.assign(
            tyear=aux_data.tyear1.fillna(aux_data["tyear"])
        )
        aux_data = aux_data.assign(
            treatment=np.where(((aux_data.tyear != 0) & (aux_data.time == aux_data.tyear)), 1, 0)
        ).reset_index(drop=True)
        aux_data["treated"] = aux_data.groupby("unit")["treatment"].transform("max")
        att = sdid(aux_data, "unit", "time", "treatment", "outcome")
        return att["att"]
    
    t = 0
    att_pb = np.array([])
    while t < n_reps:
        t += 1
        aux = theta_pb()
        att_pb = np.append(att_pb, aux)
    se_placebo = np.sqrt(1 / n_reps * np.sum((att_pb - np.sum(att_pb / n_reps)) ** 2))
    return se_placebo

def jackknife_iteration(data, time_breaks, weights, unit_index: int) -> np.ndarray:
    weighted_atts = np.array([])
    total_treated_unit_periods = data[data.treatment == 1].shape[0]

    for tyear_index, treatment_year in enumerate(time_breaks):
        tyear_data = data[data.tyear.isin([0, treatment_year])]
        N_treated = pd.unique(data[data.tyear == treatment_year].unit).shape[0]
        N_control = weights["omega"][tyear_index].shape[0]

        # --- CONTROL WEIGHTS (omega) ---
        omega_ctrl = weights["omega"][tyear_index].copy()
        if unit_index < N_control:
            omega_ctrl = np.delete(omega_ctrl, unit_index)
            omega_ctrl = sum_normalize(omega_ctrl)

        # -- TREATED WEIGHTS (uniform for the unweighted case) ---
        treated_wts = np.full(N_treated, 1/N_treated)

        # --- LAMBDA (fixed, never re-estimated) ---
        lambda_pre = weights["lambda"][tyear_index]
        T_post = pd.unique(tyear_data.time).shape[0] - lambda_pre.shape[0]
        lmd = np.concatenate([-lambda_pre, np.full(T_post, 1/T_post)])

        # --- FULL OMEGA VECTOR ---
        omg = np.concatenate([-omega_ctrl, treated_wts])

        # --- Y MATRIX ---
        tyear_data_sorted = tyear_data.sort_values(["treated", "unit", "time"])
        data_matrix = tyear_data_sorted.pivot_table(
            values ="outcome", index="unit", columns="time", sort=False
        ).to_numpy()
        
        att = omg @ data_matrix @ lmd
        att_weight = (N_treated * T_post) / total_treated_unit_periods
        weighted_atts = np.concatenate([weighted_atts, [att * att_weight]])
    
    return np.array([weighted_atts.sum()])
        
def jackknife_iteration_weighted(data, time_breaks, weights, unit_index: int, treated_weights=None) -> np.ndarray:
    weighted_atts = np.array([])
    total_treated_unit_periods = data[data.treatment == 1].shape[0]

    for tyear_index, treatment_year in enumerate(time_breaks):
        tyear_data = data[data.tyear.isin([0, treatment_year])]
        N_treated = pd.unique(data[data.tyear == treatment_year].unit).shape[0]
        N_control = weights["omega"][tyear_index].shape[0]

        # --- CONTROL WEIGHTS ---
        omega_ctrl = weights["omega"][tyear_index].copy()
        if unit_index < N_control:
            omega_ctrl = np.delete(omega_ctrl, unit_index)
            omega_ctrl = sum_normalize(omega_ctrl)

        # --- TREATED WEIGHTS ---
        if treated_weights is not None:
            tw = np.array(treated_weights).copy()

            # only adjust if a treated unit is removed
            if unit_index >= N_control:
                treated_index = unit_index - N_control
                tw = np.delete(tw, treated_index)

            # renormalize treated weights
            tw = sum_normalize(tw)
        else:
            # fallback to original implicit uniform structure
            N_treated_remaining = N_treated - (1 if unit_index >= N_control else 0)
            tw = np.full(N_treated_remaining, 1/N_treated_remaining)

        # --- LAMBDA (UNCHANGED) ---
        lambda_pre = weights["lambda"][tyear_index]
        T_post = pd.unique(tyear_data.time).shape[0] - lambda_pre.shape[0]
        lmd = np.concatenate([-lambda_pre, np.full(T_post, 1/T_post)])

        omg = np.concatenate([-omega_ctrl, tw])

        # --- Y MATRIX ---
        tyear_data_sorted = tyear_data.sort_values(["treated", "unit", "time"])
        data_matrix = tyear_data_sorted.pivot_table(
            values="outcome", index="unit", columns="time",sort=False
        ).to_numpy()

        # --- ATT COMPUTATION ---
        att = omg @ data_matrix @ lmd
        att_weight = (N_treated * T_post) / total_treated_unit_periods
        weighted_atts = np.concatenate([weighted_atts, [att * att_weight]])

    return np.array([weighted_atts.sum()])


def jackknife_se(data_ref: pd.DataFrame, time_breaks, att, weights):
    
    for tyear in time_breaks:
        if pd.unique(data_ref[data_ref.tyear == tyear].unit).shape[0] == 1:
            raise ValueError(f"Each adoption year must have more than one treated unit. Year {tyear} does not comply")
    
    control_units = np.sort(data_ref.loc[data_ref["treated"]==0, "unit"].unique())
    treated_units = np.sort(data_ref.loc[data_ref["treated"]==1, "unit"].unique())
    unique_units = np.concatenate([control_units, treated_units])
    jackknife_ates = np.array([])

    for unit_index, unit in enumerate(unique_units):
        iteration_ate = jackknife_iteration(
            data_ref[data_ref.unit != unit],
            time_breaks,
            weights,
            unit_index
        )
        jackknife_ates = np.concatenate([jackknife_ates, iteration_ate])
    
    total_units = unique_units.shape[0]
    var_jackknife = (total_units - 1) / total_units * ((jackknife_ates - att) ** 2).sum()
    se_jackknife = np.sqrt(var_jackknife)

    return se_jackknife


def jackknife_se_weighted(data_ref,time_breaks,att,weights,treated_weights=None):
        
    for tyear in time_breaks:
        if pd.unique(data_ref[data_ref.tyear == tyear].unit).shape[0] == 1:
            raise ValueError(f"Each adoption year must have more than one treated unit. Year {tyear} does not comply")
    
    control_units = np.sort(data_ref.loc[data_ref["treated"]==0, "unit"].unique())
    treated_units = np.sort(data_ref.loc[data_ref["treated"]==1, "unit"].unique())
    unique_units = np.concatenate([control_units, treated_units])
    jackknife_ates = np.array([])

    for unit_index, unit in enumerate(unique_units):
        iteration_ate = jackknife_iteration_weighted(
            data_ref[data_ref.unit != unit],
            time_breaks,
            weights,
            unit_index,
            treated_weights=treated_weights
        )
        jackknife_ates = np.concatenate([
            jackknife_ates,
            iteration_ate
        ])
    
    total_units = unique_units.shape[0]
    var_jackknife = (total_units - 1) / total_units * ((jackknife_ates - att) ** 2).sum()
    se_jackknife = np.sqrt(var_jackknife)

    return se_jackknife


def cluster_jackknife_se_weighted(data, time_breaks, att, weights, treated_weights=None, cluster_col=None):
    clusters = np.unique(data[cluster_col])
    K = len(clusters)

    if K <= 1:
        return np.nan

    theta_k = []

    for cl in clusters:
        # 1. Leave out entire cluster
        data_k = data[data[cluster_col] != cl]

        # skip degenerate cases
        if len(np.unique(data_k["unit"])) == 0:
            continue

        # 2. Recompute estimator fully
        try:
            res = sdid(
                data_k,
                unit="unit",
                time="time",
                treatment="treatment",
                outcome="outcome",
                treated_weights=treated_weights
            )
            theta_k.append(res["att"])
        except Exception:
            continue

    theta_k = np.asarray(theta_k)
    K_eff = len(theta_k)

    if K_eff < 2:
        return np.nan

    # 3. Cluster jackknife variance
    theta_bar = theta_k.mean()
    var = (K_eff - 1) / K_eff * np.sum((theta_k - theta_bar) ** 2)

    return np.sqrt(var)


class Variance:
    def vcov(self, method="placebo", n_reps=50):
        data_ref = self.data_ref
        time_break, weights = self.ttime, self.weights
        if method=="placebo":
            if self.treated_weights is not None:
                se = placebo_se_weighted(
                    data_ref,
                    self.treated_weights,
                    n_reps=n_reps
                )
            else:
                se = placebo_se(data_ref, n_reps=n_reps)
        elif method=="bootstrap":
            if self.treated_weights is not None:
                se = bootstrap_se_weighted(data_ref, self.treated_weights, weights=weights, n_reps=n_reps, cluster=self.cluster)
            else:
                se = bootstrap_se(data_ref, weights=weights, n_reps=n_reps)
        else:  # jackknife
            if self.treated_weights is not None:
                if self.cluster is not None:
                    se = cluster_jackknife_se_weighted(data_ref, time_break, self.att, weights, self.treated_weights, self.cluster)
                else:
                    se = jackknife_se_weighted(data_ref, time_break, self.att, weights, self.treated_weights)
            else:
                se = jackknife_se(data_ref, time_break, self.att, weights)
        self.se = se
        return self
