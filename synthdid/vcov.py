
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

def bootstrap_se_weighted(data_ref, treated_weights, cluster=None, n_reps=50):
    treated_units_orig = list(np.unique(data_ref[data_ref.treated == 1].unit))
    tw_dict = {u: w for u, w in zip(treated_units_orig, treated_weights)}

    def get_tw_b(sampled_df):
        sampled_treated = pd.unique(sampled_df[sampled_df.treated == 1].unit)
        orig_ids = [u.rsplit("__", 1)[0] for u in sampled_treated]
        orig_ids = [type(treated_units_orig[0])(i) for i in orig_ids]
        return sum_normalize(np.array([tw_dict[orig_id] for orig_id in orig_ids]))

    if cluster is not None:
        unique_clusters = np.unique(data_ref[cluster])
        def draw():
            sampled_clusters = np.random.choice(unique_clusters, replace=True, size=len(unique_clusters))
            pieces = []
            for idx, c in enumerate(sampled_clusters):
                chunk = data_ref[data_ref[cluster] == c].copy()
                chunk = chunk.assign(unit=chunk["unit"].astype(str) + f"__{idx}")
                pieces.append(chunk)
            return pd.concat(pieces, ignore_index=True)
    else:
        unique_units = np.unique(data_ref.unit)
        def draw():
            sampled = np.random.choice(unique_units, replace=True, size=len(unique_units))
            pieces = []
            for idx, u in enumerate(sampled):
                chunk = data_ref[data_ref.unit == u].copy()
                chunk = chunk.assign(unit=str(u) + f"__{idx}")
                pieces.append(chunk)
            return pd.concat(pieces, ignore_index=True)

    atts = []
    attempts = 0
    max_attempts = n_reps * 10
    failures = 0

    while len(atts) < n_reps and attempts < max_attempts:
        attempts += 1
        try:
            sampled_df = draw()
            if len(np.unique(sampled_df.treatment)) != 2:
                continue
            tw_b = get_tw_b(sampled_df)
            att = sdid(sampled_df, "unit", "time", "treatment", "outcome", treated_weights=tw_b)["att"]
            atts.append(att)
        except Exception:
            failures += 1

    if failures > 0:
        print(f"Warning: bootstrap_se_weighted: {failures} of {attempts} attempts failed")
    if len(atts) < n_reps:
        print(f"Warning: bootstrap_se_weighted: only {len(atts)} of {n_reps} replicates completed")

    atts = np.array(atts)
    if len(atts) == 0:
        return np.nan
    return np.sqrt(np.var(atts, ddof=0))

def bootstrap_se(data_ref, n_reps = 50):
    uniqueID = np.unique(data_ref.unit)
    N = len(uniqueID)
    def theta_bt():
        sample_id = np.random.choice(uniqueID, replace=True, size=N)
        def sample_concat(_id):
            sample_id_n = sample_id[_id]
            data_c = data_ref[data_ref.unit == sample_id_n].copy()
            data_c = data_c.assign(unit1=str(sample_id_n) + "_" + str(_id))
            return data_c
        sampled_df = pd.concat([sample_concat(i) for i in range(N)], ignore_index=True)
        if len(np.unique(sampled_df.treatment)) != 2:
            return theta_bt()
        att_aux = sdid(sampled_df, "unit1", "time", "treatment", "outcome")["att"]
        return att_aux
    t = 0
    att_bt = np.array([])
    while t < n_reps:
        t+= 1
        aux = theta_bt()
        att_bt = np.append(att_bt, aux)
    se_bootstrap = np.sqrt(1 / n_reps * np.sum((att_bt - np.sum(att_bt / n_reps)) ** 2))
    return se_bootstrap

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
        tyear_omegas =  - weights["omega"][tyear_index]
        N_control = tyear_omegas.shape[0]
        tyear_omegas = np.concatenate([tyear_omegas, np.array([1/N_treated for _ in range(N_treated)])])
        if unit_index < N_control:
            tyear_omegas = np.delete(tyear_omegas, unit_index)
        tyear_lambdas = - weights["lambda"][tyear_index]
        T_post = pd.unique(tyear_data.time).shape[0] - tyear_lambdas.shape[0]
        tyear_treated_unit_periods = N_treated * T_post
        tyear_lambdas = np.concatenate([tyear_lambdas, np.array([1 / T_post for _ in range(T_post)])])
        data_matrix = tyear_data.pivot_table(values = "outcome", index = "unit", columns = "time", sort = False).to_numpy() #type: ignore
        att = tyear_omegas @ data_matrix @ tyear_lambdas.T
        att_weight = tyear_treated_unit_periods / total_treated_unit_periods
        weighted_atts = np.concatenate([weighted_atts, [att * att_weight]])
    
    jk_iteration_att = weighted_atts.sum()
    return np.array([jk_iteration_att])
        
def jackknife_iteration_weighted(data, time_breaks, weights, unit_index: int, treated_weights=None) -> np.ndarray:
    weighted_atts = np.array([])
    total_treated_unit_periods = data[data.treatment == 1].shape[0]

    for tyear_index, treatment_year in enumerate(time_breaks):
        tyear_data = data[data.tyear.isin([0, treatment_year])]
        N_treated = pd.unique(data[data.tyear == treatment_year].unit).shape[0]

        # --- CONTROL WEIGHTS (UNCHANGED LOGIC) ---
        tyear_omegas = - weights["omega"][tyear_index]
        N_control = tyear_omegas.shape[0]

        tyear_omegas = np.concatenate([
            tyear_omegas,
            np.array([1 / N_treated for _ in range(N_treated)])
        ])

        if unit_index < N_control:
            tyear_omegas = np.delete(tyear_omegas, unit_index)

        # --- TREATED WEIGHTS (NEW ADDITION) ---
        if treated_weights is not None:
            tyear_treated_weights = np.array(treated_weights[tyear_index])

            # only adjust if a treated unit is removed
            if unit_index >= N_control:
                treated_index = unit_index - N_control
                tyear_treated_weights = np.delete(tyear_treated_weights, treated_index)

            # renormalize treated weights
            tyear_treated_weights = tyear_treated_weights / tyear_treated_weights.sum()
        else:
            # fallback to original implicit uniform structure
            tyear_treated_weights = np.array([1 / N_treated for _ in range(N_treated)])

        # --- LAMBDA (UNCHANGED) ---
        tyear_lambdas = - weights["lambda"][tyear_index]
        T_post = pd.unique(tyear_data.time).shape[0] - tyear_lambdas.shape[0]

        tyear_lambdas = np.concatenate([
            tyear_lambdas,
            np.array([1 / T_post for _ in range(T_post)])
        ])

        data_matrix = tyear_data.pivot_table(
            values="outcome",
            index="unit",
            columns="time",
            sort=False
        ).to_numpy()

        # --- ATT COMPUTATION (UPDATED STRUCTURE) ---
        # Rebuild omega vector using actual treated weights instead of uniform 1/N_treated
        N_control_remaining = len(tyear_omegas) - N_treated
        omg = np.concatenate([
            tyear_omegas[:N_control_remaining],
            tyear_treated_weights
        ])

        att = omg @ data_matrix @ tyear_lambdas.T

        att_weight = (N_treated * T_post) / total_treated_unit_periods
        weighted_atts = np.concatenate([weighted_atts, [att * att_weight]])

    jk_iteration_att = weighted_atts.sum()
    return np.array([jk_iteration_att])


def jackknife_se(data_ref: pd.DataFrame, time_breaks, att, weights):
    
    for tyear in time_breaks:
        if pd.unique(data_ref[data_ref.tyear == tyear].unit).shape[0] == 1:
            raise ValueError(f"Each adoption year must have more than one treated unit. Year {tyear} does not comply")
    
    unique_units = pd.unique(data_ref.unit.unique())
    jackknife_ates = np.array([])

    for unit_index, unit in enumerate(unique_units):
        iteration_ate = jackknife_iteration(
            data_ref[data_ref.unit != unit],
            time_breaks,
            weights,
            unit_index
        )
        jackknife_ates = np.concatenate([
            jackknife_ates,
            iteration_ate
        ])
    
    total_units = unique_units.shape[0]
    var_jackknife = (total_units - 1) / total_units * ((jackknife_ates - att) ** 2).sum()
    se_jackknife = np.sqrt(var_jackknife)

    return se_jackknife


def jackknife_se_weighted(data_ref,time_breaks,att,weights,treated_weights=None):
        
    for tyear in time_breaks:
        if pd.unique(data_ref[data_ref.tyear == tyear].unit).shape[0] == 1:
            raise ValueError(f"Each adoption year must have more than one treated unit. Year {tyear} does not comply")
    
    unique_units = pd.unique(data_ref.unit.unique())
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
                se = bootstrap_se_weighted(data_ref, self.treated_weights, n_reps=n_reps, cluster=self.cluster)
            else:
                se = bootstrap_se(data_ref, n_reps=n_reps)
        else:  # jackknife
            time_break, weights = self.ttime, self.weights
            if self.treated_weights is not None:
                if self.cluster is not None:
                    se = cluster_jackknife_se_weighted(data_ref, time_break, self.att, weights, self.treated_weights, self.cluster)
                else:
                    se = jackknife_se_weighted(data_ref, time_break, self.att, weights, self.treated_weights)
            else:
                se = jackknife_se(data_ref, time_break, self.att, weights)
        self.se = se
        return self

