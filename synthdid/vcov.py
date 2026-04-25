
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

def sum_normalize(x):
    x = np.array(x)
    if x.sum() != 0:
        return x / x.sum()
    return np.full(len(x), 1/len(x))

def bootstrap_se_weighted(data_ref, treated_weights, cluster=None, n_reps=50):
    if cluster is not None:
        unique_clusters = np.unique(data_ref[cluster])
        def theta_bt():
            sampled_clusters = np.random.choice(unique_clusters, replace=True, size=len(unique_clusters))
            pieces = []
            for idx, c in enumerate(sampled_clusters):
                chunk = data_ref[data_ref[cluster] == c].copy()
                chunk = chunk.assign(unit=chunk["unit"].astype(str) + f"__{idx}")
                pieces.append(chunk)
            sampled_df = pd.concat(pieces, ignore_index=True)
            if len(np.unique(sampled_df.treatment)) != 2:
                return theta_bt()
            # compute per-treated-unit counts for renormalization
            treated_units = np.unique(data_ref[data_ref.treated == 1].unit)
            counts = np.array([
                (sampled_df.unit.str.split("__").str[0] == str(u)).sum() / 
                len(data_ref[data_ref.unit == u])
                for u in treated_units
            ])
            tw_b = renormalize_weights(treated_weights, counts)
            return sdid(sampled_df, "unit", "time", "treatment", "outcome", treated_weights=tw_b)["att"]
    else:
        unique_units = np.unique(data_ref.unit)
        treated_units = np.unique(data_ref[data_ref.treated == 1].unit)
        def theta_bt():
            sampled = np.random.choice(unique_units, replace=True, size=len(unique_units))
            counts = np.array([(sampled == u).sum() for u in treated_units])
            tw_b = renormalize_weights(treated_weights, counts)
            pieces = []
            for idx, u in enumerate(sampled):
                chunk = data_ref[data_ref.unit == u].copy()
                chunk = chunk.assign(unit=str(u) + f"__{idx}")
                pieces.append(chunk)
            sampled_df = pd.concat(pieces, ignore_index=True)
            if len(np.unique(sampled_df.treatment)) != 2:
                return theta_bt()
            return sdid(sampled_df, "unit", "time", "treatment", "outcome", treated_weights=tw_b)["att"]

    atts = []
    while len(atts) < n_reps:
        atts.append(theta_bt())
    atts = np.array(atts)
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

def jackknife_iteration(data, time_breaks, weights, unit_index: int, treated_weights=None) -> np.ndarray:
    weighted_atts = np.array([])
    total_treated_unit_periods = data[data.treatment == 1].shape[0]

    for tyear_index, treatment_year in enumerate(time_breaks):
        tyear_data = data[data.tyear.isin([0, treatment_year])]
        N_treated = pd.unique(data[data.tyear == treatment_year].unit).shape[0]
        tyear_omegas =  - weights["omega"][tyear_index]
        N_control = tyear_omegas.shape[0]
        if treated_weights is not None:
            tw = sum_normalize(np.array(treated_weights)[unit_index < N_control:])  # subset remaining treated
            tyear_omegas = np.concatenate([tyear_omegas, tw])
        else:
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
        


def jackknife_se(data_ref: pd.DataFrame, time_breaks, att, weights, treated_weights=None):
    
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

def cluster_jackknife_se_weighted(data_ref, time_breaks, att, weights, treated_weights, cluster):
    unique_clusters = np.unique(data_ref[cluster])
    K = len(unique_clusters)
    if K <= 1:
        return np.nan
    treated_units = np.unique(data_ref[data_ref.treated == 1].unit)
    theta_k = []
    for cl in unique_clusters:
        keep = data_ref[data_ref[cluster] != cl]
        keep_treated = np.unique(keep[keep.treated == 1].unit)
        if len(keep_treated) == 0:
            continue
        tw_idx = [i for i, u in enumerate(treated_units) if u in keep_treated]
        tw_jk = sum_normalize(np.array(treated_weights)[tw_idx])
        try:
            result = sdid(keep, "unit", "time", "treatment", "outcome", treated_weights=tw_jk)
            theta_k.append(result["att"])
        except Exception:
            continue
    theta_k = np.array(theta_k)
    K_valid = len(theta_k)
    if K_valid < 2:
        return np.nan
    return np.sqrt(((K_valid - 1) / K_valid) * (K_valid - 1) * np.var(theta_k, ddof=1))

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
            se = bootstrap_se(data_ref, n_reps=n_reps)
        else:  # jackknife
            time_break, weights = self.ttime, self.weights
            if self.cluster is not None and self.treated_weights is not None:
                se = cluster_jackknife_se_weighted(data_ref, time_break, self.att, weights, self.treated_weights, self.cluster)
            else:
                se = jackknife_se(data_ref, time_break, self.att, weights, treated_weights=self.treated_weights)
        self.se = se
        return self
