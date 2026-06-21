import pandas as pd, numpy as np

def panel_matrices(data: pd.DataFrame(), unit, time, treatment, outcome, covariates = None): #-> data_prep
	if len(np.unique(data[treatment])) != 2:
		print("Error")

	data = data.reset_index()
	data_ref = pd.DataFrame()
	data_ref[["unit", "time", "outcome"]] = data[[unit, time, outcome]]
	data_ref["treatment"] = data[treatment].to_numpy()
	other = data.drop(columns=[unit, time, outcome, treatment])

	unit, time, outcome, treatment = data_ref.columns

	data_ref["treated"] = data_ref.groupby(unit)[treatment].transform("max")
	data_ref["ty"] = np.where(data_ref[treatment] == 1, data_ref[time], np.nan)
	data_ref["tyear"] = np.where(
		data_ref["treated"] == 1,
		data_ref.groupby(unit)["ty"].transform("min"),
		np.nan
	)
	data_ref = data_ref.reset_index(drop=True).sort_values(["treated", unit, time])

	break_points = np.unique(data_ref.tyear)
	break_points = break_points[~np.isnan(break_points)]

	units = data_ref[unit]
	num_col = data_ref.select_dtypes(np.number).columns
	data_ref = data_ref[num_col].fillna(0)
	data_ref[unit] = units
	data_ref = data_ref.sort_values(["treated", "time", "unit"])
	if covariates is not None:
		# return data
		data_cov = data[covariates].loc[data_ref.index]
		data_ref = pd.concat([data_ref, data_cov], axis = 1)
		# data_ref[covariates] = data_ref[covariates].fillna(0)s
		# return data_cov
		return (data_ref, break_points)

	return (data_ref, break_points)

def projected(data, outcome, unit, time, covariates):

  k = len(covariates)
  X_all = np.array(data[covariates])
  y_all = np.array(data[outcome])

  # Pick non-treated (control) units
  df_c = data[data.tyear == 0]

  # ALL year and unit FE dummies (no drop_first), matching Stata invsym approach
  year_dummies = pd.get_dummies(df_c[time]).astype(float)
  unit_dummies = pd.get_dummies(df_c[unit]).astype(float)

  y_c = df_c[outcome].to_numpy(dtype=float)
  X_cov = df_c[covariates].to_numpy(dtype=float)
  # Stack: covariates + all year dummies + all unit dummies + intercept
  X_c = np.column_stack([X_cov, year_dummies.to_numpy(), unit_dummies.to_numpy(), np.ones(len(df_c))])

  # Use lstsq (pseudoinverse) to match Stata invsym behavior with collinear dummies
  all_beta, _, _, _ = np.linalg.lstsq(X_c, y_c, rcond=None)
  beta = all_beta[:k]

  # Apply projection to all units
  Y_adj = y_all - X_all @ beta

  # output projected data
  data = data.copy()
  data[outcome] = Y_adj

  return (data, beta, X_all)

def collapse_form(Y: np.ndarray, N0: int, T0: int):
	N, T = Y.shape
	Y = pd.DataFrame(Y)
	row_mean = Y.iloc[0:N0, T0:T].mean(axis=1)
	col_mean = Y.iloc[N0:N, 0:T0].mean(axis=0)
	overall_mean = Y.iloc[N0:N, T0:T].mean().values[0]
	result_top = pd.concat([Y.iloc[0:N0, 0:T0], row_mean], axis=1)
	result_bottom = pd.concat([col_mean.T, pd.Series(overall_mean)], axis=0)
	Yc = pd.concat([result_top, pd.DataFrame(result_bottom).T], axis=0)
	return Yc

def sum_normalize(x):
    if np.sum(x) != 0:
        return x / np.sum(x)
    else:
        return np.full(len(x), 1/len(x))
    
def sparsify_function(v) -> np.array:
	v = np.where(v <= np.max(v) / 4, 0, v)
	return v / sum(v)

def varianza(x):
	n = len(x)
	media = sum(x) / n
	return sum((xi - media) ** 2 for xi in x) / (n - 1)

def collapse_form_weighted(Y, N0, T0, treated_weights=None, period_weights=None):
    N, T = Y.shape
    N1 = N - N0
    T1 = T - T0
    Y = pd.DataFrame(Y)

    if treated_weights is None:
        treated_weights = np.full(N1, 1/N1)
    else:
        treated_weights = np.array(treated_weights)
        treated_weights = treated_weights / treated_weights.sum()

    if period_weights is None:
        period_weights = np.full(T1, 1/T1)
    else:
        period_weights = np.array(period_weights)
        period_weights = period_weights / period_weights.sum()

    Y_control_pre  = Y.iloc[:N0, :T0]
    Y_control_post = Y.iloc[:N0, T0:]
    Y_treated_pre  = Y.iloc[N0:, :T0]
    Y_treated_post = Y.iloc[N0:, T0:]

    control_post_collapsed = Y_control_post.values @ period_weights      # N0 x 1
    treated_pre_collapsed  = Y_treated_pre.values.T @ treated_weights    # T0 x 1
    treated_post_collapsed = treated_weights @ Y_treated_post.values @ period_weights  # scalar

    result_top = pd.concat([Y_control_pre, pd.Series(control_post_collapsed, index=Y_control_pre.index)], axis=1)
    result_bottom = pd.DataFrame([np.append(treated_pre_collapsed, treated_post_collapsed)])
    result_bottom.index = [N0]

    return pd.concat([result_top, result_bottom])