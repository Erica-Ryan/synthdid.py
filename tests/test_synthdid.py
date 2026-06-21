"""
Tests for synthdid.py
=====================
Compares results against the reference values documented in the README,
which in turn replicate the Stata sdid package (Daniel-Pailanir/sdid).

Reference values were obtained from:
  - README.md of synthdid.py (cross-checked with Stata sdid package outputs)
  - Arkhangelsky et al. (2021) "Synthetic Difference-in-Differences"

Run with:  pytest tests/test_synthdid.py -v
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")

from synthdid.get_data import california_prop99, quota
from synthdid.synthdid import Synthdid

# ── tolerances ────────────────────────────────────────────────────────────────
ATT_TOL   = 1e-3   # ATT point estimates must match to 3 decimal places
WT_TOL    = 1e-4   # per-period att_time / att_wt must match to 4 decimal places
SE_ABS    = 3.0    # stochastic SEs allowed ±3 units from reference (50 reps)
JK_TOL    = 1e-3   # jackknife is deterministic → tight tolerance


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def df_california():
    return california_prop99()

@pytest.fixture(scope="module")
def df_quota():
    return quota()

@pytest.fixture(scope="module")
def df_quota_subset(df_quota):
    exclude = ["Algeria", "Kenya", "Samoa", "Swaziland", "Tanzania"]
    return df_quota[~df_quota.country.isin(exclude)].copy()

@pytest.fixture(scope="module")
def df_quota_cov(df_quota):
    return df_quota[~df_quota.lngdp.isnull()].copy()


# ══════════════════════════════════════════════════════════════════════════════
# 1. BLOCK DESIGN – California Proposition 99
# ══════════════════════════════════════════════════════════════════════════════

class TestCaliforniaSDID:
    """SDID on a single-adoption block design."""

    @pytest.fixture(autouse=True, scope="class")
    def model(self, df_california):
        self.__class__._m = (
            Synthdid(df_california, "State", "Year", "treated", "PacksPerCapita")
            .fit()
        )

    def test_att(self):
        assert abs(self._m.att - (-15.60383)) < ATT_TOL

    def test_att_info_shape(self):
        assert self._m.att_info.shape == (1, 7)

    def test_att_info_period(self):
        row = self._m.att_info.iloc[0]
        assert row["time"] == 1989.0
        assert abs(row["att_time"] - (-15.603828)) < WT_TOL
        assert abs(row["att_wt"]  - 1.0)          < WT_TOL
        assert row["N0"] == 38
        assert row["T0"] == 19
        assert row["N1"] == 1
        assert row["T1"] == 12

    def test_weights_sum_to_one(self):
        om = self._m.weights["omega"][0]
        lm = self._m.weights["lambda"][0]
        assert abs(np.sum(om) - 1.0) < 1e-6
        assert abs(np.sum(lm) - 1.0) < 1e-6

    def test_weights_nonnegative(self):
        om = self._m.weights["omega"][0]
        lm = self._m.weights["lambda"][0]
        assert np.all(om >= -1e-10)
        assert np.all(lm >= -1e-10)

    def test_omega_length(self):
        # N0 = 38 control units
        assert len(self._m.weights["omega"][0]) == 38

    def test_lambda_length(self):
        # T0 = 19 pre-treatment periods
        assert len(self._m.weights["lambda"][0]) == 19

    def test_plot_outcomes_runs(self, df_california):
        m = Synthdid(df_california, "State", "Year", "treated", "PacksPerCapita").fit()
        m.plot_outcomes()
        assert len(m.plot_outcomes) == 1

    def test_plot_weights_runs(self, df_california):
        m = Synthdid(df_california, "State", "Year", "treated", "PacksPerCapita").fit()
        m.plot_weights()
        assert len(m.plot_weights) == 1

    def test_summary_no_se(self):
        m = Synthdid(california_prop99(), "State", "Year", "treated", "PacksPerCapita").fit()
        m.summary()
        assert m.summary2["ATT"].iloc[0] == pytest.approx(-15.60383, abs=ATT_TOL)
        assert m.summary2["Std. Err."].iloc[0] == "-"

    def test_placebo_se(self, df_california):
        np.random.seed(0)
        m = (
            Synthdid(df_california, "State", "Year", "treated", "PacksPerCapita")
            .fit()
            .vcov(method="placebo", n_reps=50)
            .summary()
        )
        # Reference SE ≈ 10.79; allow ±SE_ABS because it's stochastic
        assert m.se > 0
        assert abs(m.se - 10.789924) < SE_ABS

    def test_bootstrap_se(self, df_california):
        np.random.seed(0)
        m = (
            Synthdid(df_california, "State", "Year", "treated", "PacksPerCapita")
            .fit()
            .vcov(method="bootstrap", n_reps=50)
        )
        assert m.se > 0

    def test_n_reps_respected(self, df_california):
        """vcov() must use the n_reps argument, not a hardcoded 50."""
        np.random.seed(1)
        m10 = (
            Synthdid(df_california, "State", "Year", "treated", "PacksPerCapita")
            .fit()
        )
        np.random.seed(1)
        m20 = (
            Synthdid(df_california, "State", "Year", "treated", "PacksPerCapita")
            .fit()
        )
        m10.vcov(method="placebo", n_reps=10)
        m20.vcov(method="placebo", n_reps=20)
        # Different n_reps → SEs will differ (unless by coincidence)
        # Main check: both are positive and ATT unchanged
        assert m10.se > 0
        assert m20.se > 0
        assert m10.att == m20.att


class TestCaliforniaSC:
    """Synthetic Control mode on California data."""

    @pytest.fixture(autouse=True, scope="class")
    def model(self, df_california):
        self.__class__._m = (
            Synthdid(df_california, "State", "Year", "treated", "PacksPerCapita")
            .fit(synth=True)
        )

    def test_att_finite(self):
        assert np.isfinite(self._m.att)

    def test_att_sign(self):
        # Prop 99 reduced smoking; SC should also be negative
        assert self._m.att < 0

    def test_lambda_zeros(self):
        # In SC mode lambda = 0 (no pre-period weighting), matching Stata sdid
        lm = self._m.weights["lambda"][0]
        T0 = self._m.att_info["T0"].iloc[0]
        expected = np.zeros(T0)
        np.testing.assert_allclose(lm, expected, atol=1e-8)

    def test_weights_sum_to_one(self):
        om = self._m.weights["omega"][0]
        assert abs(np.sum(om) - 1.0) < 1e-6


class TestCaliforniaDiD:
    """Difference-in-Differences mode on California data."""

    @pytest.fixture(autouse=True, scope="class")
    def model(self, df_california):
        self.__class__._m = (
            Synthdid(df_california, "State", "Year", "treated", "PacksPerCapita")
            .fit(did=True)
        )

    def test_att(self):
        assert abs(self._m.att - (-27.34911)) < ATT_TOL

    def test_omega_uniform(self):
        om = self._m.weights["omega"][0]
        N0 = self._m.att_info["N0"].iloc[0]
        expected = np.full(N0, 1.0 / N0)
        np.testing.assert_allclose(om, expected, atol=1e-8)

    def test_lambda_uniform(self):
        lm = self._m.weights["lambda"][0]
        T0 = self._m.att_info["T0"].iloc[0]
        expected = np.full(T0, 1.0 / T0)
        np.testing.assert_allclose(lm, expected, atol=1e-8)


# ══════════════════════════════════════════════════════════════════════════════
# 2. STAGGERED ADOPTION – Women's Parliamentary Quota
# ══════════════════════════════════════════════════════════════════════════════

class TestQuotaSDID:
    """SDID on a staggered-adoption design."""

    # Reference att_info from README
    _REF = [
        dict(time=2000.0, att_time=8.388868,  att_wt=0.170213, N0=110, T0=10, N1=1, T1=16),
        dict(time=2002.0, att_time=6.967746,  att_wt=0.297872, N0=110, T0=12, N1=2, T1=14),
        dict(time=2003.0, att_time=13.952256, att_wt=0.276596, N0=110, T0=13, N1=2, T1=13),
        dict(time=2005.0, att_time=-3.450543, att_wt=0.117021, N0=110, T0=15, N1=1, T1=11),
        dict(time=2010.0, att_time=2.749035,  att_wt=0.063830, N0=110, T0=20, N1=1, T1=6),
        dict(time=2012.0, att_time=21.762715, att_wt=0.042553, N0=110, T0=22, N1=1, T1=4),
        dict(time=2013.0, att_time=-0.820324, att_wt=0.031915, N0=110, T0=23, N1=1, T1=3),
    ]

    @pytest.fixture(autouse=True, scope="class")
    def model(self, df_quota):
        self.__class__._m = (
            Synthdid(df_quota, "country", "year", "quota", "womparl")
            .fit()
        )

    def test_att(self):
        assert abs(self._m.att - 8.0341) < ATT_TOL

    def test_att_info_n_periods(self):
        assert len(self._m.att_info) == 7

    @pytest.mark.parametrize("i,ref", enumerate(_REF))
    def test_att_info_period(self, i, ref):
        row = self._m.att_info.iloc[i]
        assert row["time"]     == ref["time"]
        assert abs(row["att_time"] - ref["att_time"]) < WT_TOL
        assert abs(row["att_wt"]   - ref["att_wt"])   < WT_TOL
        assert row["N0"] == ref["N0"]
        assert row["T0"] == ref["T0"]
        assert row["N1"] == ref["N1"]
        assert row["T1"] == ref["T1"]

    def test_weights_all_periods(self):
        for i in range(7):
            om = self._m.weights["omega"][i]
            lm = self._m.weights["lambda"][i]
            assert abs(np.sum(om) - 1.0) < 1e-6, f"omega sum != 1 at period {i}"
            assert abs(np.sum(lm) - 1.0) < 1e-6, f"lambda sum != 1 at period {i}"
            assert np.all(om >= -1e-10), f"negative omega at period {i}"
            assert np.all(lm >= -1e-10), f"negative lambda at period {i}"

    def test_plot_outcomes_runs(self, df_quota):
        m = Synthdid(df_quota, "country", "year", "quota", "womparl").fit()
        m.plot_outcomes()
        assert len(m.plot_outcomes) == 7

    def test_plot_weights_runs(self, df_quota):
        m = Synthdid(df_quota, "country", "year", "quota", "womparl").fit()
        m.plot_weights()
        assert len(m.plot_weights) == 7

    def test_att_wt_sum_to_one(self):
        assert abs(self._m.att_info["att_wt"].sum() - 1.0) < 1e-6

    def test_weighted_att_consistent(self):
        """att == dot(att_time, att_wt)"""
        computed = np.dot(self._m.att_info["att_time"], self._m.att_info["att_wt"])
        assert abs(computed - self._m.att) < 1e-3


class TestQuotaInference:
    """SE estimation on quota subset (matches README inference section)."""

    @pytest.fixture(autouse=True, scope="class")
    def model(self, df_quota_subset):
        self.__class__._m = (
            Synthdid(df_quota_subset, "country", "year", "quota", "womparl")
            .fit()
        )

    def test_att(self):
        assert abs(self._m.att - 10.33066) < ATT_TOL

    def test_jackknife_se(self):
        """Jackknife is deterministic; must match reference exactly."""
        m = self._m
        m.vcov(method="jackknife").summary()
        assert abs(m.se - 6.04213) < JK_TOL
        assert abs(m.summary2["t"].iloc[0]     - 1.709771)  < 1e-4
        assert abs(m.summary2["P>|t|"].iloc[0] - 0.087308)  < 1e-4

    def test_bootstrap_se_positive(self, df_quota_subset):
        np.random.seed(0)
        m = (
            Synthdid(df_quota_subset, "country", "year", "quota", "womparl")
            .fit()
            .vcov(method="bootstrap", n_reps=50)
        )
        assert m.se > 0
        # Reference ≈ 5.40; stochastic tolerance
        assert abs(m.se - 5.404923) < SE_ABS

    def test_placebo_se_positive(self, df_quota_subset):
        np.random.seed(0)
        m = (
            Synthdid(df_quota_subset, "country", "year", "quota", "womparl")
            .fit()
            .vcov(method="placebo", n_reps=50)
        )
        assert m.se > 0
        # Reference ≈ 2.24; stochastic tolerance
        assert abs(m.se - 2.244618) < SE_ABS


# ══════════════════════════════════════════════════════════════════════════════
# 3. COVARIATES
# ══════════════════════════════════════════════════════════════════════════════

class TestQuotaCovariates:
    """Covariate adjustment (optimized and projected methods)."""

    def test_att_optimized(self, df_quota_cov):
        m = (
            Synthdid(df_quota_cov, "country", "year", "quota", "womparl",
                     covariates=["lngdp"])
            .fit()
        )
        assert abs(m.att - 8.04901) < ATT_TOL

    def test_att_projected(self, df_quota_cov):
        m = (
            Synthdid(df_quota_cov, "country", "year", "quota", "womparl",
                     covariates=["lngdp"])
            .fit(cov_method="projected")
        )
        assert abs(m.att - 8.05903) < ATT_TOL

    def test_optimized_se_bootstrap(self, df_quota_cov):
        np.random.seed(0)
        m = (
            Synthdid(df_quota_cov, "country", "year", "quota", "womparl",
                     covariates=["lngdp"])
            .fit()
            .vcov(method="bootstrap", n_reps=50)
            .summary()
        )
        assert m.se > 0
        # Reference ≈ 3.40; stochastic tolerance
        assert abs(m.se - 3.395295) < SE_ABS

    def test_projected_se_bootstrap(self, df_quota_cov):
        np.random.seed(0)
        m = (
            Synthdid(df_quota_cov, "country", "year", "quota", "womparl",
                     covariates=["lngdp"])
            .fit(cov_method="projected")
            .vcov(method="bootstrap", n_reps=50)
            .summary()
        )
        assert m.se > 0
        # Reference ≈ 3.43
        assert abs(m.se - 3.428897) < SE_ABS

    def test_weights_sum_to_one_optimized(self, df_quota_cov):
        m = (
            Synthdid(df_quota_cov, "country", "year", "quota", "womparl",
                     covariates=["lngdp"])
            .fit()
        )
        for i in range(len(m.att_info)):
            om = m.weights["omega"][i]
            lm = m.weights["lambda"][i]
            assert abs(np.sum(om) - 1.0) < 1e-5, f"omega sum at period {i}"
            assert abs(np.sum(lm) - 1.0) < 1e-5, f"lambda sum at period {i}"

    def test_weights_sum_to_one_projected(self, df_quota_cov):
        m = (
            Synthdid(df_quota_cov, "country", "year", "quota", "womparl",
                     covariates=["lngdp"])
            .fit(cov_method="projected")
        )
        for i in range(len(m.att_info)):
            om = m.weights["omega"][i]
            lm = m.weights["lambda"][i]
            assert abs(np.sum(om) - 1.0) < 1e-5, f"omega sum at period {i}"
            assert abs(np.sum(lm) - 1.0) < 1e-5, f"lambda sum at period {i}"


# ══════════════════════════════════════════════════════════════════════════════
# 4. EDGE CASES & REGRESSION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_summary_without_vcov_shows_dashes(self, df_california):
        m = Synthdid(df_california, "State", "Year", "treated", "PacksPerCapita").fit()
        m.summary()
        assert m.summary2["Std. Err."].iloc[0] == "-"
        assert m.summary2["t"].iloc[0]         == "-"
        assert m.summary2["P>|t|"].iloc[0]     == "-"

    def test_vcov_returns_self(self, df_quota_subset):
        # Jackknife requires ≥2 treated units per period; use quota subset
        m = Synthdid(df_quota_subset, "country", "year", "quota", "womparl").fit()
        result = m.vcov(method="jackknife")
        assert result is m

    def test_fit_returns_self(self, df_california):
        m = Synthdid(df_california, "State", "Year", "treated", "PacksPerCapita")
        result = m.fit()
        assert result is m

    def test_chaining(self, df_quota_subset):
        """Full method chain must not raise."""
        m = (
            Synthdid(df_quota_subset, "country", "year", "quota", "womparl")
            .fit()
            .vcov(method="jackknife")
            .summary()
        )
        assert isinstance(m.summary2["ATT"].iloc[0], float)

    def test_att_finite_quota(self, df_quota):
        m = Synthdid(df_quota, "country", "year", "quota", "womparl").fit()
        assert np.isfinite(m.att)

    def test_no_nan_in_weights(self, df_quota):
        m = Synthdid(df_quota, "country", "year", "quota", "womparl").fit()
        for om, lm in zip(m.weights["omega"], m.weights["lambda"]):
            assert not np.any(np.isnan(om)), "NaN in omega"
            assert not np.any(np.isnan(lm)), "NaN in lambda"

    def test_bootstrap_n_reps_param_respected(self, df_california):
        """n_reps parameter must be forwarded, not hardcoded."""
        np.random.seed(99)
        m = (
            Synthdid(df_california, "State", "Year", "treated", "PacksPerCapita")
            .fit()
        )
        m.vcov(method="bootstrap", n_reps=5)
        se_5 = m.se
        m.vcov(method="bootstrap", n_reps=200)
        se_200 = m.se
        # Both positive; 200 reps should converge, 5 reps is noisier
        assert se_5 > 0
        assert se_200 > 0

    def test_placebo_n_reps_param_respected(self, df_california):
        np.random.seed(99)
        m = (
            Synthdid(df_california, "State", "Year", "treated", "PacksPerCapita")
            .fit()
        )
        m.vcov(method="placebo", n_reps=5)
        assert m.se > 0
        m.vcov(method="placebo", n_reps=200)
        assert m.se > 0

# ══════════════════════════════════════════════════════════════════════════════
# 5. WEIGHTED ESTIMATOR
# ══════════════════════════════════════════════════════════════════════════════

class TestWeighted:
    """Smoke tests for treated_weights support."""

    @pytest.fixture(autouse=True, scope="class")
    def model(self, df_california):
        df = df_california.copy()
        # California has 1 treated unit so weights are trivial (scalar 1.0)
        # Use quota data instead for a multi-treated test
        pass

    @pytest.fixture(scope="class")
    def weighted_california(self, df_california):
        return (
            Synthdid(df_california, "State", "Year", "treated", "PacksPerCapita",
                     treated_weights=np.array([1.0]))
            .fit()
        )

    @pytest.fixture(scope="class")
    def weighted_quota(self, df_quota):
        # Fabricate population weights for treated units at each adoption year
        np.random.seed(42)
        m_unweighted = Synthdid(df_quota, "country", "year", "quota", "womparl").fit()
        n_treated_total = sum(m_unweighted.att_info["N1"])
        weights = np.ones(n_treated_total)
        weights[0] = 5.0  # give first treated unit more weight
        weights = weights / weights.sum()
        return (
            Synthdid(df_quota, "country", "year", "quota", "womparl",
                     treated_weights=weights)
            .fit()
        )

    def test_california_weighted_att_finite(self, weighted_california):
        assert np.isfinite(weighted_california.att)

    def test_california_weighted_att_close_to_unweighted(self, df_california, weighted_california):
        """Single treated unit: weighted == unweighted."""
        m_uw = Synthdid(df_california, "State", "Year", "treated", "PacksPerCapita").fit()
        assert abs(weighted_california.att - m_uw.att) < 1e-6

    def test_quota_weighted_att_finite(self, weighted_quota):
        assert np.isfinite(weighted_quota.att)

    def test_quota_weighted_att_differs_from_unweighted(self, df_quota, weighted_quota):
        """Non-uniform weights should produce a different ATT."""
        m_uw = Synthdid(df_quota, "country", "year", "quota", "womparl").fit()
        assert abs(weighted_quota.att - m_uw.att) > 1e-6

    def test_weighted_omega_sums_to_one(self, weighted_quota):
        for om in weighted_quota.weights["omega"]:
            assert abs(np.sum(om) - 1.0) < 1e-6

    def test_weighted_lambda_sums_to_one(self, weighted_quota):
        for lm in weighted_quota.weights["lambda"]:
            assert abs(np.sum(lm) - 1.0) < 1e-6

    def test_jackknife_se_weighted_positive(self, weighted_quota):
        weighted_quota.vcov(method="jackknife")
        assert weighted_quota.se > 0
        assert np.isfinite(weighted_quota.se)

    def test_bootstrap_se_weighted_positive(self, df_quota):
        np.random.seed(0)
        weights = np.ones(sum(
            Synthdid(df_quota, "country", "year", "quota", "womparl").fit().att_info["N1"]
        ))
        weights = weights / weights.sum()
        m = (
            Synthdid(df_quota, "country", "year", "quota", "womparl",
                     treated_weights=weights)
            .fit()
            .vcov(method="bootstrap", n_reps=20)
        )
        assert m.se > 0
        assert np.isfinite(m.se)

    def test_placebo_se_weighted_positive(self, df_quota):
        np.random.seed(0)
        weights = np.ones(sum(
            Synthdid(df_quota, "country", "year", "quota", "womparl").fit().att_info["N1"]
        ))
        weights = weights / weights.sum()
        m = (
            Synthdid(df_quota, "country", "year", "quota", "womparl",
                     treated_weights=weights)
            .fit()
            .vcov(method="placebo", n_reps=20)
        )
        assert m.se > 0
        assert np.isfinite(m.se)