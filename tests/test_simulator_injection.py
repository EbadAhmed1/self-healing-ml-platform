"""
tests/test_simulator_injection.py
───────────────────────────────────
Tests for the failure injection engine in simulator/injection.py.

REQUIREMENT FROM SPEC:
  "Write pytest tests for the injection logic specifically (verify that
   `--drift-feature` actually shifts the distribution of what's sent, using a
   statistical check, not just 'it ran without error')."

Tests cover:
  1. Statistical shift test (Welch's t-test) for numeric drift
  2. Cumulative nature of multiple numeric drifts
  3. Reversibility (stopping an injection restores the original baseline)
  4. Chi-squared test for categorical skew
  5. Corrupt feature produces null (None) values
  6. Normal mode (no injection) leaves data untouched
  7. Independence of non-overlapping injections
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from simulator.injection import InjectionState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def base_rows() -> list[dict]:
    """200 realistic rows with tenure ~ N(30, 10) and uniform categoricals."""
    rng = np.random.default_rng(42)
    tenure_vals = rng.normal(loc=30.0, scale=10.0, size=200).tolist()
    contracts = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=200
    ).tolist()

    rows = []
    for t, c in zip(tenure_vals, contracts):
        rows.append(
            {
                "tenure": round(float(t), 2),
                "MonthlyCharges": 65.0,
                "TotalCharges": round(float(t * 65.0), 2),
                "Contract": c,
                "gender": "Female",
                "InternetService": "DSL",
            }
        )
    return rows


# ===========================================================================
# 1. Numeric Drift — Statistical Verification (Welch's t-test)
# ===========================================================================
class TestNumericDriftStatistical:
    def test_drift_feature_shifts_distribution_statistically(self, base_rows):
        """
        Verify --drift-feature actually shifts the numeric distribution.
        Uses Welch's two-sample t-test to prove the mean has shifted significantly
        (p < 0.001) and the observed shift matches the injected magnitude (+20.0).
        """
        state = InjectionState()
        # db=None skips database logging for pure unit test speed
        state.add_numeric_drift(db=None, feature="tenure", magnitude=20.0)

        original_tenure = [r["tenure"] for r in base_rows]
        shifted_rows = [state.apply(r) for r in base_rows]
        shifted_tenure = [r["tenure"] for r in shifted_rows]

        # 1. Welch's t-test (unequal variances assumed)
        t_stat, p_val = stats.ttest_ind(
            shifted_tenure, original_tenure, equal_var=False
        )

        assert (
            p_val < 0.001
        ), f"Expected significant shift (p < 0.001), got p={p_val:.6f}"
        assert t_stat > 0, "Shifted mean should be greater than original mean"

        # 2. Mean difference check
        observed_diff = np.mean(shifted_tenure) - np.mean(original_tenure)
        assert pytest.approx(observed_diff, abs=1e-5) == 20.0

    def test_numeric_drift_is_cumulative(self, base_rows):
        """
        Verify that multiple numeric drifts on the same feature stack additively.
        +10.0 + +15.0 = +25.0 total shift.
        """
        state = InjectionState()
        state.add_numeric_drift(db=None, feature="tenure", magnitude=10.0)
        state.add_numeric_drift(db=None, feature="tenure", magnitude=15.0)

        original_tenure = [r["tenure"] for r in base_rows]
        shifted_rows = [state.apply(r) for r in base_rows]
        shifted_tenure = [r["tenure"] for r in shifted_rows]

        observed_diff = np.mean(shifted_tenure) - np.mean(original_tenure)
        assert pytest.approx(observed_diff, abs=1e-5) == 25.0


# ===========================================================================
# 2. Reversibility & Normal Mode
# ===========================================================================
class TestReversibilityAndNormalMode:
    def test_no_injection_is_default(self, base_rows):
        """Empty InjectionState returns exact copies of the original rows."""
        state = InjectionState()

        for original in base_rows:
            transformed = state.apply(original)
            assert transformed == original
            assert transformed is not original  # must be a copy

    def test_drift_is_reversible(self, base_rows):
        """
        Stopping an active injection must restore baseline outputs.
        """
        state = InjectionState()
        inj = state.add_numeric_drift(db=None, feature="tenure", magnitude=20.0)

        # Active state: modified
        assert state.apply(base_rows[0])["tenure"] == base_rows[0]["tenure"] + 20.0

        # Remove state: restored
        removed = state.remove(db=None, injection=inj)
        assert removed is True
        assert state.apply(base_rows[0])["tenure"] == base_rows[0]["tenure"]


# ===========================================================================
# 3. Categorical Skew — Statistical Verification (Chi-Squared test)
# ===========================================================================
class TestCategorySkewStatistical:
    def test_category_skew_changes_distribution(self, base_rows):
        """
        Verify --drift-category skews categorical distribution.
        Uses Chi-squared test to prove the post-injection category frequency
        differs significantly from baseline (p < 0.001) and target category
        reaches ~80% proportion.
        """
        state = InjectionState()
        target_category = "Month-to-month"
        state.add_category_skew(
            db=None,
            feature="Contract",
            target_value=target_category,
            skew_prob=0.8,
        )

        transformed_rows = [state.apply(r) for r in base_rows]
        post_contracts = [r["Contract"] for r in transformed_rows]

        target_count = post_contracts.count(target_category)
        target_freq = target_count / len(post_contracts)

        # Should be around 80% + baseline contribution
        assert (
            target_freq >= 0.70
        ), f"Expected high target frequency, got {target_freq:.2f}"

        # Chi-squared test against uniform baseline distribution
        # Expected counts under 1/3 uniform assumption:
        n = len(post_contracts)
        expected = [n / 3, n / 3, n / 3]
        observed = [
            post_contracts.count("Month-to-month"),
            post_contracts.count("One year"),
            post_contracts.count("Two year"),
        ]

        chi2_stat, p_val = stats.chisquare(f_obs=observed, f_exp=expected)
        assert p_val < 0.001, f"Expected chi-squared p < 0.001, got p={p_val:.6f}"


# ===========================================================================
# 4. Feature Corruption & Independence
# ===========================================================================
class TestCorruptionAndIndependence:
    def test_corrupt_feature_produces_null(self, base_rows):
        """
        Verify --corrupt-feature sets the target feature value to None (null).
        This triggers Pydantic 422 at the API layer.
        """
        state = InjectionState()
        state.add_corrupt_feature(db=None, feature="tenure")

        for r in base_rows:
            transformed = state.apply(r)
            assert transformed["tenure"] is None
            assert (
                transformed["MonthlyCharges"] == r["MonthlyCharges"]
            )  # other fields intact

    def test_multiple_injections_are_independent(self, base_rows):
        """
        Injections on different features operate independently.
        """
        state = InjectionState()
        state.add_numeric_drift(db=None, feature="tenure", magnitude=15.0)
        state.add_category_skew(
            db=None, feature="Contract", target_value="Two year", skew_prob=1.0
        )

        transformed = state.apply(base_rows[0])
        assert transformed["tenure"] == base_rows[0]["tenure"] + 15.0
        assert transformed["Contract"] == "Two year"
        assert (
            transformed["MonthlyCharges"] == base_rows[0]["MonthlyCharges"]
        )  # unaffected
