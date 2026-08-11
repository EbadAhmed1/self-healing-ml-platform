"""
monitoring/psi.py
─────────────────
Population Stability Index (PSI) calculation engine built from scratch.

PSI MATHEMATICS:
  PSI = Σ (Actual_i - Expected_i) * ln(Actual_i / Expected_i)

  Where:
    - Expected_i (E_i) = proportion of baseline data in bin i
    - Actual_i (A_i)   = proportion of current production data in bin i
    - k                = number of bins

BUSINESS THRESHOLDS:
  PSI < 0.10       → Stable (no significant distribution change)
  0.10 <= PSI < 0.25 → Moderate drift (warning — monitor closely)
  PSI >= 0.25      → Significant drift (critical alert — investigation required)

EDGE CASES HANDLED EXPLICITLY:
  1. Zero-count bins: Epsilon smoothing (EPSILON = 1e-4) prevents division by zero
     and ln(0) without corrupting the proportion sum.
  2. Small samples: If current sample size < MIN_SAMPLE_SIZE (default: 30), return
     status = "insufficient_data" and PSI = 0.0 to prevent false alerts.
  3. All-null or constant series: If a feature is 100% missing or has 0 variance,
     the binning logic applies epsilon smoothing safely without raising exceptions.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Named Threshold Constants (SSoT for drift classification across platform)
# ---------------------------------------------------------------------------
PSI_STABLE_THRESHOLD: float = 0.10
PSI_MODERATE_THRESHOLD: float = 0.25
MIN_SAMPLE_SIZE: int = 30
EPSILON: float = 1e-4  # smoothing factor for zero-count bins


# ---------------------------------------------------------------------------
# Classification helper
# ---------------------------------------------------------------------------
def classify_psi(score: float, sample_size: int) -> str:
    """
    Classify a PSI score into a standard status string.

    Statuses:
      'insufficient_data' — sample_size < MIN_SAMPLE_SIZE
      'stable'            — score < 0.10
      'moderate'          — 0.10 <= score < 0.25
      'significant'       — score >= 0.25
    """
    if sample_size < MIN_SAMPLE_SIZE:
        return "insufficient_data"
    if score < PSI_STABLE_THRESHOLD:
        return "stable"
    if score < PSI_MODERATE_THRESHOLD:
        return "moderate"
    return "significant"


# ---------------------------------------------------------------------------
# Core PSI formula (proportions → float score)
# ---------------------------------------------------------------------------
def _compute_psi_from_proportions(
    expected_props: np.ndarray, actual_props: np.ndarray
) -> float:
    """
    Calculate PSI score from two probability distribution arrays.

    Applies EPSILON smoothing to prevent zero-division or ln(0).
    Formula: Σ (actual_i - expected_i) * ln(actual_i / expected_i)
    """
    # 1. Epsilon smoothing for zero entries
    e = np.where(expected_props == 0, EPSILON, expected_props)
    a = np.where(actual_props == 0, EPSILON, actual_props)

    # 2. Re-normalize to ensure sum(P) == 1.0
    e = e / np.sum(e)
    a = a / np.sum(a)

    # 3. Calculate element-wise PSI components
    psi_components = (a - e) * np.log(a / e)
    score = float(np.sum(psi_components))

    # Guard against float representation noise producing negative zero (-0.0)
    return max(0.0, round(score, 6))


# ---------------------------------------------------------------------------
# Categorical Feature PSI
# ---------------------------------------------------------------------------
def calculate_categorical_psi(
    baseline_value_counts: dict[str, int] | pd.Series | Sequence[str],
    current_series: Sequence[str] | pd.Series,
) -> float:
    """
    Calculate PSI for a categorical feature.

    Args:
        baseline_value_counts: dict of category counts from metadata snapshot,
                               OR raw baseline sequence/Series.
        current_series: current production values sequence or Series.

    Returns:
        PSI float score.
    """
    curr = pd.Series(list(current_series)).astype(str)

    if len(curr) == 0:
        return 0.0

    # Normalize baseline inputs to a dictionary of counts
    if isinstance(baseline_value_counts, dict):
        if "value_counts" in baseline_value_counts and isinstance(
            baseline_value_counts["value_counts"], dict
        ):
            baseline_value_counts = baseline_value_counts["value_counts"]
        base_counts = {
            str(k): int(v)
            for k, v in baseline_value_counts.items()
            if not isinstance(v, dict)
        }
    else:
        base_series = pd.Series(list(baseline_value_counts)).astype(str)
        base_counts = base_series.value_counts().to_dict()

    base_total = sum(base_counts.values())
    if base_total == 0:
        return 0.0

    curr_counts = curr.value_counts().to_dict()
    curr_total = len(curr)

    # Collect all unique category labels (baseline + current)
    all_categories = sorted(set(base_counts.keys()).union(curr_counts.keys()))
    if not all_categories:
        return 0.0

    expected_props = np.array(
        [base_counts.get(c, 0) / base_total for c in all_categories], dtype=float
    )
    actual_props = np.array(
        [curr_counts.get(c, 0) / curr_total for c in all_categories], dtype=float
    )

    return _compute_psi_from_proportions(expected_props, actual_props)


# ---------------------------------------------------------------------------
# Numeric Feature PSI
# ---------------------------------------------------------------------------
def calculate_numeric_psi(
    baseline_input: list[float] | dict | pd.Series | Sequence[float],
    current_series: Sequence[float] | pd.Series,
    num_bins: int = 10,
) -> float:
    """
    Calculate PSI for a continuous numeric feature.

    Args:
        baseline_input:
          - Option A: raw baseline sequence or Series of numbers
          - Option B: snapshot dict with 'deciles' or ('p25', 'median', 'p75', etc.)
        current_series: current production values sequence or Series.
        num_bins: number of quantile bins (default: 10).

    Returns:
        PSI float score.
    """
    curr = pd.Series(list(current_series), dtype=float).dropna()

    if len(curr) == 0:
        return 0.0

    # 1. Determine bin edges from baseline
    bin_edges: np.ndarray

    if isinstance(baseline_input, dict) and "deciles" in baseline_input:
        # Use deciles pre-calculated during training metadata export
        deciles = baseline_input["deciles"]
        bin_edges = np.array(deciles, dtype=float)
        # Ensure bin edges are strictly monotonically increasing
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            val = float(bin_edges[0]) if len(bin_edges) > 0 else 0.0
            bin_edges = np.array([val - 1.0, val + 1.0])
    elif isinstance(baseline_input, dict):
        # Fallback for old metadata snapshots missing explicit deciles:
        # construct approximate bin edges from p25, median, p75, mean, std
        median = float(baseline_input.get("median", 0.0))
        std = float(baseline_input.get("std", 1.0))
        if std == 0.0:
            std = 1.0
        bin_edges = np.linspace(median - 3 * std, median + 3 * std, num_bins + 1)
    else:
        # Raw baseline sequence
        base = pd.Series(list(baseline_input), dtype=float).dropna()
        if len(base) == 0:
            return 0.0
        # Compute quantile cutoffs
        percentiles = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.percentile(base, percentiles)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            val = float(base.iloc[0]) if len(base) > 0 else 0.0
            bin_edges = np.array([val - 1.0, val + 1.0])

    # Extend outer boundaries to [-inf, inf] so outliers fall into edge bins
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # 2. Bin baseline values
    if isinstance(baseline_input, (list, tuple, np.ndarray, pd.Series)):
        base = pd.Series(list(baseline_input), dtype=float).dropna()
        expected_counts, _ = np.histogram(base, bins=bin_edges)
    else:
        # If baseline is snapshot dict without raw data, assign equal weights across bins
        expected_counts = np.ones(len(bin_edges) - 1)

    expected_total = np.sum(expected_counts)
    if expected_total == 0:
        expected_counts = np.ones(len(bin_edges) - 1)
        expected_total = len(bin_edges) - 1

    expected_props = expected_counts / expected_total

    # 3. Bin current values
    actual_counts, _ = np.histogram(curr, bins=bin_edges)
    actual_total = np.sum(actual_counts)
    if actual_total == 0:
        return 0.0

    actual_props = actual_counts / actual_total

    # 4. Compute PSI
    return _compute_psi_from_proportions(expected_props, actual_props)
