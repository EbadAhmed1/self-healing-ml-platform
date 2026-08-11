"""
simulator/split.py
──────────────────
Deterministic dataset split utility.

SPLIT STRATEGY — three non-overlapping slices from one CSV:
  ┌─────────────────────────────────────────────────────────────┐
  │  All rows  (sorted by customerID — alphabetically stable)   │
  ├──────────────────────────────┬──────────────┬───────────────┤
  │  Train-eligible (68%)        │  Test (17%)  │  Sim (15%)    │
  └──────────────────────────────┴──────────────┴───────────────┘
  • Sorting by customerID mimics a temporal cutoff: IDs issued
    sequentially → later IDs = "newer customers", a realistic
    production scenario where the sim set is temporally after training.
  • The split is 100% deterministic — no random_state needed here.
    The train/test random split happens WITHIN the train-eligible slice
    inside models/train.py (random_state=42 preserved).
  • sim_fraction default = 0.15 → ~1057 rows from the full 7043-row
    Telco dataset.

This function is the single source of truth for all three slices.
train.py imports and calls it so training and simulation always agree.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from models.feature_config import ID_COL

log = logging.getLogger(__name__)

# Fraction of total data reserved for simulation (never seen during training)
SIM_FRACTION = 0.15
SIM_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "simulation"
SIM_DATA_PATH = SIM_DATA_DIR / "sim_data.csv"


def make_splits(
    df: pd.DataFrame,
    sim_fraction: float = SIM_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a raw Telco DataFrame into (train_eligible, simulation) slices.

    Args:
        df: Raw DataFrame with customerID column present.
        sim_fraction: Fraction of rows to reserve for simulation.

    Returns:
        (train_eligible_df, sim_df)
        train_eligible_df → passed into train_test_split() in train.py
        sim_df            → saved to data/simulation/sim_data.csv
    """
    if ID_COL not in df.columns:
        raise ValueError(f"Expected column '{ID_COL}' not found in DataFrame.")

    # Sort by customerID for deterministic, stable ordering
    df_sorted = df.sort_values(ID_COL, ascending=True).reset_index(drop=True)

    sim_start_idx = int(len(df_sorted) * (1 - sim_fraction))
    sim_df = df_sorted.iloc[sim_start_idx:].copy().reset_index(drop=True)
    train_eligible_df = df_sorted.iloc[:sim_start_idx].copy().reset_index(drop=True)

    log.info(
        "Dataset split: %d train-eligible / %d simulation (%.0f%% of %d total)",
        len(train_eligible_df),
        len(sim_df),
        sim_fraction * 100,
        len(df_sorted),
    )
    return train_eligible_df, sim_df


def save_sim_split(df: pd.DataFrame, output_path: Path = SIM_DATA_PATH) -> Path:
    """
    Compute and persist the simulation split to disk.

    Args:
        df: Raw DataFrame (full dataset before any splits).
        output_path: Where to write the CSV.

    Returns:
        Path to the written CSV.
    """
    _, sim_df = make_splits(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sim_df.to_csv(output_path, index=False)
    log.info("Simulation split saved → %s (%d rows)", output_path, len(sim_df))
    return output_path


def load_sim_data(path: Path = SIM_DATA_PATH) -> pd.DataFrame:
    """
    Load the simulation split CSV. Fails fast with a helpful error if missing.

    IMPORTANT: TotalCharges arrives as object dtype in the raw Telco CSV
    (spaces for new customers). We coerce it to float here exactly as
    train.py's preprocess_raw() does, so the simulator works with clean
    numeric values.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"\n\nSimulation data not found at: {path}\n"
            f"Generate it first by running:\n"
            f"  python models/train.py --save-sim-split\n"
        )

    df = pd.read_csv(path)
    # Coerce TotalCharges: spaces in raw data → NaN → fill with 0.0 for
    # brand-new customers (tenure=0, no charges yet)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    log.info("Loaded simulation data: %d rows from %s", len(df), path)
    return df
