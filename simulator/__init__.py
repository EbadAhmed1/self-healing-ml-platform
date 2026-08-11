"""
simulator
─────────
Traffic simulator and failure injection engine for the self-healing ML platform.

Modules:
  split          — deterministic 15% simulation dataset split
  injection      — failure injection engine (numeric drift, category skew, corrupt, delay spike)
  outcome_writer — background thread for writing delayed ground-truth labels
  db_models      — ORM models for `outcomes` and `simulation_events` tables
  run            — CLI runner with HTTP retry and runtime stdin control
"""

from simulator.injection import Injection, InjectionState
from simulator.outcome_writer import OutcomeWriter
from simulator.split import load_sim_data, make_splits, save_sim_split

__all__ = [
    "Injection",
    "InjectionState",
    "OutcomeWriter",
    "load_sim_data",
    "make_splits",
    "save_sim_split",
]
