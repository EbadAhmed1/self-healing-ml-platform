"""
simulator/injection.py
──────────────────────
Failure injection engine for the traffic simulator.

ARCHITECTURE:
  • Each active injection is an `Injection` dataclass holding:
      - Metadata (event_type, feature, parameters, db_id, started_at)
      - A `_transform` callable: dict → dict  (pure function, no side effects)
  • `InjectionState` holds the ordered list of active injections behind a
    threading.Lock so the main loop and stdin-command thread can both access it.
  • `apply(row)` runs all transforms in insertion order → injections are
    CUMULATIVE (two numeric drifts on the same feature stack additively).
  • `stop_injection(db, state, event_type, feature)` removes the injection
    and writes `ended_at` to simulation_events → REVERSIBLE.

INJECTION MODES:
  drift_feature     — adds `magnitude` to a numeric feature (additive, cumulative)
  drift_category    — replaces a categorical feature with `value` with probability
                      `skew_prob` (default 0.8); valid category, no 422
  label_delay_spike — does NOT modify the row; sets a multiplier on outcome_delay
  corrupt_feature   — sets a field to None (null) → API returns 422 (by design)

DESIGN DECISION on corrupt_feature:
  We send None for the field rather than an out-of-range numeric value.
  Reason: None gives a clean, deterministic Pydantic "field required" 422.
  An out-of-range value (e.g., tenure=-999) also works for numeric fields
  but is ambiguous for categoricals. None is type-agnostic and unambiguous.
  The simulator handles the 422 response gracefully and logs it as expected.

DESIGN DECISION on cumulative drift:
  Two drift_feature injections on the same feature accumulate because
  `apply()` runs all transforms in order. This lets you model "slow creep":
  add +5 drift, observe, add another +5, observe, etc.
  If you want to REPLACE the drift value instead of accumulate, stop the
  first injection before adding the second.
"""

from __future__ import annotations

import logging
import random
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

log = logging.getLogger(__name__)

# Valid event type strings (enforced at injection creation time)
VALID_EVENT_TYPES = frozenset(
    {"drift_feature", "drift_category", "label_delay_spike", "corrupt_feature"}
)


# ---------------------------------------------------------------------------
# Transform factories (pure functions — no side effects, fully testable)
# ---------------------------------------------------------------------------
def _make_numeric_drift(feature: str, magnitude: float) -> Callable[[dict], dict]:
    """
    Add `magnitude` to `feature` in a feature dict.
    If the field is None (already corrupted), leave it as-is.
    """

    def transform(row: dict) -> dict:
        if feature in row and row[feature] is not None:
            try:
                row[feature] = float(row[feature]) + magnitude
            except (TypeError, ValueError):
                pass  # non-numeric value; leave unchanged
        return row

    return transform


def _make_category_skew(
    feature: str, target_value: str, skew_prob: float = 0.8
) -> Callable[[dict], dict]:
    """
    With probability `skew_prob`, replace `feature` with `target_value`.
    `target_value` MUST be a valid category (to avoid triggering 422).
    This shifts the marginal distribution without making every row identical.
    """

    def transform(row: dict) -> dict:
        if feature in row:
            if random.random() < skew_prob:
                row[feature] = target_value
        return row

    return transform


def _make_corrupt_feature(feature: str) -> Callable[[dict], dict]:
    """
    Set `feature` to None (null). The API will return 422 because the
    Pydantic schema marks all fields as required. This is intentional —
    corrupt_feature tests the API's input validation path, not the model.
    The simulator handles the 422 gracefully and does NOT queue an outcome.
    """

    def transform(row: dict) -> dict:
        row[feature] = None
        return row

    return transform


def _make_label_delay_spike(_multiplier: float) -> Callable[[dict], dict]:
    """
    Label delay spike does NOT modify the feature row.
    The multiplier is stored in Injection.parameters and read by the
    outcome writer when computing write_at_time. This transform is a no-op
    on the row — it exists so the injection participates in InjectionState
    like all other injection types.
    """

    def transform(row: dict) -> dict:
        return row  # no-op; multiplier applied by outcome_writer

    return transform


# ---------------------------------------------------------------------------
# Injection dataclass
# ---------------------------------------------------------------------------
@dataclass
class Injection:
    """
    A single active injection.

    Fields:
      event_type  — one of VALID_EVENT_TYPES
      feature     — name of the affected feature (or "" for label_delay_spike)
      parameters  — raw parameters dict stored in simulation_events
      db_id       — UUID of the simulation_events row for this injection
      started_at  — wall-clock time injection started
      _transform  — pure function applied per row by InjectionState.apply()
    """

    event_type: str
    feature: str
    parameters: dict
    db_id: str
    started_at: datetime
    _transform: Callable[[dict], dict] = field(repr=False)

    def apply(self, row: dict) -> dict:
        """Apply this injection's transform to a copy of `row`."""
        return self._transform(row)

    def __repr__(self) -> str:
        return (
            f"Injection(type={self.event_type!r}, feature={self.feature!r}, "
            f"params={self.parameters}, db_id={self.db_id!r})"
        )


# ---------------------------------------------------------------------------
# InjectionState — thread-safe container for active injections
# ---------------------------------------------------------------------------
class InjectionState:
    """
    Thread-safe container for all currently active injections.

    Usage:
        state = InjectionState()
        inj = state.add_numeric_drift(db, "tenure", 20.0)
        modified_row = state.apply(row)
        state.remove(db, inj)  # logs ended_at, stops drift
    """

    def __init__(self) -> None:
        self._injections: list[Injection] = []
        self._lock = threading.Lock()

    # ── Core apply ──────────────────────────────────────────────────────────

    def apply(self, row: dict) -> dict:
        """
        Apply all active injections to `row` in insertion order.
        Makes a shallow copy before the first transform so the original
        row dict is never mutated.
        CUMULATIVE: transforms stack in order (two numeric drifts add up).
        """
        with self._lock:
            active = list(self._injections)

        if not active:
            return dict(row)  # always return a copy

        result = dict(row)
        for inj in active:
            result = inj.apply(result)
        return result

    # ── Inspection ──────────────────────────────────────────────────────────

    def active_count(self) -> int:
        with self._lock:
            return len(self._injections)

    def active_types(self) -> list[str]:
        with self._lock:
            return [i.event_type for i in self._injections]

    @property
    def has_label_delay_spike(self) -> bool:
        with self._lock:
            return any(i.event_type == "label_delay_spike" for i in self._injections)

    @property
    def delay_multiplier(self) -> float:
        """Return the first label_delay_spike multiplier, or 1.0 if none active."""
        with self._lock:
            for i in self._injections:
                if i.event_type == "label_delay_spike":
                    return float(i.parameters.get("multiplier", 10.0))
        return 1.0

    @property
    def has_corrupt_feature(self) -> bool:
        with self._lock:
            return any(i.event_type == "corrupt_feature" for i in self._injections)

    def list_injections(self) -> list[Injection]:
        with self._lock:
            return list(self._injections)

    # ── Add injections ───────────────────────────────────────────────────────

    def _add(self, injection: Injection) -> Injection:
        with self._lock:
            self._injections.append(injection)
        log.info("Injection STARTED: %s", injection)
        return injection

    def add_numeric_drift(
        self,
        db,
        feature: str,
        magnitude: float,
    ) -> Injection:
        """
        Shift a numeric feature by `magnitude` for all subsequent requests.
        Cumulative: adding two drifts on the same feature stacks them.
        """
        params = {"feature": feature, "magnitude": magnitude}
        inj = Injection(
            event_type="drift_feature",
            feature=feature,
            parameters=params,
            db_id=_log_event_start(db, "drift_feature", params),
            started_at=datetime.now(timezone.utc),
            _transform=_make_numeric_drift(feature, magnitude),
        )
        return self._add(inj)

    def add_category_skew(
        self,
        db,
        feature: str,
        target_value: str,
        skew_prob: float = 0.8,
    ) -> Injection:
        """
        Over-represent `target_value` in `feature` with probability `skew_prob`.
        `target_value` must be a valid category to avoid triggering 422.
        """
        params = {
            "feature": feature,
            "target_value": target_value,
            "skew_prob": skew_prob,
        }
        inj = Injection(
            event_type="drift_category",
            feature=feature,
            parameters=params,
            db_id=_log_event_start(db, "drift_category", params),
            started_at=datetime.now(timezone.utc),
            _transform=_make_category_skew(feature, target_value, skew_prob),
        )
        return self._add(inj)

    def add_label_delay_spike(
        self,
        db,
        multiplier: float = 10.0,
    ) -> Injection:
        """
        Multiply the outcome delay by `multiplier` for all subsequent rows.
        Does not modify feature rows — affects only the outcome writer.
        """
        params = {"multiplier": multiplier}
        inj = Injection(
            event_type="label_delay_spike",
            feature="",
            parameters=params,
            db_id=_log_event_start(db, "label_delay_spike", params),
            started_at=datetime.now(timezone.utc),
            _transform=_make_label_delay_spike(multiplier),
        )
        return self._add(inj)

    def add_corrupt_feature(self, db, feature: str) -> Injection:
        """
        Send None (null) for `feature` → API returns 422 (expected).
        The simulator logs 422 as expected behavior and skips outcome queuing.
        """
        params = {"feature": feature, "corruption_type": "null"}
        inj = Injection(
            event_type="corrupt_feature",
            feature=feature,
            parameters=params,
            db_id=_log_event_start(db, "corrupt_feature", params),
            started_at=datetime.now(timezone.utc),
            _transform=_make_corrupt_feature(feature),
        )
        return self._add(inj)

    # ── Remove / stop ────────────────────────────────────────────────────────

    def remove(self, db, injection: Injection) -> bool:
        """
        Stop an injection: remove from active list and write ended_at to DB.
        Returns True if found and removed, False if already gone.
        """
        with self._lock:
            try:
                self._injections.remove(injection)
            except ValueError:
                return False

        _log_event_stop(db, injection.db_id)
        log.info("Injection STOPPED: %s", injection)
        return True

    def remove_by_type_and_feature(self, db, event_type: str, feature: str) -> bool:
        """
        Convenience: stop the first injection matching (event_type, feature).
        Used by stdin command handler.
        """
        with self._lock:
            for inj in self._injections:
                if inj.event_type == event_type and inj.feature == feature:
                    self._injections.remove(inj)
                    _log_event_stop(db, inj.db_id)
                    log.info("Injection STOPPED via command: %s", inj)
                    return True
        return False

    def stop_all(self, db) -> int:
        """Stop all active injections. Returns count stopped."""
        with self._lock:
            to_stop = list(self._injections)
            self._injections.clear()

        for inj in to_stop:
            _log_event_stop(db, inj.db_id)
            log.info("Injection STOPPED (stop_all): %s", inj)

        return len(to_stop)


# ---------------------------------------------------------------------------
# DB helpers (write simulation_events rows)
# ---------------------------------------------------------------------------
def _log_event_start(db, event_type: str, parameters: dict) -> str:
    """
    Insert a simulation_events row for a new injection.
    Returns the new row's UUID (stored as Injection.db_id).
    If db is None (e.g., tests without DB), returns a fresh UUID without writing.
    """
    new_id = str(uuid.uuid4())

    if db is None:
        return new_id

    try:
        from simulator.db_models import SimulationEvent

        event = SimulationEvent(
            id=new_id,
            event_type=event_type,
            started_at=datetime.now(timezone.utc),
            ended_at=None,
            parameters=parameters,
        )
        db.add(event)
        db.commit()
    except Exception:
        log.exception("Failed to log injection start to DB — continuing")
        try:
            db.rollback()
        except Exception:
            pass

    return new_id


def _log_event_stop(db, db_id: str) -> None:
    """Write ended_at to the simulation_events row identified by db_id."""
    if db is None:
        return

    try:
        from simulator.db_models import SimulationEvent

        event = db.get(SimulationEvent, db_id)
        if event:
            event.ended_at = datetime.now(timezone.utc)
            db.commit()
    except Exception:
        log.exception("Failed to log injection stop to DB — continuing")
        try:
            db.rollback()
        except Exception:
            pass
