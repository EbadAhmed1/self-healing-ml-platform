"""
simulator/outcome_writer.py
────────────────────────────
Delayed ground-truth outcome writer.

DESIGN: separate thread + priority queue (heapq)

The main simulation loop queues `PendingOutcome` objects with a `write_at`
monotonic timestamp. The writer thread wakes up, drains the incoming queue
into a min-heap sorted by `write_at`, then writes any items whose
`write_at` has passed.

EDGE CASES handled explicitly:
  1. prediction_id not in DB — possible if API was down and prediction was
     never logged. We skip gracefully and log a WARNING (not an error).
  2. DB error during write — log and continue; the thread never crashes.
  3. Shutdown signal — `stop()` sets an event; thread drains remaining
     queue before exiting (best-effort flush).
  4. Label delay spike — the main loop passes `delay_seconds` which already
     incorporates the multiplier; the writer just uses the value as-is.
"""

from __future__ import annotations

import heapq
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

log = logging.getLogger(__name__)

# How often the writer thread wakes up to check for ready outcomes
# (even if queue is empty, to catch any that slipped through)
_POLL_INTERVAL_S = 0.5


# ---------------------------------------------------------------------------
# PendingOutcome
# ---------------------------------------------------------------------------
@dataclass(order=True)
class PendingOutcome:
    """
    A future outcome waiting to be written.

    `write_at` is a monotonic timestamp (time.monotonic()) so sleep
    calculations are immune to wall-clock jumps.
    """

    write_at: float  # time.monotonic() value — used for heap ordering
    prediction_id: str = field(compare=False)
    true_label: bool = field(compare=False)
    delay_seconds: float = field(compare=False)


# ---------------------------------------------------------------------------
# OutcomeWriter
# ---------------------------------------------------------------------------
class OutcomeWriter:
    """
    Background thread that writes delayed outcomes to the `outcomes` table.

    Usage:
        writer = OutcomeWriter(db_session_factory)
        writer.start()
        writer.enqueue("pred-uuid-123", true_label=True, delay_seconds=300)
        # ... later ...
        writer.stop()
    """

    def __init__(self, db_session_factory) -> None:
        """
        Args:
            db_session_factory: Callable that returns a new SQLAlchemy Session.
                                 The writer uses its own session (not the main
                                 loop's session) to avoid threading issues.
        """
        self._factory = db_session_factory
        self._incoming: queue.Queue[PendingOutcome | None] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="outcome-writer", daemon=True
        )

    def start(self) -> None:
        self._thread.start()
        log.info("OutcomeWriter thread started")

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the thread to stop and wait for it to finish."""
        self._stop_event.set()
        self._incoming.put(None)  # unblock the get() call
        self._thread.join(timeout=timeout)
        log.info("OutcomeWriter thread stopped")

    def enqueue(
        self,
        prediction_id: str,
        true_label: bool,
        delay_seconds: float,
    ) -> None:
        """
        Schedule an outcome to be written after `delay_seconds`.
        Thread-safe: called from the main simulation loop.
        """
        write_at = time.monotonic() + delay_seconds
        self._incoming.put(
            PendingOutcome(
                write_at=write_at,
                prediction_id=prediction_id,
                true_label=true_label,
                delay_seconds=delay_seconds,
            )
        )

    # ── Internal thread ──────────────────────────────────────────────────────

    def _run(self) -> None:
        """Main loop: drain incoming queue into heap, write ready items."""
        heap: list[PendingOutcome] = []

        while not self._stop_event.is_set():
            # 1. Drain incoming queue into heap (non-blocking after first item)
            self._drain_incoming(heap)

            # 2. Write all items whose write_at has passed
            now = time.monotonic()
            while heap and heap[0].write_at <= now:
                pending = heapq.heappop(heap)
                self._write_outcome(pending)

            # 3. Sleep until the next scheduled item (or poll interval)
            if heap:
                sleep_for = min(heap[0].write_at - time.monotonic(), _POLL_INTERVAL_S)
                sleep_for = max(sleep_for, 0.05)  # never busy-loop
            else:
                sleep_for = _POLL_INTERVAL_S

            # Interruptible sleep — checks stop_event every 0.1s
            self._stop_event.wait(timeout=sleep_for)

        # Best-effort flush: write any items that are already overdue
        log.info("OutcomeWriter flushing remaining %d items…", len(heap))
        for pending in sorted(heap):
            self._write_outcome(pending)

    def _drain_incoming(self, heap: list) -> None:
        """Move all items currently in the incoming queue into the heap."""
        try:
            while True:
                item = self._incoming.get_nowait()
                if item is not None:
                    heapq.heappush(heap, item)
        except queue.Empty:
            pass

    def _write_outcome(self, pending: PendingOutcome) -> None:
        """
        Write one outcome row to the database.

        EDGE CASE: If prediction_id is not in the predictions table (API was
        down and never logged it), we log a WARNING and skip. This is not an
        error — it's expected when the API is flaky.
        """
        db = self._factory()
        try:
            from simulator.db_models import Outcome

            outcome = Outcome(
                prediction_id=pending.prediction_id,
                true_label=pending.true_label,
                delay_seconds=pending.delay_seconds,
                observed_at=datetime.now(timezone.utc),
            )
            db.add(outcome)
            db.commit()
            log.debug(
                "Outcome written: prediction_id=%s true_label=%s delay=%.1fs",
                pending.prediction_id,
                pending.true_label,
                pending.delay_seconds,
            )
        except Exception:
            log.exception(
                "Failed to write outcome for prediction_id=%s — skipping",
                pending.prediction_id,
            )
            try:
                db.rollback()
            except Exception:
                pass
        finally:
            db.close()
