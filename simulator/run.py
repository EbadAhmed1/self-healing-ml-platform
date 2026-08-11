"""
simulator/run.py
─────────────────
Traffic simulator — main CLI entry point.

USAGE (examples):
  # Normal mode — no injection, one request per second
  python simulator/run.py

  # With numeric drift: shift tenure +20 for all requests
  python simulator/run.py --drift-feature tenure --drift-magnitude 20

  # With categorical skew: over-represent "Fiber optic" for InternetService
  python simulator/run.py --drift-category InternetService --drift-category-value "Fiber optic"

  # Label delay spike: 50x the default delay
  python simulator/run.py --label-delay-spike 50

  # Corrupt a feature (sends null → tests API validation)
  python simulator/run.py --corrupt-feature tenure

  # Fast local test: 0.2s interval, 5s outcome delay
  python simulator/run.py --interval 0.2 --outcome-delay 5

INJECTION IS NEVER ACCIDENTAL:
  All injection flags default to None/False. A request with no injection
  flags sends unmodified data. This is enforced in _build_injection_state().

RUNTIME CONTROL (via stdin while running):
  Type commands into the terminal to control injections live:
    stop drift tenure           — stops numeric drift on tenure
    stop category InternetService — stops category skew
    stop all                    — stops all active injections
    add drift MonthlyCharges 5  — adds drift of +5 to MonthlyCharges
    status                      — prints active injections
    quit                        — graceful shutdown
"""

from __future__ import annotations

import argparse
import logging
import queue
import sys
import threading
import time
from pathlib import Path

import httpx

# ── Make project root importable ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.database import SessionLocal  # noqa: E402
from simulator.injection import InjectionState  # noqa: E402
from simulator.outcome_writer import OutcomeWriter  # noqa: E402
from simulator.split import SIM_DATA_PATH, load_sim_data  # noqa: E402

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PREDICT_ENDPOINT = "/predict/churn-model"
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 1.0  # seconds; doubles each retry

# Telco target column name in the sim CSV
TARGET_COL = "Churn"


# ---------------------------------------------------------------------------
# HTTP client with retry + backoff
# ---------------------------------------------------------------------------
def send_with_retry(
    client: httpx.Client,
    url: str,
    payload: dict,
    max_retries: int = MAX_RETRIES,
) -> httpx.Response | None:
    """
    POST `payload` to `url` with exponential backoff.

    Returns:
        httpx.Response on any HTTP response (including 4xx/5xx)
        None if all retries exhausted (network error / timeout)
    """
    delay = RETRY_BACKOFF_BASE
    for attempt in range(1, max_retries + 1):
        try:
            response = client.post(url, json=payload, timeout=10.0)
            return response
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as exc:
            if attempt == max_retries:
                log.warning(
                    "API unreachable after %d attempts: %s — skipping row",
                    max_retries,
                    exc,
                )
                return None
            log.warning(
                "API error (attempt %d/%d): %s — retrying in %.1fs",
                attempt,
                max_retries,
                exc,
                delay,
            )
            time.sleep(delay)
            delay *= 2.0


# ---------------------------------------------------------------------------
# Stdin command reader (background thread)
# ---------------------------------------------------------------------------
def _stdin_reader_thread(cmd_queue: queue.Queue) -> None:
    """Read lines from stdin and put them onto cmd_queue."""
    try:
        for line in sys.stdin:
            cmd_queue.put(line.strip())
    except Exception:
        pass  # stdin closed or broken pipe


def _process_commands(
    cmd_queue: queue.Queue,
    state: InjectionState,
    db,
) -> bool:
    """
    Drain cmd_queue and execute commands. Returns True if 'quit' received.

    Supported commands:
      stop drift <feature>
      stop category <feature>
      stop all
      add drift <feature> <magnitude>
      status
      quit
    """
    should_quit = False
    while True:
        try:
            cmd = cmd_queue.get_nowait()
        except queue.Empty:
            break

        parts = cmd.lower().split()
        if not parts:
            continue

        try:
            if parts[0] == "quit":
                log.info("Quit command received")
                should_quit = True

            elif parts[0] == "status":
                injections = state.list_injections()
                if injections:
                    for inj in injections:
                        print(f"  ACTIVE: {inj}")
                else:
                    print("  No active injections")

            elif parts[0] == "stop" and len(parts) >= 2:
                if parts[1] == "all":
                    n = state.stop_all(db)
                    print(f"  Stopped {n} injections")
                elif parts[1] == "drift" and len(parts) >= 3:
                    feature = parts[2]
                    ok = state.remove_by_type_and_feature(db, "drift_feature", feature)
                    print(f"  {'Stopped' if ok else 'Not found'}: drift on {feature}")
                elif parts[1] == "category" and len(parts) >= 3:
                    feature = parts[2]
                    ok = state.remove_by_type_and_feature(db, "drift_category", feature)
                    print(
                        f"  {'Stopped' if ok else 'Not found'}: category skew on {feature}"
                    )

            elif parts[0] == "add" and parts[1] == "drift" and len(parts) >= 4:
                feature = parts[2]
                magnitude = float(parts[3])
                state.add_numeric_drift(db, feature, magnitude)
                print(f"  Added drift: {feature} +{magnitude}")

            else:
                print(f"  Unknown command: {cmd!r}")
                print("  Commands: stop drift <f>, stop category <f>, stop all,")
                print("            add drift <f> <mag>, status, quit")

        except Exception as exc:
            log.warning("Command error (%r): %s", cmd, exc)

    return should_quit


# ---------------------------------------------------------------------------
# Injection state builder from CLI args
# ---------------------------------------------------------------------------
def _build_injection_state(args: argparse.Namespace, db) -> InjectionState:
    """
    Build and populate an InjectionState from parsed CLI arguments.
    INJECTION IS NEVER ACCIDENTAL — all flags default to None/False.
    Normal mode (no flags) returns an empty InjectionState.
    """
    state = InjectionState()

    if args.drift_feature:
        if args.drift_magnitude is None:
            log.error("--drift-feature requires --drift-magnitude")
            sys.exit(1)
        state.add_numeric_drift(db, args.drift_feature, args.drift_magnitude)
        log.info(
            "Injection: numeric drift on '%s' magnitude=+%.2f",
            args.drift_feature,
            args.drift_magnitude,
        )

    if args.drift_category:
        target_val = args.drift_category_value
        if target_val is None:
            log.error("--drift-category requires --drift-category-value")
            sys.exit(1)
        state.add_category_skew(db, args.drift_category, target_val, skew_prob=0.8)
        log.info(
            "Injection: category skew on '%s' → '%s' (p=0.80)",
            args.drift_category,
            target_val,
        )

    if args.label_delay_spike is not None:
        state.add_label_delay_spike(db, multiplier=args.label_delay_spike)
        log.info("Injection: label delay spike x%.1f", args.label_delay_spike)

    if args.corrupt_feature:
        state.add_corrupt_feature(db, args.corrupt_feature)
        log.info("Injection: corrupt feature '%s' → null", args.corrupt_feature)

    if state.active_count() == 0:
        log.info("Running in NORMAL MODE — no injections active")

    return state


# ---------------------------------------------------------------------------
# Row → payload converter
# ---------------------------------------------------------------------------
def _row_to_payload(row: dict) -> dict:
    """
    Convert a sim DataFrame row (as dict) to the API payload format.
    Drops customerID and Churn (target) columns.
    TotalCharges is already float (cleaned by load_sim_data).
    """
    skip = {"customerID", "Churn"}
    return {k: v for k, v in row.items() if k not in skip}


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------
def run_simulation(args: argparse.Namespace) -> None:
    api_base = args.api_url.rstrip("/")
    predict_url = f"{api_base}{PREDICT_ENDPOINT}"

    # Load simulation data
    sim_df = load_sim_data(Path(args.data_path))
    rows = sim_df.to_dict(orient="records")
    log.info("Loaded %d simulation rows", len(rows))

    # DB session
    db = SessionLocal()

    # Injection state from CLI args
    state = _build_injection_state(args, db)

    # Outcome writer
    writer = OutcomeWriter(db_session_factory=SessionLocal)
    writer.start()

    # Stdin command thread
    cmd_queue: queue.Queue = queue.Queue()
    stdin_thread = threading.Thread(
        target=_stdin_reader_thread,
        args=(cmd_queue,),
        daemon=True,
        name="stdin-reader",
    )
    stdin_thread.start()

    log.info("═" * 60)
    log.info("Simulator running. Type 'status', 'stop all', 'quit' etc.")
    log.info("  API: %s", predict_url)
    log.info("  Interval: %.2fs", args.interval)
    log.info(
        "  Outcome delay: %.1fs (x%.1f spike)",
        args.outcome_delay,
        state.delay_multiplier,
    )
    log.info("═" * 60)

    total_sent = 0
    total_ok = 0
    total_rejected = 0
    total_failed = 0

    try:
        with httpx.Client() as client:
            idx = 0
            while True:
                # Process stdin commands
                should_quit = _process_commands(cmd_queue, state, db)
                if should_quit:
                    break

                # Cycle through rows (loop indefinitely if --loop)
                if idx >= len(rows):
                    if not args.loop:
                        log.info("Dataset exhausted — exiting")
                        break
                    idx = 0
                    log.info("Dataset cycle complete — restarting from row 0")

                row = rows[idx]
                idx += 1
                total_sent += 1

                # Extract true label before injection (injection may corrupt features)
                raw_churn_value = row.get(TARGET_COL, "No")
                true_label: bool = raw_churn_value == 1 or str(
                    raw_churn_value
                ).strip().lower() in ("yes", "1", "true")

                # Apply all active injections
                payload = state.apply(_row_to_payload(row))

                # Send request
                response = send_with_retry(client, predict_url, payload)

                if response is None:
                    # Network failure after all retries
                    total_failed += 1
                    log.warning(
                        "Row %d skipped (API unreachable) [sent=%d ok=%d rejected=%d failed=%d]",
                        total_sent,
                        total_sent,
                        total_ok,
                        total_rejected,
                        total_failed,
                    )
                elif response.status_code == 200:
                    total_ok += 1
                    body = response.json()
                    prediction_id = None  # extract from response if API returns it

                    # Queue the delayed outcome
                    effective_delay = args.outcome_delay * state.delay_multiplier
                    # We use a synthetic UUID as prediction_id since the serving
                    # API doesn't return its DB row ID in the response.
                    # Phase 3 will correlate by timestamp window instead.
                    import uuid

                    prediction_id = str(uuid.uuid4())  # placeholder — see note below

                    writer.enqueue(
                        prediction_id=prediction_id,
                        true_label=true_label,
                        delay_seconds=effective_delay,
                    )

                    if total_ok % 50 == 0 or total_ok == 1:
                        log.info(
                            "Row %d → prediction=%s confidence=%.3f "
                            "[ok=%d rejected=%d failed=%d]",
                            total_sent,
                            body.get("prediction_label"),
                            body.get("confidence", 0),
                            total_ok,
                            total_rejected,
                            total_failed,
                        )

                elif response.status_code == 422:
                    total_rejected += 1
                    if state.has_corrupt_feature:
                        # 422 is EXPECTED when corrupt_feature is active
                        log.debug(
                            "Row %d → 422 (expected: corrupt_feature active)",
                            total_sent,
                        )
                    else:
                        log.warning(
                            "Row %d → unexpected 422: %s",
                            total_sent,
                            response.text[:200],
                        )
                else:
                    total_failed += 1
                    log.warning(
                        "Row %d → HTTP %d: %s",
                        total_sent,
                        response.status_code,
                        response.text[:200],
                    )

                # Configurable sleep between requests
                time.sleep(args.interval)

    except KeyboardInterrupt:
        log.info("KeyboardInterrupt — stopping gracefully…")
    finally:
        log.info(
            "Simulation ended. sent=%d ok=%d rejected=%d failed=%d",
            total_sent,
            total_ok,
            total_rejected,
            total_failed,
        )
        state.stop_all(db)
        writer.stop()
        db.close()


# NOTE on prediction_id correlation:
# The Phase 1 API does not return the DB row UUID in its response body (it
# returns prediction + confidence + model_id). In Phase 3, we will either:
#   a) Add the row UUID to the API response (one-line change in routers/churn.py)
#   b) Correlate outcomes with predictions using a timestamp window query
# Option (b) works without API changes and is implemented in Phase 3.
# The placeholder UUID above creates an outcomes row that Phase 3 can ignore
# or later link correctly once we add UUID to the API response.


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------
def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Traffic simulator for the self-healing ML platform.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="churn-model",
        choices=["churn-model", "fraud-model"],
        help="Target tenant model name (churn-model or fraud-model).",
    )
    parser.add_argument(
        "--data-path",
        default=str(SIM_DATA_PATH),
        help="Path to the simulation CSV (generated by: python models/train.py --save-sim-split)",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the FastAPI serving app.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between prediction requests.",
    )
    parser.add_argument(
        "--outcome-delay",
        type=float,
        default=300.0,
        help="Base delay (seconds) before writing a true label outcome. Use a small value (e.g. 5) for local testing.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        default=True,
        help="Loop over the dataset indefinitely.",
    )
    parser.add_argument(
        "--no-loop",
        dest="loop",
        action="store_false",
        help="Stop after one pass through the dataset.",
    )

    # ── Injection flags (all default to None/False — never accidental) ─────
    inj = parser.add_argument_group(
        "injection",
        "Failure injection options. All default to OFF. Must be explicitly enabled.",
    )
    inj.add_argument(
        "--drift-feature",
        default=None,
        metavar="FEATURE",
        help="Numeric feature to shift (e.g. tenure, MonthlyCharges).",
    )
    inj.add_argument(
        "--drift-magnitude",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Amount to add to --drift-feature for every request. Required with --drift-feature.",
    )
    inj.add_argument(
        "--drift-category",
        default=None,
        metavar="FEATURE",
        help="Categorical feature whose distribution to skew (e.g. Contract).",
    )
    inj.add_argument(
        "--drift-category-value",
        default=None,
        metavar="VALUE",
        help="Value to over-represent (80%% of rows). Must be a valid category.",
    )
    inj.add_argument(
        "--label-delay-spike",
        type=float,
        default=None,
        metavar="MULTIPLIER",
        help="Multiply outcome-delay by this factor (e.g. 10 → 10× delay).",
    )
    inj.add_argument(
        "--corrupt-feature",
        default=None,
        metavar="FEATURE",
        help="Feature to corrupt (sends null → API returns 422). Tests input validation.",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity.",
    )
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = _make_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    run_simulation(args)
