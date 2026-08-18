"""
agent/run_agent.py
───────────────────
Diagnosis Agent Service — Main CLI Runner.

USAGE:
  python agent/run_agent.py [--once] [--interval 10]

WORKFLOW PER UNPROCESSED ALERT:
  1. Fetch unprocessed alerts (`alerts.processed == False`).
  2. Check deduplication (`find_duplicate_incident` within 15m window).
  3. If novel alert:
       a. Gather evidence from DB (deployments, injections, drift, accuracy context).
       b. Formulate root cause hypothesis using priority-ranked rules.
       c. Evaluate confidence-gated remediations (rollback vs escalation).
       d. Execute automated rollback if confidence >= 0.80 and action == 'rollback'.
       e. Record Incident ticket and update alert `processed = True`.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from sqlalchemy import select

# ── Make project root importable when running as a script ──────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.dedup import attach_duplicate_alert, find_duplicate_incident  # noqa: E402
from agent.diagnosis import DiagnosisEngine  # noqa: E402
from agent.evidence import EvidenceGatherer  # noqa: E402
from agent.remediation import RemediationEngine  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from monitoring.db_models import Alert  # noqa: E402

log = logging.getLogger("agent.runner")


def process_unprocessed_alerts(db_session_factory=SessionLocal) -> list:
    """
    Process all unprocessed alerts in the database.

    Returns:
        List of created or updated Incident ORM instances.
    """
    db = db_session_factory()
    incidents = []

    try:
        # Fetch unprocessed alerts
        stmt = (
            select(Alert)
            .where(Alert.processed.is_(False))
            .order_by(Alert.triggered_at.asc())
        )
        unprocessed_alerts = db.scalars(stmt).all()

        if not unprocessed_alerts:
            log.debug("No unprocessed alerts found.")
            return []

        log.info("Found %d unprocessed alert(s) to diagnose.", len(unprocessed_alerts))

        gatherer = EvidenceGatherer(db)
        diagnoser = DiagnosisEngine()
        remediator = RemediationEngine(db)

        for alert in unprocessed_alerts:
            log.info(
                "Processing Alert id=%s model=%s type=%s",
                alert.id,
                alert.model_id,
                alert.alert_type,
            )

            # 1. Deduplication check
            duplicate = find_duplicate_incident(db, alert)
            if duplicate:
                attach_duplicate_alert(db, duplicate, alert)
                incidents.append(duplicate)
                continue

            # 2. Gather Evidence
            evidence = gatherer.gather(alert)

            # 3. Diagnose Root Cause Hypothesis
            diagnosis = diagnoser.diagnose(evidence)

            # 4. Evaluate & Execute Remediation
            incident = remediator.process_incident(
                alert_id=alert.id,
                model_id=alert.model_id,
                diagnosis=diagnosis,
                evidence=evidence,
            )

            # 5. LLM Enrichment Layer for Escalated Incidents (Phase 5)
            if incident.status == "escalated":
                try:
                    from agent.llm import LLMReasoningEngine

                    llm_engine = LLMReasoningEngine()
                    llm_engine.enrich_incident(incident, evidence, db)
                except Exception as exc:
                    log.warning("LLM reasoning enrichment failed gracefully: %s", exc)

            # 6. Send Slack alert notification (Phase 9)
            try:
                from agent.notifications import send_slack_alert

                send_slack_alert(incident)
            except Exception as exc:
                log.warning("Slack notification failed gracefully: %s", exc)

            # 7. Mark alert as processed
            alert.processed = True
            db.commit()

            incidents.append(incident)

        # Expunge incidents so returned objects are safe to access after session close
        for inc in incidents:
            try:
                db.refresh(inc)
                db.expunge(inc)
            except Exception:
                pass

        return incidents

    except Exception:
        log.exception("Error processing alerts in Diagnosis Agent")
        db.rollback()
        raise
    finally:
        db.close()


def run_agent_loop(once: bool = False, interval: float = 10.0) -> None:
    """Run agent loop."""
    log.info("═" * 60)
    log.info(
        "Starting Diagnosis Agent Service (once=%s, interval=%.1fs)", once, interval
    )
    log.info("═" * 60)

    while True:
        try:
            process_unprocessed_alerts()
        except Exception as exc:
            log.error("Error in agent iteration: %s", exc)

        if once:
            log.info("Agent run_once complete. Exiting.")
            break

        time.sleep(interval)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the autonomous rule-based diagnosis agent service.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single pass over unprocessed alerts and exit.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Polling interval in seconds when running continuously.",
    )
    return parser


if __name__ == "__main__":
    parser = _make_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    run_agent_loop(once=args.once, interval=args.interval)
