"""
agent/dedup.py
──────────────
Alert Deduplication Helper.

DEDUPLICATION LOGIC & RATIONALE:
  ---------------------------------------------------------------------------
  When a model experiences an anomaly, multiple monitoring checks (e.g. 16
  different feature drift checks, prediction drift check, or consecutive batch
  accuracy checks) often trigger multiple `Alert` rows in rapid succession.

  Creating a separate `Incident` ticket for every individual alert row within
  a short window would flood the incident log and cause race conditions in
  automated remediations (e.g. attempting multiple rollbacks for 1 issue).

  Deduplication Strategy:
    1. Look back `DEDUP_WINDOW_MINUTES` (15 minutes) for existing `Incident`
       records for the same `model_id`.
    2. If an open, auto_resolved, or recently escalated incident exists within
       this window, mark the new `Alert` as `processed = True`.
    3. Append the new alert's ID to `primary_incident.evidence["deduped_alert_ids"]`
       and log the deduplication.
    4. Return the primary `Incident` without creating a duplicate ticket.
  ---------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from datetime import timedelta, timezone

from sqlalchemy import select

from agent.db_models import Incident
from monitoring.db_models import Alert

log = logging.getLogger("agent.dedup")

DEDUP_WINDOW_MINUTES = 15


def find_duplicate_incident(db, alert: Alert) -> Incident | None:
    """
    Check if an incident was created for `alert.model_id` within the dedup window.

    Returns:
        Existing Incident ORM object if duplicate, or None if novel issue.
    """
    triggered_at = alert.triggered_at
    if triggered_at.tzinfo is None:
        triggered_at = triggered_at.replace(tzinfo=timezone.utc)

    cutoff = triggered_at - timedelta(minutes=DEDUP_WINDOW_MINUTES)

    stmt = (
        select(Incident)
        .where(Incident.model_id == alert.model_id)
        .where(Incident.created_at >= cutoff)
        .order_by(Incident.created_at.desc())
    )
    existing_incident = db.scalars(stmt).first()

    if existing_incident:
        log.info(
            "Alert %s deduplicated against existing Incident %s for %s (within %dm window)",
            alert.id,
            existing_incident.id,
            alert.model_id,
            DEDUP_WINDOW_MINUTES,
        )
        return existing_incident

    return None


def attach_duplicate_alert(db, incident: Incident, alert: Alert) -> None:
    """
    Mark `alert` as processed and append its ID to `incident.evidence["deduped_alert_ids"]`.
    """
    alert.processed = True

    evidence = dict(incident.evidence)
    dedup_ids = evidence.get("deduped_alert_ids", [])
    if alert.id not in dedup_ids:
        dedup_ids.append(alert.id)
    evidence["deduped_alert_ids"] = dedup_ids

    # Update SQLAlchemy JSON field explicitly
    incident.evidence = evidence
    db.commit()
