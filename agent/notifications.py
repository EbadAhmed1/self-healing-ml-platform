"""
agent/notifications.py
────────────────────
Slack / Discord Webhook notification dispatcher for the Self-Healing ML Platform.

Posts formatted alert cards to Slack when a new incident ticket is created or updated.

FAILURE SAFETY:
  - Network timeouts or HTTP errors are caught and logged as warnings.
  - Incident creation/remediation workflows NEVER crash if Slack is down.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx

from app.config import get_settings

if TYPE_CHECKING:
    from agent.db_models import Incident

log = logging.getLogger("agent.notifications")


def send_slack_alert(incident: Incident, dashboard_url: str = "") -> bool:
    """
    Send a formatted incident alert message to a Slack / Discord webhook URL.

    Args:
        incident: Incident ORM object containing incident details.
        dashboard_url: Optional override URL for the monitoring dashboard.

    Returns:
        bool: True if alert was sent successfully, False otherwise.
    """
    settings = get_settings()
    webhook_url = settings.slack_webhook_url
    if not webhook_url:
        log.info("SLACK_WEBHOOK_URL is not set — skipping Slack notification.")
        return False

    dash_link = dashboard_url or settings.api_base_url
    explanation_text = incident.llm_explanation or "Auto-resolved by rule engine."

    # Build Slack Block Kit payload
    payload = {
        "text": f"🚨 ML Incident #{incident.id} [{incident.model_id}] — {incident.hypothesis}",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "🚨 Self-Healing ML Platform Alert",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Incident ID:*\n#{incident.id}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Tenant Model:*\n`{incident.model_id}`",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Hypothesis:*\n{incident.hypothesis}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Confidence:*\n`{incident.confidence:.2f}`",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Status:*\n`{incident.status}`",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Action Taken:*\n`{getattr(incident, 'recommended_action', getattr(incident, 'action_taken', 'N/A'))}`",
                    },
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*🤖 LLM Explanation:*\n>{explanation_text}",
                },
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "📊 Open Monitoring Dashboard",
                        },
                        "url": dash_link,
                        "style": "primary",
                    }
                ],
            },
        ],
    }

    try:
        response = httpx.post(webhook_url, json=payload, timeout=5.0)
        if response.status_code in (200, 201, 204):
            log.info("Successfully sent Slack alert for incident #%s", incident.id)
            return True
        else:
            log.warning(
                "Slack webhook returned status %d: %s",
                response.status_code,
                response.text[:200],
            )
            return False
    except Exception as exc:
        log.warning("Failed to send Slack alert for incident #%s: %s", incident.id, exc)
        return False
