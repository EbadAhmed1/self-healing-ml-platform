"""
tests/test_notifications.py
────────────────────────────
Unit tests for the Slack / Discord webhook notification dispatcher.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent.db_models import Incident
from agent.notifications import send_slack_alert


class TestSlackNotifications:
    def test_send_slack_alert_skips_when_webhook_empty(self):
        incident = Incident(
            id="1",
            alert_id="alert-1",
            model_id="churn-model:v1",
            hypothesis="Upstream schema drift",
            confidence=0.92,
            recommended_action="monitor",
            status="investigating",
        )
        with patch("agent.notifications.get_settings") as mock_settings:
            mock_settings.return_value.slack_webhook_url = ""
            res = send_slack_alert(incident)
            assert res is False

    def test_send_slack_alert_posts_payload_successfully(self):
        incident = Incident(
            id="42",
            alert_id="alert-42",
            model_id="churn-model:v1",
            hypothesis="Model quality degraded",
            confidence=0.88,
            recommended_action="escalate_to_human",
            status="escalated",
            llm_explanation="PSI score exceeded 0.35 on monthly charges.",
        )
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("agent.notifications.get_settings") as mock_settings, patch(
            "httpx.post", return_value=mock_response
        ) as mock_post:
            mock_settings.return_value.slack_webhook_url = (
                "https://hooks.slack.com/services/TEST/WEBHOOK"
            )
            mock_settings.return_value.api_base_url = "http://localhost:8000"

            res = send_slack_alert(incident)

            assert res is True
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            payload = call_args.kwargs["json"]
            assert "Incident #42" in payload["text"]
            assert payload["blocks"][0]["type"] == "header"

    def test_send_slack_alert_handles_network_failure_gracefully(self):
        incident = Incident(
            id="99",
            alert_id="alert-99",
            model_id="fraud-model:v1",
            hypothesis="Drift detected",
            confidence=0.75,
            recommended_action="monitor",
            status="investigating",
        )

        with patch("agent.notifications.get_settings") as mock_settings, patch(
            "httpx.post", side_effect=Exception("Slack network timeout")
        ):
            mock_settings.return_value.slack_webhook_url = (
                "https://hooks.slack.com/services/TEST/WEBHOOK"
            )

            res = send_slack_alert(incident)
            assert res is False
