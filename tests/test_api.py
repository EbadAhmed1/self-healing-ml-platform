"""
tests/test_api.py
──────────────────
API contract tests for the serving endpoints.

Tests are grouped by contract guarantee:
  1. Happy path — valid input produces a correctly shaped 200 response
  2. Input validation — malformed input ALWAYS returns 422 (not 500)
  3. Health endpoint — returns 200 and reports model status
"""

from __future__ import annotations


# ===========================================================================
# Happy path
# ===========================================================================
class TestPredictChurnHappyPath:
    def test_valid_input_returns_200(self, client, valid_payload):
        response = client.post("/predict/churn-model", json=valid_payload)
        assert response.status_code == 200, response.text

    def test_response_has_required_fields(self, client, valid_payload):
        response = client.post("/predict/churn-model", json=valid_payload)
        body = response.json()
        assert "prediction" in body
        assert "prediction_label" in body
        assert "confidence" in body
        assert "model_id" in body

    def test_prediction_is_bool(self, client, valid_payload):
        response = client.post("/predict/churn-model", json=valid_payload)
        body = response.json()
        assert isinstance(body["prediction"], bool)

    def test_confidence_is_between_0_and_1(self, client, valid_payload):
        response = client.post("/predict/churn-model", json=valid_payload)
        body = response.json()
        assert 0.0 <= body["confidence"] <= 1.0

    def test_prediction_label_matches_prediction(self, client, valid_payload):
        response = client.post("/predict/churn-model", json=valid_payload)
        body = response.json()
        expected_label = "Churn" if body["prediction"] else "No Churn"
        assert body["prediction_label"] == expected_label

    def test_model_id_is_non_empty_string(self, client, valid_payload):
        response = client.post("/predict/churn-model", json=valid_payload)
        body = response.json()
        assert isinstance(body["model_id"], str)
        assert len(body["model_id"]) > 0


# ===========================================================================
# Input validation — must return 422 (not 500) for all malformed inputs
# ===========================================================================
class TestPredictChurnValidation:
    def test_missing_required_field_returns_422(self, client, valid_payload):
        """Omit 'tenure' — Pydantic must catch this before the model runs."""
        bad = {k: v for k, v in valid_payload.items() if k != "tenure"}
        response = client.post("/predict/churn-model", json=bad)
        assert response.status_code == 422, response.text

    def test_wrong_type_for_numeric_returns_422(self, client, valid_payload):
        """Send a string for tenure — must return 422."""
        bad = {**valid_payload, "tenure": "not-a-number"}
        response = client.post("/predict/churn-model", json=bad)
        assert response.status_code == 422, response.text

    def test_invalid_categorical_value_returns_422(self, client, valid_payload):
        """
        Send a category value not in the Literal enum.
        Pydantic's Literal validation should catch this BEFORE the model
        (which would map it to the -1 unknown bucket).
        This is intentional: we reject clearly invalid categories at the API
        boundary; the OrdinalEncoder unknown bucket is for categories that were
        valid at training time but aren't in the current Pydantic Literal list.
        """
        bad = {**valid_payload, "gender": "NonBinary"}  # not in Literal
        response = client.post("/predict/churn-model", json=bad)
        assert response.status_code == 422, response.text

    def test_negative_tenure_returns_422(self, client, valid_payload):
        """tenure has ge=0 constraint."""
        bad = {**valid_payload, "tenure": -1}
        response = client.post("/predict/churn-model", json=bad)
        assert response.status_code == 422, response.text

    def test_negative_monthly_charges_returns_422(self, client, valid_payload):
        bad = {**valid_payload, "MonthlyCharges": -10.0}
        response = client.post("/predict/churn-model", json=bad)
        assert response.status_code == 422, response.text

    def test_empty_body_returns_422(self, client):
        response = client.post("/predict/churn-model", json={})
        assert response.status_code == 422, response.text

    def test_invalid_senior_citizen_value_returns_422(self, client, valid_payload):
        """SeniorCitizen must be 0 or 1, not 2."""
        bad = {**valid_payload, "SeniorCitizen": 2}
        response = client.post("/predict/churn-model", json=bad)
        assert response.status_code == 422, response.text

    def test_missing_multiple_fields_returns_422(self, client, valid_payload):
        """Only send three fields — must still return 422."""
        response = client.post(
            "/predict/churn-model",
            json={"tenure": 12, "MonthlyCharges": 50.0, "TotalCharges": 600.0},
        )
        assert response.status_code == 422, response.text

    def test_422_response_body_has_detail(self, client, valid_payload):
        """FastAPI 422 responses must include a 'detail' key."""
        bad = {k: v for k, v in valid_payload.items() if k != "tenure"}
        response = client.post("/predict/churn-model", json=bad)
        assert "detail" in response.json()


# ===========================================================================
# Health endpoint
# ===========================================================================
class TestHealth:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_status_field(self, client):
        body = client.get("/health").json()
        assert "status" in body

    def test_health_model_loaded_true(self, client):
        """The mock pipeline is set in conftest, so model_loaded should be True."""
        body = client.get("/health").json()
        assert body["model_loaded"] is True

    def test_health_model_id_present(self, client):
        body = client.get("/health").json()
        assert body.get("model_id") is not None
