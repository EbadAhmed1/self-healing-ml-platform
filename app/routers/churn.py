"""
app/routers/churn.py
─────────────────────
Router for the churn-model prediction endpoint.

ENDPOINT: POST /predict/churn-model

Namespaced from day one (not a generic /predict) so that adding a second
model (e.g., POST /predict/ltv-model) never requires breaking URL changes.

REQUEST FLOW:
  1. FastAPI + Pydantic validate input → 422 if malformed (before model runs)
  2. input.to_dataframe() builds a single-row DataFrame in training feature order
  3. pipeline.predict() + predict_proba() produce prediction + confidence
  4. Result is logged to Postgres (fire-and-forget within the request)
  5. Response returned to caller
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app import model_loader
from app.database import get_db
from app.db_models import Prediction
from app.schemas import ChurnInput, ChurnPrediction

log = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["predictions"])


@router.post(
    "/churn-model",
    response_model=ChurnPrediction,
    summary="Predict customer churn",
    description=(
        "Run the current production churn model against a single customer's "
        "feature vector. Returns a boolean prediction and a confidence score."
    ),
    status_code=status.HTTP_200_OK,
)
def predict_churn(
    payload: ChurnInput,
    db: Session = Depends(get_db),
) -> ChurnPrediction:
    """
    Predict churn for a single customer.

    Pydantic validates the request body BEFORE this function is called.
    Any missing field or wrong type returns a 422 automatically — no
    defensive try/except needed for input validation here.
    """
    pipeline, model_id = model_loader.get_pipeline_for_request()

    # Convert validated Pydantic model → single-row DataFrame
    X = payload.to_dataframe()

    try:
        pred_int = int(pipeline.predict(X)[0])
        prob_array = pipeline.predict_proba(X)[0]  # [p_no_churn, p_churn]
        confidence = float(prob_array[pred_int])
    except Exception as exc:
        log.exception("Pipeline inference failed for model_id=%s", model_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model inference error: {exc}",
        )

    prediction_bool = bool(pred_int)

    # Log to Postgres
    try:
        import uuid

        record = Prediction(
            id=str(uuid.uuid4()),
            model_id=model_id,
            input_features=payload.model_dump(),
            prediction=prediction_bool,
            confidence=confidence,
        )
        db.add(record)
        db.commit()
    except Exception:
        log.exception("Failed to log prediction to DB — continuing without logging")
        db.rollback()
        # We intentionally do NOT raise here: a DB logging failure should
        # not cause the prediction to fail. The result is still returned.

    log.info(
        "prediction=%s confidence=%.4f model_id=%s",
        prediction_bool,
        confidence,
        model_id,
    )

    return ChurnPrediction(
        prediction=prediction_bool,
        prediction_label="Churn" if prediction_bool else "No Churn",
        confidence=confidence,
        model_id=model_id,
        prediction_id=record.id if "record" in locals() else None,
    )
