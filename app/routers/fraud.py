"""
app/routers/fraud.py
────────────────────
Router for Tenant #2: Fraud Detection Model endpoint (POST /predict/fraud-model).

Reuses the same Postgres `predictions` table — filtered by model_id.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app import model_loader
from app.database import get_db
from app.db_models import Prediction
from app.schemas import FraudInput, FraudPrediction

log = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["predictions"])


@router.post(
    "/fraud-model",
    response_model=FraudPrediction,
    summary="Predict transaction fraud (Tenant #2)",
    description="Run the fraud detection model against a single transaction vector.",
    status_code=status.HTTP_200_OK,
)
def predict_fraud(
    payload: FraudInput,
    db: Session = Depends(get_db),
) -> FraudPrediction:
    """Predict transaction fraud for Tenant #2."""
    try:
        pipeline, model_id = model_loader.get_pipeline_for_tenant("fraud-model")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Fraud model not loaded: {exc}",
        )

    X = payload.to_dataframe()

    try:
        pred_int = int(pipeline.predict(X)[0])
        prob_array = pipeline.predict_proba(X)[0]
        confidence = float(prob_array[pred_int])
    except Exception as exc:
        log.exception("Inference failed for fraud-model model_id=%s", model_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model inference error: {exc}",
        )

    prediction_bool = bool(pred_int)

    # Log to shared predictions table with model_id="fraud-model:v1"
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
        log.exception("Failed to log fraud prediction to DB — continuing")
        db.rollback()

    log.info(
        "fraud_prediction=%s confidence=%.4f model_id=%s",
        prediction_bool,
        confidence,
        model_id,
    )

    return FraudPrediction(
        prediction=prediction_bool,
        prediction_label="Fraud" if prediction_bool else "Legitimate",
        confidence=confidence,
        model_id=model_id,
    )
