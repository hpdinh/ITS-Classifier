# api/routes_health.py
import logging
import os
import datetime
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse

from core.dependencies import get_classifier
from services.classifier_service import ClassifierService
from schemas.debug import ModelInfo, HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix='/api')


@router.get("/health", response_class=HTMLResponse)
async def health(classifier: ClassifierService = Depends(get_classifier)):
    """
    Simple health check:
      - Verifies classifier service is running
      - Runs a dummy prediction to confirm model works
    """
    try:
        test_prediction = await classifier.predict("Healthcheck #### Ticket body")
        if not isinstance(test_prediction, list) or not all(isinstance(p, str) for p in test_prediction):
            raise HTTPException(status_code=503, detail="Prediction malformed")
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Model prediction failed")

    return HealthResponse("ok")


@router.get("/model", response_model=ModelInfo)
async def model_info(classifier: ClassifierService = Depends(get_classifier)):
    """
    Return basic classifier metadata:
      - number of labels
      - last updated timestamp
    """
    try:
        last_updated = datetime.datetime.fromtimestamp(
            os.path.getmtime(classifier.cfg.model_path)
        ).strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        logger.error(f"Error fetching model last updated time: {e}")
        raise HTTPException(status_code=500, detail="Unable to fetch model information")

    return ModelInfo(
        number_of_labels=len(classifier.id2label),
        model_last_updated=last_updated,
    )


@router.get("/debug/headers")
async def debug_headers(request: Request):
    # Only return selected headers
    safe_headers = ["x-forwarded-for", "x-real-ip", "host", "user-agent"]
    return {k: v for k, v in request.headers.items() if k.lower() in safe_headers}