# api/routes_classify.py
"""
Classification Routes API.

Defines FastAPI endpoints for text classification using a classifier
service (local or API-backed). 

Endpoint:
    - POST `/api/classify/` → Classify input text and return top predictions.
"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from schemas.classify import ClassificationRequest, ClassificationResponse
from services.classifier_service import ClassifierService
from core.dependencies import get_classifier

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/classify", tags=['classify'])


@router.post("/", response_model=ClassificationResponse)
async def classify(
    request: ClassificationRequest,
    classifier: ClassifierService = Depends(get_classifier),
):
    """
    Classify input text and return top predictions.

    Args:
        request (ClassificationRequest): Input payload containing text to classify
            and the number of predictions to return (`top_k`).
        classifier (ClassifierService): Dependency-injected classifier service.

    Returns:
        ClassificationResponse: Object containing the list of top predictions.

    Raises:
        HTTPException: 500 if the classifier fails to produce predictions.
    """
    try:
        prediction = await classifier.predict(request.text, top_k = request.top_k)
        logger.info(f"Predictions: {prediction}")
        return ClassificationResponse(predictions=prediction)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
