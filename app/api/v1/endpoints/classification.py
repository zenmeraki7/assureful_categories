"""
app/api/v1/endpoints/classification.py

The main user-facing API endpoints for classification and feedback.
"""

import logging
from fastapi import (
    APIRouter, 
    Depends, 
    HTTPException, 
    status,
    BackgroundTasks
)
from app.schemas import (
    ProductInput, 
    ClassificationResponse,
    FeedbackInput
)
from app.services.classification_service import (
    ClassificationService,
    ClassificationError
)
from app.services.feedback_service import FeedbackService
from app.dependencies import (
    get_classification_service,
    get_feedback_service
)
from app.utils.prometheus_utils import APP_ERRORS, REQUEST_LATENCY


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/classify",
    response_model=ClassificationResponse,
    summary="Classify a product into a category",
    description="Receives product data and returns the most likely category "
                "using a retrieve-and-rerank ensemble."
)
@REQUEST_LATENCY.labels(endpoint="/classify", stage="full_request").time()
async def handle_classify(
    product: ProductInput,
    classifier: ClassificationService = Depends(get_classification_service)
):
    """
    Handles the end-to-end classification request.
    """
    try:
        response = await classifier.classify_product(product)
        return response
        
    except ClassificationError as e:
        # This is a "clean" error (e.g., "No match found")
        logger.warning(f"Classification failed for '{product.title}': {e}")
        APP_ERRORS.labels(endpoint="/classify", error_type="no_match").inc()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        # This is an "unclean" error (e.g., bug)
        logger.error(
            f"Unhandled error during classification: {e}", 
            exc_info=True
        )
        APP_ERRORS.labels(
            endpoint="/classify", 
            error_type="unhandled_exception"
        ).inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred."
        )


@router.post(
    "/feedback",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit feedback for a classification"
)
async def handle_feedback(
    feedback: FeedbackInput,
    background_tasks: BackgroundTasks,
    feedback_svc: FeedbackService = Depends(get_feedback_service)
):
    """
    Receives classification corrections and logs them in the background.
    """
    background_tasks.add_task(
        feedback_svc.log_feedback,
        feedback.product_input,
        feedback.predicted_path,
        feedback.correct_path
    )
    return {"message": "Feedback received. Thank you!"}
