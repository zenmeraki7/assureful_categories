"""
app/services/feedback_service.py

Handles the logging of user-submitted feedback.
In a production system, this would write to a message queue or
a dedicated "ground_truth" database.
"""

import logging
from app.schemas import ProductInput  # We'll use this schema

# Create a dedicated logger for feedback
feedback_logger = logging.getLogger("feedback_loop")

class FeedbackService:
    """
    Logs feedback for analysis and model retraining.
    """

    def __init__(self):
        logger = logging.getLogger(__name__)
        logger.info("FeedbackService initialized.")
        
    async def log_feedback(
        self,
        product: ProductInput,
        predicted_path: str,
        correct_path: str
    ) -> None:
        """
        Logs the "hard" example to a structured log.
        This is designed to be run as a background task.
        """
        try:
            # Log as a structured JSON object for easy parsing
            feedback_logger.info({
                "type": "user_correction",
                "product_title": product.title,
                "product_vendor": product.vendor,
                "product_description": product.description,
                "predicted_path": predicted_path,
                "correct_path": correct_path,
            })
        except Exception as e:
            # Don't let feedback logging fail the request
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to log feedback: {e}", exc_info=True)