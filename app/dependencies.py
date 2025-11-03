from app.services.classification_service import ClassificationService
from app.services.feedback_service import FeedbackService
from app.services.embedding_service import EmbeddingService

def get_classification_service():
    return ClassificationService()

def get_feedback_service():
    return FeedbackService()

def get_embedding_service():
    return EmbeddingService()
