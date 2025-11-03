"""
app/api/v1/endpoints/health.py

API endpoint for liveness and readiness probes.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from app.core.active_indices import get_active_indices, IndexNotReadyError
from app.services.embedding_service import EmbeddingService
from app.dependencies import get_embedding_service


router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    services: dict


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Get system health status"
)
async def health_check(
    embed_svc: EmbeddingService = Depends(get_embedding_service)
):
    """
    Performs a health check on all critical services.
    """
    services = {
        "embeddings": "OK",
        "cross_encoder": "OK",
        "faiss_index": "OK"
    }
    overall_status = "OK"

    # Check Embedding Models (via EmbeddingService)
    if not embed_svc.model_ensemble.models:
        services["embeddings"] = "DEGRADED: Models not loaded."
        overall_status = "DEGRADED"

    # Check Faiss Index (via ActiveIndices getter)
    try:
        indices = get_active_indices()
        if indices.total_vectors == 0:
            services["faiss_index"] = "DEGRADED: Index is empty."
            overall_status = "DEGRADED"
    except IndexNotReadyError:
        services["faiss_index"] = "DEGRADED: Index not ready."
        overall_status = "DEGRADED"

    response = HealthResponse(status=overall_status, services=services)
    
    if overall_status == "DEGRADED":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=response.model_dump()
        )
        
    return response
