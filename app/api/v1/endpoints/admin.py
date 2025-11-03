"""
app/api/v1/endpoints/admin.py

Admin-only endpoints for managing the application.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from app.services.faiss_service import FaissIndexService, get_faiss_service

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post(
    "/admin/rebuild-index",
    status_code=status.HTTP_202_ACCEPTED,
    summary="[Admin] Rebuild FAISS and vocabularies"
)
async def handle_rebuild_index(
    background_tasks: BackgroundTasks,
    # TODO: Add a dependency for API key auth
    faiss_svc: FaissIndexService = Depends(get_faiss_service)
):
    """
    Triggers a full, asynchronous rebuild of the FAISS index cache.
    The server remains operational using the old index during the rebuild.
    """
    logger.warning("Manual index rebuild triggered via API.")
    
    # We ask the service to clear its *own* cache first
    await faiss_svc.clear_cache()
    
    # Then we trigger a rebuild, which will build from scratch
    background_tasks.add_task(faiss_svc.build_or_load_indices)
    
    return {
        "message": "Cache cleared. Index rebuild process started in "
                   "the background."
    }