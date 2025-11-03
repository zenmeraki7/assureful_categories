"""
app/ml/cross_encoder_service.py

This service loads and manages the cross-encoder model
for the Stage 2 re-ranking process.
"""

import logging
import numpy as np
import asyncio
from sentence_transformers.cross_encoder import CrossEncoder
from app.core.runtime import RuntimeConfig
from app.utils.prometheus_utils import REQUEST_LATENCY

logger = logging.getLogger(__name__)

class CrossEncoderService:
    """
    A service that wraps the CrossEncoder model for
    asynchronous, non-blocking scoring.
    """
    
    def __init__(self, cfg: RuntimeConfig):
        """
        Loads the cross-encoder model into memory on startup.
        This is a blocking, one-time operation.
        """
        self.model_name = cfg.cross_encoder_model_name
        self.device = cfg.device
        logger.info(
            f"--- 🤖 Loading Cross-Encoder Model: {self.model_name} ---"
        )
        try:
            # max_length is important for performance
            self.model = CrossEncoder(
                self.model_name, 
                device=self.device,
                max_length=512
            )
            logger.info("--- ✅ Cross-Encoder Model Loaded ---")
        except Exception as e:
            logger.critical(
                f"FATAL: Failed to load cross-encoder model: {e}", 
                exc_info=True
            )
            raise
            
    @REQUEST_LATENCY.labels(endpoint="/classify", stage="rerank_score").time()
    async def score(self, query: str, documents: list[str]) -> np.ndarray:
        """
        Asynchronously predicts similarity scores for a query
        and a list of documents.
        
        Args:
            query: The single product query string.
            documents: A list of candidate category strings.
            
        Returns:
            A 1D numpy array of scores (0.0 to 1.0),
            one for each document.
        """
        if not documents:
            return np.array([])
            
        # Create the pairs for the model
        model_input = [(query, doc) for doc in documents]
        
        loop = asyncio.get_running_loop()
        
        try:
            # self.model.predict is a blocking, CPU/GPU-bound operation
            scores = await loop.run_in_executor(
                None,  # Default ThreadPoolExecutor
                self.model.predict,
                model_input,
                {"show_progress_bar": False}
            )
            return scores
            
        except Exception as e:
            logger.error(
                f"Cross-encoder scoring failed: {e}", exc_info=True
            )
            return np.array([])