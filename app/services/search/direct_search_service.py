"""
app/services/search/direct_search_service.py

Contains the DirectSearchService, which performs a k-NN search
across the main, "flat" FAISS index.
"""

import logging
import numpy as np
import asyncio
from typing import Dict, Any, Callable
from app.core.runtime import RuntimeConfig
from app.core.active_indices import FaissIndices, get_active_indices, IndexNotReadyError
from app.utils.prometheus_utils import FAISS_SEARCH_LATENCY, REQUEST_LATENCY

logger = logging.getLogger(__name__)

class DirectSearchService:
    """
    Implements the "Direct" search strategy.
    
    This service is initialized with a dependency on the RuntimeConfig
    and the function to retrieve the active indices.
    """
    
    def __init__(
        self,
        cfg: RuntimeConfig,
        indices_getter: Callable[[], FaissIndices] = get_active_indices
    ):
        """
        Args:
            cfg: The application's runtime configuration.
            indices_getter: A callable (like a function) that returns
                            the active FaissIndices object.
        """
        self.top_k = cfg.progressive.top_k_per_level  # Use configured top_k
        self.get_active_indices = indices_getter
        logger.info(
            f"DirectSearchService initialized. Will retrieve top_k={self.top_k}."
        )

    @REQUEST_LATENCY.labels(endpoint="/classify", stage="search_direct").time()
    async def search(
        self, 
        query_text: str, # Added, but unused
        query_embedding: np.ndarray
    ) -> Dict[str, Any] | None:
        """
        Asynchronously searches the main FAISS index.
        (Ignores query_text, uses embedding only)
        """
        try:
            # 1. Get the live, in-memory index
            active_indices = self.get_active_indices()
            main_index = active_indices.main_index
            
            if main_index is None:
                raise IndexNotReadyError("Main index is not loaded.")
                
            # Ensure embedding is 2D, float32
            if query_embedding.ndim == 1:
                query_embedding = np.expand_dims(query_embedding, axis=0)
            
            query_vector = query_embedding.astype(np.float32)

            # 2. Run the blocking search in an executor
            loop = asyncio.get_running_loop()
            
            with FAISS_SEARCH_LATENCY.time():
                distances, indices = await loop.run_in_executor(
                    None,  # Default ThreadPoolExecutor
                    main_index.search,
                    query_vector,
                    self.top_k
                )
            
            # 3. Process results
            if indices.size == 0 or indices[0][0] == -1:
                logger.warning("DirectSearchService: No results found.")
                return None
            
            best_idx = int(indices[0][0])
            best_similarity = float(distances[0][0])
            
            alternatives = [
                {'index': int(indices[0][i]), 'similarity': float(distances[0][i])}
                for i in range(1, len(indices[0]))
                if indices[0][i] != -1
            ]
            
            return {
                'index': best_idx,
                'similarity': best_similarity,
                'method': 'direct',
                'confidence_weight': 0.3,  # Low confidence
                'alternatives': alternatives
            }
            
        except IndexNotReadyError:
            logger.error("DirectSearchService: Search failed, index not ready.")
            return None
        except Exception as e:
            logger.error(
                f"DirectSearchService: Unhandled error during search: {e}", 
                exc_info=True
            )
            return None