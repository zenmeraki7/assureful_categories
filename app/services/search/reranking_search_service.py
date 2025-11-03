"""
app/services/search/reranking_search_service.py

Implements the Two-Stage "Retrieve & Re-rank" search strategy.
"""

import logging
import numpy as np
import asyncio
from typing import Dict, Any, Callable, List
from app.core.runtime import RuntimeConfig
from app.core.active_indices import (
    FaissIndices, 
    get_active_indices, 
    IndexNotReadyError
)
from app.ml.cross_encoder_service import CrossEncoderService
from app.utils.prometheus_utils import FAISS_SEARCH_LATENCY, REQUEST_LATENCY

logger = logging.getLogger(__name__)

class ReRankingSearchService:
    """
    Implements the "Retrieve & Re-rank" strategy.
    """
    
    def __init__(
        self,
        cfg: RuntimeConfig,
        cross_encoder_service: CrossEncoderService,
        indices_getter: Callable[[], FaissIndices] = get_active_indices
    ):
        self.cfg = cfg
        self.cross_encoder_service = cross_encoder_service
        self.get_active_indices = indices_getter
        
        # How many candidates to fetch from FAISS
        self.retrieval_k = cfg.rerank_retrieval_k
        logger.info(
            f"ReRankingSearchService initialized. "
            f"(Retrieval K: {self.retrieval_k})"
        )

    @REQUEST_LATENCY.labels(endpoint="/classify", stage="search_rerank").time()
    async def search(
        self, 
        query_text: str, 
        query_embedding: np.ndarray
    ) -> Dict[str, Any] | None:
        """
        Asynchronously runs the full two-stage pipeline.
        """
        try:
            # --- STAGE 1: RETRIEVE ---
            stage1_indices, stage1_sims = await self._retrieve_candidates(
                query_embedding
            )
            
            if not stage1_indices:
                logger.warning("Re-ranker: Stage 1 (Retrieval) found no candidates.")
                return None

            # --- STAGE 2: RE-RANK ---
            category_texts = self.cfg.category_df.loc[
                stage1_indices, 'enhanced_text'
            ].tolist()

            rerank_scores = await self.cross_encoder_service.score(
                query_text, category_texts
            )

            if rerank_scores.size == 0:
                logger.warning("Re-ranker: Stage 2 (Scoring) failed.")
                return None

            # --- Process Results ---
            best_local_idx = np.argmax(rerank_scores)
            best_global_idx = stage1_indices[best_local_idx]
            best_score = float(rerank_scores[best_local_idx])
            
            scored_indices = sorted(
                zip(rerank_scores, stage1_indices), 
                reverse=True
            )
            
            alternatives = [
                {'index': int(g_idx), 'similarity': float(score)}
                for score, g_idx in scored_indices[1:self.cfg.progressive.top_k_per_level] # Limit alternatives
            ]

            return {
                'index': int(best_global_idx),
                'similarity': best_score,
                'method': 're-rank',
                'confidence_weight': 1.0, # Highest confidence
                'alternatives': alternatives
            }

        except IndexNotReadyError:
            logger.error("Re-ranker: Search failed, index not ready.")
            return None
        except Exception as e:
            logger.error(
                f"Re-ranker: Unhandled error during search: {e}", 
                exc_info=True
            )
            return None

    @FAISS_SEARCH_LATENCY.time()
    async def _retrieve_candidates(
        self, 
        query_embedding: np.ndarray
    ) -> tuple[List[int], List[float]]:
        """Stage 1: Fetches candidate IDs from FAISS."""
        
        active_indices = self.get_active_indices()
        main_index = active_indices.main_index
        
        if main_index is None:
            raise IndexNotReadyError("Main index is not loaded.")
            
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)
        query_vector = query_embedding.astype(np.float32)

        loop = asyncio.get_running_loop()
        distances, indices = await loop.run_in_executor(
            None, main_index.search, query_vector, self.retrieval_k
        )
        
        global_indices = [int(i) for i in indices[0] if i != -1]
        similarities = [float(d) for d in distances[0] if d != -np.inf]
        
        return global_indices, similarities