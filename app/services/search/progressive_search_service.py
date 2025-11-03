"""
app/services/search/progressive_search_service.py

Contains the ProgressiveSearchService, which performs a
hierarchical, level-by-level search.
"""

import logging
import numpy as np
import asyncio
import pandas as pd
import faiss
from typing import Dict, Any, Callable, List, Set
from app.core.runtime import RuntimeConfig
from app.core.active_indices import FaissIndices, get_active_indices, IndexNotReadyError
from app.utils.prometheus_utils import FAISS_SEARCH_LATENCY, REQUEST_LATENCY

logger = logging.getLogger(__name__)

class ProgressiveSearchService:
    """
    Implements the "Progressive" hierarchical search strategy.
    
    Searches L1, finds top candidates, searches their children in L2, etc.
    """
    
    def __init__(
        self,
        cfg: RuntimeConfig,
        indices_getter: Callable[[], FaissIndices] = get_active_indices
    ):
        """
        Args:
            cfg: The application's runtime configuration.
            indices_getter: A callable that returns the active FaissIndices.
        """
        self.get_active_indices = indices_getter
        self.categories_df: pd.DataFrame = cfg.category_df
        self.max_depth: int = cfg.max_depth
        self.min_similarity: float = cfg.progressive.min_similarity
        self.top_k: int = cfg.progressive.top_k_per_level
        
        # We search wider to ensure we find valid children
        self.wide_search_k: int = self.top_k * 20
        
        logger.info(
            f"ProgressiveSearchService initialized. "
            f"(TopK: {self.top_k}, MinSim: {self.min_similarity})"
        )

    @REQUEST_LATENCY.labels(endpoint="/classify", stage="search_progressive").time()
    async def search(
        self, 
        query_text: str, # Added, but unused
        query_embedding: np.ndarray
    ) -> Dict[str, Any] | None:
        """
        Asynchronously searches the hierarchy level by level.
        (Ignores query_text, uses embedding only)
        """
        
        # Ensure embedding is 2D, float32
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)
        query_vector = query_embedding.astype(np.float32)

        current_candidates_set: Set[str] = set()
        best_match = None
        level_history = []
        
        try:
            active_indices = self.get_active_indices()
        except IndexNotReadyError:
            logger.error("ProgressiveSearch: Search failed, indices not ready.")
            return None

        for level in range(1, self.max_depth + 1):
            level_index = active_indices.level_indices.get(level)
            level_data = active_indices.level_data.get(level)
            
            if level_index is None or level_data is None:
                break
            
            if level == 1:
                results = await self._search_full_level(
                    query_vector, level_index, level_data
                )
            else:
                results = await self._search_children_only(
                    query_vector, level, level_index, level_data, current_candidates_set
                )

            if not results['indices']:
                break
                
            best_idx = results['indices'][0]
            similarity = results['similarities'][0]
            
            if similarity < self.min_similarity:
                break
                
            best_match = {'index': best_idx, 'similarity': similarity, 'level': level}
            level_history.append(best_match)
            
            top_k_global_indices = results['indices'][:self.top_k]
            
            level_path_col = f'level_{level}_path'
            if level_path_col not in self.categories_df.columns:
                break # Should not happen if data_loader worked
                
            paths = self.categories_df.loc[
                top_k_global_indices, level_path_col
            ]
            
            current_candidates_set = set(paths.dropna())
            if not current_candidates_set:
                break

        if best_match:
            return {
                'index': best_match['index'],
                'similarity': best_match['similarity'],
                'method': 'progressive',
                'confidence_weight': 0.5, # Medium confidence
                'level_history': level_history,
                'final_level': best_match['level']
            }
        
        logger.warning("ProgressiveSearch: No match found.")
        return None

    @FAISS_SEARCH_LATENCY.time()
    async def _search_full_level(
        self, 
        query_vector: np.ndarray, 
        level_index: faiss.Index, 
        level_data: Dict[str, Any]
    ) -> Dict[str, list]:
        """Searches an entire level index (used for L1)."""
        
        loop = asyncio.get_running_loop()
        distances, local_indices = await loop.run_in_executor(
            None, level_index.search, query_vector, self.top_k
        )
        
        global_indices_map = level_data['indices']
        global_indices = [
            global_indices_map[i] 
            for i in local_indices[0] 
            if i != -1 and i < len(global_indices_map)
        ]
        similarities = [float(d) for d, i in zip(distances[0], local_indices[0]) if i != -1]
        
        return {'indices': global_indices, 'similarities': similarities}

    @FAISS_SEARCH_LATENCY.time()
    async def _search_children_only(
        self, 
        query_vector: np.ndarray, 
        level: int,
        level_index: faiss.Index,
        level_data: Dict[str, Any],
        parent_paths_set: Set[str]
    ) -> Dict[str, list]:
        """
        OPTIMIZED: Searches a full level, then filters results
        to include only children of the parent candidates.
        """
        
        loop = asyncio.get_running_loop()
        
        distances, local_indices = await loop.run_in_executor(
            None, level_index.search, query_vector, self.wide_search_k
        )
        
        local_idx_list = local_indices[0]
        dist_list = distances[0]
        
        global_indices_map = level_data['indices']
        global_indices_to_check = [
            global_indices_map[i] 
            for i in local_idx_list 
            if i != -1 and i < len(global_indices_map)
        ]
        
        if not global_indices_to_check:
            return {'indices': [], 'similarities': []}
            
        def _filter_by_parent():
            """This sync function runs in a thread."""
            parent_col = f'level_{level-1}_path'
            if parent_col not in self.categories_df.columns:
                return [], []
            
            parent_paths_of_results = self.categories_df.loc[
                global_indices_to_check, parent_col
            ]
            
            final_global_indices = []
            final_similarities = []
            
            for i, global_idx in enumerate(global_indices_to_check):
                parent_path = parent_paths_of_results.iloc[i]
                
                if parent_path in parent_paths_set:
                    final_global_indices.append(global_idx)
                    final_similarities.append(float(dist_list[i]))
                    
                    if len(final_global_indices) >= self.top_k:
                        break
                        
            return final_global_indices, final_similarities

        global_indices, similarities = await loop.run_in_executor(
            None, _filter_by_parent
        )
        
        return {'indices': global_indices, 'similarities': similarities}