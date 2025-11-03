"""
app/services/classification_service.py

This service is the main orchestrator for classifying a product.
It replaces the original `Predictor` class.
"""

import logging
import pandas as pd
import numpy as np
import asyncio
from typing import List, Dict, Any, Protocol
from app.core.runtime import RuntimeConfig
from app.services.embedding_service import EmbeddingService
from app.services.text_enhancer_service import TextEnhancerService
from app.schemas import (
    ProductInput,
    ClassificationResponse,
    LevelResult,
    AlternativeResult
)
from app.utils.prometheus_utils import REQUEST_LATENCY

# --- Define Strategy Interfaces (for type hinting) ---
class SearchStrategy(Protocol):
    async def search(
        self, 
        query_text: str, 
        query_embedding: np.ndarray
    ) -> Dict[str, Any] | None:
        ...

class CombiningStrategy(Protocol):
    async def combine(
        self, 
        results: List[Dict[str, Any] | None]
    ) -> Dict[str, Any] | None:
        ...

logger = logging.getLogger(__name__)

class ClassificationError(Exception):
    """Custom exception for errors during the classification process."""
    pass

class ClassificationService:
    """
    Orchestrates the classification process by coordinating embedding,
    multiple search strategies, and response formatting.
    """
    
    def __init__(
        self,
        cfg: RuntimeConfig,
        embedding_service: EmbeddingService,
        text_enhancer_service: TextEnhancerService,
        direct_strategy: SearchStrategy,
        progressive_strategy: SearchStrategy,
        reranking_strategy: SearchStrategy,
        ensemble_strategy: CombiningStrategy
    ):
        """
        All dependencies are injected via the constructor.
        """
        logger.info("Initializing ClassificationService...")
        self.cfg = cfg
        self.embedding_service = embedding_service
        self.text_enhancer_service = text_enhancer_service
        self.direct_strategy = direct_strategy
        self.progressive_strategy = progressive_strategy
        self.reranking_strategy = reranking_strategy
        self.ensemble_strategy = ensemble_strategy
        
        # Keep references from config for quick access
        self.categories_df: pd.DataFrame = cfg.category_df
        self.max_depth: int = cfg.max_depth
        self.thresholds = cfg.thresholds
        self.level_decay: float = cfg.progressive.level_decay

    @REQUEST_LATENCY.labels(endpoint="/classify", stage="orchestration").time()
    async def classify_product(
        self, 
        product: ProductInput
    ) -> ClassificationResponse:
        """
        Main asynchronous method to classify a product.
        """
        
        # 1. Enhance Text
        with REQUEST_LATENCY.labels(endpoint="/classify", stage="text_enhance").time():
            product_text = self.text_enhancer_service.enhance_product_text(product)
            if not product_text:
                raise ClassificationError("No text content found in product.")

        # 2. Generate Embedding (Async)
        # This is timed inside the service
        product_embedding = await self.embedding_service.generate_single_embedding(
            product_text
        )

        # 3. Run Strategies in Parallel (Async)
        with REQUEST_LATENCY.labels(endpoint="/classify", stage="search_all").time():
            direct_task = self.direct_strategy.search(product_text, product_embedding)
            progressive_task = self.progressive_strategy.search(product_text, product_embedding)
            rerank_task = self.reranking_strategy.search(product_text, product_embedding)

            (direct_result, progressive_result, rerank_result) = await asyncio.gather(
                direct_task,
                progressive_task,
                rerank_task
            )

        # 4. Combine Results (Async)
        best_result = await self.ensemble_strategy.combine(
            [direct_result, progressive_result, rerank_result]
        )

        if best_result is None:
            logger.warning(
                f"No prediction could be made for: {product.title}"
            )
            raise ClassificationError("No category match found.")

        # 5. Build and return the Pydantic response
        return self._build_response(best_result, product.title)

    def _build_response(
        self, 
        prediction: Dict[str, Any], 
        product_title: str
    ) -> ClassificationResponse:
        """
        Formats the raw prediction dictionary into a validated
        Pydantic response model.
        """
        try:
            category_idx = int(prediction['index'])
            category = self.categories_df.iloc[category_idx]
            base_similarity = float(prediction['similarity'])
            
            levels = {}
            category_path = str(category.get('Category_path', ''))
            
            if category_path:
                parts = category_path.split('/')
                for i in range(1, len(parts) + 1):
                    level_name = category.get(f'level_{i}')
                    if pd.isna(level_name) or not str(level_name).strip():
                        continue

                    decay_factor = self.level_decay ** (i - 1)
                    similarity = base_similarity * decay_factor
                    
                    levels[f'level_{i}'] = LevelResult(
                        name=str(level_name).strip(),
                        path=str(category.get(f'level_{i}_path', '/'.join(parts[:i]))),
                        similarity=similarity,
                        confidence=self._get_confidence(similarity)
                    )
            
            alternatives = self._format_alternatives(
                prediction.get('alternatives', [])
            )
            
            response = ClassificationResponse(
                product_title=product_title,
                category_id=category.get('Category_ID', ''),
                category_path=category_path,
                similarity=base_similarity,
                confidence=self._get_confidence(base_similarity),
                depth=category.get('depth', 0),
                max_depth=self.max_depth,
                prediction_method=prediction.get('method', 'unknown'),
                levels=levels,
                alternatives=alternatives
            )
            return response
            
        except (KeyError, IndexError, TypeError) as e:
            logger.error(
                f"Failed to build response: {e}", exc_info=True
            )
            raise ClassificationError(f"Failed to format result: {e}") from e

    def _get_confidence(self, similarity: float) -> str:
        """Maps a float similarity score to a confidence string."""
        if similarity >= self.thresholds.high:
            return 'high'
        if similarity >= self.thresholds.medium:
            return 'medium'
        if similarity >= self.thresholds.low:
            return 'low'
        return 'very_low'

    def _format_alternatives(
        self, 
        alternatives: List[Dict[str, Any]]
    ) -> List[AlternativeResult]:
        """Formats the raw alternatives list."""
        formatted = []
        for alt in alternatives[:5]:  # Limit to 5
            try:
                cat_idx = int(alt['index'])
                cat = self.categories_df.iloc[cat_idx]
                sim = float(alt['similarity'])
                
                formatted.append(
                    AlternativeResult(
                        category_id=cat.get('Category_ID', ''),
                        category_path=cat.get('Category_path', ''),
                        similarity=sim,
                        confidence=self._get_confidence(sim),
                        depth=int(cat.get('depth', 0))
                    )
                )
            except Exception:
                logger.warning(f"Could not format alternative: {alt}", exc_info=True)
                
        return formatted