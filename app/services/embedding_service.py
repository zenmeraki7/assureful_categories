"""
app/services/embedding_service.py

This service orchestrates the generation and caching of category
embeddings using the ModelEnsemble.
"""

import logging
import numpy as np
import asyncio
from pathlib import Path
from app.core.runtime import RuntimeConfig
from app.ml.model_ensemble import ModelEnsemble
from app.utils.prometheus_utils import (
    EMBEDDING_LATENCY, 
    INDEX_REBUILD_LATENCY,
    REQUEST_LATENCY 
)

logger = logging.getLogger(__name__)

class ClassificationError(Exception):
    """Custom exception for errors during the classification process."""
    pass

class EmbeddingService:
    """
    A service for managing the lifecycle of category embeddings,
    including generation, caching, and loading.
    """
    def __init__(self, cfg: RuntimeConfig):
        self.cfg = cfg
        self.embedding_file: Path = cfg.embedding_file
        self.model_ensemble = ModelEnsemble(
            models_config=cfg.models,
            weights_config=cfg.model_weights,
            device=cfg.device,
            batch_size=cfg.batch_size
        )
        
    def load_models(self) -> None:
        """
        Synchronous method to load models.
        Called once at application startup.
        """
        self.model_ensemble.load_models()

    def get_embedding_dim(self) -> int:
        """Helper to get the final embedding dimension from the ensemble."""
        return self.model_ensemble.get_embedding_dim()

    async def get_category_embeddings(self) -> np.ndarray:
        """
        Main orchestration method.
        Tries to load embeddings from cache first. If cache is
        missing or disabled, it generates new ones.
        """
        if self.cfg.cache_embeddings:
            embeddings = await self.load_from_cache()
            if embeddings is not None:
                logger.info(
                    f"Loaded {embeddings.shape[0]:,} embeddings from cache."
                )
                return embeddings
            
            logger.info(
                f"Embedding cache file not found: {self.embedding_file.name}"
            )
        
        logger.warning(
            "Cache miss or caching disabled. "
            "Generating new category embeddings..."
        )
        
        # Get texts from the DataFrame loaded at startup
        texts = self.cfg.category_df['enhanced_text'].tolist()
        
        # Generate new embeddings
        embeddings = await self.generate_embeddings(texts)
        
        # Save to cache if enabled
        if self.cfg.cache_embeddings:
            await self.save_to_cache(embeddings)
            
        return embeddings

    @REQUEST_LATENCY.labels(endpoint="/classify", stage="embedding").time()
    async def generate_single_embedding(self, text: str) -> np.ndarray:
        """
        Generates an embedding for a single query string.
        """
        loop = asyncio.get_running_loop()
        
        try:
            # Run the blocking encode task in an executor
            embeddings = await loop.run_in_executor(
                None,  # Default ThreadPoolExecutor
                self.model_ensemble.encode,
                [text],  # Pass text as a list
                False  # show_progress=False
            )
            
            # Return the first (and only) embedding
            if embeddings.size == 0:
                 raise ClassificationError("Embedding generation returned empty array.")
            return embeddings[0]
            
        except Exception as e:
            logger.error(f"Error during single embedding generation: {e}", exc_info=True)
            raise ClassificationError("Embedding generation failed") from e

    async def load_from_cache(self) -> np.ndarray | None:
        """
        Asynchronously loads embeddings from a .npy file.
        """
        if not self.embedding_file.exists():
            return None
        
        logger.info(f"Loading embeddings from cache: {self.embedding_file}...")
        try:
            loop = asyncio.get_running_loop()
            embeddings = await loop.run_in_executor(
                None, np.load, self.embedding_file
            )
            return embeddings
        except Exception as e:
            logger.error(
                f"Error loading embedding cache file: {e}. "
                "Will regenerate.",
                exc_info=True
            )
            return None

    @INDEX_REBUILD_LATENCY.time()
    async def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Asynchronously generates embeddings by running the
        blocking ModelEnsemble.encode method in an executor.
        """
        loop = asyncio.get_running_loop()
        
        logger.info(
            f"Starting async embedding generation for {len(texts):,} texts..."
        )
        
        with EMBEDDING_LATENCY.time():
            embeddings = await loop.run_in_executor(
                None,
                self.model_ensemble.encode,
                texts,
                True  # show_progress
            )
        
        logger.info(
            f"Async embedding generation complete. Shape: {embeddings.shape}"
        )
        return embeddings

    async def save_to_cache(self, embeddings: np.ndarray) -> None:
        """
Async-friendly save method
        """
        logger.info(f"Saving {embeddings.shape[0]:,} embeddings to cache: {self.embedding_file}")
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, np.save, self.embedding_file, embeddings
            )
            logger.info("Embeddings saved to cache successfully.")
        except Exception as e:
            logger.error(f"Failed to save embedding cache: {e}", exc_info=True)