"""
app/ml/model_ensemble.py

Manages the loading and execution of the multi-model ensemble
using a weighted-average strategy.
"""

import logging
import numpy as np
import faiss  # Using faiss for fast L2 normalization
from sentence_transformers import SentenceTransformer
from app.config import Models, ModelWeights

logger = logging.getLogger(__name__)

class ModelEnsemble:
    """
    Loads and manages multiple SentenceTransformer models for
    weighted average encoding.
    """
    def __init__(
        self,
        models_config: Models,
        weights_config: ModelWeights,
        device: str,
        batch_size: int
    ):
        self.models_config = models_config.model_dump()
        self.weights_config = weights_config.model_dump()
        self.device = device
        self.batch_size = batch_size
        self.models: dict[str, SentenceTransformer] = {}
        self.target_dim: int = 0  # The max dimension for padding

    def load_models(self) -> None:
        """
        Loads all models specified in the configuration into memory.
        Calculates the target dimension for alignment.
        
        This is a heavy, one-time operation called at startup.
        """
        logger.info("--- 🤖 Loading Model Ensemble ---")
        if not self.models_config:
            raise ValueError("No models configured in settings.MODELS")

        loaded_model_dims = {}

        for key, model_name in self.models_config.items():
            weight = self.weights_config.get(key, 0.0)
            logger.info(
                f"Loading model '{key}' (weight: {weight:.2f}): "
                f"{model_name} on device '{self.device}'..."
            )

            if weight == 0.0:
                logger.warning(f"Skipping model '{key}' as weight is 0.")
                continue

            try:
                model = SentenceTransformer(model_name, device=self.device)
                model.eval()  # Set to evaluation mode
                self.models[key] = model
                loaded_model_dims[key] = model.get_sentence_embedding_dimension()
            except Exception as e:
                logger.critical(
                    f"Failed to load model {model_name}: {e}", exc_info=True
                )
                raise

        if not self.models:
            logger.warning("No models were loaded (all weights might be 0).")
            return

        dims = list(loaded_model_dims.values())
        self.target_dim = max(dims)
        
        if len(set(dims)) > 1:
            logger.warning(
                f"Models have different dimensions: {loaded_model_dims}. "
                f"All vectors will be padded with zeros to {self.target_dim}."
            )
        else:
            logger.info(f"All models loaded. Target dimension: {self.target_dim}")
            
        logger.info("--- ✅ Model Ensemble Loaded Successfully ---")

    def get_embedding_dim(self) -> int:
        """Returns the final dimension of the ensemble's output vector."""
        if self.target_dim == 0:
            logger.warning("Models not loaded or all weights are zero.")
            # Try to get dim from config
            try:
                any_model_name = next(iter(self.models_config.values()))
                return SentenceTransformer(any_model_name).get_sentence_embedding_dimension()
            except:
                 raise RuntimeError("Could not determine embedding dimension.")
        return self.target_dim

    def encode(
        self,
        texts: list[str],
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encodes a list of texts using the weighted-average ensemble.
        
        This is a BLOCKING, CPU/GPU-bound method. It should be
        run in an asyncio executor.
        """
        if not self.models or self.target_dim == 0:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        # Initialize an empty array for the weighted sum
        weighted_sum_embeddings = np.zeros(
            (len(texts), self.target_dim), 
            dtype=np.float32
        )
        
        for key, model in self.models.items():
            weight = self.weights_config[key]
            
            if show_progress:
                logger.info(f"Encoding with model '{key}' (weight: {weight})...")
            
            # 1. Encode and normalize this model's output
            embeddings = model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=True,  # Normalize individual model output
                convert_to_numpy=True
            )
            
            # 2. Apply weight
            weighted_embeddings = embeddings * weight
            
            # 3. Add to the sum, padding if necessary
            current_dim = embeddings.shape[1]
            weighted_sum_embeddings[:, :current_dim] += weighted_embeddings
            
        # 4. Final normalization
        faiss.normalize_L2(weighted_sum_embeddings)
        
        if show_progress:
            logger.info(
                f"Ensemble encoding complete. "
                f"Final shape: {weighted_sum_embeddings.shape}"
            )
        
        return weighted_sum_embeddings