"""
app/core/system_setup.py

Handles the application startup logic:
1. Detects hardware (GPU/CPU).
2. Computes optimal batch size.
3. Computes the versioned embedding cache key.
4. Creates all necessary directories.
5. Loads and parses category data.
6. Bundles all settings into a single RuntimeConfig object.
"""

import torch
import hashlib
import logging
from pathlib import Path
from app.config import settings, Models, ModelWeights
from app.core.runtime import RuntimeConfig
from app.core.data_loader import CategoryLoader

# Use the application's root logger
logger = logging.getLogger(__name__)

def _get_optimal_batch_size(device: str) -> int:
    """
    Calculates the optimal batch size based on available hardware.
    """
    if device == 'cuda':
        try:
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            if gpu_mem_gb < 6:
                batch_size = 32  # Small GPU (4-6 GB)
            elif gpu_mem_gb < 12:
                batch_size = 64  # Medium GPU (6-12 GB)
            elif gpu_mem_gb < 24:
                batch_size = 128 # Large GPU (12-24 GB)
            else:
                batch_size = 256 # Very large GPU (24+ GB)
            
            logger.info(f"Detected {gpu_mem_gb:.1f}GB GPU. Setting batch size to {batch_size}.")
            return batch_size
            
        except Exception as e:
            logger.warning(f'Could not detect GPU memory: {e}. Using safe default batch size: 32')
            return 32
    else:
        logger.info("No CUDA device found. Using CPU batch size: 16")
        return 16  # CPU fallback

def _get_embedding_cache_key(models: Models, weights: ModelWeights) -> str:
    """
    Generates a unique cache key based on model configuration.
    This invalidates the cache if models or weights change.
    """
    # Use pydantic's model_dump_json for a stable, sorted string
    model_string = models.model_dump_json(sort_keys=True)
    weight_string = weights.model_dump_json(sort_keys=True)
    config_string = f"{model_string}|{weight_string}"
    
    hash_obj = hashlib.md5(config_string.encode())
    return hash_obj.hexdigest()[:8] # 8-char hash is plenty

def create_runtime_config() -> RuntimeConfig:
    """
    The main setup function. Called once at application startup.
    Generates and returns the immutable RuntimeConfig.
    """
    logger.info("--- Initializing System Runtime ---")

    # 1. Create directories
    logger.info(f"Ensuring data directory exists: {settings.DATA_DIR}")
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensuring cache directory exists: {settings.CACHE_DIR}")
    settings.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Load Category Data
    category_data = CategoryLoader.load(settings.CATEGORIES_JSON)

    # 3. Compute dynamic values
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = _get_optimal_batch_size(device)
    cache_key = _get_embedding_cache_key(settings.MODELS, settings.MODEL_WEIGHTS)
    embedding_file = settings.CACHE_DIR / f'embeddings_{cache_key}.npy'

    # 4. Create the immutable runtime config object
    runtime_cfg = RuntimeConfig(
        # Dynamic values
        device=device,
        batch_size=batch_size,
        embedding_file=embedding_file,
        category_df=category_data.dataframe,
        max_depth=category_data.max_depth,
        
        # Static values
        cache_embeddings=settings.CACHE_EMBEDDINGS,
        categories_json=settings.CATEGORIES_JSON,
        model_weights=settings.MODEL_WEIGHTS,
        models=settings.MODELS,
        thresholds=settings.THRESHOLDS,
        progressive=settings.PROGRESSIVE,
        cross_encoder_model_name=settings.CROSS_ENCODER_MODEL_NAME,
        rerank_retrieval_k=settings.RERANK_RETRIEVAL_K
    )
    
    logger.info("--- System Runtime Initialized Successfully ---")
    
    return runtime_cfg