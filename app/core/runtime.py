"""
app/core/runtime.py

Defines the RuntimeConfig class.
This class holds configuration values that are computed at startup,
(e.g., detected device, optimal batch size, cache-versioned paths).
It is created by system_setup.py and injected into services.
"""

from pathlib import Path
from pydantic import BaseModel, ConfigDict
import pandas as pd
from app.config import Models, ModelWeights, Thresholds, ProgressiveSearch, AppSettings

class RuntimeConfig(BaseModel):
    """
    A read-only container for all static and dynamic config
    needed by the application at runtime.
    """
    # Allow pandas.DataFrame
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    # Dynamic (computed) values
    device: str
    batch_size: int
    embedding_file: Path
    category_df: pd.DataFrame
    max_depth: int
    
    # Static (copied from AppSettings)
    cache_embeddings: bool
    categories_json: Path
    model_weights: ModelWeights
    models: Models
    thresholds: Thresholds
    progressive: ProgressiveSearch
    
    # Added from AppSettings for the reranker
    cross_encoder_model_name: str
    rerank_retrieval_k: int