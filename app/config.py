"""
app/config.py

Defines all static, loadable application settings using Pydantic.
This class reads from environment variables and .env files.
It validates the configuration on load.
"""

from pathlib import Path
from pydantic import BaseModel, Field, model_validator
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except Exception:
    # Fallback for environments where the separate 'pydantic_settings' package
    # is not installed (e.g., using pydantic v1 or a linter that can't resolve it).
    # Provide minimal compatible aliases so model_config = SettingsConfigDict(...) works.
    try:
        from pydantic import BaseSettings  # type: ignore
    except Exception:
        # Last-resort minimal fallback BaseSettings to avoid NameError at import-time.
        class BaseSettings:  # type: ignore
            pass
    SettingsConfigDict = dict
import logging
import sys

# --- 1. Nested Configuration Models ---
# These provide structure and allow for nested env vars
# e.g., APP_MODEL_WEIGHTS__FAST=0.2

class ModelWeights(BaseModel):
    fast: float = 0.3
    accurate: float = 0.45
    specialized: float = 0.25

    @model_validator(mode='after')
    def validate_weights_sum(self) -> 'ModelWeights':
        """Validates that model weights sum to 1.0"""
        total = self.fast + self.accurate + self.specialized
        if not (0.99 <= total <= 1.01):  # Float tolerance
            raise ValueError(f"Model weights must sum to 1.0, got {total:.4f}")
        return self

class Models(BaseModel):
    fast: str = 'sentence-transformers/all-MiniLM-L6-v2'
    accurate: str = 'sentence-transformers/all-mpnet-base-v2'
    specialized: str = 'sentence-transformers/msmarco-distilbert-base-v4'

class Thresholds(BaseModel):
    high: float = 0.75
    medium: float = 0.60
    low: float = 0.45

    @model_validator(mode='after')
    def validate_threshold_logic(self) -> 'Thresholds':
        """Validates the logical order of thresholds."""
        if not (self.high > self.medium > self.low):
            raise ValueError(
                f"Thresholds must be high > medium > low. Got: "
                f"high={self.high}, medium={self.medium}, low={self.low}"
            )
        return self

class ProgressiveSearch(BaseModel):
    top_k_per_level: int = Field(5, gt=0)
    min_similarity: float = Field(0.40, gt=0, lt=1)
    level_decay: float = Field(0.98, gt=0, le=1)


# --- 2. Main Application Settings ---

class AppSettings(BaseSettings):
    """
    Main settings class. Loads from .env and environment variables.
    Prefix: 'APP_' (e.g., APP_LOG_LEVEL='DEBUG')
    """
    model_config = SettingsConfigDict(
        env_prefix='APP_',
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        env_nested_delimiter='__'  # Allows APP_THRESHOLDS__HIGH=0.8
    )

    # --- Paths (Static) ---
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / 'data'
    CACHE_DIR: Path = PROJECT_ROOT / 'cache'
    CATEGORIES_JSON: Path = DATA_DIR / 'categories.json'

    # --- App Behavior (Configurable) ---
    LOG_LEVEL: str = "INFO"
    JSON_LOGS: bool = True
    CACHE_EMBEDDINGS: bool = True

    # --- Nested Settings Models (Configurable) ---
    MODEL_WEIGHTS: ModelWeights = Field(default_factory=ModelWeights)
    MODELS: Models = Field(default_factory=Models)
    THRESHOLDS: Thresholds = Field(default_factory=Thresholds)
    PROGRESSIVE: ProgressiveSearch = Field(default_factory=ProgressiveSearch)

    # --- NEW: Re-ranking & Cross-Encoder Settings ---
    CROSS_ENCODER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # How many candidates to "retrieve" in Stage 1
    RERANK_RETRIEVAL_K: int = 50 


# --- 3. Singleton Instance ---
# This single instance is imported by other modules.
# Validation runs automatically on instantiation.
try:
    settings = AppSettings()
except ValueError as e:
    # Fail-fast on config error
    logging.basicConfig(level="CRITICAL")
    logging.critical(f"FATAL: Configuration validation failed:\n{e}")
    sys.exit(1)