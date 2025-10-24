# # core/config.py
# '''Configuration settings'''

# import torch
# from pathlib import Path

# class Config:
#     # Directories
#     PROJECT_ROOT = Path(__file__).parent.parent
#     DATA_DIR = PROJECT_ROOT / 'data'
#     CACHE_DIR = PROJECT_ROOT / 'cache'
    
#     # Files
#     CATEGORIES_JSON = DATA_DIR / 'categories.json'
#     EMBEDDING_FILE = CACHE_DIR / 'embeddings_v2.npy'
    
#     # Device
#     DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     # Processing - INCREASED FOR SPEED!
#     BATCH_SIZE = 128  # Changed from 32 → 50% faster! ⚡
    
#     # Model ensemble weights
#     MODEL_WEIGHTS = {
#         'fast': 0.3,
#         'accurate': 0.45,
#         'specialized': 0.25
#     }
    
#     # Models
#     MODELS = {
#         'fast': 'sentence-transformers/all-MiniLM-L6-v2',
#         'accurate': 'sentence-transformers/all-mpnet-base-v2',
#         'specialized': 'sentence-transformers/msmarco-distilbert-base-v4'
#     }
    
#     # Similarity thresholds
#     THRESHOLDS = {
#         'high': 0.75,
#         'medium': 0.60,
#         'low': 0.45
#     }
    
#     # Progressive search settings
#     PROGRESSIVE = {
#         'top_k_per_level': 5,
#         'min_similarity': 0.40,
#         'level_decay': 0.98
#     }
    
#     # Caching
#     CACHE_EMBEDDINGS = True
    
#     @classmethod
#     def setup(cls):
#         '''Create necessary directories'''
#         cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
#         cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
#         print('\n' + '='*70)
#         print('SETUP')
#         print('='*70)
#         print(f'Project root: {cls.PROJECT_ROOT}')
#         print(f'Data dir: {cls.DATA_DIR}')
#         print(f'Cache dir: {cls.CACHE_DIR}')
#         print(f'Device: {cls.DEVICE}')
#         print(f'Batch size: {cls.BATCH_SIZE} (optimized for speed!)')
#         print('='*70 + '\n')


# core/config.py
'''Configuration settings with validation'''

import torch
import hashlib
import os
from pathlib import Path
from typing import Dict

class Config:
    # Directories
    PROJECT_ROOT = Path(os.getenv('PROJECT_ROOT', Path(__file__).parent.parent))
    DATA_DIR = PROJECT_ROOT / 'data'
    CACHE_DIR = PROJECT_ROOT / 'cache'
    
    # Files
    CATEGORIES_JSON = DATA_DIR / 'categories.json'
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model ensemble weights
    MODEL_WEIGHTS = {
        'fast': 0.3,
        'accurate': 0.45,
        'specialized': 0.25
    }
    
    # Models
    MODELS = {
        'fast': 'sentence-transformers/all-MiniLM-L6-v2',
        'accurate': 'sentence-transformers/all-mpnet-base-v2',
        'specialized': 'sentence-transformers/msmarco-distilbert-base-v4'
    }
    
    # Similarity thresholds
    THRESHOLDS = {
        'high': 0.75,
        'medium': 0.60,
        'low': 0.45
    }
    
    # Progressive search settings
    PROGRESSIVE = {
        'top_k_per_level': 5,
        'min_similarity': 0.40,
        'level_decay': 0.98
    }
    
    # Caching
    CACHE_EMBEDDINGS = True
    
    # ✅ NEW: Dynamic batch size (set after validation)
    BATCH_SIZE = None
    
    # ✅ NEW: Dynamic embedding file with cache versioning
    EMBEDDING_FILE = None
    
    @classmethod
    def get_optimal_batch_size(cls) -> int:
        """
        Calculate optimal batch size based on available hardware
        
        Returns:
            Optimal batch size for current device
        """
        if cls.DEVICE == 'cuda':
            try:
                gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                
                if gpu_mem_gb < 6:
                    return 32  # Small GPU (4-6 GB) - RTX 3050
                elif gpu_mem_gb < 12:
                    return 64  # Medium GPU (6-12 GB)
                elif gpu_mem_gb < 24:
                    return 128  # Large GPU (12-24 GB)
                else:
                    return 256  # Very large GPU (24+ GB)
            except Exception as e:
                print(f'⚠️  Warning: Could not detect GPU memory: {e}')
                return 32  # Safe default
        else:
            return 16  # CPU fallback
    
    @classmethod
    def get_embedding_cache_key(cls) -> str:
        """
        Generate cache key based on model configuration
        
        Returns:
            Unique hash that changes when models change
        """
        # Create deterministic string from model config
        model_string = '|'.join(sorted(f"{k}:{v}" for k, v in cls.MODELS.items()))
        weight_string = '|'.join(sorted(f"{k}:{v}" for k, v in cls.MODEL_WEIGHTS.items()))
        config_string = f"{model_string}|{weight_string}"
        
        # Generate short hash
        hash_obj = hashlib.md5(config_string.encode())
        return hash_obj.hexdigest()[:8]
    
    @classmethod
    def validate_model_weights(cls):
        """Validate that model weights sum to 1.0"""
        total = sum(cls.MODEL_WEIGHTS.values())
        
        if not (0.99 <= total <= 1.01):  # Float tolerance
            raise ValueError(
                f"MODEL_WEIGHTS must sum to 1.0, got {total:.4f}. "
                f"Current weights: {cls.MODEL_WEIGHTS}"
            )
        
        # Check all weights are positive
        for key, weight in cls.MODEL_WEIGHTS.items():
            if weight < 0:
                raise ValueError(f"Model weight '{key}' cannot be negative: {weight}")
            if weight > 1:
                raise ValueError(f"Model weight '{key}' cannot exceed 1.0: {weight}")
    
    @classmethod
    def validate_thresholds(cls):
        """Validate that all thresholds are between 0 and 1"""
        for key, threshold in cls.THRESHOLDS.items():
            if not (0 < threshold < 1):
                raise ValueError(
                    f"Threshold '{key}' must be between 0 and 1, got {threshold}"
                )
        
        # Check logical ordering
        if cls.THRESHOLDS['high'] <= cls.THRESHOLDS['medium']:
            raise ValueError(
                f"'high' threshold ({cls.THRESHOLDS['high']}) must be greater than "
                f"'medium' threshold ({cls.THRESHOLDS['medium']})"
            )
        
        if cls.THRESHOLDS['medium'] <= cls.THRESHOLDS['low']:
            raise ValueError(
                f"'medium' threshold ({cls.THRESHOLDS['medium']}) must be greater than "
                f"'low' threshold ({cls.THRESHOLDS['low']})"
            )
    
    @classmethod
    def validate_progressive_settings(cls):
        """Validate progressive search settings"""
        prog = cls.PROGRESSIVE
        
        # Validate top_k_per_level
        if not isinstance(prog['top_k_per_level'], int) or prog['top_k_per_level'] < 1:
            raise ValueError(
                f"top_k_per_level must be a positive integer, got {prog['top_k_per_level']}"
            )
        
        # Validate min_similarity
        if not (0 < prog['min_similarity'] < 1):
            raise ValueError(
                f"min_similarity must be between 0 and 1, got {prog['min_similarity']}"
            )
        
        # Validate level_decay
        if not (0 < prog['level_decay'] <= 1):
            raise ValueError(
                f"level_decay must be between 0 and 1, got {prog['level_decay']}"
            )
        
        # Check if decay is too aggressive
        # After 10 levels: min_similarity * decay^10
        final_threshold = prog['min_similarity'] * (prog['level_decay'] ** 10)
        if final_threshold < 0.20:
            print(f"⚠️  Warning: Progressive decay is aggressive. "
                  f"After 10 levels, threshold will be {final_threshold:.3f}")
    
    @classmethod
    def validate_models(cls):
        """Validate model configuration"""
        # Check that all models are specified
        required_keys = set(cls.MODEL_WEIGHTS.keys())
        model_keys = set(cls.MODELS.keys())
        
        if required_keys != model_keys:
            missing = required_keys - model_keys
            extra = model_keys - required_keys
            
            error_parts = []
            if missing:
                error_parts.append(f"Missing models: {missing}")
            if extra:
                error_parts.append(f"Extra models: {extra}")
            
            raise ValueError(
                f"MODEL_WEIGHTS and MODELS keys must match. {' '.join(error_parts)}"
            )
        
        # Check model names are valid
        for key, model_name in cls.MODELS.items():
            if not isinstance(model_name, str) or not model_name.strip():
                raise ValueError(f"Invalid model name for '{key}': {model_name}")
    
    @classmethod
    def validate(cls):
        """
        Validate all configuration settings
        
        Raises:
            ValueError: If any configuration is invalid
        """
        try:
            cls.validate_model_weights()
            cls.validate_thresholds()
            cls.validate_progressive_settings()
            cls.validate_models()
            
        except ValueError as e:
            print(f"\n{'='*70}")
            print("❌ CONFIGURATION ERROR")
            print(f"{'='*70}")
            print(f"Error: {e}")
            print(f"{'='*70}\n")
            raise
    
    @classmethod
    def setup(cls):
        """
        Initialize configuration with validation
        
        This should be called once at application startup
        """
        print('\n' + '='*70)
        print('CONFIGURATION SETUP')
        print('='*70 + '\n')
        
        try:
            # ✅ Run validation first
            print('🔍 Validating configuration...')
            cls.validate()
            print('✅ Configuration valid\n')
            
            # ✅ Set dynamic values
            cls.BATCH_SIZE = cls.get_optimal_batch_size()
            cache_key = cls.get_embedding_cache_key()
            cls.EMBEDDING_FILE = cls.CACHE_DIR / f'embeddings_{cache_key}.npy'
            
            # ✅ Create directories
            print('📁 Creating directories...')
            cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
            cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            print('✅ Directories ready\n')
            
            # ✅ Print configuration summary
            cls._print_summary()
            
        except Exception as e:
            print(f'\n❌ Setup failed: {e}\n')
            raise
    
    @classmethod
    def _print_summary(cls):
        """Print configuration summary"""
        print('📊 Configuration Summary:')
        print(f'   Project root: {cls.PROJECT_ROOT}')
        print(f'   Data directory: {cls.DATA_DIR}')
        print(f'   Cache directory: {cls.CACHE_DIR}')
        print(f'   Categories file: {cls.CATEGORIES_JSON.name}')
        print(f'   Embeddings cache: {cls.EMBEDDING_FILE.name}')
        print()
        
        print('🖥️  Hardware:')
        print(f'   Device: {cls.DEVICE.upper()}')
        
        if cls.DEVICE == 'cuda':
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f'   GPU: {gpu_name}')
                print(f'   GPU Memory: {gpu_mem:.1f} GB')
            except:
                pass
        
        print(f'   Batch size: {cls.BATCH_SIZE} (auto-detected)')
        print()
        
        print('🤖 Model Ensemble:')
        for key, model_name in cls.MODELS.items():
            weight = cls.MODEL_WEIGHTS[key]
            short_name = model_name.split('/')[-1]
            print(f'   {key:12s}: {short_name:30s} (weight: {weight:.2f})')
        print()
        
        print('🎯 Thresholds:')
        for key, value in cls.THRESHOLDS.items():
            print(f'   {key:8s}: {value:.2f}')
        print()
        
        print('🔍 Progressive Search:')
        for key, value in cls.PROGRESSIVE.items():
            print(f'   {key:20s}: {value}')
        
        print('\n' + '='*70 + '\n')
    
    @classmethod
    def get_config_dict(cls) -> Dict:
        """
        Get configuration as dictionary (useful for logging/debugging)
        
        Returns:
            Dictionary with all configuration values
        """
        return {
            'device': cls.DEVICE,
            'batch_size': cls.BATCH_SIZE,
            'model_weights': cls.MODEL_WEIGHTS,
            'models': cls.MODELS,
            'thresholds': cls.THRESHOLDS,
            'progressive': cls.PROGRESSIVE,
            'cache_embeddings': cls.CACHE_EMBEDDINGS,
            'embedding_file': str(cls.EMBEDDING_FILE) if cls.EMBEDDING_FILE else None,
        }


# ✅ AUTO-VALIDATE: Run validation when module is imported
# This catches configuration errors immediately
try:
    Config.validate()
except ValueError as e:
    # Re-raise with helpful message
    raise ValueError(
        f"Invalid configuration in config.py. Please fix the following:\n{e}"
    ) from e