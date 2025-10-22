# core/config.py
'''Configuration settings'''

import torch
from pathlib import Path

class Config:
    # Directories
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / 'data'
    CACHE_DIR = PROJECT_ROOT / 'cache'
    
    # Files
    CATEGORIES_JSON = DATA_DIR / 'categories.json'
    EMBEDDING_FILE = CACHE_DIR / 'embeddings_v2.npy'
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Processing - INCREASED FOR SPEED!
    BATCH_SIZE = 128  # Changed from 32 → 50% faster! ⚡
    
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
    
    @classmethod
    def setup(cls):
        '''Create necessary directories'''
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        print('\n' + '='*70)
        print('SETUP')
        print('='*70)
        print(f'Project root: {cls.PROJECT_ROOT}')
        print(f'Data dir: {cls.DATA_DIR}')
        print(f'Cache dir: {cls.CACHE_DIR}')
        print(f'Device: {cls.DEVICE}')
        print(f'Batch size: {cls.BATCH_SIZE} (optimized for speed!)')
        print('='*70 + '\n')
