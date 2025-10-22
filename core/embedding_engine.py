# core/embedding_engine.py
'''Embedding generation engine'''

import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import Config

class EmbeddingEngine:
    def __init__(self, model_ensemble):
        self.model_ensemble = model_ensemble
        self.models = model_ensemble.get_models()
        self.weights = model_ensemble.get_weights()
    
    def generate(self, texts, show_progress=True):
        '''Generate embeddings using model ensemble'''
        
        if show_progress:
            print(f'\n🔢 Generating embeddings for {len(texts)} texts...')
        
        # Use the model_ensemble's encode method which handles everything
        embeddings = self.model_ensemble.encode(texts, show_progress=show_progress)
        
        if show_progress:
            print(f'✅ Generated embeddings: {embeddings.shape}\n')
        
        return embeddings
    
    def save(self, embeddings, filepath):
        '''Save embeddings to file'''
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        print(f'💾 Saving embeddings to {filepath}...')
        np.save(filepath, embeddings)
        print(f'✅ Embeddings saved\n')
    
    def load(self, filepath):
        '''Load embeddings from file'''
        filepath = Path(filepath)
        
        if not filepath.exists():
            return None
        
        try:
            embeddings = np.load(filepath)
            return embeddings
        except Exception as e:
            print(f'⚠️  Error loading embeddings: {e}')
            return None
