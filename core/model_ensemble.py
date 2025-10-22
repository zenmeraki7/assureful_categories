# core/model_ensemble.py
'''Triple model ensemble for embeddings'''

from sentence_transformers import SentenceTransformer
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import Config

class ModelEnsemble:
    def __init__(self):
        self.models = {}
        self.weights = Config.MODEL_WEIGHTS
        self.device = Config.DEVICE
        
        self.model_configs = {
            'fast': {'name': Config.MODELS['fast']},
            'accurate': {'name': Config.MODELS['accurate']},
            'specialized': {'name': Config.MODELS['specialized']}
        }
    
    def load(self):
        print('\n' + '='*70)
        print('LOADING MODEL ENSEMBLE')
        print('='*70 + '\n')
        
        for key, config in self.model_configs.items():
            model_name = config['name']
            weight = self.weights[key]
            
            print(f'Loading {key} model: {model_name}')
            print(f'  Weight: {weight}')
            
            try:
                model = SentenceTransformer(model_name, device=self.device)
                self.models[key] = model
                print(f'  Status: Loaded successfully\n')
            except Exception as e:
                print(f'  Error: {e}\n')
                raise
        
        print(f'All models loaded on {self.device}')
        print('='*70 + '\n')
    
    def get_models(self):
        '''Return the loaded models dictionary'''
        return self.models
    
    def get_weights(self):
        '''Return the model weights dictionary'''
        return self.weights
    
    def encode(self, texts, show_progress=True):
        if not self.models:
            raise RuntimeError('Models not loaded. Call load() first.')
        
        all_embeddings = []
        
        for key, model in self.models.items():
            weight = self.weights[key]
            
            if show_progress:
                print(f'Encoding with {key} model (weight {weight})...')
            
            embeddings = model.encode(
                texts,
                show_progress_bar=show_progress,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            weighted_embeddings = embeddings * weight
            all_embeddings.append(weighted_embeddings)
        
        combined = np.concatenate(all_embeddings, axis=1)
        
        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        combined = combined / (norms + 1e-8)
        
        if show_progress:
            print(f'Combined embeddings shape: {combined.shape}\n')
        
        return combined
    
    def get_embedding_dim(self):
        if not self.models:
            raise RuntimeError('Models not loaded')
        
        total_dim = 0
        for key, model in self.models.items():
            total_dim += model.get_sentence_embedding_dimension()
        
        return total_dim
