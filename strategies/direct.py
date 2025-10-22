# strategies/direct.py
import numpy as np
from typing import Dict

class DirectStrategy:
    def __init__(self, search_builder):
        self.main_index = search_builder.get_main_index()
    
    def search(self, product_embedding: np.ndarray, top_k: int = 10):
        distances, indices = self.main_index.search(product_embedding, top_k)
        
        best_idx = int(indices[0][0])
        best_similarity = float(distances[0][0])
        
        alternatives = []
        for i in range(1, len(indices[0])):
            alternatives.append({
                'index': int(indices[0][i]),
                'similarity': float(distances[0][i])
            })
        
        return {
            'index': best_idx,
            'similarity': best_similarity,
            'method': 'direct',
            'confidence_weight': 0.3,
            'alternatives': alternatives
        }
