# core/predictor.py
import numpy as np
from typing import Dict, List
from .config import Config
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.direct import DirectStrategy
from strategies.progressive import ProgressiveStrategy
from strategies.ensemble import EnsembleStrategy

class Predictor:
    def __init__(self, embedding_engine, search_builder, categories_df, embeddings):
        self.embedding_engine = embedding_engine
        self.categories_df = categories_df
        self.max_depth = search_builder.max_depth
        
        self.direct_strategy = DirectStrategy(search_builder)
        self.progressive_strategy = ProgressiveStrategy(search_builder, categories_df, embeddings)
    
    def predict(self, title: str, description: str = '', tags: str = '', product_type: str = '', vendor: str = ''):
        from .text_enhancer import TextEnhancer
        
        product_text = TextEnhancer.enhance_product_text(title, description, tags, product_type, vendor)
        product_embedding = self.embedding_engine.generate([product_text], show_progress=False)
        
        direct_result = self.direct_strategy.search(product_embedding)
        progressive_result = self.progressive_strategy.search(product_embedding)
        
        best_result = EnsembleStrategy.combine([direct_result, progressive_result])
        
        if best_result is None:
            raise ValueError('No prediction could be made')
        
        final_result = self._build_result(best_result, title)
        return final_result
    
    def _build_result(self, prediction: Dict, product_title: str):
        # Ensure index is int
        category_idx = int(prediction['index'])
        category = self.categories_df.iloc[category_idx]
        
        levels = {}
        category_path = str(category.get('Category_path', ''))
        
        # Ensure similarity is float
        base_similarity = float(prediction['similarity'])
        
        if category_path:
            parts = category_path.split('/')
            
            for i in range(1, len(parts) + 1):
                level_name = category.get(f'level_{i}', '')
                if not level_name or str(level_name).strip() == '':
                    continue
                
                level_path = category.get(f'level_{i}_path', '/'.join(parts[:i]))
                
                # Calculate similarity with explicit float conversion
                decay_factor = float(0.98 ** (i - 1))
                similarity = base_similarity * decay_factor
                
                levels[f'level_{i}'] = {
                    'name': str(level_name).strip(),
                    'path': str(level_path),
                    'similarity': round(float(similarity), 4),
                    'confidence': self._get_confidence(similarity)
                }
        
        result = {
            'product_title': str(product_title),
            'category_id': str(category.get('Category_ID', '')),
            'category_path': str(category_path),
            'similarity': round(float(base_similarity), 4),
            'confidence': self._get_confidence(base_similarity),
            'depth': int(category.get('depth', 0)),
            'max_depth': int(self.max_depth),
            'prediction_method': str(prediction['method']),
            'levels': levels
        }
        
        if 'alternatives' in prediction:
            result['alternatives'] = self._format_alternatives(prediction['alternatives'])
        
        return result
    
    def _get_confidence(self, similarity: float):
        similarity = float(similarity)
        thresholds = Config.THRESHOLDS
        
        if similarity >= thresholds['high']:
            return 'high'
        elif similarity >= thresholds['medium']:
            return 'medium'
        elif similarity >= thresholds['low']:
            return 'low'
        else:
            return 'very_low'
    
    def _format_alternatives(self, alternatives: List[Dict]):
        formatted = []
        for alt in alternatives[:5]:
            cat_idx = int(alt['index'])
            cat = self.categories_df.iloc[cat_idx]
            sim = float(alt['similarity'])
            
            formatted.append({
                'category_id': str(cat.get('Category_ID', '')),
                'category_path': str(cat.get('Category_path', '')),
                'similarity': round(sim, 4),
                'confidence': self._get_confidence(sim),
                'depth': int(cat.get('depth', 0))
            })
        
        return formatted
