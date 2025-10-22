# strategies/progressive.py
import numpy as np
import faiss
from typing import Dict, List, Tuple
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import Config

class ProgressiveStrategy:
    def __init__(self, search_builder, categories_df, embeddings):
        self.search_builder = search_builder
        self.categories_df = categories_df
        self.embeddings = embeddings
        self.max_depth = search_builder.max_depth
    
    def search(self, product_embedding: np.ndarray):
        current_candidates = None
        best_match = None
        level_history = []
        
        for level in range(1, self.max_depth + 1):
            level_index = self.search_builder.get_level_index(level)
            
            if level_index is None:
                break
            
            if current_candidates is None:
                results = self._search_full_level(product_embedding, level)
            else:
                results = self._search_children_only(product_embedding, level, current_candidates)
            
            if not results or len(results['indices']) == 0:
                break
            
            best_idx = results['indices'][0]
            similarity = float(results['similarities'][0])
            
            if similarity < Config.PROGRESSIVE['min_similarity']:
                break
            
            best_match = {'index': best_idx, 'similarity': similarity, 'level': level}
            level_history.append({'level': level, 'index': best_idx, 'similarity': similarity})
            
            top_k = Config.PROGRESSIVE['top_k_per_level']
            current_candidates = []
            
            for idx in results['indices'][:top_k]:
                cat = self.categories_df.iloc[idx]
                path = cat.get(f'level_{level}_path', '')
                if path:
                    current_candidates.append(path)
        
        if best_match:
            return {
                'index': best_match['index'],
                'similarity': best_match['similarity'],
                'method': 'progressive',
                'confidence_weight': 0.5,
                'level_history': level_history,
                'final_level': best_match['level']
            }
        
        return None
    
    def _search_full_level(self, embedding: np.ndarray, level: int):
        level_index = self.search_builder.get_level_index(level)
        level_data = self.search_builder.get_level_data(level)
        
        top_k = Config.PROGRESSIVE['top_k_per_level']
        distances, indices = level_index.search(embedding, top_k)
        
        global_indices = [level_data['indices'][i] for i in indices[0]]
        similarities = [float(d) for d in distances[0]]
        
        return {'indices': global_indices, 'similarities': similarities}
    
    def _search_children_only(self, embedding: np.ndarray, level: int, parent_paths: List[str]):
        level_data = self.search_builder.get_level_data(level)
        level_df = level_data['dataframe']
        
        parent_col = f'level_{level-1}_path'
        
        if parent_col not in level_df.columns:
            return {'indices': [], 'similarities': []}
        
        children_mask = level_df[parent_col].isin(parent_paths)
        
        if children_mask.sum() == 0:
            return {'indices': [], 'similarities': []}
        
        children_df = level_df[children_mask]
        children_global_indices = children_df.index.tolist()
        children_embeddings = self.embeddings[children_global_indices]
        
        temp_index = faiss.IndexFlatIP(children_embeddings.shape[1])
        temp_index.add(children_embeddings)
        
        top_k = min(Config.PROGRESSIVE['top_k_per_level'], len(children_global_indices))
        distances, temp_indices = temp_index.search(embedding, top_k)
        
        global_indices = [children_global_indices[i] for i in temp_indices[0]]
        similarities = [float(d) for d in distances[0]]
        
        return {'indices': global_indices, 'similarities': similarities}
