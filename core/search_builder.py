# core/search_builder.py
import faiss
import numpy as np
import pandas as pd
from typing import Dict

class SearchBuilder:
    def __init__(self, embeddings, categories_df, max_depth):
        self.embeddings = embeddings
        self.categories_df = categories_df
        self.max_depth = max_depth
        
        self.main_index = None
        self.level_indices = {}
        self.level_data = {}
    
    def build_all(self):
        print('\n' + '='*70)
        print('BUILDING SEARCH INDICES')
        print('='*70 + '\n')
        
        self._build_main_index()
        self._build_level_indices()
        
        print('All indices built\n')
    
    def _build_main_index(self):
        print('Building main index...')
        
        embedding_dim = self.embeddings.shape[1]
        self.main_index = faiss.IndexFlatIP(embedding_dim)
        self.main_index.add(self.embeddings)
        
        print(f'Main index: {len(self.embeddings)} categories\n')
    
    def _build_level_indices(self):
        print('Building level-specific indices...')
        
        for level in range(1, self.max_depth + 1):
            level_col = f'level_{level}'
            
            if level_col not in self.categories_df.columns:
                continue
            
            mask = self.categories_df[level_col].notna() & (self.categories_df[level_col] != '')
            
            if mask.sum() == 0:
                continue
            
            level_df = self.categories_df[mask].copy()
            level_indices = level_df.index.tolist()
            level_embeddings = self.embeddings[level_indices]
            
            embedding_dim = level_embeddings.shape[1]
            level_index = faiss.IndexFlatIP(embedding_dim)
            level_index.add(level_embeddings)
            
            self.level_indices[level] = level_index
            self.level_data[level] = {
                'indices': level_indices,
                'dataframe': level_df
            }
            
            print(f'   Level {level:2d}: {len(level_indices):5d} categories')
        
        print()
    
    def get_main_index(self):
        return self.main_index
    
    def get_level_index(self, level: int):
        return self.level_indices.get(level)
    
    def get_level_data(self, level: int):
        return self.level_data.get(level)
