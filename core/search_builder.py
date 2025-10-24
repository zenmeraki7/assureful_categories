# # core/search_builder.py
# import faiss
# import numpy as np
# import pandas as pd
# from typing import Dict

# class SearchBuilder:
#     def __init__(self, embeddings, categories_df, max_depth):
#         self.embeddings = embeddings
#         self.categories_df = categories_df
#         self.max_depth = max_depth
        
#         self.main_index = None
#         self.level_indices = {}
#         self.level_data = {}
    
#     def build_all(self):
#         print('\n' + '='*70)
#         print('BUILDING SEARCH INDICES')
#         print('='*70 + '\n')
        
#         self._build_main_index()
#         self._build_level_indices()
        
#         print('All indices built\n')
    
#     def _build_main_index(self):
#         print('Building main index...')
        
#         embedding_dim = self.embeddings.shape[1]
#         self.main_index = faiss.IndexFlatIP(embedding_dim)
#         self.main_index.add(self.embeddings)
        
#         print(f'Main index: {len(self.embeddings)} categories\n')
    
#     def _build_level_indices(self):
#         print('Building level-specific indices...')
        
#         for level in range(1, self.max_depth + 1):
#             level_col = f'level_{level}'
            
#             if level_col not in self.categories_df.columns:
#                 continue
            
#             mask = self.categories_df[level_col].notna() & (self.categories_df[level_col] != '')
            
#             if mask.sum() == 0:
#                 continue
            
#             level_df = self.categories_df[mask].copy()
#             level_indices = level_df.index.tolist()
#             level_embeddings = self.embeddings[level_indices]
            
#             embedding_dim = level_embeddings.shape[1]
#             level_index = faiss.IndexFlatIP(embedding_dim)
#             level_index.add(level_embeddings)
            
#             self.level_indices[level] = level_index
#             self.level_data[level] = {
#                 'indices': level_indices,
#                 'dataframe': level_df
#             }
            
#             print(f'   Level {level:2d}: {len(level_indices):5d} categories')
        
#         print()
    
#     def get_main_index(self):
#         return self.main_index
    
#     def get_level_index(self, level: int):
#         return self.level_indices.get(level)
    
#     def get_level_data(self, level: int):
#         return self.level_data.get(level)


# core/search_builder.py
import faiss
import numpy as np
import pandas as pd
import pickle
import hashlib
from typing import Dict, Optional
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import Config

class SearchBuilder:
    def __init__(self, embeddings, categories_df, max_depth):
        self.embeddings = embeddings
        self.categories_df = categories_df
        self.max_depth = max_depth
        
        self.main_index = None
        self.level_indices = {}
        self.level_data = {}
        
        # ✅ NEW: Cache configuration
        self.cache_dir = Config.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def build_all(self, force_rebuild: bool = False):
        """
        Build all indices with automatic caching
        
        Args:
            force_rebuild: If True, rebuild even if cache exists
        """
        print('\n' + '='*70)
        print('BUILDING SEARCH INDICES')
        print('='*70 + '\n')
        
        # ✅ NEW: Try loading from cache first
        if not force_rebuild and self._load_from_cache():
            print('✅ Loaded all indices from cache (instant startup!)\n')
            return
        
        # Cache miss or force rebuild - build from scratch
        print('⚠️  Cache miss - building indices from scratch...\n')
        self._build_main_index()
        self._build_level_indices()
        
        # ✅ NEW: Save to cache for next time
        self._save_to_cache()
        
        print('✅ All indices built and cached\n')
    
    def _build_main_index(self):
        """Build main FAISS index"""
        print('Building main index...')
        
        embedding_dim = self.embeddings.shape[1]
        self.main_index = faiss.IndexFlatIP(embedding_dim)
        self.main_index.add(self.embeddings)
        
        print(f'   Main index: {len(self.embeddings):,} categories\n')
    
    def _build_level_indices(self):
        """Build level-specific FAISS indices"""
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
            
            print(f'   Level {level:2d}: {len(level_indices):5,} categories')
        
        print()
    
    # ✅ NEW: Cache saving functionality
    def _save_to_cache(self):
        """Save all indices and metadata to cache"""
        print('💾 Saving indices to cache...')
        
        try:
            # Save main index
            main_index_path = self.cache_dir / 'faiss_main_index.bin'
            faiss.write_index(self.main_index, str(main_index_path))
            print(f'   ✓ Main index saved')
            
            # Save level indices
            level_indices_dir = self.cache_dir / 'level_indices'
            level_indices_dir.mkdir(exist_ok=True)
            
            for level, index in self.level_indices.items():
                level_path = level_indices_dir / f'level_{level}_index.bin'
                faiss.write_index(index, str(level_path))
            
            print(f'   ✓ {len(self.level_indices)} level indices saved')
            
            # Save metadata (level_data without dataframes to save space)
            metadata = {
                'max_depth': self.max_depth,
                'embedding_shape': self.embeddings.shape,
                'level_data': {
                    level: {
                        'indices': data['indices'],
                        'num_categories': len(data['dataframe'])
                    }
                    for level, data in self.level_data.items()
                },
                'cache_version': self._get_cache_version()
            }
            
            metadata_path = self.cache_dir / 'search_indices_metadata.pkl'
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            print(f'   ✓ Metadata saved')
            print(f'✅ Cache saved successfully\n')
            
        except Exception as e:
            print(f'⚠️  Warning: Failed to save cache: {e}\n')
    
    # ✅ NEW: Cache loading functionality
    def _load_from_cache(self) -> bool:
        """
        Load indices from cache if available and valid
        
        Returns:
            True if successfully loaded from cache, False otherwise
        """
        try:
            # Check if cache exists
            main_index_path = self.cache_dir / 'faiss_main_index.bin'
            metadata_path = self.cache_dir / 'search_indices_metadata.pkl'
            
            if not main_index_path.exists() or not metadata_path.exists():
                return False
            
            # Load and validate metadata
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            # Validate cache version
            if metadata.get('cache_version') != self._get_cache_version():
                print('⚠️  Cache invalidated: embeddings have changed')
                return False
            
            # Validate embedding shape
            if metadata.get('embedding_shape') != self.embeddings.shape:
                print('⚠️  Cache invalidated: embedding dimensions changed')
                return False
            
            print('📂 Loading indices from cache...')
            
            # Load main index
            self.main_index = faiss.read_index(str(main_index_path))
            print(f'   ✓ Main index loaded ({self.main_index.ntotal:,} vectors)')
            
            # Load level indices
            level_indices_dir = self.cache_dir / 'level_indices'
            if level_indices_dir.exists():
                for level, level_meta in metadata['level_data'].items():
                    level_path = level_indices_dir / f'level_{level}_index.bin'
                    
                    if level_path.exists():
                        self.level_indices[level] = faiss.read_index(str(level_path))
                        
                        # Reconstruct level_data
                        level_indices = level_meta['indices']
                        level_df = self.categories_df.iloc[level_indices].copy()
                        
                        self.level_data[level] = {
                            'indices': level_indices,
                            'dataframe': level_df
                        }
                
                print(f'   ✓ {len(self.level_indices)} level indices loaded')
            
            return True
            
        except Exception as e:
            print(f'⚠️  Failed to load cache: {e}')
            return False
    
    # ✅ NEW: Cache version tracking
    def _get_cache_version(self) -> str:
        """
        Generate cache version based on embeddings
        
        Returns:
            Hash string that changes when embeddings change
        """
        # Create hash of embedding metadata
        version_string = f"{self.embeddings.shape}_{self.embeddings.dtype}"
        
        # Sample first and last embeddings for quick validation
        if len(self.embeddings) > 0:
            sample = np.concatenate([
                self.embeddings[0].flatten()[:10],
                self.embeddings[-1].flatten()[:10]
            ])
            version_string += f"_{sample.sum():.6f}"
        
        return hashlib.md5(version_string.encode()).hexdigest()[:12]
    
    # ✅ NEW: Force cache rebuild
    def clear_cache(self):
        """Clear all cached indices"""
        print('🗑️  Clearing cache...')
        
        cache_files = [
            self.cache_dir / 'faiss_main_index.bin',
            self.cache_dir / 'search_indices_metadata.pkl'
        ]
        
        for file in cache_files:
            if file.exists():
                file.unlink()
                print(f'   ✓ Deleted {file.name}')
        
        level_indices_dir = self.cache_dir / 'level_indices'
        if level_indices_dir.exists():
            for file in level_indices_dir.glob('*.bin'):
                file.unlink()
            level_indices_dir.rmdir()
            print(f'   ✓ Deleted level indices')
        
        print('✅ Cache cleared\n')
    
    # Existing getters (unchanged)
    def get_main_index(self):
        return self.main_index
    
    def get_level_index(self, level: int):
        return self.level_indices.get(level)
    
    def get_level_data(self, level: int):
        return self.level_data.get(level)
    
    # ✅ NEW: Utility methods
    def get_cache_info(self) -> Dict:
        """Get information about cache status"""
        main_index_path = self.cache_dir / 'faiss_main_index.bin'
        metadata_path = self.cache_dir / 'search_indices_metadata.pkl'
        
        info = {
            'cache_exists': main_index_path.exists() and metadata_path.exists(),
            'cache_dir': str(self.cache_dir),
            'main_index_size': main_index_path.stat().st_size if main_index_path.exists() else 0,
        }
        
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            info['cached_categories'] = metadata.get('embedding_shape', (0,))[0]
            info['cache_version'] = metadata.get('cache_version', 'unknown')
        
        return info
