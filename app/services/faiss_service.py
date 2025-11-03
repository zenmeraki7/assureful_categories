# """
# app/services/faiss_service.py

# This service handles building or loading the FAISS indices from cache.
# """

# import logging
# import faiss
# import numpy as np
# import pandas as pd
# import pickle
# import hashlib
# import asyncio
# import shutil
# from pathlib import Path
# from app.core.runtime import RuntimeConfig
# from app.core.active_indices import FaissIndices, swap_active_indices
# from app.config import settings
# from app.utils.prometheus_utils import INDEX_VECTORS_TOTAL

# logger = logging.getLogger(__name__)

# class FaissIndexService:
#     """
#     Builds the main and per-level FAISS indices, using a
#     content-aware cache for instant startups.
#     """
#     def __init__(self, cfg: RuntimeConfig, embeddings: np.ndarray):
#         self.cfg = cfg
#         self.embeddings = embeddings
#         self.categories_df = cfg.category_df
#         self.max_depth = cfg.max_depth
        
#         # Cache paths
#         self.cache_dir = settings.CACHE_DIR
#         self.metadata_path = self.cache_dir / 'search_indices_metadata.pkl'
#         self.main_index_path = self.cache_dir / 'faiss_main_index.bin'
#         self.level_indices_dir = self.cache_dir / 'level_indices'
        
#         self.cache_version = self._get_cache_version()
#         logger.info(f"Current embedding cache version: {self.cache_version}")

#     def _get_cache_version(self) -> str:
#         """
#         Generates a cache version hash based on embedding metadata
#         and content.
#         """
#         version_string = f"{self.embeddings.shape}_{self.embeddings.dtype}"
        
#         if len(self.embeddings) > 0:
#             sample = np.concatenate([
#                 self.embeddings[0].flatten()[:10],
#                 self.embeddings[-1].flatten()[:10]
#             ])
#             version_string += f"_{sample.sum():.6f}"
            
#         return hashlib.md5(version_string.encode()).hexdigest()[:12]

#     async def build_or_load_indices(self) -> None:
#         """
#         Main public method. Orchestrates loading from cache or
#         building from scratch, then hot-swaps the result.
#         """
#         logger.info("--- 🔎 Initializing FAISS Indices ---")
        
#         cached_indices = await self._load_from_cache()
        
#         if cached_indices:
#             logger.info("✅ Successfully loaded all indices from cache.")
#             indices_to_activate = cached_indices
#         else:
#             logger.warning("⚠️ Cache miss or invalid. Building indices from scratch...")
#             indices_to_activate = await self._build_from_scratch()
#             await self._save_to_cache(indices_to_activate)
        
#         # Atomically swap the new indices into the live state
#         await swap_active_indices(indices_to_activate)
        
#         INDEX_VECTORS_TOTAL.observe(indices_to_activate.total_vectors)
#         logger.info(
#             f"--- ✅ FAISS Indices are live ({indices_to_activate.total_vectors} vectors) ---"
#         )

#     async def _load_from_cache(self) -> FaissIndices | None:
#         """
#         Asynchronously loads all indices from cache.
#         Returns a populated FaissIndices object or None on failure.
#         """
#         if not self.main_index_path.exists() or not self.metadata_path.exists():
#             logger.info("Cache not found (missing index or metadata file).")
#             return None
            
#         loop = asyncio.get_running_loop()
        
#         try:
#             # 1. Load and validate metadata
#             def _load_meta():
#                 with open(self.metadata_path, 'rb') as f:
#                     return pickle.load(f)
            
#             metadata = await loop.run_in_executor(None, _load_meta)
            
#             if metadata.get('cache_version') != self.cache_version:
#                 logger.warning("Cache invalidated: Embeddings version mismatch.")
#                 return None
#             if metadata.get('embedding_shape') != self.embeddings.shape:
#                 logger.warning("Cache invalidated: Embedding shape mismatch.")
#                 return None

#             logger.info("Cache metadata validated. Loading indices...")
            
#             # 2. Load main index
#             def _read_index(path):
#                 return faiss.read_index(str(path))
            
#             main_index = await loop.run_in_executor(None, _read_index, self.main_index_path)
#             logger.info(f"Main index loaded ({main_index.ntotal:,} vectors).")
            
#             # 3. Load level indices
#             level_indices = {}
#             level_data = {}
#             for level, level_meta in metadata['level_data'].items():
#                 level_path = self.level_indices_dir / f'level_{level}_index.bin'
#                 if level_path.exists():
#                     level_indices[level] = await loop.run_in_executor(
#                         None, _read_index, level_path
#                     )
                    
#                     # Reconstruct level_data
#                     level_indices_list = level_meta['indices']
#                     level_df = self.categories_df.iloc[level_indices_list].copy()
#                     level_data[level] = {
#                         'indices': level_indices_list,
#                         'dataframe': level_df
#                     }
            
#             logger.info(f"{len(level_indices)} level indices loaded.")
            
#             return FaissIndices(
#                 main_index=main_index,
#                 level_indices=level_indices,
#                 level_data=level_data,
#                 total_vectors=main_index.ntotal
#             )

#         except Exception as e:
#             logger.error(f"Failed to load cache: {e}", exc_info=True)
#             return None

#     async def _build_from_scratch(self) -> FaissIndices:
#         """
#         Runs the CPU-bound index-building process in an executor.
#         """
#         loop = asyncio.get_running_loop()
#         return await loop.run_in_executor(None, self._build_indices_sync)

#     def _build_indices_sync(self) -> FaissIndices:
#         """
#         The synchronous, blocking logic for building all indices.
#         """
        
#         # 1. Build main index
#         logger.info("Building main index...")
#         embedding_dim = self.embeddings.shape[1]
#         main_index = faiss.IndexFlatIP(embedding_dim)
#         main_index.add(self.embeddings.astype(np.float32))
#         logger.info(f"Main index built ({main_index.ntotal:,} vectors).")
        
#         # 2. Build level indices
#         logger.info("Building level-specific indices...")
#         level_indices = {}
#         level_data = {}
        
#         for level in range(1, self.max_depth + 1):
#             level_col = f'level_{level}'
#             if level_col not in self.categories_df.columns:
#                 continue
                
#             mask = self.categories_df[level_col].notna()
#             if mask.sum() == 0:
#                 continue
                
#             level_df = self.categories_df[mask].copy()
#             level_indices_list = level_df.index.tolist()
#             level_embeddings = self.embeddings[level_indices_list]
            
#             level_index = faiss.IndexFlatIP(embedding_dim)
#             level_index.add(level_embeddings.astype(np.float32))
            
#             level_indices[level] = level_index
#             level_data[level] = {
#                 'indices': level_indices_list,
#                 'dataframe': level_df
#             }
#             logger.info(f"  Level {level:2d}: {len(level_indices_list):,} categories")
            
#         return FaissIndices(
#             main_index=main_index,
#             level_indices=level_indices,
#             level_data=level_data,
#             total_vectors=main_index.ntotal
#         )

#     async def _save_to_cache(self, indices: FaissIndices) -> None:
#         """
#         Asynchronously saves all built indices and metadata to disk.
#         """
#         logger.info("💾 Saving indices to cache...")
#         loop = asyncio.get_running_loop()
        
#         try:
#             # Create directories
#             self.cache_dir.mkdir(parents=True, exist_ok=True)
#             self.level_indices_dir.mkdir(parents=True, exist_ok=True)

#             # 1. Save main index (blocking I/O)
#             def _write_index(index, path):
#                 faiss.write_index(index, str(path))
            
#             await loop.run_in_executor(
#                 None, _write_index, indices.main_index, self.main_index_path
#             )
            
#             # 2. Save level indices (blocking I/O)
#             for level, index in indices.level_indices.items():
#                 level_path = self.level_indices_dir / f'level_{level}_index.bin'
#                 await loop.run_in_executor(None, _write_index, index, level_path)
            
#             # 3. Save metadata (blocking I/O)
#             metadata = {
#                 'max_depth': self.max_depth,
#                 'embedding_shape': self.embeddings.shape,
#                 'level_data': {
#                     level: {'indices': data['indices']} 
#                     for level, data in indices.level_data.items()
#                 },
#                 'cache_version': self.cache_version
#             }
            
#             def _write_meta():
#                 with open(self.metadata_path, 'wb') as f:
#                     pickle.dump(metadata, f)
            
#             await loop.run_in_executor(None, _write_meta)
            
#             logger.info("✅ Cache saved successfully.")
            
#         except Exception as e:
#             logger.warning(f"Failed to save cache: {e}", exc_info=True)

#     async def clear_cache(self) -> None:
#         """Asynchronously clear all cached indices from disk."""
#         logger.info("🗑️  Clearing FAISS index cache...")
#         loop = asyncio.get_running_loop()
        
#         async def _rm_file(path):
#             try:
#                 if path.exists():
#                     await loop.run_in_executor(None, path.unlink)
#                     logger.info(f"Deleted {path.name}")
#             except Exception as e:
#                 logger.error(f"Failed to delete {path.name}: {e}")

#         async def _rm_dir(path):
#             try:
#                 if path.exists():
#                     await loop.run_in_executor(None, shutil.rmtree, path)
#                     logger.info(f"Deleted directory {path.name}")
#             except Exception as e:
#                 logger.error(f"Failed to delete dir {path.name}: {e}")

#         await _rm_file(self.main_index_path)
#         await _rm_file(self.metadata_path)
#         await _rm_dir(self.level_indices_dir)
        
#         logger.info("✅ Cache cleared.")

"""
app/services/faiss_service.py

This service handles building or loading the FAISS indices from cache.
"""

import logging
import faiss
import numpy as np
import pickle
import hashlib
import asyncio
import shutil
from app.core.runtime import RuntimeConfig
from app.core.active_indices import FaissIndices, swap_active_indices
from app.config import settings
from app.utils.prometheus_utils import INDEX_VECTORS_TOTAL
from pathlib import Path

logger = logging.getLogger(__name__)


class FaissIndexService:
    """
    Builds the main and per-level FAISS indices, using a
    content-aware cache for instant startups.
    """

    def __init__(self, cfg: RuntimeConfig, embeddings: np.ndarray):
        self.cfg = cfg
        self.embeddings = embeddings
        self.categories_df = cfg.category_df
        self.max_depth = cfg.max_depth

        # Cache paths
        self.cache_dir = settings.CACHE_DIR
        self.metadata_path = self.cache_dir / 'search_indices_metadata.pkl'
        self.main_index_path = self.cache_dir / 'faiss_main_index.bin'
        self.level_indices_dir = self.cache_dir / 'level_indices'

        self.cache_version = self._get_cache_version()
        logger.info(f"Current embedding cache version: {self.cache_version}")

    def _get_cache_version(self) -> str:
        """Generates a cache version hash based on embedding metadata and content."""
        version_string = f"{self.embeddings.shape}_{self.embeddings.dtype}"

        if len(self.embeddings) > 0:
            sample = np.concatenate([
                self.embeddings[0].flatten()[:10],
                self.embeddings[-1].flatten()[:10]
            ])
            version_string += f"_{sample.sum():.6f}"

        return hashlib.md5(version_string.encode()).hexdigest()[:12]

    async def build_or_load_indices(self) -> None:
        """Main public method. Orchestrates loading from cache or building from scratch."""
        logger.info("--- 🔎 Initializing FAISS Indices ---")

        cached_indices = await self._load_from_cache()

        if cached_indices:
            logger.info("✅ Successfully loaded all indices from cache.")
            indices_to_activate = cached_indices
        else:
            logger.warning("⚠️ Cache miss or invalid. Building indices from scratch...")
            indices_to_activate = await self._build_from_scratch()
            await self._save_to_cache(indices_to_activate)

        # Atomically swap the new indices into the live state
        await swap_active_indices(indices_to_activate)

        INDEX_VECTORS_TOTAL.observe(indices_to_activate.total_vectors)
        logger.info(
            f"--- ✅ FAISS Indices are live ({indices_to_activate.total_vectors} vectors) ---"
        )

    async def _load_from_cache(self) -> FaissIndices | None:
        """Asynchronously loads all indices from cache."""
        if not self.main_index_path.exists() or not self.metadata_path.exists():
            logger.info("Cache not found (missing index or metadata file).")
            return None

        loop = asyncio.get_running_loop()

        try:
            # 1. Load and validate metadata
            def _load_meta():
                with open(self.metadata_path, 'rb') as f:
                    return pickle.load(f)

            metadata = await loop.run_in_executor(None, _load_meta)

            if metadata.get('cache_version') != self.cache_version:
                logger.warning("Cache invalidated: Embeddings version mismatch.")
                return None
            if metadata.get('embedding_shape') != self.embeddings.shape:
                logger.warning("Cache invalidated: Embedding shape mismatch.")
                return None

            logger.info("Cache metadata validated. Loading indices...")

            # 2. Load main index
            def _read_index(path):
                return faiss.read_index(str(path))

            main_index = await loop.run_in_executor(None, _read_index, self.main_index_path)
            logger.info(f"Main index loaded ({main_index.ntotal:,} vectors).")

            # 3. Load level indices
            level_indices = {}
            level_data = {}
            for level, level_meta in metadata['level_data'].items():
                level_path = self.level_indices_dir / f'level_{level}_index.bin'
                if level_path.exists():
                    level_indices[level] = await loop.run_in_executor(
                        None, _read_index, level_path
                    )

                    # Reconstruct level_data
                    level_indices_list = level_meta['indices']
                    level_df = self.categories_df.iloc[level_indices_list].copy()
                    level_data[level] = {
                        'indices': level_indices_list,
                        'dataframe': level_df
                    }

            logger.info(f"{len(level_indices)} level indices loaded.")

            return FaissIndices(
                main_index=main_index,
                level_indices=level_indices,
                level_data=level_data,
                total_vectors=main_index.ntotal
            )

        except Exception as e:
            logger.error(f"Failed to load cache: {e}", exc_info=True)
            return None

    async def _build_from_scratch(self) -> FaissIndices:
        """Runs the CPU-bound index-building process in an executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._build_indices_sync)

    def _build_indices_sync(self) -> FaissIndices:
        """The synchronous logic for building all indices."""

        # 1. Build main index
        logger.info("Building main index...")
        embedding_dim = self.embeddings.shape[1]
        main_index = faiss.IndexFlatIP(embedding_dim)
        main_index.add(self.embeddings.astype(np.float32))
        logger.info(f"Main index built ({main_index.ntotal:,} vectors).")

        # 2. Build level indices
        logger.info("Building level-specific indices...")
        level_indices = {}
        level_data = {}

        for level in range(1, self.max_depth + 1):
            level_col = f'level_{level}'
            if level_col not in self.categories_df.columns:
                continue

            mask = self.categories_df[level_col].notna()
            if mask.sum() == 0:
                continue

            level_df = self.categories_df[mask].copy()
            level_indices_list = level_df.index.tolist()
            level_embeddings = self.embeddings[level_indices_list]

            level_index = faiss.IndexFlatIP(embedding_dim)
            level_index.add(level_embeddings.astype(np.float32))

            level_indices[level] = level_index
            level_data[level] = {
                'indices': level_indices_list,
                'dataframe': level_df
            }
            logger.info(f"  Level {level:2d}: {len(level_indices_list):,} categories")

        return FaissIndices(
            main_index=main_index,
            level_indices=level_indices,
            level_data=level_data,
            total_vectors=main_index.ntotal
        )

    async def _save_to_cache(self, indices: FaissIndices) -> None:
        """Asynchronously saves all built indices and metadata to disk."""
        logger.info("💾 Saving indices to cache...")
        loop = asyncio.get_running_loop()

        try:
            # Create directories
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.level_indices_dir.mkdir(parents=True, exist_ok=True)

            # 1. Save main index
            def _write_index(index, path):
                faiss.write_index(index, str(path))

            await loop.run_in_executor(
                None, _write_index, indices.main_index, self.main_index_path
            )

            # 2. Save level indices
            for level, index in indices.level_indices.items():
                level_path = self.level_indices_dir / f'level_{level}_index.bin'
                await loop.run_in_executor(None, _write_index, index, level_path)

            # 3. Save metadata
            metadata = {
                'max_depth': self.max_depth,
                'embedding_shape': self.embeddings.shape,
                'level_data': {
                    level: {'indices': data['indices']}
                    for level, data in indices.level_data.items()
                },
                'cache_version': self.cache_version
            }

            def _write_meta():
                with open(self.metadata_path, 'wb') as f:
                    pickle.dump(metadata, f)

            await loop.run_in_executor(None, _write_meta)
            logger.info("✅ Cache saved successfully.")

        except Exception as e:
            logger.warning(f"Failed to save cache: {e}", exc_info=True)

    async def clear_cache(self) -> None:
        """Asynchronously clear all cached indices from disk."""
        logger.info("🗑️  Clearing FAISS index cache...")
        loop = asyncio.get_running_loop()

        async def _rm_file(path):
            try:
                if path.exists():
                    await loop.run_in_executor(None, path.unlink)
                    logger.info(f"Deleted {path.name}")
            except Exception as e:
                logger.error(f"Failed to delete {path.name}: {e}")

        async def _rm_dir(path):
            try:
                if path.exists():
                    await loop.run_in_executor(None, shutil.rmtree, path)
                    logger.info(f"Deleted directory {path.name}")
            except Exception as e:
                logger.error(f"Failed to delete dir {path.name}: {e}")

        await _rm_file(self.main_index_path)
        await _rm_file(self.metadata_path)
        await _rm_dir(self.level_indices_dir)

        logger.info("✅ Cache cleared.")


# ============================================================
# Dependency Injection Helper
# ============================================================

def get_faiss_service():
    """Dependency injection for FaissIndexService."""
    from app.core.runtime import RuntimeConfig
    import numpy as np

    dummy_cfg = RuntimeConfig()
    dummy_embeddings = np.zeros((1, 768), dtype=np.float32)
    return FaissIndexService(dummy_cfg, dummy_embeddings)
    return FaissIndexService(dummy_cfg, dummy_embeddings)