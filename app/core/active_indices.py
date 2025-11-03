"""
app/core/active_indices.py

Manages the global, in-memory state of all active FAISS indices
(main and per-level).

Ensures atomic "hot-swap" of the indices using asyncio.Lock.
"""

import asyncio
import faiss
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any, List

# Get a logger for this module
import logging
logger = logging.getLogger(__name__)

@dataclass
class FaissIndices:
    """
    A data container holding all live FAISS indices and their
    associated metadata.
    """
    main_index: faiss.Index | None = None
    level_indices: Dict[int, faiss.Index] = field(default_factory=dict)
    level_data: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    total_vectors: int = 0

# --- In-Memory State ---
_global_state = FaissIndices()
_index_write_lock = asyncio.Lock()

class IndexNotReadyError(Exception):
    """Raised when a search is attempted before indices are loaded."""
    pass

def get_active_indices() -> FaissIndices:
    """
    Safely retrieves the current active FAISS indices and metadata.
    
    This is a non-locking, atomic read of the global state.

    Returns:
        FaissIndices: A dataclass with the active index components.
    
    Raises:
        IndexNotReadyError: If the indices have not been loaded yet.
    """
    current_indices = _global_state
    
    if (
        current_indices.main_index is None or 
        current_indices.total_vectors == 0
    ):
        raise IndexNotReadyError(
            "The FAISS indices are not loaded or are empty."
        )
        
    return current_indices


async def swap_active_indices(new_indices: FaissIndices) -> None:
    """
    Atomically swaps the new indices into the global state.

    This function acquires an asyncio.Lock, making it safe to call
    from background tasks.
    
    Args:
        new_indices: The newly built FaissIndices object.
    """
    global _global_state
    
    async with _index_write_lock:
        logger.info(
            f"Acquired index lock. Swapping to new indices with "
            f"{new_indices.total_vectors} vectors."
        )
        
        # This assignment is atomic.
        _global_state = new_indices
        
        logger.info("Atomic index swap complete. Releasing lock.")