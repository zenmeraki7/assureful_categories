"""
app/services/search/ensemble_strategy_service.py

Contains the EnsembleStrategyService, which combines results
from multiple search strategies (e.g., direct, progressive)
and selects the best one based on a weighted score.
"""

import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class EnsembleStrategyService:
    """
    Combines results from multiple search strategies.
    It selects the best result based on a weighted score:
    (similarity * confidence_weight)
    """
    
    def __init__(self):
        """
        This service is stateless and requires no configuration.
        """
        logger.info("EnsembleStrategyService initialized.")

    async def combine(
        self, 
        results: List[Optional[Dict[str, Any]]]
    ) -> Dict[str, Any] | None:
        """
        Asynchronously combines a list of strategy results.
        """
        
        # 1. Filter out any 'None' results from failed strategies
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            logger.warning("EnsembleStrategy: No valid results to combine.")
            return None
            
        # 2. Score each result
        scored_results = [
            (
                # Calculate weighted score
                r.get('similarity', 0.0) * r.get('confidence_weight', 0.0),
                r 
            )
            for r in valid_results
        ]
        
        # 3. Sort by score, descending
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        # 4. Select the best one
        best_score, best_result = scored_results[0]
        
        # Log which strategy won
        method = best_result.get('method', 'unknown')
        sim = best_result.get('similarity', 0.0)
        logger.info(
            f"Ensemble selected: '{method}' "
            f"(Final Score: {best_score:.4f}, Original Sim: {sim:.4f})"
        )
        
        return best_result