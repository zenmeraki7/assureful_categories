# strategies/ensemble.py
from typing import Dict, List, Optional

class EnsembleStrategy:
    @staticmethod
    def combine(results: List[Optional[Dict]]):
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            return None
        
        scored_results = []
        for result in valid_results:
            weighted_score = result['similarity'] * result['confidence_weight']
            scored_results.append((weighted_score, result))
        
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        return scored_results[0][1]
