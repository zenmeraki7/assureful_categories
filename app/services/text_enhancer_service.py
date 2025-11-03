"""
app/services/text_enhancer_service.py

This service handles the real-time, per-request logic for
text enhancement and feature engineering. It depends on the
pre-computed vocabularies from the VocabularyService.
"""

import logging
import re
from typing import List, Set
from app.services.vocabulary_service import VocabularyService
from app.schemas import ProductInput # Using our Pydantic model

logger = logging.getLogger(__name__)

class TextEnhancerService:
    """
    Performs real-time text enhancement using pre-computed vocabularies.
    """
    
    def __init__(self, vocab_service: VocabularyService):
        """
        Injects the VocabularyService.
        """
        logger.info("Initializing TextEnhancerService...")
        self.vocab = vocab_service
        self.level_weights = {
            10: 9, 9: 8, 8: 7, 7: 6, 6: 5,
            5: 4, 4: 3, 3: 3, 2: 2, 1: 2
        }

    def enhance_product_text(self, product: ProductInput) -> str:
        """
        Enhances product text with 10-LEVEL PRIORITY weighting.
        """
        
        # Combine all input fields
        combined_text = " ".join(filter(None, [
            product.title, 
            product.description, 
            product.product_type, 
            product.vendor,
            product.tags
        ]))
        combined_lower = combined_text.lower().strip()
        
        if not combined_lower:
            return "" # Return early if no text
            
        # Extract unique words (3+ chars) from the input
        input_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', combined_lower))
        
        # --- 1. Find all matches ---
        
        # Match complete multi-word phrases
        matched_complete_products = {
            phrase for phrase in self.vocab.complete_products 
            if phrase in combined_lower
        }
        
        # Detect gender from title
        detected_gender = self._detect_gender_from_title(
            product.title, product.description
        )
        
        # Match single-word keywords
        matched_types = input_words & self.vocab.product_types
        matched_brands = input_words & self.vocab.brands
        matched_keywords = input_words & self.vocab.category_keywords
        
        # Match level-specific keywords
        matched_levels: dict[int, set[str]] = {
            level_num: input_words & keywords
            for level_num, keywords in self.vocab.level_keywords.items()
        }
        
        # Match gender/age keywords
        matched_gender_age = {
            word for keywords in self.vocab.gender_age_keywords.values()
            for word in (input_words & keywords)
        }

        # --- 2. Build weighted string ---
        enhanced_parts = [product.title] # Start with the original title
        
        # P1: Complete Product Names (15X)
        enhanced_parts.extend(list(matched_complete_products) * 15)
        
        # P2: Gender Detection (12X)
        enhanced_parts.extend(detected_gender * 12)
        
        # P3: Product Types (10X)
        enhanced_parts.extend(list(matched_types) * 10)
        
        # P4-13: Levels 10 down to 1
        for level_num, weight in self.level_weights.items():
            if level_num in matched_levels:
                enhanced_parts.extend(list(matched_levels[level_num]) * weight)
        
        # P15: Gender/Age keywords (2X)
        enhanced_parts.extend(list(matched_gender_age) * 2)
        
        # P16: All keywords (1X)
        enhanced_parts.extend(list(matched_keywords))
        
        # P17: Brands (1X)
        enhanced_parts.extend(list(matched_brands))
        
        # Add description at the end (1X)
        if product.description:
            enhanced_parts.append(product.description)
            
        return ' '.join(enhanced_parts).strip()

    def _detect_gender_from_title(
        self, 
        title: str, 
        description: str | None
    ) -> list[str]:
        """Detects gender using pre-compiled regex patterns."""
        text = f"{title} {description or ''}".lower()
        detected = []

        # Check in priority order
        for category, pattern in self.vocab.gender_patterns.items():
            if pattern.search(text):
                detected.append(category)
                
        # Remove duplicates while preserving order
        unique_detected = list(dict.fromkeys(detected))
        
        # Default to unisex if no other gender is found
        if not unique_detected:
            return ['unisex']
            
        return unique_detected

    def generate_smart_tags(self, product: ProductInput) -> list[str]:
        """
        Generates tags that match the category vocabularies.
        """
        text = " ".join(filter(None, [
            product.title, product.description, product.vendor
        ])).lower()
        
        input_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', text))
        tags = []

        # P1: Gender/Sex from title
        tags.extend(
            self._detect_gender_from_title(product.title, product.description)
        )
        
        # P2: Complete multi-word product names
        tags.extend(sorted(
            phrase for phrase in self.vocab.complete_products 
            if phrase in text and ' ' in phrase
        ))
        
        # P3: Product types
        tags.extend(sorted(input_words & self.vocab.product_types))
        
        # P4: Levels 10 down to 1
        for level_num in sorted(self.vocab.level_keywords.keys(), reverse=True):
            tags.extend(sorted(input_words & self.vocab.level_keywords[level_num]))
            
        # P5: Brands
        tags.extend(sorted(input_words & self.vocab.brands))

        # De-duplicate while preserving priority order
        unique_tags = list(dict.fromkeys(tags))
        return unique_tags[:25] # Return top 25