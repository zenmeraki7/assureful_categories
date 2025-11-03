"""
app/services/vocabulary_service.py

This service is initialized once at startup. It performs the
expensive, one-time analysis of the categories DataFrame to
build all necessary vocabularies (keywords, phrases, etc.)
for text enhancement.
"""

import logging
import re
import pandas as pd
from app.core.runtime import RuntimeConfig

logger = logging.getLogger(__name__)

class VocabularyService:
    """
    Holds all pre-computed vocabularies extracted from the
    category data for fast, real-time lookups.
    """
    
    def __init__(self, cfg: RuntimeConfig):
        """
        Initializes and runs the full extraction pipeline.
        """
        logger.info("--- 🧠 Initializing VocabularyService ---")
        self.categories_df = cfg.category_df
        
        # Define hardcoded sets
        self.brands = self._get_brands()
        self.stop_words = self._get_stop_words()
        self.gender_patterns = self._build_gender_patterns()
        self.gender_indicators = self._get_gender_indicators()

        # --- Run Extraction Pipeline ---
        logger.info(f"Extracting vocabularies from {len(self.categories_df):,} categories...")
        
        # 1. Extract complete phrases (multi-word)
        self.level_complete_names: dict[int, set[str]] = {}
        for i in range(1, cfg.max_depth + 1):
            self.level_complete_names[i] = self._extract_level_complete_names(f'level_{i}')
        
        self.complete_products = self._extract_all_complete_products(cfg.max_depth)

        # 2. Extract single keywords
        self.level_keywords: dict[int, set[str]] = {}
        for i in range(1, cfg.max_depth + 1):
            self.level_keywords[i] = self._extract_level_keywords(f'level_{i}')
        
        self.category_keywords = self._extract_all_category_keywords()
        self.product_types = self.category_keywords.copy() # Start with all keywords
        
        # 3. Extract specialized keywords
        self.gender_age_keywords = self._extract_gender_age_keywords()
        
        self._log_statistics()

    def _get_stop_words(self) -> set[str]:
        return {
            'and', 'the', 'for', 'with', 'from', 'other', 'more', 
            'all', 'new', 'used', 'best', 'top', 'high', 'low', 'sale'
        }

    def _get_brands(self) -> set[str]:
        return {
            'fossil', 'rolex', 'casio', 'timex', 'seiko', 'citizen', 'omega',
            'apple', 'samsung', 'sony', 'lg', 'dell', 'hp', 'lenovo', 'asus',
            'acer', 'microsoft', 'google', 'amazon', 'huawei', 'xiaomi',
            'bosch', 'whirlpool', 'ge', 'frigidaire', 'kitchenaid', 'maytag',
            'nike', 'adidas', 'puma', 'reebok', 'under', 'armour',
            'dewalt', 'black', 'decker', 'craftsman', 'stanley',
            'dove', 'olay', 'nivea', 'loreal', 'maybelline',
            'toyota', 'honda', 'ford', 'bmw', 'mercedes', 'benz'
        }

    def _build_gender_patterns(self) -> dict[str, re.Pattern]:
        """Builds compiled regex patterns for fast gender detection."""
        patterns = {
            'men': [
                r'\bmen\b', r'\bmens\b', r"\bmen's\b", r'\bmale\b',
                r'\bgentleman\b', r'\bgents?\b', r'\bboys?\b',
                r'\bhim\b', r'\bhis\b', r'\bfather\b', r'\bdad\b'
            ],
            'women': [
                r'\bwomen\b', r'\bwomens\b', r"\bwomen's\b", r'\bfemale\b',
                r'\blad(?:y|ies)\b', r'\bgirls?\b', r'\bher\b',
                r'\bmother\b', r'\bmom\b', r'\bmum\b'
            ],
            'boys': [r'\bboys?\b', r"\bboy's\b", r'\blads?\b'],
            'girls': [r'\bgirls?\b', r"\bgirl's\b", r'\blass(?:es)?\b'],
            'kids': [
                r'\bkids?\b', r'\bchildren\b', r'\bchild\b',
                r'\btoddlers?\b', r'\binfants?\b', r'\bbab(?:y|ies)\b',
                r'\bjuniors?\b', r'\byouth\b'
            ],
            'unisex': [r'\bunisex\b', r'\beveryone\b', r'\badult\b']
        }
        # Compile patterns for speed
        return {
            category: re.compile('|'.join(p for p in pats), re.IGNORECASE)
            for category, pats in patterns.items()
        }

    def _get_gender_indicators(self) -> dict[str, list[str]]:
        """Keywords to find gendered categories."""
        return {
            'men': ['men', 'mens', "men's", 'male', 'gentleman', 'gent'],
            'women': ['women', 'womens', "women's", 'female', 'lady', 'ladies'],
            'kids': ['kids', 'children', 'child', 'toddler', 'infant', 'baby'],
            'boys': ['boys', "boy's", 'lad'],
            'girls': ['girls', "girl's", 'lass'],
            'unisex': ['unisex', 'adult', 'everyone']
        }

    def _extract_level_complete_names(self, level_col: str) -> set[str]:
        """Extracts complete, multi-word phrases from a level column."""
        if level_col not in self.categories_df.columns:
            return set()
            
        unique_phrases = self.categories_df[level_col].dropna().astype(str).unique()
        return {
            p.lower().strip() 
            for p in unique_phrases 
            if len(p.strip()) >= 3
        }

    def _extract_all_complete_products(self, max_depth: int) -> set[str]:
        """Merges all level complete names and path parts."""
        complete_products = set()
        for i in range(1, max_depth + 1):
            if i in self.level_complete_names:
                complete_products.update(self.level_complete_names[i])
            
        # Also extract from Category_path
        if 'Category_path' in self.categories_df.columns:
            path_parts = self.categories_df['Category_path'].str.split('/')
            phrases = {
                part.lower().strip()
                for part_list in path_parts.dropna()
                for part in part_list
                if len(part.strip()) >= 3
            }
            complete_products.update(phrases)
            
        return complete_products

    def _extract_level_keywords(self, level_col: str) -> set[str]:
        """Vectorized extraction of single words from a level column."""
        if level_col not in self.categories_df.columns:
            return set()
        
        # Find all 3+ letter words
        series_of_lists = self.categories_df[level_col].str.findall(r'\b[a-zA-Z]{3,}\b')
        
        # Flatten the list of lists into a single set
        keywords = {
            word.lower() 
            for word_list in series_of_lists.dropna() 
            for word in word_list
        }
        return keywords - self.stop_words

    def _extract_all_category_keywords(self) -> set[str]:
        """Vectorized extraction of all single words from all paths."""
        if 'Category_path' not in self.categories_df.columns:
            return set()
            
        all_text_series = self.categories_df['Category_path'].dropna().astype(str)
        series_of_lists = all_text_series.str.findall(r'\b[a-zA-Z]{3,}\b')
        
        keywords = {
            word.lower() 
            for word_list in series_of_lists.dropna() 
            for word in word_list
        }
        return keywords - self.stop_words

    def _extract_gender_age_keywords(self) -> dict[str, set[str]]:
        """Vectorized extraction of gender-related keywords."""
        gender_age = {k: set() for k in self.gender_indicators.keys()}
        
        if 'Category_path' not in self.categories_df.columns:
            return gender_age
            
        all_paths_lower = self.categories_df['Category_path'].dropna().astype(str).str.lower()

        for category, keywords in self.gender_indicators.items():
            # Create a regex pattern like '(men|mens|male)'
            pattern = r'(' + '|'.join(re.escape(k) for k in keywords) + r')'
            
            # Find all paths that match this pattern
            matching_paths = all_paths_lower[all_paths_lower.str.contains(pattern, na=False)]
            
            if matching_paths.empty:
                continue

            # Extract all words from *only* these matching paths
            series_of_lists = matching_paths.str.findall(r'\b[a-zA-Z]{3,}\b')
            words = {
                word for word_list in series_of_lists.dropna() 
                for word in word_list
            }
            gender_age[category].update(words - self.stop_words)
            
        return gender_age

    def _log_statistics(self) -> None:
        """Logs the summary of extracted vocabularies."""
        logger.info("--- 📊 VocabularyService Statistics ---")
        logger.info(f"Complete Products (phrases): {len(self.complete_products):,}")
        logger.info(f"Total Keywords (single words): {len(self.category_keywords):,}")
        logger.info(f"Total Brands: {len(self.brands):,}")
        
        gender_stats = [
            f"{k.capitalize()}: {len(v):,}" 
            for k, v in self.gender_age_keywords.items()
        ]
        logger.info(f"Gender/Age Keywords: {', '.join(gender_stats)}")
        
        level_stats = []
        for i in sorted(self.level_keywords.keys()):
            kw_count = len(self.level_keywords.get(i, set()))
            name_count = len(self.level_complete_names.get(i, set()))
            if kw_count > 0 or name_count > 0:
                level_stats.append(f"L{i} (K:{kw_count:,} | P:{name_count:,})")
        logger.info("Level Stats (Keywords | Phrases):\n" + " | ".join(level_stats))
        logger.info("--- ✅ VocabularyService Initialized ---")