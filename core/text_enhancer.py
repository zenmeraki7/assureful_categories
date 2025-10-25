"""
SUPER ENHANCED Text Enhancer
- Extracts COMPLETE PRODUCT NAMES from ALL 10 levels (1-10)
- Better gender/sex detection directly from title
- Maximum coverage: ALL products from categories_fixed.json
"""

import re
import pandas as pd
from typing import Optional


class SuperEnhancedTextEnhancer:
    """
    SUPER ENHANCED - Extracts EVERYTHING from your 33,304 categories
    - Complete product names (not just keywords)
    - ALL levels 1-10 (including level 1)
    - Better gender detection from title
    """
    
    def __init__(self, categories_df: pd.DataFrame):
        """Initialize with your categories data"""
        self.categories_df = categories_df
        
        print("\n" + "=" * 70)
        print("🚀 SUPER ENHANCED - Extracting EVERYTHING from ALL 10 LEVELS")
        print("=" * 70)
        
        # Extract COMPLETE product names from ALL levels (1-10)
        self.complete_products = self._extract_complete_products()
        
        # Extract keywords (individual words) for fallback
        self.category_keywords = self._extract_all_category_keywords()
        self.product_types = self._extract_product_types()
        
        # Extract each level separately for priority weighting
        self.level_keywords = {}
        self.level_complete_names = {}  # NEW: Store complete names per level
        for i in range(1, 11):  # Levels 1-10 (INCLUDING LEVEL 1!)
            level_col = f'level_{i}'
            if level_col in self.categories_df.columns:
                self.level_keywords[i] = self._extract_level_keywords(level_col)
                self.level_complete_names[i] = self._extract_level_complete_names(level_col)
        
        # Enhanced gender/sex detection
        self.gender_patterns = self._build_gender_patterns()
        self.gender_age_keywords = self._extract_gender_age_keywords()
        
        # Brands
        self.brands = {
            'fossil', 'rolex', 'casio', 'timex', 'seiko', 'citizen', 'omega',
            'apple', 'samsung', 'sony', 'lg', 'dell', 'hp', 'lenovo', 'asus',
            'acer', 'microsoft', 'google', 'amazon', 'huawei', 'xiaomi',
            'bosch', 'whirlpool', 'ge', 'frigidaire', 'kitchenaid', 'maytag',
            'nike', 'adidas', 'puma', 'reebok', 'under', 'armour',
            'dewalt', 'black', 'decker', 'craftsman', 'stanley',
            'dove', 'olay', 'nivea', 'loreal', 'maybelline',
            'toyota', 'honda', 'ford', 'bmw', 'mercedes', 'benz'
        }
        
        print(f"✅ Extracted from YOUR 33,304 categories:")
        print(f"   • Complete products: {len(self.complete_products)}")
        print(f"   • Total keywords: {len(self.category_keywords)}")
        print(f"   • Product types: {len(self.product_types)}")
        for level_num in sorted(self.level_keywords.keys()):
            print(f"   • Level {level_num}: {len(self.level_keywords[level_num])} keywords, {len(self.level_complete_names[level_num])} complete names")
        print(f"   • Gender/Age keywords: {sum(len(v) for v in self.gender_age_keywords.values())}")
        print(f"   • Brands: {len(self.brands)}")
        print("=" * 70)
    
    def _extract_complete_products(self) -> set[str]:
        """
        Extract COMPLETE product names from ALL levels (1-10)
        Including full multi-word product names
        """
        complete_products = set()
        
        # Extract from ALL levels (1-10)
        for i in range(1, 11):
            level_col = f'level_{i}'
            if level_col in self.categories_df.columns:
                items = self.categories_df[level_col].dropna().astype(str).unique()
                for item in items:
                    # Add complete name (lowercased, cleaned)
                    clean_item = item.lower().strip()
                    if clean_item and len(clean_item) >= 3:
                        complete_products.add(clean_item)
        
        # Also extract from Category_path (full paths)
        if 'Category_path' in self.categories_df.columns:
            paths = self.categories_df['Category_path'].dropna().astype(str).unique()
            for path in paths:
                # Split by / and add each segment
                parts = path.split('/')
                for part in parts:
                    clean_part = part.lower().strip()
                    if clean_part and len(clean_part) >= 3:
                        complete_products.add(clean_part)
        
        return complete_products
    
    def _extract_level_complete_names(self, level_col: str) -> set[str]:
        """Extract COMPLETE names (not just keywords) from specific level"""
        complete_names = set()
        
        if level_col in self.categories_df.columns:
            items = self.categories_df[level_col].dropna().astype(str).unique()
            for item in items:
                clean_item = item.lower().strip()
                if clean_item and len(clean_item) >= 3:
                    complete_names.add(clean_item)
        
        return complete_names
    
    def _extract_all_category_keywords(self) -> set[str]:
        """Extract ALL keywords (individual words) from category paths"""
        keywords = set()
        
        if 'Category_path' in self.categories_df.columns:
            paths = self.categories_df['Category_path'].dropna().astype(str)
            
            for path in paths:
                # Split by / and extract all words
                parts = path.split('/')
                for part in parts:
                    # Extract individual words (3+ letters)
                    words = re.findall(r'\b[a-zA-Z]{3,}\b', part)
                    for word in words:
                        keywords.add(word.lower())
        
        # Remove common stop words
        stop_words = {
            'and', 'the', 'for', 'with', 'from', 'other', 'more', 
            'all', 'new', 'used', 'best', 'top', 'high', 'low', 'sale'
        }
        keywords = keywords - stop_words
        
        return keywords
    
    def _extract_product_types(self) -> set[str]:
        """
        Extract product types from ALL levels (1-10)
        """
        product_types = set()
        
        # Check ALL levels (1-10) - CHANGED from 2-10 to 1-10
        for i in range(1, 11):
            col = f'level_{i}'
            if col in self.categories_df.columns:
                items = self.categories_df[col].dropna().astype(str)
                for item in items:
                    words = re.findall(r'\b[a-zA-Z]{3,}\b', item)
                    product_types.update([w.lower() for w in words])
        
        return product_types
    
    def _extract_level_keywords(self, level_col: str) -> set[str]:
        """Extract keywords (individual words) from specific level"""
        keywords = set()
        
        if level_col in self.categories_df.columns:
            items = self.categories_df[level_col].dropna().astype(str).unique()
            for item in items:
                words = re.findall(r'\b[a-zA-Z]{3,}\b', item)
                keywords.update([w.lower() for w in words])
        
        return keywords
    
    def _build_gender_patterns(self) -> dict[str, list[str]]:
        """
        Build comprehensive gender/sex detection patterns
        These are used to detect gender directly from title
        """
        return {
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
            'boys': [
                r'\bboys?\b', r"\bboy's\b", r'\blads?\b',
                r'\bkid.*boy\b', r'\bboy.*kid\b'
            ],
            'girls': [
                r'\bgirls?\b', r"\bgirl's\b", r'\blass(?:es)?\b',
                r'\bkid.*girl\b', r'\bgirl.*kid\b'
            ],
            'kids': [
                r'\bkids?\b', r'\bchildren\b', r'\bchild\b',
                r'\btoddlers?\b', r'\binfants?\b', r'\bbab(?:y|ies)\b',
                r'\bjuniors?\b', r'\byouth\b'
            ],
            'unisex': [
                r'\bunisex\b', r'\beveryone\b', r'\badult\b'
            ]
        }
    
    def _extract_gender_age_keywords(self) -> dict[str, set[str]]:
        """
        Extract gender and age-related keywords from categories
        Returns dict with categories: men, women, kids, boys, girls, unisex
        """
        gender_age = {
            'men': set(), 'women': set(), 'kids': set(),
            'boys': set(), 'girls': set(), 'unisex': set()
        }
        
        # Gender/age indicators in category paths
        indicators = {
            'men': ['men', 'mens', "men's", 'male', 'gentleman', 'gent'],
            'women': ['women', 'womens', "women's", 'female', 'lady', 'ladies'],
            'kids': ['kids', 'children', 'child', 'toddler', 'infant', 'baby'],
            'boys': ['boys', "boy's", 'lad'],
            'girls': ['girls', "girl's", 'lass'],
            'unisex': ['unisex', 'adult', 'everyone']
        }
        
        # Extract from category paths
        if 'Category_path' in self.categories_df.columns:
            paths = self.categories_df['Category_path'].dropna().astype(str)
            
            for path in paths:
                path_lower = path.lower()
                
                for category, keywords in indicators.items():
                    for keyword in keywords:
                        if keyword in path_lower:
                            # Extract all words from this category
                            words = re.findall(r'\b[a-zA-Z]{3,}\b', path_lower)
                            gender_age[category].update(words)
        
        return gender_age
    
    def detect_gender_from_title(self, title: str, description: str = "") -> list[str]:
        """
        Detect gender/sex directly from title using enhanced patterns
        Returns list of detected categories: men, women, boys, girls, kids, unisex
        """
        text = f"{title} {description}".lower()
        detected = []
        
        # Check patterns in priority order
        for category, patterns in self.gender_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    detected.append(category)
                    break  # Only add category once
        
        # Remove duplicates while preserving order
        detected = list(dict.fromkeys(detected))
        
        # If no gender detected, default to unisex
        if not detected:
            detected.append('unisex')
        
        return detected
    
    def enhance_product_text(self, title: str, description: str = '',
                            tags: str = '', product_type: str = '',
                            vendor: str = '') -> str:
        """
        Enhance product text with 10-LEVEL PRIORITY weighting + COMPLETE PRODUCTS
        
        PRIORITY ORDER (highest to lowest):
        1. Complete Product Names (15X) - FULL product names from categories
        2. Gender/Sex Detection (12X) - Direct gender from title
        3. Product Types (10X) - Most specific from levels 1-10
        4. Level 10 Keywords (9X) - Deepest level
        5. Level 9 Keywords (8X)
        6. Level 8 Keywords (7X)
        7. Level 7 Keywords (6X)
        8. Level 6 Keywords (5X)
        9. Level 5 Keywords (4X)
        10. User Tags (4X) - Same as level 5
        11. Level 4 Keywords (3X)
        12. Level 3 Keywords (3X)
        13. Level 2 Keywords (2X)
        14. Level 1 Keywords (2X) - NOW INCLUDED!
        15. Gender/Age Keywords (2X)
        16. All Keywords (1X)
        17. Brands (1X)
        """
        
        combined = f'{title} {description} {product_type} {vendor}'
        combined_lower = combined.lower().strip()
        
        # Extract words from input
        input_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', combined_lower))
        
        # NEW: Match complete product names
        matched_complete_products = set()
        for product in self.complete_products:
            if product in combined_lower:
                matched_complete_products.add(product)
        
        # Detect gender directly from title
        detected_gender = self.detect_gender_from_title(title, description)
        
        # Match against product types
        matched_types = input_words & self.product_types
        
        # Match against each level
        matched_levels = {}
        for level_num, keywords in self.level_keywords.items():
            matched_levels[level_num] = input_words & keywords
        
        # Match gender/age keywords
        matched_gender_age = set()
        for category, keywords in self.gender_age_keywords.items():
            matched = input_words & keywords
            matched_gender_age.update(matched)
        
        # Match other keywords
        matched_keywords = input_words & self.category_keywords
        matched_brands = input_words & self.brands
        
        # User tags
        tags_lower = tags.lower()
        user_tags = set(re.findall(r'\b[a-zA-Z]{3,}\b', tags_lower))
        
        # Build enhanced text with priority weights
        enhanced_parts = [title]  # Start with title
        
        # Priority 1: Complete Product Names (15X) - HIGHEST PRIORITY!
        if matched_complete_products:
            enhanced_parts.extend(list(matched_complete_products) * 15)
        
        # Priority 2: Gender Detection (12X)
        if detected_gender:
            enhanced_parts.extend(detected_gender * 12)
        
        # Priority 3: Product types (10X)
        if matched_types:
            enhanced_parts.extend(list(matched_types) * 10)
        
        # Priority 4-13: Levels 10 down to 1 (9X to 2X)
        level_weights = {
            10: 9, 9: 8, 8: 7, 7: 6, 6: 5,
            5: 4, 4: 3, 3: 3, 2: 2, 1: 2  # Level 1 NOW INCLUDED!
        }
        
        for level_num in sorted(self.level_keywords.keys(), reverse=True):
            if level_num in matched_levels and matched_levels[level_num]:
                weight = level_weights.get(level_num, 1)
                enhanced_parts.extend(list(matched_levels[level_num]) * weight)
        
        # Priority 10: User tags (4X)
        if user_tags:
            enhanced_parts.extend(list(user_tags) * 4)
        
        # Priority 15: Gender/Age keywords (2X)
        if matched_gender_age:
            enhanced_parts.extend(list(matched_gender_age) * 2)
        
        # Priority 16: All keywords (1X)
        if matched_keywords:
            enhanced_parts.extend(list(matched_keywords))
        
        # Priority 17: Brands (1X)
        if matched_brands:
            enhanced_parts.extend(list(matched_brands))
        
        # Add description at the end (1X)
        if description:
            enhanced_parts.append(description)
        
        return ' '.join(enhanced_parts).strip()
    
    def generate_smart_tags(self, title: str, description: str = "",
                           vendor: str = "") -> list[str]:
        """
        Generate tags that match YOUR actual categories
        Uses COMPLETE products + ALL levels (1-10) + Gender detection
        Returns tags in priority order
        """
        
        text = f"{title} {description} {vendor}".lower()
        input_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', text))
        
        tags = []
        
        # Priority 1: Gender/Sex from title
        detected_gender = self.detect_gender_from_title(title, description)
        tags.extend(detected_gender)
        
        # Priority 2: Complete product names that match
        for product in self.complete_products:
            if product in text and len(product) > 3:
                # Add meaningful complete products
                if ' ' in product:  # Multi-word products
                    tags.append(product)
        
        # Priority 3: Product types
        matched_types = input_words & self.product_types
        tags.extend(sorted(matched_types))
        
        # Priority 4: Levels 10 down to 1 (ALL LEVELS!)
        for level_num in sorted(self.level_keywords.keys(), reverse=True):
            matched = input_words & self.level_keywords[level_num]
            tags.extend(sorted(matched))
        
        # Priority 5: All other keywords
        matched_keywords = input_words & self.category_keywords
        remaining = matched_keywords - set(tags)
        tags.extend(sorted(remaining))
        
        # Priority 6: Brands
        matched_brands = input_words & self.brands
        tags.extend(sorted(matched_brands))
        
        # Remove duplicates while preserving order
        unique_tags = list(dict.fromkeys(tags))
        
        # Return top 25 (more tags for better coverage)
        return unique_tags[:25]
    
    def detect_gender_age(self, title: str, description: str = "") -> list[str]:
        """
        Detect gender/age category: Men, Women, Kids, Boys, Girls, Unisex
        Uses enhanced pattern matching
        """
        return self.detect_gender_from_title(title, description)
    
    @staticmethod
    def _extract_model_numbers(text: str) -> list[str]:
        """Extract model numbers"""
        patterns = [
            r'\b[A-Z]\d+\b',
            r'\b[A-Z]{2,}-\d+[A-Z]*\d*\b',
            r'\b\d{3,}\b'
        ]
        
        models = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            models.update(matches)
        
        return list(models)
    
    def get_statistics(self) -> dict:
        """Get statistics about extracted data"""
        stats = {
            'complete_products': len(self.complete_products),
            'total_category_keywords': len(self.category_keywords),
            'product_types': len(self.product_types),
            'brands': len(self.brands),
            'gender_age_categories': {k: len(v) for k, v in self.gender_age_keywords.items()},
            'sample_complete_products': list(self.complete_products)[:30]
        }
        
        # Add level statistics
        for level_num in sorted(self.level_keywords.keys()):
            stats[f'level_{level_num}_keywords'] = len(self.level_keywords[level_num])
            stats[f'level_{level_num}_complete_names'] = len(self.level_complete_names[level_num])
            stats[f'sample_level_{level_num}'] = list(self.level_complete_names[level_num])[:10]
        
        return stats


# For backward compatibility
EnhancedUltimateTextEnhancer = SuperEnhancedTextEnhancer


# ============================================================================
# STANDALONE USAGE - Test your categories
# ============================================================================

if __name__ == "__main__":
    """
    Test script to see what gets extracted from ALL 10 levels
    """
    import pandas as pd
    
    print("=" * 70)
    print("SUPER ENHANCED TEXT ENHANCER - Complete Product Analysis")
    print("=" * 70)
    
    try:
        # Try to load your categories
        paths = [
            'categories_fixed.json',
            'data/categories_fixed.json',
            'categories.json',
            'data/categories.json'
        ]
        
        df = None
        for path in paths:
            try:
                df = pd.read_json(path)
                print(f"✅ Loaded categories from: {path}")
                break
            except:
                continue
        
        if df is None:
            print("❌ Could not find categories file")
            print("Place this script in your project folder")
            exit()
        
        # Initialize enhancer
        enhancer = SuperEnhancedTextEnhancer(df)
        
        # Show statistics
        print("\n" + "=" * 70)
        print("EXTRACTED DATA STATISTICS")
        print("=" * 70)
        stats = enhancer.get_statistics()
        
        print(f"\n📦 Complete Products: {stats['complete_products']}")
        print(f"📊 Total Keywords: {stats['total_category_keywords']}")
        print(f"🏷️  Product Types: {stats['product_types']}")
        print(f"🏭 Brands: {stats['brands']}")
        
        print(f"\n👥 Gender/Age Keywords:")
        for category, count in stats['gender_age_categories'].items():
            print(f"   • {category.capitalize()}: {count}")
        
        print(f"\n📁 Level Statistics (Keywords | Complete Names):")
        for i in range(1, 11):
            if f'level_{i}_keywords' in stats:
                kw_count = stats[f'level_{i}_keywords']
                name_count = stats[f'level_{i}_complete_names']
                print(f"   • Level {i}: {kw_count} keywords | {name_count} complete names")
        
        print(f"\n🏷️ Sample Complete Products (first 30):")
        for i, product in enumerate(stats['sample_complete_products'][:30], 1):
            print(f"   {i}. {product}")
        
        # Test examples
        print("\n" + "=" * 70)
        print("TESTING EXAMPLES")
        print("=" * 70)
        
        examples = [
            ("Men's Nike Running Shoes", "Athletic footwear for men", "Nike"),
            ("Women's Cotton Dress", "Summer dress for ladies", ""),
            ("Kids Cricket Bat", "Professional grade bat for children", ""),
            ("Boys Sports Watch", "Digital watch for boys", "Casio"),
            ("Girls Pink T-Shirt", "Cotton shirt for girls", ""),
            ("iPhone 13 Pro", "5G smartphone", "Apple"),
            ("Wireless Earbuds Bluetooth", "True wireless earphones", ""),
        ]
        
        for title, desc, vendor in examples:
            print(f"\n📝 Input: {title}")
            print(f"   Description: {desc}")
            
            # Detect gender/age
            gender_age = enhancer.detect_gender_from_title(title, desc)
            print(f"   👥 Gender Detected: {', '.join(gender_age)}")
            
            # Generate tags
            tags = enhancer.generate_smart_tags(title, desc, vendor)
            print(f"   🏷️  Generated Tags: {', '.join(tags[:15])}")
            
            # Show enhanced text preview
            enhanced = enhancer.enhance_product_text(title, desc, ', '.join(tags), "", vendor)
            print(f"   ✨ Enhanced (first 200 chars): {enhanced[:200]}...")
        
        print("\n" + "=" * 70)
        print("✅ Analysis Complete!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()