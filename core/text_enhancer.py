# core/text_enhancer.py
'''Text enhancement for better matching'''

import re
from typing import List

class TextEnhancer:
    '''Enhance text for better embeddings'''
    
    # EXPANDED: All brands that might confuse with categories
    BRANDS = {
        # Watch brands
        'fossil', 'rolex', 'casio', 'timex', 'seiko', 'citizen', 'omega',
        
        # Tech brands
        'apple', 'samsung', 'sony', 'lg', 'dell', 'hp', 'lenovo', 'asus', 
        'acer', 'microsoft', 'google', 'amazon', 'huawei', 'xiaomi',
        
        # Appliance brands
        'bosch', 'whirlpool', 'ge', 'frigidaire', 'kitchenaid', 'maytag', 
        'electrolux', 'panasonic', 'kenmore',
        
        # Fashion/Sports brands
        'nike', 'adidas', 'puma', 'reebok', 'under', 'armour', 'north', 'face',
        'columbia', 'patagonia', 'levi', 'levis', 'gap', 'zara',
        
        # Beauty brands
        'dove', 'olay', 'nivea', 'loreal', 'maybelline', 'covergirl',
        
        # Automotive brands
        'shell', 'mobil', 'castrol', 'valvoline', 'pennzoil', 'jaguar',
        'toyota', 'honda', 'ford', 'chevrolet', 'bmw', 'mercedes',
        
        # Tool brands
        'dewalt', 'black', 'decker', 'craftsman', 'stanley', 'makita',
        
        # Kitchen brands
        'cuisinart', 'ninja', 'breville', 'vitamix', 'oxo',
        
        # Other confusables
        'delta', 'corona', 'titan', 'atlas', 'orion', 'mercury'
    }
    
    # EXPANDED: Product types (PRIORITY!)
    PRODUCT_TYPES = {
        # Electronics
        'phone', 'smartphone', 'iphone', 'android', 'mobile', 'cellphone',
        'laptop', 'computer', 'notebook', 'desktop', 'pc', 'mac',
        'tablet', 'ipad', 'kindle',
        'watch', 'wristwatch', 'smartwatch', 'timepiece', 'chronograph',
        'headphones', 'earbuds', 'earphones', 'airpods',
        'speaker', 'soundbar', 'subwoofer',
        'camera', 'camcorder', 'dslr', 'mirrorless',
        'tv', 'television', 'monitor', 'display', 'screen',
        'keyboard', 'mouse', 'webcam',
        'printer', 'scanner', 'copier',
        'router', 'modem', 'wifi',
        
        # Appliances
        'dishwasher', 'refrigerator', 'fridge', 'freezer',
        'oven', 'stove', 'range', 'cooktop',
        'microwave', 'toaster', 'blender', 'mixer',
        'washer', 'dryer', 'laundry',
        'vacuum', 'cleaner',
        'purifier', 'humidifier', 'dehumidifier',
        'heater', 'fan', 'conditioner', 'thermostat',
        
        # Sports
        'bat', 'cricket', 'football', 'soccer', 'basketball',
        'tennis', 'racket', 'golf', 'club',
        'bicycle', 'bike', 'cycle',
        'treadmill', 'dumbbell', 'weights',
        'shoes', 'sneakers', 'boots', 'sandals',
        
        # Automotive
        'tire', 'tyre', 'wheel', 'rim',
        'battery', 'alternator', 'brake',
        'oil', 'filter', 'spark', 'plug',
        'bumper', 'hood', 'door', 'mirror',
        
        # Home
        'sofa', 'couch', 'chair', 'table', 'desk',
        'bed', 'mattress', 'pillow', 'blanket',
        'lamp', 'light', 'bulb', 'fixture',
        'curtain', 'blind', 'rug', 'carpet',
        
        # Kitchen
        'pot', 'pan', 'skillet', 'wok',
        'knife', 'cutting', 'board',
        'plate', 'bowl', 'cup', 'glass',
        'spoon', 'fork', 'spatula',
        
        # Tools
        'drill', 'saw', 'hammer', 'screwdriver',
        'wrench', 'pliers', 'level',
        
        # Gaming
        'gaming', 'xbox', 'playstation', 'nintendo',
        'console', 'controller', 'joystick',
        
        # Others
        'book', 'toy', 'game', 'puzzle',
        'clothing', 'shirt', 'pants', 'dress',
        'bag', 'backpack', 'luggage', 'wallet'
    }
    
    @staticmethod
    def enhance_product_text(title: str, description: str = '',
                            tags: str = '', product_type: str = '',
                            vendor: str = '') -> str:   
        '''
        Enhance product text for better matching
        
        PRIORITY: Product Type >> Tags >> Brands   
        '''
        
        combined = f'{title} {description} {product_type} {vendor}'
        combined = combined.lower().strip()
        
        tags_cleaned = tags.lower().strip()
        
        brands = TextEnhancer._extract_brands(combined)
        types = TextEnhancer._extract_product_types(combined)
        models = TextEnhancer._extract_model_numbers(combined)
        
        enhanced_parts = [combined]
        
        # ⭐ PRIORITY 1: Product types (5X weight!)
        if types:
            enhanced_parts.extend(list(types) * 5)
        
        # ⭐ PRIORITY 2: Tags (3X weight!)
        if tags_cleaned:
            enhanced_parts.append(tags_cleaned)
            enhanced_parts.append(tags_cleaned)
            enhanced_parts.append(tags_cleaned)
        
        # PRIORITY 3: Brands (1X weight only!)
        if brands:
            enhanced_parts.extend(list(brands))
        
        # PRIORITY 4: Model numbers
        if models:
            enhanced_parts.extend(list(models))
        
        enhanced = ' '.join(enhanced_parts)
        enhanced = re.sub(r'\s+', ' ', enhanced).strip()
        
        return enhanced
    
    @staticmethod
    def _extract_brands(text: str) -> List[str]:
        '''Extract brand names'''
        words = set(re.findall(r'\b\w+\b', text.lower()))
        return list(words & TextEnhancer.BRANDS)
    
    @staticmethod
    def _extract_product_types(text: str) -> List[str]:
        '''Extract product types - HIGHEST PRIORITY!'''
        words = set(re.findall(r'\b\w+\b', text.lower()))
        return list(words & TextEnhancer.PRODUCT_TYPES)
    
    @staticmethod
    def _extract_model_numbers(text: str) -> List[str]:
        '''Extract model numbers'''
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
