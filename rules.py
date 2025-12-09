
# import json
# import os
# import pandas as pd
# import re
# from collections import Counter

# # ---------------------------------------------------------
# # 1. CONFIGURATION
# # ---------------------------------------------------------
# INPUT_CSV = "data/categories.csv"  # Ensure this path is correct
# OUTPUT_FILE = "data/rules.json"

# # ---------------------------------------------------------
# # 2. YOUR "GOLDEN" MANUAL RULES (High Priority)
# # ---------------------------------------------------------
# # These will NEVER be overwritten by the auto-generator.
# manual_domain_keywords = {
#     "beauty": [
#       "chebe", "tallow", "karkar", "rice water", 
#       "hair", "face", "skin", "body", "scalp",
#       "paste", "pomade", "grease", "butter", "moisturizer", "lotion", 
#       "cream", "serum", "oil", "soap", "shampoo", "conditioner", "balm", 
#       "scrub", "mask", "treatment", "perfume", "makeup", "cosmetic", 
#       "spa", "grooming", "detangler",
#       "alopecia", "regrowth", "thickening", "thinning", "growth",
#       "locks", "tresses", "coils", "curls", 
#       "styling", "gel", "wax", "clay", "putty", "mousse", "foam", "spray"
#     ],
#     "aviation": [
#       "aviation", "pilot", "aircraft", "cockpit", "headset", "fly", 
#       "flight", "airplane", "cessna", "piper", "intercom", "ka-1"
#     ],
#     "computers": [
#       "laptop", "computer", "pc", "macbook", "notebook", "desktop", 
#       "monitor", "screen", "keyboard", "mouse", "webcam", "drive", 
#       "usb", "processor", "cpu", "ram", "memory", "graphics", "stand", 
#       "riser", "mount", "sleeve", "case"
#     ]
# }

# # Words to ignore when auto-generating keywords
# IGNORE_WORDS = {
#     "and", "for", "with", "the", "set", "pack", "kit", "size", "color", 
#     "men", "women", "of", "in", "control", "digital", "auto", "manual", 
#     "electric", "star", "drive", "design", "replacement", "part", "parts", 
#     "universal", "remote", "system", "quality", "cleaner", "wash", 
#     "washable", "machine", "teen", "young", "adult", "kid", "kids", 
#     "child", "children", "mini", "micro", "smart", "pro", "max", "ultra",
#     "series", "edition", "collection", "gift", "premium", "luxury",
#     "sale", "free", "shipping", "oz", "ml", "lb", "kg", "authentic",
#     "supplies", "accessories", "products", "other", "generic", "general"
# }

# # ---------------------------------------------------------
# # 3. HELPER FUNCTIONS
# # ---------------------------------------------------------
# def clean_and_tokenize(text):
#     """Splits text into words, removing special chars."""
#     text = str(text).lower()
#     # Keep only letters
#     words = re.findall(r'\b[a-z]{3,}\b', text)
#     return [w for w in words if w not in IGNORE_WORDS]

# def map_category_name(raw_name):
#     """Maps CSV top-level names to simpler JSON keys."""
#     n = raw_name.lower()
#     if "clothing" in n: return "clothing"
#     if "grocery" in n or "food" in n: return "food"
#     if "health" in n: return "health"
#     if "home" in n or "kitchen" in n: return "home"
#     if "pet" in n: return "pets"
#     if "auto" in n: return "automotive"
#     if "toy" in n: return "toys"
#     if "art" in n or "craft" in n: return "arts_crafts"
#     if "electronics" in n: return "electronics"
#     if "sports" in n: return "sports"
#     if "tool" in n: return "tools"
#     if "industrial" in n: return "industrial"
#     if "office" in n: return "office"
#     return n.replace(" & ", "_").replace(" ", "_").replace(",", "")

# # ---------------------------------------------------------
# # 4. MAIN GENERATOR LOGIC
# # ---------------------------------------------------------
# def generate_rules():
#     print(f"📂 Reading {INPUT_CSV}...")
    
#     # Try different encodings
#     try:
#         df = pd.read_csv(INPUT_CSV, encoding='utf-8', dtype=str).fillna("")
#     except:
#         df = pd.read_csv(INPUT_CSV, encoding='latin1', dtype=str).fillna("")

#     # Identify the Top-Level Column
#     # Usually it's "Top-Level Category (Level 1)" or the first column after ID
#     col_level1 = "Top-Level Category (Level 1)"
#     if col_level1 not in df.columns:
#         # Fallback: Derive from path
#         print("⚠️ 'Level 1' column not found. Deriving from Path...")
#         path_col = df.columns[1]
#         df[col_level1] = df[path_col].apply(lambda x: str(x).split('/')[0])

#     # Initialize domain keywords with manual rules
#     final_domains = manual_domain_keywords.copy()

#     # Iterate over every unique top-level category in the CSV
#     unique_cats = df[col_level1].unique()
#     print(f"🔍 Found {len(unique_cats)} Top-Level Categories.")

#     for cat in unique_cats:
#         if not cat: continue
        
#         # 1. Map to simple key (e.g. "Pet Supplies" -> "pets")
#         simple_key = map_category_name(cat)
        
#         # 2. Get all text for this category
#         subset = df[df[col_level1] == cat]
#         # We use the full path to find common keywords
#         all_text = " ".join(subset.iloc[:, 1].astype(str).tolist())
        
#         # 3. Extract Keywords
#         words = clean_and_tokenize(all_text)
#         common_words = [w for w, count in Counter(words).most_common(60)]
        
#         # 4. Merge into final dict
#         if simple_key not in final_domains:
#             final_domains[simple_key] = []
        
#         # Add new keywords if they aren't already there
#         current_set = set(final_domains[simple_key])
#         for w in common_words:
#             if w not in current_set:
#                 final_domains[simple_key].append(w)
#                 current_set.add(w)

#     # ---------------------------------------------------------
#     # 5. FINAL JSON STRUCTURE
#     # ---------------------------------------------------------
#     full_rules = {
#         "ignore_words": list(IGNORE_WORDS),
#         "domain_keywords": final_domains,
#         "critical_ids": {
#             "iphone_books": "6133978011",
#             "washing_machine": "2383576011",
#             "carrier_phone": "2407748011",
#             "unlocked_phone": "2407749011"
#         }
#     }

#     # Save to file
#     with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#         json.dump(full_rules, f, indent=2)
    
#     print(f"✅ SUCCESSFULLY GENERATED {OUTPUT_FILE}")
#     print(f"📊 Total Categories Covered: {len(final_domains)}")
#     print(f"   Includes specific logic for: {list(manual_domain_keywords.keys())}")
#     print(f"   Plus auto-extracted logic for: {list(final_domains.keys())}")

# if __name__ == "__main__":
#     generate_rules()



import json
import os
import pandas as pd
import re
from collections import Counter

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
INPUT_CSV = "data/categories.csv"
OUTPUT_FILE = "data/rules.json"

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# ---------------------------------------------------------
# 2. THE "GOLDEN RULES" (Manual High-Priority Logic)
# ---------------------------------------------------------
# These keywords ensure your specific problem products (Chebe, Headsets, Gaming) 
# are always routed correctly.
manual_domain_keywords = {
    "beauty": [
        # --- Specific Products ---
        "chebe", "tallow", "karkar", "rice water", 
        # --- Base Terms ---
        "hair", "face", "skin", "body", "scalp", 
        # --- Formats ---
        "paste", "pomade", "grease", "butter", "moisturizer", "lotion", "cream", 
        "serum", "oil", "soap", "shampoo", "conditioner", "balm", "scrub", 
        "mask", "treatment", "perfume", "makeup", "cosmetic", "spa", "grooming", 
        "detangler",
        # --- Hair Specifics (The "Alopecia" Fix) ---
        "alopecia", "regrowth", "thickening", "thinning", "growth", "locks", 
        "tresses", "coils", "curls", "styling", "gel", "wax", "clay", "putty", 
        "mousse", "foam", "spray"
    ],
    "aviation": [
        # --- The "Headset" Fix ---
        "aviation", "pilot", "aircraft", "cockpit", "headset", "fly", "flight", 
        "airplane", "cessna", "piper", "intercom", "ka-1"
    ],
    "computers": [
        # --- The "Gaming" Fix ---
        "gaming", "gamer", "nvidia", "geforce", "rtx", "gtx", "radeon", 
        "alienware", "rog", "msi",
        # --- The "Processor" Fix ---
        "intel", "amd", "ryzen", "core", "ghz", "threadripper",
        # --- Standard Terms ---
        "laptop", "computer", "pc", "macbook", "notebook", "desktop", "monitor", 
        "screen", "keyboard", "mouse", "webcam", "drive", "usb", "processor", 
        "cpu", "ram", "memory", "graphics", "stand", "riser", "mount", 
        "sleeve", "case"
    ],
    "automotive": [
        # --- Common confusions ---
        "oil filter", "air filter", "spark plug", "wiper", "tire", "engine"
    ],
    "pet_supplies": [
        # --- Common confusions ---
        "dog", "cat", "chew", "leash", "collar", "aquarium"
    ]
}

# Words to ignore when auto-learning from CSV
IGNORE_WORDS = {
    "and", "for", "with", "the", "set", "pack", "kit", "size", "color", 
    "men", "women", "of", "in", "control", "digital", "auto", "manual", 
    "electric", "star", "drive", "design", "replacement", "part", "parts", 
    "universal", "remote", "system", "quality", "cleaner", "wash", 
    "washable", "machine", "teen", "young", "adult", "kid", "kids", 
    "child", "children", "mini", "micro", "smart", "pro", "max", "ultra",
    "series", "edition", "collection", "gift", "premium", "luxury",
    "sale", "free", "shipping", "oz", "ml", "lb", "kg", "authentic",
    "supplies", "accessories", "products", "other", "generic", "general",
    "type", "style", "brand", "new", "used"
}

# ---------------------------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------
def clean_and_tokenize(text):
    """Splits text into words, keeps only significant ones."""
    text = str(text).lower()
    words = re.findall(r'\b[a-z]{3,}\b', text)
    return [w for w in words if w not in IGNORE_WORDS]

def map_category_name(raw_name):
    """Maps CSV top-level names to cleaner JSON keys."""
    n = raw_name.lower()
    if "clothing" in n: return "clothing"
    if "grocery" in n or "food" in n: return "food"
    if "health" in n: return "health"
    if "home" in n or "kitchen" in n: return "home"
    if "pet" in n: return "pet_supplies"
    if "auto" in n: return "automotive"
    if "toy" in n: return "toys"
    if "art" in n or "craft" in n: return "arts_crafts"
    if "electronics" in n: return "electronics"
    if "sports" in n: return "sports"
    if "tool" in n: return "tools"
    if "industrial" in n: return "industrial"
    if "office" in n: return "office"
    if "appliance" in n: return "appliances"
    if "music" in n: return "musical_instruments"
    if "baby" in n: return "baby"
    return n.replace(" & ", "_").replace(" ", "_").replace(",", "")

# ---------------------------------------------------------
# 4. MAIN GENERATOR LOGIC
# ---------------------------------------------------------
def generate_rules():
    print(f"📂 Reading {INPUT_CSV}...")
    
    # Robust CSV Reading
    try:
        df = pd.read_csv(INPUT_CSV, encoding='utf-8', dtype=str).fillna("")
    except:
        try:
            df = pd.read_csv(INPUT_CSV, encoding='latin1', dtype=str).fillna("")
        except:
            print("❌ Critical Error: Could not read CSV file. Check format.")
            return

    # Identify the Top-Level Column
    col_level1 = "Top-Level Category (Level 1)"
    if col_level1 not in df.columns:
        # Fallback if column missing: derive from path
        print("⚠️ 'Level 1' column not found. Deriving from Category_path...")
        if len(df.columns) > 1:
            path_col = df.columns[1]
            df[col_level1] = df[path_col].apply(lambda x: str(x).split('/')[0])
        else:
            print("❌ Critical Error: CSV structure unknown.")
            return

    # Start with Manual Rules
    final_domains = manual_domain_keywords.copy()
    
    # Get all unique categories from the file
    unique_cats = df[col_level1].unique()
    print(f"🔍 Found {len(unique_cats)} unique Top-Level Categories.")

    # Loop through every category to learn keywords
    for cat in unique_cats:
        if not cat: continue
        
        # Map simple key (e.g. "Pet Supplies" -> "pet_supplies")
        simple_key = map_category_name(cat)
        
        # Filter data for this category
        subset = df[df[col_level1] == cat]
        
        # Combine all category paths into one giant text
        # (We use column 1 usually for the full path)
        path_col_idx = 1 if len(df.columns) > 1 else 0
        all_text = " ".join(subset.iloc[:, path_col_idx].astype(str).tolist())
        
        # Find top 60 most common words
        words = clean_and_tokenize(all_text)
        common_words = [w for w, count in Counter(words).most_common(60)]
        
        # Initialize list if not present
        if simple_key not in final_domains:
            final_domains[simple_key] = []
        
        # Merge new words (avoiding duplicates)
        current_set = set(final_domains[simple_key])
        for w in common_words:
            if w not in current_set:
                final_domains[simple_key].append(w)
                current_set.add(w)

    # ---------------------------------------------------------
    # 5. SAVE
    # ---------------------------------------------------------
    full_rules = {
        "ignore_words": list(IGNORE_WORDS),
        "domain_keywords": final_domains,
        "critical_ids": {
            "iphone_books": "6133978011",
            "washing_machine": "2383576011",
            "carrier_phone": "2407748011",
            "unlocked_phone": "2407749011"
        }
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(full_rules, f, indent=2)
    
    print(f"✅ SUCCESSFULLY GENERATED {OUTPUT_FILE}")
    print(f"📊 Total Domains: {len(final_domains)}")
    print(f"   Manual Logic Preserved: {list(manual_domain_keywords.keys())}")

if __name__ == "__main__":
    generate_rules()