import pandas as pd
import json
import re
import os

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
INPUT_CSV = 'data/categories.csv'      # Your source data (34k rows)
OUTPUT_JSON = 'data/tags.json'         # The file your classifier needs
OUTPUT_CSV = 'data/categories_enriched.csv' # (Optional) CSV with keywords added for inspection

# ---------------------------------------------------------
# KEYWORD LOGIC
# ---------------------------------------------------------

# Words that are too generic to be useful on their own.
# We will COMBINE these with their parent category (e.g. "Parts" -> "Dishwasher Parts")
GENERIC_TERMS = {
    "accessories", "parts", "supplies", "system", "systems", "kit", "kits",
    "set", "sets", "pack", "packs", "other", "general", "replacement",
    "guide", "guides", "manual", "manuals", "cable", "cables", "wire", "wires",
    "mount", "mounts", "stand", "stands", "holder", "holders", "cover", "covers",
    "case", "cases", "bag", "bags", "protection", "guard", "guards",
    "switch", "switches", "remote", "remotes", "control", "controls",
    "adapter", "adapters", "connector", "connectors", "charger", "chargers",
    "component", "components", "unit", "units", "tool", "tools",
    "hardware", "device", "devices", "equipment", "apparel", "clothing",
    "shoe", "shoes", "boot", "boots", "glove", "gloves", "mask", "masks"
}

def clean_text(text):
    """Cleans text: removes (...) and keeps only letters/numbers."""
    if not isinstance(text, str): return ""
    # Remove text inside parentheses e.g. "Cables (Coaxial)" -> "Cables"
    text = re.sub(r'\([^)]*\)', '', text)
    text = text.replace('&', ' ')
    # Keep only alphanumeric and spaces
    text = re.sub(r'[^a-zA-Z0-9\s-]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip().lower()

def generate_keywords_for_path(path):
    """
    The Core Logic:
    Takes a category path string and returns a list of smart keywords.
    """
    if not isinstance(path, str): return []
    
    parts = path.split('/')
    if not parts: return []
    
    # Clean the path parts
    cleaned_parts = [clean_text(p) for p in parts]
    leaf = cleaned_parts[-1]
    parent = cleaned_parts[-2] if len(cleaned_parts) > 1 else ""
    top_level = cleaned_parts[0] if cleaned_parts else ""
    
    if not leaf: return []
    
    keywords = set()
    
    # 1. Add the leaf term itself (e.g., "Dishwashers")
    keywords.add(leaf)
    
    # 2. Add individual words from the leaf (e.g., "Dishwashers" -> "Dishwashers")
    leaf_words = leaf.split()
    stopwords = {"and", "or", "the", "for", "of", "in", "with", "a", "an", "to", "at", "by", "&"}
    keywords.update([w for w in leaf_words if w not in stopwords])

    # 3. Context Logic: Fix "Generic" words
    # If leaf is "Parts" or "Accessories", grab the parent!
    is_generic = leaf in GENERIC_TERMS or any(w in GENERIC_TERMS for w in leaf_words)
    is_single_word = len(leaf_words) == 1
    
    if (is_generic or is_single_word) and parent:
        # Create combo: "Dishwasher" + "Parts" -> "Dishwasher Parts"
        if parent not in leaf:
            keywords.add(f"{parent} {leaf}")
        
        # Add parent as a standalone keyword if it's specific enough
        if parent not in GENERIC_TERMS:
            keywords.add(parent)

    # 4. CRITICAL FIX: Keyboard Ambiguity
    # Splits "Keyboard" into "Music" or "Computer" based on the folder path
    if "keyboard" in leaf:
        if "musical" in top_level or "instrument" in top_level:
            keywords.add("musical keyboard")
            keywords.add("piano")
        elif "computer" in top_level or "electronics" in top_level:
            keywords.add("computer keyboard")
            keywords.add("pc keyboard")
            keywords.add("typing")

    # 5. CRITICAL FIX: Laundry / Washing Machines
    # Ensures "Godrej Washing Machine" hits the Laundry category
    if "washing machine" in path.lower() or "laundry" in path.lower():
        keywords.add("washing machine")
        keywords.add("laundry")
        keywords.add("washer")

    # Final Cleanup: Remove duplicates and tiny words
    final_list = {k for k in keywords if k not in stopwords and len(k) > 1}
    return sorted(list(final_list))

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: '{INPUT_CSV}' not found. Please place it in this folder.")
        return

    print(f"📂 Loading {INPUT_CSV}...")
    # Load CSV. Handle cases where it might not have headers.
    try:
        df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)
        # Fallback: If columns are numbers (0, 1), rename them
        if 'Category_path' not in df.columns:
            print("   (Detecting headerless CSV structure...)")
            df = pd.read_csv(INPUT_CSV, header=None, dtype=str, keep_default_na=False)
            df.rename(columns={0: 'Category_ID', 1: 'Category_path'}, inplace=True)
    except Exception as e:
        print(f"❌ CSV Read Error: {e}")
        return

    print(f"⚙️  Processing {len(df)} rows to generate keywords...")
    
    # Apply the generator to every row
    df['Generated_Keywords'] = df['Category_path'].apply(generate_keywords_for_path)

    # Create the Dictionary for JSON
    tags_dict = {}
    for _, row in df.iterrows():
        cid = str(row['Category_ID']).strip()
        # Only add valid entries
        if cid and row['Generated_Keywords']:
            tags_dict[cid] = row['Generated_Keywords']

    # OPTIONAL: MERGE WITH OLD TAGS (If you really want to keep "buy/best")
    # Uncomment lines below to merge an existing 'old_tags.json'
    if os.path.exists('tags.json'):
        with open('tags.json', 'r') as f:
            old_tags = json.load(f)
        for cid, old_list in old_tags.items():
            if cid in tags_dict:
                tags_dict[cid] = list(set(tags_dict[cid] + old_list)) # Merge unique
            else:
                tags_dict[cid] = old_list

    print(f"💾 Saving {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(tags_dict, f, indent=4)

    print(f"💾 Saving {OUTPUT_CSV} (for your reference)...")
    # Flatten list to string for CSV readability
    df['Generated_Keywords'] = df['Generated_Keywords'].apply(lambda x: ", ".join(x))
    df.to_csv(OUTPUT_CSV, index=False)

    print("✅ Success! 'tags.json' is ready.")

if __name__ == "__main__":
    main()