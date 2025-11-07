# import pandas as pd
# import json
# import re
# from tqdm import tqdm
# import os

# # ✅ Path setup
# DATA_PATH = "data/category_id_path_only.csv"
# OUTPUT_PATH = "data/auto_tags.json"

# # ✅ Make sure file exists
# if not os.path.exists(DATA_PATH):
#     raise FileNotFoundError(f"❌ File not found: {DATA_PATH}")

# print("📥 Reading category data...")
# df = pd.read_csv(DATA_PATH)

# # Ensure correct column names
# if 'Category_ID' not in df.columns or 'Category_path' not in df.columns:
#     raise ValueError("❌ CSV must contain 'Category_ID' and 'Category_path' columns.")

# # ✅ Function to clean and extract tags from category path
# def extract_tags(category_path):
#     # Split path by '/' and commas
#     parts = re.split(r"[/,]", str(category_path))
    
#     # Clean tokens: remove symbols, trim spaces, lowercase
#     clean_parts = []
#     for p in parts:
#         token = re.sub(r"[^a-zA-Z0-9&+ ]", "", p).strip().lower()
#         if len(token) > 1:
#             clean_parts.append(token)

#     # Remove duplicates while preserving order
#     seen = set()
#     tags = []
#     for x in clean_parts:
#         if x not in seen:
#             seen.add(x)
#             tags.append(x)
    
#     return tags

# # ✅ Generate auto-tags for all categories
# auto_tags_dict = {}

# print("⚙️ Generating auto-tags for all categories...")
# for _, row in tqdm(df.iterrows(), total=len(df)):
#     cat_id = str(row['Category_ID']).strip()
#     path = row['Category_path']
#     tags = extract_tags(path)
#     auto_tags_dict[cat_id] = tags

# # ✅ Save as JSON
# print(f"💾 Saving auto-tags to {OUTPUT_PATH} ...")
# with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
#     json.dump(auto_tags_dict, f, indent=2, ensure_ascii=False)

# print(f"✅ Auto-tags generated successfully for {len(auto_tags_dict)} categories!")
# print(f"📁 Output saved to: {OUTPUT_PATH}")


import pandas as pd
import json
import re
from tqdm import tqdm
import os

# ✅ Path setup
DATA_PATH = "data/category_id_path_only.csv"
OUTPUT_PATH = "data/tags.json"

# ✅ Make sure file exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ File not found: {DATA_PATH}")

print("📥 Reading category data...")
df = pd.read_csv(DATA_PATH)

# Ensure correct column names
if 'Category_ID' not in df.columns or 'Category_path' not in df.columns:
    raise ValueError("❌ CSV must contain 'Category_ID' and 'Category_path' columns.")

# ✅ Function to clean and extract tags from category path
def extract_tags(category_path):
    # Split path by '/'
    parts = str(category_path).split('/')
    
    if len(parts) == 0:
        return []
    
    # Separate all words except last one
    tags = []
    for part in parts[:-1]:
        # Split by spaces, remove symbols, lowercase
        words = re.findall(r'\w+', part.lower())
        tags.extend(words)
    
    # Keep the last part as-is (product or final category)
    last_part = parts[-1].strip().lower()
    if last_part:
        tags.append(last_part)
    
    # Remove duplicates while preserving order
    seen = set()
    final_tags = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            final_tags.append(t)
    
    return final_tags

# ✅ Generate auto-tags for all categories
auto_tags_dict = {}

print("⚙️ Generating auto-tags for all categories...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    cat_id = str(row['Category_ID']).strip()
    path = row['Category_path']
    tags = extract_tags(path)
    auto_tags_dict[cat_id] = tags

# ✅ Save as JSON
print(f"💾 Saving auto-tags to {OUTPUT_PATH} ...")
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(auto_tags_dict, f, indent=2, ensure_ascii=False)

print(f"✅ Auto-tags generated successfully for {len(auto_tags_dict)} categories!")
print(f"📁 Output saved to: {OUTPUT_PATH}")
