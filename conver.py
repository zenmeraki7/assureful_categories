import pandas as pd

# === 1. Load your dataset ===
file_path = r"C:\Users\user\assureful_categories\data\categories.csv"
df = pd.read_csv(file_path, low_memory=False)

# === 2. Select only Category_ID and Category_path ===
required_cols = ['Category_ID', 'Category_path']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"⚠️ '{col}' column not found in CSV!")

category_data = df[required_cols].dropna().drop_duplicates().reset_index(drop=True)

print("\n✅ Loaded Category_ID and Category_path columns successfully!")
print(category_data.head())

# === 3. Save the new file ===
output_path = r"C:\Users\user\assureful_categories\data\category_id_path_only.csv"
category_data.to_csv(output_path, index=False)

print(f"\n✅ Saved '{output_path}' with only Category_ID and Category_path columns.")
print(f"🧩 Total unique Category_IDs: {category_data['Category_ID'].nunique()}")
print(f"🧠 Total unique Category_paths: {category_data['Category_path'].nunique()}")

