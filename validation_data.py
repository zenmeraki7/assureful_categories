"""
📊 VALIDATION DATA CREATOR
===========================
Helper script to create validation CSV for confidence calibration.

Two modes:
1. Sample from existing categories (automated)
2. Manual entry (interactive)

Output format:
    product_title,true_category_id
    "Oxygen Sensor Tool",12345
    "Hydraulic Oil Additive",67890

Usage:
    # Automated sampling:
    python create_validation_data.py auto data/category_id_path_only.csv
    
    # Manual entry:
    python create_validation_data.py manual
"""

import pandas as pd
import sys
from pathlib import Path
import random


def sample_from_categories(csv_path, num_samples=100, output_file='data/validation.csv'):
    """
    Automatically create validation data by sampling from categories
    and generating product titles based on category paths.
    """
    print("\n" + "="*80)
    print("📊 AUTO-GENERATING VALIDATION DATA")
    print("="*80 + "\n")
    
    # Load categories
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if len(df.columns) < 2:
        print("❌ CSV must have at least 2 columns (category_id, category_path)")
        return False
    
    df.columns = ['category_id', 'category_path'] + list(df.columns[2:])
    df = df.dropna(subset=['category_path'])
    
    print(f"✅ Loaded {len(df):,} categories\n")
    
    # Sample categories
    sample_size = min(num_samples, len(df))
    sampled = df.sample(n=sample_size, random_state=42)
    
    print(f"📝 Generating {sample_size} validation entries...\n")
    
    validation_data = []
    
    for idx, row in sampled.iterrows():
        cat_id = str(row['category_id'])
        cat_path = str(row['category_path'])
        
        # Generate product title from category path
        levels = cat_path.split('/')
        
        # Use last 2-3 levels as product title
        if len(levels) >= 3:
            title_parts = levels[-3:]
        elif len(levels) >= 2:
            title_parts = levels[-2:]
        else:
            title_parts = levels
        
        # Clean and combine
        title = ' '.join(title_parts).strip()
        
        # Add some variation
        variations = [
            title,
            f"{title} kit",
            f"{title} tool",
            f"{title} set",
            f"professional {title}",
            f"{title} replacement",
        ]
        
        product_title = random.choice(variations)
        
        validation_data.append({
            'product_title': product_title,
            'true_category_id': cat_id
        })
    
    # Create DataFrame
    val_df = pd.DataFrame(validation_data)
    
    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    val_df.to_csv(output_path, index=False)
    
    print(f"✅ Created validation file: {output_path}")
    print(f"   Entries: {len(val_df):,}")
    
    # Show samples
    print("\n📝 Sample entries:")
    for i, row in val_df.head(5).iterrows():
        print(f"   {i+1}. \"{row['product_title']}\" → {row['true_category_id']}")
    
    print("\n" + "="*80)
    print("✅ VALIDATION DATA CREATED!")
    print("="*80)
    print(f"\nNext step: Train with calibration")
    print(f"   python train_fixed_v2.py data/category_id_path_only.csv data/tags.json {output_path}")
    print("="*80 + "\n")
    
    return True


def manual_entry(output_file='data/validation_manual.csv'):
    """
    Interactive mode to manually create validation data.
    """
    print("\n" + "="*80)
    print("📝 MANUAL VALIDATION DATA ENTRY")
    print("="*80)
    print("\nEnter product titles and their correct category IDs.")
    print("Press CTRL+C when done.\n")
    
    validation_data = []
    
    try:
        while True:
            print(f"\n--- Entry #{len(validation_data) + 1} ---")
            
            title = input("Product title: ").strip()
            if not title:
                print("⚠️  Title cannot be empty")
                continue
            
            cat_id = input("Category ID: ").strip()
            if not cat_id:
                print("⚠️  Category ID cannot be empty")
                continue
            
            validation_data.append({
                'product_title': title,
                'true_category_id': cat_id
            })
            
            print(f"✅ Added: \"{title}\" → {cat_id}")
            
    except KeyboardInterrupt:
        print("\n\n📊 Entry complete!")
    
    if not validation_data:
        print("❌ No entries created")
        return False
    
    # Create DataFrame
    val_df = pd.DataFrame(validation_data)
    
    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    val_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Created validation file: {output_path}")
    print(f"   Entries: {len(val_df):,}")
    
    print("\n" + "="*80)
    print("✅ VALIDATION DATA CREATED!")
    print("="*80)
    print(f"\nNext step: Train with calibration")
    print(f"   python train_fixed_v2.py data/category_id_path_only.csv data/tags.json {output_path}")
    print("="*80 + "\n")
    
    return True


def verify_validation_file(validation_csv, categories_csv):
    """
    Verify that validation data references valid category IDs.
    """
    print("\n" + "="*80)
    print("🔍 VERIFYING VALIDATION DATA")
    print("="*80 + "\n")
    
    # Load validation data
    print(f"Loading validation: {validation_csv}")
    val_df = pd.read_csv(validation_csv)
    
    if 'product_title' not in val_df.columns or 'true_category_id' not in val_df.columns:
        print("❌ Validation CSV must have: product_title, true_category_id")
        return False
    
    print(f"✅ Loaded {len(val_df):,} validation entries\n")
    
    # Load categories
    print(f"Loading categories: {categories_csv}")
    cat_df = pd.read_csv(categories_csv)
    cat_df.columns = ['category_id', 'category_path'] + list(cat_df.columns[2:])
    
    valid_ids = set(cat_df['category_id'].astype(str))
    print(f"✅ Loaded {len(valid_ids):,} valid category IDs\n")
    
    # Verify
    print("Checking validation entries...")
    invalid_count = 0
    
    for idx, row in val_df.iterrows():
        cat_id = str(row['true_category_id'])
        title = row['product_title']
        
        if cat_id not in valid_ids:
            print(f"❌ Invalid ID: {cat_id} for \"{title}\"")
            invalid_count += 1
    
    if invalid_count == 0:
        print("✅ All validation entries are valid!")
    else:
        print(f"\n⚠️  Found {invalid_count} invalid entries")
    
    # Summary
    print("\n" + "="*80)
    print("📊 VALIDATION DATA SUMMARY")
    print("="*80)
    print(f"Total entries: {len(val_df):,}")
    print(f"Valid entries: {len(val_df) - invalid_count:,}")
    print(f"Invalid entries: {invalid_count}")
    print("="*80 + "\n")
    
    return invalid_count == 0


def main():
    print("\n" + "="*80)
    print("📊 VALIDATION DATA CREATOR")
    print("="*80 + "\n")
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python create_validation_data.py auto <csv_path> [num_samples] [output_file]")
        print("  python create_validation_data.py manual [output_file]")
        print("  python create_validation_data.py verify <validation_csv> <categories_csv>")
        print("\nExamples:")
        print("  # Auto-generate 100 samples:")
        print("  python create_validation_data.py auto data/category_id_path_only.csv")
        print()
        print("  # Auto-generate 200 samples:")
        print("  python create_validation_data.py auto data/category_id_path_only.csv 200")
        print()
        print("  # Manual entry:")
        print("  python create_validation_data.py manual")
        print()
        print("  # Verify validation file:")
        print("  python create_validation_data.py verify data/validation.csv data/category_id_path_only.csv")
        print()
        return
    
    mode = sys.argv[1].lower()
    
    if mode == 'auto':
        if len(sys.argv) < 3:
            print("❌ CSV path required for auto mode")
            print("   python create_validation_data.py auto data/category_id_path_only.csv")
            return
        
        csv_path = sys.argv[2]
        num_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        output_file = sys.argv[4] if len(sys.argv) > 4 else 'data/validation.csv'
        
        if not Path(csv_path).exists():
            print(f"❌ File not found: {csv_path}")
            return
        
        sample_from_categories(csv_path, num_samples, output_file)
    
    elif mode == 'manual':
        output_file = sys.argv[2] if len(sys.argv) > 2 else 'data/validation_manual.csv'
        manual_entry(output_file)
    
    elif mode == 'verify':
        if len(sys.argv) < 4:
            print("❌ Both validation CSV and categories CSV required")
            print("   python create_validation_data.py verify data/validation.csv data/category_id_path_only.csv")
            return
        
        validation_csv = sys.argv[2]
        categories_csv = sys.argv[3]
        
        if not Path(validation_csv).exists():
            print(f"❌ File not found: {validation_csv}")
            return
        
        if not Path(categories_csv).exists():
            print(f"❌ File not found: {categories_csv}")
            return
        
        verify_validation_file(validation_csv, categories_csv)
    
    else:
        print(f"❌ Unknown mode: {mode}")
        print("   Use: auto, manual, or verify")


if __name__ == "__main__":
    main()