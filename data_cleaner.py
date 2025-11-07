"""
🧹 DATA CLEANER & OPTIMIZER
===========================
Cleans your raw category CSV and prepares it for optimal training.

Features:
- Removes extra columns (keeps only ID and path)
- Validates data quality
- Removes duplicates
- Creates clean training CSV
- Generates data statistics

Usage:
    python data_cleaner.py input.csv output.csv
    python data_cleaner.py data/raw_categories.csv data/category_id_path_only.csv
"""

import pandas as pd
import sys
from pathlib import Path
from collections import Counter
import re


def clean_category_data(input_file, output_file):
    """Clean and optimize category data for training"""
    
    print("\n" + "="*80)
    print("🧹 DATA CLEANING & OPTIMIZATION")
    print("="*80 + "\n")
    
    # Load data
    print(f"📂 Loading: {input_file}")
    df = pd.read_csv(input_file, low_memory=False)
    
    print(f"✅ Loaded {len(df):,} rows with {len(df.columns)} columns\n")
    
    # Show sample
    print("📊 Sample of first 3 rows:")
    print(df.head(3).to_string())
    print()
    
    # Extract only category_id and category_path (first 2 columns)
    print("🔍 Extracting category_id and category_path...")
    clean_df = df.iloc[:, :2].copy()
    clean_df.columns = ['category_id', 'category_path']
    
    # Remove rows with missing values
    before_dropna = len(clean_df)
    clean_df = clean_df.dropna()
    dropped_na = before_dropna - len(clean_df)
    if dropped_na > 0:
        print(f"   Removed {dropped_na:,} rows with missing values")
    
    # Remove duplicates based on category_id
    before_dedup = len(clean_df)
    clean_df = clean_df.drop_duplicates(subset=['category_id'])
    dropped_dup = before_dedup - len(clean_df)
    if dropped_dup > 0:
        print(f"   Removed {dropped_dup:,} duplicate category IDs")
    
    # Clean category_path - remove extra whitespace
    clean_df['category_path'] = clean_df['category_path'].str.strip()
    
    # Validate paths
    print("\n🔍 Validating category paths...")
    invalid_paths = []
    for idx, row in clean_df.iterrows():
        path = str(row['category_path'])
        if not path or path == 'nan':
            invalid_paths.append(idx)
        elif '/' not in path:
            print(f"   ⚠️  Warning: No hierarchy separator in: {path}")
    
    if invalid_paths:
        print(f"   Removing {len(invalid_paths)} invalid paths")
        clean_df = clean_df.drop(invalid_paths)
    
    # Analyze hierarchy depth
    print("\n📊 ANALYZING CATEGORY STRUCTURE")
    print("="*80)
    
    depths = []
    final_products = []
    
    for _, row in clean_df.iterrows():
        path = str(row['category_path'])
        levels = [l.strip() for l in path.split('/') if l.strip()]
        depths.append(len(levels))
        if levels:
            final_products.append(levels[-1])
    
    # Statistics
    depth_counts = Counter(depths)
    
    print(f"\n📈 Hierarchy Depth Distribution:")
    for depth in sorted(depth_counts.keys()):
        count = depth_counts[depth]
        bar = "█" * min(50, count // 100)
        print(f"   Level {depth}: {count:>6,} categories {bar}")
    
    print(f"\n📊 Summary Statistics:")
    print(f"   Total categories: {len(clean_df):,}")
    print(f"   Min depth: {min(depths)}")
    print(f"   Max depth: {max(depths)}")
    print(f"   Avg depth: {sum(depths)/len(depths):.1f}")
    print(f"   Unique final products: {len(set(final_products)):,}")
    
    # Show sample final products
    print(f"\n📝 Sample Final Products (the last word in path):")
    product_counts = Counter(final_products)
    for product, count in product_counts.most_common(15):
        print(f"   • {product}: {count:,} categories")
    
    # Show sample paths
    print(f"\n📂 Sample Category Paths:")
    for path in clean_df['category_path'].head(10):
        print(f"   • {path}")
    
    # Check for problematic patterns
    print(f"\n🔍 Data Quality Checks:")
    
    # Check for paths with special characters that might cause issues
    special_chars = clean_df['category_path'].str.contains(r'[^\w\s/&,\'-]', regex=True)
    if special_chars.any():
        print(f"   ⚠️  Found {special_chars.sum()} paths with special characters")
    
    # Check for very long paths
    long_paths = clean_df['category_path'].str.len() > 150
    if long_paths.any():
        print(f"   ⚠️  Found {long_paths.sum()} very long paths (>150 chars)")
        print("      Sample:", clean_df[long_paths]['category_path'].iloc[0][:100] + "...")
    
    # Check for paths with too many levels
    too_deep = [d > 8 for d in depths]
    if any(too_deep):
        print(f"   ⚠️  Found {sum(too_deep)} paths with >8 levels (might be over-specific)")
    
    # Save cleaned data
    print(f"\n💾 Saving cleaned data to: {output_file}")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    clean_df.to_csv(output_file, index=False)
    
    print(f"✅ Saved {len(clean_df):,} clean categories")
    
    # Verify saved file
    verify_df = pd.read_csv(output_file)
    print(f"✅ Verified: {len(verify_df):,} rows, {len(verify_df.columns)} columns")
    
    print("\n" + "="*80)
    print("✅ DATA CLEANING COMPLETE!")
    print("="*80)
    
    print("\n🎯 Next Steps:")
    print("   1. Generate synonyms:")
    print(f"      python synonym_manager.py autobuild {output_file}")
    print("   2. Train model:")
    print(f"      python train.py {output_file}")
    print("   3. Start API server:")
    print("      python api_server.py")
    print("\n" + "="*80 + "\n")
    
    return clean_df


def search_categories(csv_file, search_term):
    """Search for categories containing a term"""
    
    print(f"\n🔍 Searching for: '{search_term}'")
    print("="*80)
    
    df = pd.read_csv(csv_file)
    
    # Case-insensitive search in category_path
    matches = df[df['category_path'].str.contains(search_term, case=False, na=False)]
    
    if len(matches) == 0:
        print(f"❌ No matches found for '{search_term}'")
        return
    
    print(f"✅ Found {len(matches)} matches:\n")
    
    for idx, row in matches.head(20).iterrows():
        cat_id = row['category_id']
        path = row['category_path']
        levels = path.split('/')
        final = levels[-1] if levels else ''
        
        print(f"ID: {cat_id}")
        print(f"   Path: {path}")
        print(f"   Final: {final}")
        print()
    
    if len(matches) > 20:
        print(f"... and {len(matches) - 20} more matches")
    
    print("="*80 + "\n")


def main():
    """Main entry point"""
    
    if len(sys.argv) < 2:
        print("\n❌ Error: Input file required")
        print("\nUsage:")
        print("   python data_cleaner.py <input_csv> <output_csv>")
        print("   python data_cleaner.py <csv_file> search <term>")
        print("\nExamples:")
        print("   python data_cleaner.py data/raw.csv data/category_id_path_only.csv")
        print("   python data_cleaner.py data/category_id_path_only.csv search shoes")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Check if it's a search command
    if len(sys.argv) >= 4 and sys.argv[2].lower() == 'search':
        search_term = sys.argv[3]
        search_categories(input_file, search_term)
        return
    
    # Clean and prepare data
    if len(sys.argv) < 3:
        # Generate output filename
        input_path = Path(input_file)
        output_file = input_path.parent / 'category_id_path_only.csv'
    else:
        output_file = sys.argv[2]
    
    if not Path(input_file).exists():
        print(f"\n❌ Error: Input file not found: {input_file}")
        sys.exit(1)
    
    clean_category_data(input_file, output_file)


if __name__ == "__main__":
    main()