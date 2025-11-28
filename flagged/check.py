"""
🔧 DIAGNOSTIC AND FIX TOOL
===========================
Analyzes your trained model and fixes common issues causing low confidence.

Issues it detects and fixes:
1. Column name mismatches (Category_ID vs category_id)
2. Missing or corrupted tags.json
3. Wrong metadata format in cache
4. FAISS index mismatch

Usage:
    python diagnose_and_fix.py
"""

import pickle
import json
import pandas as pd
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
import sys

def check_cache_files():
    """Check what files exist in cache"""
    cache_dir = Path('cache')
    
    print("\n" + "="*80)
    print("🔍 STEP 1: CHECKING CACHE FILES")
    print("="*80 + "\n")
    
    required_files = {
        'main_index.faiss': cache_dir / 'main_index.faiss',
        'metadata.pkl': cache_dir / 'metadata.pkl',
        'model_info.json': cache_dir / 'model_info.json',
    }
    
    optional_files = {
        'parent_embeddings.pkl': cache_dir / 'parent_embeddings.pkl',
        'calibrator.pkl': cache_dir / 'calibrator.pkl',
        'cross_store_synonyms.pkl': cache_dir / 'cross_store_synonyms.pkl',
    }
    
    issues = []
    
    print("Required files:")
    for name, path in required_files.items():
        if path.exists():
            size = path.stat().st_size / (1024 * 1024)  # MB
            print(f"  ✅ {name} ({size:.2f} MB)")
        else:
            print(f"  ❌ {name} - MISSING")
            issues.append(f"Missing required file: {name}")
    
    print("\nOptional files:")
    for name, path in optional_files.items():
        if path.exists():
            size = path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {name} ({size:.2f} MB)")
        else:
            print(f"  ⚠️  {name} - not found")
    
    return issues


def check_csv_format():
    """Check CSV file format"""
    print("\n" + "="*80)
    print("🔍 STEP 2: CHECKING CSV FORMAT")
    print("="*80 + "\n")
    
    csv_path = Path('data/category_only_path.csv')
    
    if not csv_path.exists():
        print("❌ CSV not found at: data/category_only_path.csv")
        return ["CSV file not found"]
    
    try:
        df = pd.read_csv(csv_path, nrows=5)
        
        print(f"Columns found: {list(df.columns)}")
        print(f"Total rows: {len(pd.read_csv(csv_path)):,}")
        
        print("\nFirst 3 rows:")
        print(df.head(3).to_string())
        
        # Check column names
        if 'Category_ID' in df.columns and 'Category_path' in df.columns:
            print("\n✅ Column format: Uppercase (Category_ID, Category_path)")
            return []
        elif 'category_id' in df.columns and 'category_path' in df.columns:
            print("\n✅ Column format: Lowercase (category_id, category_path)")
            return []
        else:
            print("\n❌ Unexpected column names!")
            return ["CSV has wrong column names"]
    
    except Exception as e:
        print(f"\n❌ Error reading CSV: {e}")
        return [f"CSV read error: {e}"]


def check_metadata():
    """Check metadata format"""
    print("\n" + "="*80)
    print("🔍 STEP 3: CHECKING METADATA FORMAT")
    print("="*80 + "\n")
    
    meta_path = Path('cache/metadata.pkl')
    
    if not meta_path.exists():
        print("❌ Metadata file not found")
        return ["Metadata missing"]
    
    try:
        with open(meta_path, 'rb') as f:
            metadata = pickle.load(f)
        
        print(f"Metadata entries: {len(metadata):,}")
        
        if metadata:
            sample = metadata[0]
            print(f"\nSample entry:")
            print(f"  Keys: {list(sample.keys())}")
            print(f"  category_id: {sample.get('category_id', 'MISSING')}")
            print(f"  category_path: {sample.get('category_path', 'MISSING')[:50]}...")
            
            # Check if all entries have required fields
            missing_fields = []
            for i, entry in enumerate(metadata[:100]):
                if 'category_id' not in entry:
                    missing_fields.append(f"Entry {i}: missing category_id")
                if 'category_path' not in entry:
                    missing_fields.append(f"Entry {i}: missing category_path")
            
            if missing_fields:
                print(f"\n❌ Found {len(missing_fields)} entries with missing fields")
                return missing_fields[:5]  # Return first 5
            else:
                print("\n✅ All entries have required fields")
                return []
        else:
            print("❌ Metadata is empty!")
            return ["Empty metadata"]
    
    except Exception as e:
        print(f"❌ Error reading metadata: {e}")
        return [f"Metadata error: {e}"]


def check_faiss_index():
    """Check FAISS index"""
    print("\n" + "="*80)
    print("🔍 STEP 4: CHECKING FAISS INDEX")
    print("="*80 + "\n")
    
    index_path = Path('cache/main_index.faiss')
    meta_path = Path('cache/metadata.pkl')
    
    if not index_path.exists():
        print("❌ FAISS index not found")
        return ["FAISS index missing"]
    
    try:
        index = faiss.read_index(str(index_path))
        print(f"FAISS index vectors: {index.ntotal:,}")
        print(f"Dimension: {index.d}")
        
        with open(meta_path, 'rb') as f:
            metadata = pickle.load(f)
        
        print(f"Metadata entries: {len(metadata):,}")
        
        if index.ntotal != len(metadata):
            print(f"\n❌ MISMATCH!")
            print(f"   FAISS has {index.ntotal:,} vectors")
            print(f"   Metadata has {len(metadata):,} entries")
            return ["FAISS-metadata count mismatch"]
        else:
            print("\n✅ FAISS and metadata counts match")
            return []
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return [f"FAISS error: {e}"]


def check_tags_json():
    """Check tags.json"""
    print("\n" + "="*80)
    print("🔍 STEP 5: CHECKING TAGS.JSON")
    print("="*80 + "\n")
    
    tags_path = Path('data/tags.json')
    
    if not tags_path.exists():
        print("⚠️  tags.json not found - this will reduce accuracy!")
        print("   Expected location: data/tags.json")
        return ["tags.json missing"]
    
    try:
        with open(tags_path, 'r') as f:
            tags = json.load(f)
        
        print(f"Tags for {len(tags):,} categories")
        
        if tags:
            sample_key = list(tags.keys())[0]
            sample_tags = tags[sample_key]
            
            print(f"\nSample category: {sample_key}")
            print(f"Tags ({len(sample_tags)}): {', '.join(sample_tags[:5])}...")
            
            # Check average tags per category
            tag_counts = [len(t) for t in tags.values() if isinstance(t, list)]
            avg_tags = sum(tag_counts) / len(tag_counts) if tag_counts else 0
            
            print(f"\nAverage tags per category: {avg_tags:.1f}")
            
            if avg_tags < 10:
                print("⚠️  Very few tags - this will reduce accuracy")
                return ["Too few tags per category"]
            else:
                print("✅ Tags look good")
                return []
        else:
            print("❌ tags.json is empty!")
            return ["Empty tags.json"]
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return [f"tags.json error: {e}"]


def test_prediction():
    """Test a sample prediction"""
    print("\n" + "="*80)
    print("🔍 STEP 6: TESTING PREDICTION")
    print("="*80 + "\n")
    
    try:
        print("Loading model...")
        encoder = SentenceTransformer('intfloat/e5-base-v2')
        
        print("Loading FAISS index...")
        index = faiss.read_index('cache/main_index.faiss')
        
        print("Loading metadata...")
        with open('cache/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        # Test query
        test_query = "query: built in dishwasher"
        
        print(f"\nTest query: \"{test_query}\"")
        print("Encoding...")
        
        query_emb = encoder.encode(test_query, convert_to_numpy=True, normalize_embeddings=True)
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        
        print("Searching...")
        distances, indices = index.search(query_emb.astype('float32'), 5)
        
        print("\nTop 5 results:")
        for i in range(5):
            idx = indices[0][i]
            score = distances[0][i]
            meta = metadata[idx]
            
            print(f"\n{i+1}. Score: {score:.4f}")
            print(f"   ID: {meta.get('category_id', 'N/A')}")
            print(f"   Path: {meta.get('category_path', 'N/A')[:60]}...")
        
        best_score = float(distances[0][0])
        
        if best_score < 0.3:
            print(f"\n❌ VERY LOW CONFIDENCE: {best_score:.4f}")
            print("   This indicates a serious problem with training!")
            return ["Very low prediction scores"]
        elif best_score < 0.5:
            print(f"\n⚠️  LOW CONFIDENCE: {best_score:.4f}")
            print("   Model needs improvement")
            return ["Low prediction scores"]
        else:
            print(f"\n✅ GOOD CONFIDENCE: {best_score:.4f}")
            return []
    
    except Exception as e:
        print(f"\n❌ Prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        return [f"Prediction error: {e}"]


def generate_fix_commands(all_issues):
    """Generate commands to fix issues"""
    print("\n" + "="*80)
    print("🔧 RECOMMENDED FIXES")
    print("="*80 + "\n")
    
    if not all_issues:
        print("✅ No critical issues found!")
        print("\nIf you're still experiencing low confidence:")
        print("  1. Make sure you're using tags.json")
        print("  2. Check if validation.csv is being used for calibration")
        print("  3. Verify CSV has correct column names")
        return
    
    print("Issues found:")
    for i, issue in enumerate(all_issues, 1):
        print(f"  {i}. {issue}")
    
    print("\n" + "="*80)
    print("FIX STEPS:")
    print("="*80 + "\n")
    
    if any('missing' in issue.lower() or 'mismatch' in issue.lower() or 'low' in issue.lower() for issue in all_issues):
        print("🔄 RE-TRAINING REQUIRED")
        print("\nRun these commands in order:\n")
        
        print("# Step 1: Generate tags (if missing)")
        print("python generate_hybrid_tags.py data/category_only_path.csv data/tags.json")
        print()
        
        print("# Step 2: Generate validation data (for calibration)")
        print("python create_validation_data.py auto data/category_only_path.csv 200")
        print()
        
        print("# Step 3: Train with ALL fixes")
        print("python train_fixed_v2.py data/category_only_path.csv data/tags.json data/validation.csv")
        print()
    else:
        print("✅ No retraining needed - minor issues only")


def main():
    print("\n" + "="*80)
    print("🔧 DIAGNOSTIC AND FIX TOOL")
    print("="*80)
    print("\nThis tool will analyze your model and identify issues\n")
    
    all_issues = []
    
    # Run all checks
    all_issues.extend(check_cache_files())
    all_issues.extend(check_csv_format())
    all_issues.extend(check_metadata())
    all_issues.extend(check_faiss_index())
    all_issues.extend(check_tags_json())
    all_issues.extend(test_prediction())
    
    # Generate fixes
    generate_fix_commands(all_issues)
    
    print("\n" + "="*80)
    print("📊 DIAGNOSIS COMPLETE")
    print("="*80)
    print(f"\nTotal issues found: {len(all_issues)}")
    print("\n")


if __name__ == "__main__":
    main()