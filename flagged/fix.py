"""
🔧 AUTOMATIC EMBEDDING & INDEX FIXER
====================================
Fixes common issues causing low confidence scores

Usage:
    python fix_embeddings.py normalize          # Fix normalization
    python fix_embeddings.py rebuild-index      # Rebuild FAISS
    python fix_embeddings.py full-fix           # Do everything
"""

import numpy as np
import faiss
import pickle
import sys
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class EmbeddingFixer:
    def __init__(self, cache_dir='cache'):
        self.cache_dir = Path(cache_dir)
        
    def banner(self, text):
        print("\n" + "="*80)
        print(f"🔧 {text}")
        print("="*80 + "\n")
    
    def backup_files(self):
        """Backup existing files"""
        self.banner("CREATING BACKUPS")
        
        backup_dir = self.cache_dir / 'backup'
        backup_dir.mkdir(exist_ok=True)
        
        files_to_backup = [
            'embeddings.npy',
            'main_index.faiss',
            'metadata.pkl'
        ]
        
        for filename in files_to_backup:
            src = self.cache_dir / filename
            if src.exists():
                dst = backup_dir / filename
                import shutil
                shutil.copy2(src, dst)
                print(f"✅ Backed up: {filename}")
        
        print(f"\n📁 Backups saved to: {backup_dir}")
    
    def normalize_embeddings(self):
        """Normalize embeddings to unit length"""
        self.banner("NORMALIZING EMBEDDINGS")
        
        emb_path = self.cache_dir / 'embeddings.npy'
        
        if not emb_path.exists():
            print("❌ embeddings.npy not found!")
            return False
        
        print("Loading embeddings...")
        embeddings = np.load(emb_path)
        
        print(f"Original shape: {embeddings.shape}")
        
        # Check current normalization
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        print(f"Mean norm before: {norms.mean():.6f}")
        print(f"Std norm before: {norms.std():.6f}")
        
        # Normalize
        print("\nNormalizing...")
        embeddings_normalized = embeddings / (norms + 1e-8)
        
        # Verify
        norms_after = np.linalg.norm(embeddings_normalized, axis=1)
        print(f"Mean norm after: {norms_after.mean():.6f}")
        print(f"Std norm after: {norms_after.std():.6f}")
        
        # Save
        output_path = self.cache_dir / 'embeddings.npy'
        np.save(output_path, embeddings_normalized.astype('float32'))
        print(f"\n✅ Saved normalized embeddings: {output_path}")
        
        return True
    
    def rebuild_faiss_index(self):
        """Rebuild FAISS index with correct metric"""
        self.banner("REBUILDING FAISS INDEX")
        
        emb_path = self.cache_dir / 'embeddings.npy'
        
        if not emb_path.exists():
            print("❌ embeddings.npy not found!")
            return False
        
        print("Loading embeddings...")
        embeddings = np.load(emb_path).astype('float32')
        
        print(f"Shape: {embeddings.shape}")
        
        # Ensure normalized
        norms = np.linalg.norm(embeddings, axis=1)
        if abs(norms.mean() - 1.0) > 0.01:
            print("⚠️  Embeddings not normalized, normalizing now...")
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            np.save(emb_path, embeddings)
        
        dimension = embeddings.shape[1]
        
        print(f"\nBuilding FAISS index...")
        print(f"   Dimension: {dimension}")
        print(f"   Vectors: {len(embeddings):,}")
        print(f"   Metric: INNER_PRODUCT")
        
        # Create index with INNER_PRODUCT metric
        index = faiss.IndexFlatIP(dimension)
        
        # Add vectors
        print("\nAdding vectors...")
        index.add(embeddings)
        
        # Save
        index_path = self.cache_dir / 'main_index.faiss'
        faiss.write_index(index, str(index_path))
        
        print(f"\n✅ Saved FAISS index: {index_path}")
        print(f"   Total vectors: {index.ntotal:,}")
        
        return True
    
    def verify_fixes(self):
        """Verify that fixes worked"""
        self.banner("VERIFYING FIXES")
        
        try:
            # Check embeddings
            embeddings = np.load(self.cache_dir / 'embeddings.npy')
            norms = np.linalg.norm(embeddings, axis=1)
            
            print("📊 Embeddings:")
            print(f"   Mean norm: {norms.mean():.6f}")
            print(f"   Std norm: {norms.std():.6f}")
            
            if abs(norms.mean() - 1.0) < 0.01 and norms.std() < 0.01:
                print("   ✅ Properly normalized")
            else:
                print("   ❌ Still not normalized properly")
                return False
            
            # Check FAISS
            index = faiss.read_index(str(self.cache_dir / 'main_index.faiss'))
            
            print(f"\n📊 FAISS Index:")
            print(f"   Vectors: {index.ntotal:,}")
            print(f"   Dimension: {index.d}")
            
            metric = index.metric_type
            if metric == faiss.METRIC_INNER_PRODUCT:
                print("   ✅ Using INNER_PRODUCT")
            else:
                print(f"   ❌ Wrong metric: {metric}")
                return False
            
            # Test search
            print("\n🔍 Testing search...")
            query = embeddings[0:1]
            distances, indices = index.search(query, 5)
            
            print(f"   Top result index: {indices[0][0]}")
            print(f"   Top result score: {distances[0][0]:.6f}")
            
            if distances[0][0] > 0.95:  # Should match itself almost perfectly
                print("   ✅ Search working correctly")
            else:
                print("   ⚠️  Unexpected similarity score")
            
            print("\n✅ ALL CHECKS PASSED!")
            return True
            
        except Exception as e:
            print(f"\n❌ Verification failed: {e}")
            return False
    
    def full_fix(self):
        """Run all fixes"""
        self.banner("RUNNING FULL FIX")
        
        print("This will:")
        print("1. Backup existing files")
        print("2. Normalize embeddings")
        print("3. Rebuild FAISS index")
        print("4. Verify fixes")
        
        print("\nStarting in 3 seconds...")
        import time
        time.sleep(3)
        
        # Backup
        self.backup_files()
        
        # Fix embeddings
        if not self.normalize_embeddings():
            print("\n❌ Failed to normalize embeddings")
            return False
        
        # Rebuild index
        if not self.rebuild_faiss_index():
            print("\n❌ Failed to rebuild index")
            return False
        
        # Verify
        if not self.verify_fixes():
            print("\n❌ Fixes did not work properly")
            return False
        
        print("\n" + "="*80)
        print("✅ ALL FIXES COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nNext steps:")
        print("1. Restart your API server: python api_server.py")
        print("2. Test classification with a known category")
        print("3. Check confidence scores")
        print("\nIf issues persist, run diagnostics:")
        print("   python diagnose_and_fix.py")
        print("="*80 + "\n")
        
        return True


def main():
    if len(sys.argv) < 2:
        print("\n" + "="*80)
        print("🔧 EMBEDDING & INDEX FIXER")
        print("="*80)
        print("\nUsage:")
        print("   python fix_embeddings.py normalize       # Fix normalization only")
        print("   python fix_embeddings.py rebuild-index   # Rebuild FAISS index")
        print("   python fix_embeddings.py full-fix        # Do everything (recommended)")
        print("\nExample:")
        print("   python fix_embeddings.py full-fix")
        print("="*80 + "\n")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    fixer = EmbeddingFixer()
    
    if command == 'normalize':
        fixer.backup_files()
        fixer.normalize_embeddings()
        fixer.verify_fixes()
    
    elif command == 'rebuild-index':
        fixer.backup_files()
        fixer.rebuild_faiss_index()
        fixer.verify_fixes()
    
    elif command == 'full-fix':
        fixer.full_fix()
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Use: normalize, rebuild-index, or full-fix")
        sys.exit(1)


if __name__ == "__main__":
    main()