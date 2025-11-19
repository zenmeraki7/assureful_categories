"""
🔬 ADVANCED MODEL DIAGNOSTICS & AUTOMATIC FIXES
===============================================
Diagnoses and fixes common issues causing low confidence/accuracy

Usage:
    python diagnose_and_fix.py
"""

import numpy as np
import pandas as pd
import pickle
import json
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from collections import defaultdict, Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class ModelDiagnostics:
    def __init__(self, cache_dir='cache', data_dir='data'):
        self.cache_dir = Path(cache_dir)
        self.data_dir = Path(data_dir)
        self.issues = []
        self.fixes_applied = []
        
    def banner(self, text):
        print("\n" + "="*80)
        print(f"🔍 {text}")
        print("="*80 + "\n")
    
    def check_embedding_normalization(self):
        """Check if embeddings are properly normalized"""
        self.banner("CHECKING EMBEDDING NORMALIZATION")
        
        try:
            embeddings = np.load(self.cache_dir / 'embeddings.npy')
            
            # Check norms
            norms = np.linalg.norm(embeddings, axis=1)
            
            print(f"📊 Embedding Statistics:")
            print(f"   Shape: {embeddings.shape}")
            print(f"   Mean norm: {norms.mean():.6f}")
            print(f"   Std norm: {norms.std():.6f}")
            print(f"   Min norm: {norms.min():.6f}")
            print(f"   Max norm: {norms.max():.6f}")
            
            # Should be ~1.0 if normalized
            if abs(norms.mean() - 1.0) > 0.01 or norms.std() > 0.01:
                self.issues.append({
                    'type': 'CRITICAL',
                    'issue': 'Embeddings not normalized',
                    'details': f'Mean norm: {norms.mean():.6f} (should be ~1.0)',
                    'fix': 'Re-normalize embeddings'
                })
                print("   ❌ ISSUE: Embeddings are NOT normalized!")
                print("   This causes incorrect similarity scores")
                return False
            else:
                print("   ✅ Embeddings properly normalized")
                return True
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    def check_faiss_metric(self):
        """Check FAISS index metric type"""
        self.banner("CHECKING FAISS INDEX METRIC")
        
        try:
            index = faiss.read_index(str(self.cache_dir / 'main_index.faiss'))
            
            metric = index.metric_type
            
            print(f"📊 FAISS Index:")
            print(f"   Vectors: {index.ntotal:,}")
            print(f"   Dimension: {index.d}")
            print(f"   Metric type: {metric}")
            
            if metric == faiss.METRIC_INNER_PRODUCT:
                print("   ✅ Using INNER_PRODUCT (correct for normalized vectors)")
                return True
            elif metric == faiss.METRIC_L2:
                self.issues.append({
                    'type': 'CRITICAL',
                    'issue': 'Wrong FAISS metric',
                    'details': 'Using L2 distance instead of inner product',
                    'fix': 'Rebuild index with METRIC_INNER_PRODUCT'
                })
                print("   ❌ ISSUE: Using L2 distance!")
                print("   Should use INNER_PRODUCT for normalized vectors")
                return False
            else:
                print(f"   ⚠️  Unknown metric: {metric}")
                return False
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    def check_text_weighting(self):
        """Check if text is properly weighted"""
        self.banner("CHECKING TEXT CONSTRUCTION")
        
        try:
            with open(self.cache_dir / 'metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
            
            # Analyze a sample
            sample = metadata[0]
            
            print(f"📊 Sample Category:")
            print(f"   ID: {sample.get('category_id')}")
            print(f"   Path: {sample.get('category_path')}")
            print(f"   Depth: {sample.get('depth')}")
            print(f"   Levels: {sample.get('levels')}")
            
            # Check if we have tags
            if 'auto_tags' in sample and sample['auto_tags']:
                print(f"   Tags: {len(sample['auto_tags'])} tags")
                print(f"   Sample tags: {sample['auto_tags'][:5]}")
                print("   ✅ Auto-tags present")
            else:
                self.issues.append({
                    'type': 'WARNING',
                    'issue': 'Missing auto-tags',
                    'details': 'Categories lack auto-generated tags',
                    'fix': 'Generate tags from category paths'
                })
                print("   ⚠️  No auto-tags found")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    def test_predictions(self, num_samples=100):
        """Test prediction accuracy on random samples"""
        self.banner("TESTING PREDICTION ACCURACY")
        
        try:
            # Load model
            print("Loading model and index...")
            encoder = SentenceTransformer('intfloat/e5-base-v2')
            index = faiss.read_index(str(self.cache_dir / 'main_index.faiss'))
            
            with open(self.cache_dir / 'metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
            
            # Load CSV
            csv_files = list(self.data_dir.glob('*.csv'))
            if not csv_files:
                print("   ❌ No CSV files found in data/")
                return False
            
            df = pd.read_csv(csv_files[0])
            
            # Sample categories
            samples = df.sample(min(num_samples, len(df)))
            
            correct = 0
            confidence_scores = []
            rank_positions = []
            
            print(f"Testing {len(samples)} random categories...\n")
            
            for idx, row in tqdm(samples.iterrows(), total=len(samples)):
                cat_id = str(row.iloc[0])  # First column
                cat_path = str(row.iloc[1])  # Second column
                
                # Get leaf category (final product)
                leaf = cat_path.split('/')[-1].strip()
                
                # Build query
                query = f"query: {leaf}"
                
                # Encode
                query_emb = encoder.encode(query, normalize_embeddings=True)
                query_emb = query_emb.reshape(1, -1).astype('float32')
                
                # Search
                distances, indices = index.search(query_emb, 10)
                
                # Check if correct category is in top results
                found_rank = None
                for rank, idx in enumerate(indices[0]):
                    pred_id = str(metadata[idx]['category_id'])
                    if pred_id == cat_id:
                        found_rank = rank + 1
                        correct += 1
                        confidence_scores.append(float(distances[0][rank]))
                        break
                
                if found_rank:
                    rank_positions.append(found_rank)
                else:
                    rank_positions.append(11)  # Not in top 10
            
            # Calculate metrics
            accuracy = (correct / len(samples)) * 100
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
            
            print(f"\n📊 Results:")
            print(f"   Accuracy (Top-1): {accuracy:.2f}%")
            print(f"   Correct predictions: {correct}/{len(samples)}")
            print(f"   Average confidence: {avg_confidence:.4f}")
            
            if confidence_scores:
                print(f"   Min confidence: {min(confidence_scores):.4f}")
                print(f"   Max confidence: {max(confidence_scores):.4f}")
            
            # Rank distribution
            rank_counts = Counter(rank_positions)
            print(f"\n   Rank Distribution:")
            for rank in sorted(rank_counts.keys())[:5]:
                count = rank_counts[rank]
                pct = (count / len(samples)) * 100
                print(f"   Rank {rank}: {count} ({pct:.1f}%)")
            
            if accuracy < 70:
                self.issues.append({
                    'type': 'CRITICAL',
                    'issue': 'Low prediction accuracy',
                    'details': f'Only {accuracy:.1f}% accuracy',
                    'fix': 'Retrain with better text weighting'
                })
                print(f"\n   ❌ ISSUE: Low accuracy ({accuracy:.1f}%)")
                return False
            elif accuracy < 85:
                self.issues.append({
                    'type': 'WARNING',
                    'issue': 'Moderate accuracy',
                    'details': f'Accuracy: {accuracy:.1f}%',
                    'fix': 'Consider retraining with optimizations'
                })
                print(f"\n   ⚠️  Moderate accuracy ({accuracy:.1f}%)")
                return True
            else:
                print(f"\n   ✅ Good accuracy ({accuracy:.1f}%)")
                return True
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def analyze_category_distribution(self):
        """Analyze category depth and structure"""
        self.banner("ANALYZING CATEGORY STRUCTURE")
        
        try:
            with open(self.cache_dir / 'metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
            
            depths = [m.get('depth', 0) for m in metadata]
            
            print(f"📊 Category Structure:")
            print(f"   Total categories: {len(metadata):,}")
            print(f"   Average depth: {np.mean(depths):.2f}")
            print(f"   Min depth: {min(depths)}")
            print(f"   Max depth: {max(depths)}")
            
            # Depth distribution
            depth_counts = Counter(depths)
            print(f"\n   Depth Distribution:")
            for depth in sorted(depth_counts.keys())[:8]:
                count = depth_counts[depth]
                pct = (count / len(metadata)) * 100
                print(f"   Depth {depth}: {count:,} ({pct:.1f}%)")
            
            # Check for imbalance
            if max(depths) - min(depths) > 5:
                self.issues.append({
                    'type': 'WARNING',
                    'issue': 'Large depth variation',
                    'details': f'Depth ranges from {min(depths)} to {max(depths)}',
                    'fix': 'Consider depth-based weighting'
                })
                print(f"\n   ⚠️  Large depth variation detected")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    def check_duplicate_embeddings(self):
        """Check for duplicate or near-duplicate embeddings"""
        self.banner("CHECKING FOR DUPLICATE EMBEDDINGS")
        
        try:
            embeddings = np.load(self.cache_dir / 'embeddings.npy')
            
            # Sample check (checking all would be too slow)
            sample_size = min(1000, len(embeddings))
            sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
            sample_embs = embeddings[sample_indices]
            
            # Compute pairwise similarities
            similarities = np.dot(sample_embs, sample_embs.T)
            
            # Count very high similarities (excluding diagonal)
            np.fill_diagonal(similarities, 0)
            high_sim = (similarities > 0.99).sum() // 2  # Divide by 2 for symmetry
            
            print(f"📊 Duplicate Check (sample of {sample_size}):")
            print(f"   Very similar pairs (>0.99): {high_sim}")
            
            if high_sim > sample_size * 0.05:  # >5% duplicates
                self.issues.append({
                    'type': 'WARNING',
                    'issue': 'Many duplicate embeddings',
                    'details': f'{high_sim} pairs with >0.99 similarity',
                    'fix': 'Check for duplicate categories or improve text diversity'
                })
                print(f"   ⚠️  Many near-duplicates detected")
                return False
            else:
                print(f"   ✅ Low duplicate rate")
                return True
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    def generate_report(self):
        """Generate diagnostic report"""
        self.banner("DIAGNOSTIC REPORT")
        
        if not self.issues:
            print("✅ NO ISSUES FOUND!")
            print("\nYour model appears to be properly configured.")
            return
        
        # Group by severity
        critical = [i for i in self.issues if i['type'] == 'CRITICAL']
        warnings = [i for i in self.issues if i['type'] == 'WARNING']
        
        if critical:
            print("🔴 CRITICAL ISSUES:")
            for i, issue in enumerate(critical, 1):
                print(f"\n{i}. {issue['issue']}")
                print(f"   Details: {issue['details']}")
                print(f"   Fix: {issue['fix']}")
        
        if warnings:
            print("\n🟡 WARNINGS:")
            for i, issue in enumerate(warnings, 1):
                print(f"\n{i}. {issue['issue']}")
                print(f"   Details: {issue['details']}")
                print(f"   Fix: {issue['fix']}")
        
        print(f"\n📊 Summary:")
        print(f"   Critical issues: {len(critical)}")
        print(f"   Warnings: {len(warnings)}")
    
    def suggest_fixes(self):
        """Suggest fixes based on issues found"""
        self.banner("RECOMMENDED FIXES")
        
        if not self.issues:
            print("✅ No fixes needed!")
            return
        
        print("Run these commands to fix issues:\n")
        
        # Check for critical issues
        critical = [i for i in self.issues if i['type'] == 'CRITICAL']
        
        if any('normalization' in i['issue'].lower() for i in critical):
            print("1️⃣ Fix embedding normalization:")
            print("   python fix_embeddings.py normalize")
            print()
        
        if any('faiss' in i['issue'].lower() for i in critical):
            print("2️⃣ Rebuild FAISS index with correct metric:")
            print("   python fix_embeddings.py rebuild-index")
            print()
        
        if any('accuracy' in i['issue'].lower() for i in critical):
            print("3️⃣ Retrain with improved settings:")
            print("   python train_fixed_v2.py data/categories.csv data/tags.json")
            print()
        
        if any('tags' in i['issue'].lower() for i in self.issues):
            print("4️⃣ Generate missing tags:")
            print("   python generate_tags.py data/categories.csv")
            print()
    
    def run_full_diagnostics(self):
        """Run all diagnostic checks"""
        print("\n" + "="*80)
        print("🔬 COMPREHENSIVE MODEL DIAGNOSTICS")
        print("="*80)
        
        # Run all checks
        self.check_embedding_normalization()
        self.check_faiss_metric()
        self.check_text_weighting()
        self.analyze_category_distribution()
        self.check_duplicate_embeddings()
        self.test_predictions(num_samples=50)
        
        # Generate report
        self.generate_report()
        self.suggest_fixes()
        
        print("\n" + "="*80)
        print("🎯 DIAGNOSTICS COMPLETE")
        print("="*80 + "\n")


if __name__ == "__main__":
    diagnostics = ModelDiagnostics()
    diagnostics.run_full_diagnostics()