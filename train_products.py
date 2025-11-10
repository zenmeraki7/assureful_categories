
# """
# 🎯 ENHANCED TRAINING SYSTEM - Optimized for High Confidence
# ===========================================================
# ✅ Better final product emphasis (10x weight on last word)
# ✅ Improved synonym integration
# ✅ Optimized for 85%+ confidence scores
# ✅ Validates data quality before training

# Usage:
#     python train_enhanced.py data/category_id_path_only.csv
#     python train_enhanced.py data/category_id_path_only.csv data/tags.json
# """

# import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import faiss
# import pickle
# import json
# from pathlib import Path
# from tqdm import tqdm
# import re
# from collections import defaultdict
# import warnings
# import sys
# warnings.filterwarnings('ignore')


# class EnhancedTrainer:
#     """Enhanced trainer with stronger final product emphasis"""
    
#     def __init__(self, cache_dir='cache'):
#         self.cache_dir = Path(cache_dir)
#         self.cache_dir.mkdir(parents=True, exist_ok=True)
        
#         print("\n" + "="*80)
#         print("🎯 ENHANCED CATEGORY PREDICTION SYSTEM")
#         print("="*80)
#         print("✅ 10x emphasis on final product (last word)")
#         print("✅ Optimized for 85%+ confidence")
#         print("✅ AI-powered synonym loading")
#         print("="*80 + "\n")
        
#         # Model (single powerful one for speed)
#         self.model_name = 'sentence-transformers/all-mpnet-base-v2'
#         self.encoder = None
        
#         # Data
#         self.df = None
#         self.embeddings = None
#         self.auto_tags = {}
        
#         # Load AI-powered synonyms
#         self.cross_store_synonyms = self._load_cross_store_synonyms()
        
#         # Other attributes
#         self.path_depth = {}
#         self.path_hierarchy = {}
#         self.max_depth = 0
    
#     def _load_cross_store_synonyms(self):
#         """Load AI-generated synonyms from synonym_manager.py"""
#         synonyms_file = self.cache_dir / 'cross_store_synonyms.pkl'
        
#         if synonyms_file.exists():
#             print("📥 Loading AI-generated synonyms...")
#             try:
#                 with open(synonyms_file, 'rb') as f:
#                     synonyms = pickle.load(f)
#                 print(f"✅ Loaded {len(synonyms):,} AI-generated synonym mappings\n")
#                 return synonyms
#             except Exception as e:
#                 print(f"⚠️  Error loading AI synonyms: {e}")
#                 print("   Falling back to basic synonyms...\n")
#         else:
#             print("⚠️  AI-generated synonyms not found")
#             print("   💡 Generate with: python synonym_manager.py autobuild data/category_id_path_only.csv")
#             print("   Using basic synonyms for now...\n")
        
#         return self._build_basic_synonyms()
    
#     def _build_basic_synonyms(self):
#         """Basic fallback synonyms"""
#         synonyms = {
#             # Footwear
#             'shoes': {'footwear', 'sneakers', 'boots'},
#             'sneakers': {'shoes', 'trainers', 'athletic shoes', 'running shoes'},
#             'boots': {'footwear', 'shoes'},
#             'sandals': {'footwear', 'shoes', 'flip flops'},
            
#             # Clothing
#             'pants': {'trousers', 'slacks', 'bottoms'},
#             'shirt': {'top', 'blouse', 'tee'},
#             'jacket': {'coat', 'outerwear'},
#             'dress': {'frock', 'gown'},
            
#             # Electronics
#             'tv': {'television', 'smart tv'},
#             'phone': {'mobile', 'smartphone', 'cell phone'},
#             'laptop': {'notebook', 'computer'},
            
#             # Appliances
#             'washing machine': {'washer', 'laundry machine'},
#             'refrigerator': {'fridge', 'cooler'},
            
#             # General
#             'kids': {'children', 'childrens', 'youth'},
#             'women': {'womens', 'ladies', 'female'},
#             'men': {'mens', 'male', 'guys'},
#         }
        
#         # Build bidirectional
#         expanded = {}
#         for term, syns in synonyms.items():
#             expanded[term] = syns.copy()
#             for syn in syns:
#                 if syn not in expanded:
#                     expanded[syn] = set()
#                 expanded[syn].add(term)
#                 expanded[syn].update(syns - {syn})
        
#         print(f"📚 Using {len(expanded)} basic synonym mappings\n")
#         return expanded
    
#     def load_data(self, filepath):
#         """Load and validate category data"""
#         print("📂 Loading category data...")
        
#         filepath = Path(filepath)
#         if not filepath.exists():
#             raise FileNotFoundError(f"CSV file not found: {filepath}")
        
#         # Read CSV - only first 2 columns
#         self.df = pd.read_csv(filepath, usecols=[0, 1], names=['category_id', 'category_path'], 
#                              header=0, low_memory=False)
        
#         # Remove any NaN
#         before = len(self.df)
#         self.df = self.df.dropna()
#         after = len(self.df)
        
#         if before != after:
#             print(f"   Removed {before - after} rows with missing data")
        
#         print(f"✅ Loaded {len(self.df):,} valid categories")
        
#         # Validate paths
#         sample_paths = self.df['category_path'].head(5).tolist()
#         print(f"\n📝 Sample paths:")
#         for path in sample_paths:
#             print(f"   • {path}")
#         print()
        
#         return True
    
#     def load_auto_tags(self, json_path):
#         """Load auto-generated tags from JSON"""
#         print("📂 Loading auto-generated tags...")
        
#         json_path = Path(json_path)
#         if not json_path.exists():
#             print(f"⚠️  Auto-tags file not found: {json_path}")
#             print("   Continuing without auto-tags...\n")
#             return False
        
#         try:
#             with open(json_path, 'r', encoding='utf-8') as f:
#                 self.auto_tags = json.load(f)
            
#             print(f"✅ Loaded tags for {len(self.auto_tags):,} categories\n")
#             return True
#         except Exception as e:
#             print(f"⚠️  Error loading auto-tags: {e}")
#             return False
    
#     def load_model(self):
#         """Load sentence transformer"""
#         print(f"🤖 Loading model: {self.model_name}")
#         print("   (This may take a minute on first run...)\n")
        
#         self.encoder = SentenceTransformer(self.model_name)
#         print("✅ Model loaded\n")
    
#     def clean_text(self, text):
#         """Clean text"""
#         if pd.isna(text):
#             return ""
#         text = str(text).lower()
#         text = re.sub(r'[^\w\s-]', ' ', text)
#         text = re.sub(r'\s+', ' ', text).strip()
#         return text
    
#     def split_path(self, path):
#         """Split path into levels"""
#         if pd.isna(path):
#             return []
#         path = str(path).strip()
#         levels = [l.strip() for l in path.split('/') if l.strip()]
#         return levels
    
#     def extract_terms_with_synonyms(self, text):
#         """Extract terms and expand with synonyms"""
#         cleaned = self.clean_text(text)
#         words = cleaned.split()
        
#         all_terms = set()
#         all_terms.add(cleaned)
        
#         # Single words + synonyms
#         for word in words:
#             if len(word) > 2:
#                 all_terms.add(word)
#                 if word in self.cross_store_synonyms:
#                     all_terms.update(list(self.cross_store_synonyms[word])[:5])
        
#         # 2-word phrases + synonyms
#         for i in range(len(words) - 1):
#             if len(words[i]) > 2 and len(words[i+1]) > 2:
#                 phrase = f"{words[i]} {words[i+1]}"
#                 all_terms.add(phrase)
#                 if phrase in self.cross_store_synonyms:
#                     all_terms.update(list(self.cross_store_synonyms[phrase])[:5])
        
#         return list(all_terms)
    
#     def analyze_paths(self):
#         """Analyze all paths"""
#         print("🔍 ANALYZING CATEGORY STRUCTURE")
#         print("="*80)
        
#         for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Analyzing"):
#             cat_path = str(row['category_path'])
#             levels = self.split_path(cat_path)
            
#             self.path_hierarchy[cat_path] = levels
#             self.path_depth[cat_path] = len(levels)
#             self.max_depth = max(self.max_depth, len(levels))
        
#         print(f"\n✅ Analysis complete!")
#         print(f"   Max depth: {self.max_depth}")
#         print(f"   Synonym terms: {len(self.cross_store_synonyms):,}")
#         print("="*80 + "\n")
    
#     def build_enhanced_text(self, row):
#         """Build training text with HEAVY emphasis on final product"""
#         cat_path = str(row['category_path'])
#         cat_id = str(row['category_id'])
#         levels = self.path_hierarchy.get(cat_path, [])
        
#         if not levels:
#             return "unknown"
        
#         components = []
        
#         # 1. FINAL PRODUCT - MASSIVE EMPHASIS (the most important!)
#         final_product = levels[-1]
#         final_product_clean = self.clean_text(final_product)
        
#         # Repeat final product 15x for maximum weight
#         components.append(' '.join([final_product_clean] * 15))
        
#         # Add synonym variations of final product 5x
#         final_product_terms = self.extract_terms_with_synonyms(final_product)
#         components.append(' '.join(final_product_terms * 5))
        
#         # 2. Auto-tags (if available) - high weight
#         auto_tags = self.auto_tags.get(cat_id, []) or self.auto_tags.get(cat_path, [])
#         if auto_tags:
#             components.append(' '.join(auto_tags[:30]))
#             components.append(' '.join(auto_tags[:15]))
        
#         # 3. Parent level (second to last) - medium weight
#         if len(levels) >= 2:
#             parent = self.clean_text(levels[-2])
#             components.append(' '.join([parent] * 3))
            
#             # Combine parent + final
#             combined = f"{parent} {final_product_clean}"
#             components.append(' '.join([combined] * 3))
        
#         # 4. Full hierarchy with emphasis on deeper levels
#         for i, level in enumerate(levels):
#             cleaned = self.clean_text(level)
#             if i == len(levels) - 1:  # Last level (already heavily weighted above)
#                 components.append(cleaned)
#             elif i == len(levels) - 2:  # Parent (already added above)
#                 components.append(cleaned)
#             elif i == len(levels) - 3:  # Grandparent
#                 components.append(' '.join([cleaned] * 2))
#             else:  # Top levels
#                 components.append(cleaned)
        
#         # 5. Synonym expansion for all levels
#         for level in levels:
#             terms = self.extract_terms_with_synonyms(level)
#             components.append(' '.join(terms[:10]))
        
#         # 6. Full path for context
#         components.append(cat_path.lower())
        
#         return ' '.join(components)
    
#     def prepare_texts(self):
#         """Prepare all training texts"""
#         print("📝 PREPARING ENHANCED TEXTS")
#         print("="*80)
        
#         texts = []
#         for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing"):
#             enhanced = self.build_enhanced_text(row)
#             texts.append(enhanced)
        
#         print(f"\n✅ Prepared {len(texts):,} enhanced texts")
        
#         # Show sample
#         if texts:
#             print("\n📝 Sample enhanced text (first 200 chars):")
#             print(f"   {texts[0][:200]}...")
        
#         print("="*80 + "\n")
        
#         return texts
    
#     def encode_texts(self, texts, batch_size=32):
#         """Encode texts with single powerful model"""
#         print("🔄 ENCODING TEXTS")
#         print("="*80)
        
#         embeddings = self.encoder.encode(
#             texts,
#             batch_size=batch_size,
#             show_progress_bar=True,
#             convert_to_numpy=True,
#             normalize_embeddings=True
#         )
        
#         embeddings = np.array(embeddings, dtype='float32')
        
#         print(f"\n✅ Encoded to shape: {embeddings.shape}")
#         print("="*80 + "\n")
        
#         return embeddings
    
#     def build_faiss_index(self):
#         """Build FAISS index"""
#         print("🔍 BUILDING FAISS INDEX")
#         print("="*80)
        
#         dimension = self.embeddings.shape[1]
        
#         # Use Inner Product for cosine similarity (embeddings are normalized)
#         index = faiss.IndexFlatIP(dimension)
#         index.add(self.embeddings)
        
#         # Save index
#         index_path = self.cache_dir / 'main_index.faiss'
#         faiss.write_index(index, str(index_path))
        
#         print(f"✅ Built index with {index.ntotal:,} vectors")
#         print(f"✅ Saved to: {index_path}")
#         print("="*80 + "\n")
    
#     def save_all_data(self):
#         """Save all training data"""
#         print("💾 SAVING TRAINING DATA")
#         print("="*80)
        
#         # Save embeddings
#         emb_path = self.cache_dir / 'embeddings.npy'
#         np.save(emb_path, self.embeddings)
#         print(f"✅ Saved: {emb_path}")
        
#         # Save metadata
#         metadata = []
#         for idx, row in self.df.iterrows():
#             cat_id = str(row['category_id'])
#             cat_path = str(row['category_path'])
            
#             metadata.append({
#                 'category_id': cat_id,
#                 'category_path': cat_path,
#                 'auto_tags': self.auto_tags.get(cat_id, []) or self.auto_tags.get(cat_path, []),
#                 'depth': self.path_depth.get(cat_path, 0),
#                 'levels': self.path_hierarchy.get(cat_path, [])
#             })
        
#         meta_path = self.cache_dir / 'metadata.pkl'
#         with open(meta_path, 'wb') as f:
#             pickle.dump(metadata, f)
#         print(f"✅ Saved: {meta_path} ({len(metadata):,} entries)")
        
#         # Save synonyms
#         syn_path = self.cache_dir / 'cross_store_synonyms.pkl'
#         with open(syn_path, 'wb') as f:
#             pickle.dump(self.cross_store_synonyms, f)
#         print(f"✅ Saved: {syn_path} ({len(self.cross_store_synonyms):,} terms)")
        
#         print("="*80 + "\n")
    
#     def train(self, csv_path, json_path=None):
#         """Complete training pipeline"""
#         print("\n" + "="*80)
#         print("🚀 STARTING ENHANCED TRAINING")
#         print("="*80 + "\n")
        
#         # Load data
#         self.load_data(csv_path)
        
#         # Load auto-tags if provided
#         if json_path:
#             self.load_auto_tags(json_path)
        
#         # Analyze paths
#         self.analyze_paths()
        
#         # Load model
#         self.load_model()
        
#         # Prepare texts
#         texts = self.prepare_texts()
        
#         # Encode
#         self.embeddings = self.encode_texts(texts)
        
#         # Build FAISS index
#         self.build_faiss_index()
        
#         # Save everything
#         self.save_all_data()
        
#         # Summary
#         print("\n" + "="*80)
#         print("✅ TRAINING COMPLETE!")
#         print("="*80)
#         print(f"📊 Summary:")
#         print(f"   Categories trained: {len(self.df):,}")
#         print(f"   Max hierarchy depth: {self.max_depth}")
#         print(f"   Synonym terms: {len(self.cross_store_synonyms):,}")
#         print(f"   Auto-tags loaded: {'Yes' if self.auto_tags else 'No'}")
#         print(f"   Embedding dimension: {self.embeddings.shape[1]}")
#         print(f"\n🎯 Optimizations:")
#         print(f"   ✅ 15x emphasis on final product")
#         print(f"   ✅ 5x synonym expansion")
#         print(f"   ✅ Cross-store intelligence")
#         print(f"   ✅ Optimized for 85%+ confidence")
#         print(f"\n📁 Saved files:")
#         print(f"   {self.cache_dir}/main_index.faiss")
#         print(f"   {self.cache_dir}/metadata.pkl")
#         print(f"   {self.cache_dir}/embeddings.npy")
#         print(f"   {self.cache_dir}/cross_store_synonyms.pkl")
#         print("="*80 + "\n")
        
#         print("🚀 Next: Start API server")
#         print("   python api_server.py\n")


# def main():
#     """Main entry point"""
#     if len(sys.argv) < 2:
#         print("\n❌ Error: CSV file path required")
#         print("\nUsage:")
#         print("   python train_enhanced.py <csv_path> [json_path]")
#         print("\nExamples:")
#         print("   python train_enhanced.py data/category_id_path_only.csv")
#         print("   python train_enhanced.py data/category_id_path_only.csv data/tags.json")
#         sys.exit(1)
    
#     csv_path = sys.argv[1]
#     json_path = sys.argv[2] if len(sys.argv) > 2 else None
    
#     if not Path(csv_path).exists():
#         print(f"\n❌ Error: CSV file not found: {csv_path}")
#         sys.exit(1)
    
#     trainer = EnhancedTrainer()
#     trainer.train(csv_path, json_path)


# if __name__ == "__main__":
#     main()






"""
🎯 FIXED TRAINING SYSTEM (Windows + NVIDIA GPU)
================================================
✅ Uses e5-base-v2 (lower memory, 768D)
✅ Windows + NVIDIA GPU optimized
✅ Proper error handling
✅ 15x emphasis on final product

Usage:
    python train_fixed.py data/category_id_path_only.csv
    python train_fixed.py data/category_id_path_only.csv data/tags.json
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import json
from pathlib import Path
from tqdm import tqdm
import re
from collections import defaultdict
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# Fix Windows CUDA issues
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available")


class FixedTrainer:
    """Fixed trainer for Windows + NVIDIA GPU"""
    
    def __init__(self, cache_dir='cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*80)
        print("🎯 CATEGORY PREDICTION TRAINING (Windows + NVIDIA GPU)")
        print("="*80)
        print("✅ E5-Base-v2 (768D, memory-efficient)")
        print("✅ Proper E5 formatting (passage: prefix)")
        print("✅ 15x emphasis on final product")
        print("✅ NVIDIA GPU acceleration")
        print("="*80 + "\n")
        
        # Use e5-base-v2 for better memory efficiency
        self.model_name = 'intfloat/e5-base-v2'
        self.encoder = None
        self.device = "cpu"
        
        # Data
        self.df = None
        self.embeddings = None
        self.auto_tags = {}
        
        # Load synonyms
        self.cross_store_synonyms = self._load_cross_store_synonyms()
        
        # Analysis
        self.path_depth = {}
        self.path_hierarchy = {}
        self.max_depth = 0
    
    def _load_cross_store_synonyms(self):
        """Load AI-generated synonyms"""
        synonyms_file = self.cache_dir / 'cross_store_synonyms.pkl'
        
        if synonyms_file.exists():
            print("📥 Loading AI-generated synonyms...")
            try:
                with open(synonyms_file, 'rb') as f:
                    synonyms = pickle.load(f)
                
                # Handle different formats
                if synonyms and list(synonyms.values()):
                    first_val = next(iter(synonyms.values()))
                    
                    if isinstance(first_val, list) and first_val:
                        if isinstance(first_val[0], tuple):
                            # New format: extract just the synonym words
                            cleaned = {}
                            for term, syn_list in synonyms.items():
                                cleaned[term] = {syn for syn, conf, src in syn_list}
                            synonyms = cleaned
                        elif isinstance(first_val[0], str):
                            # Already good
                            pass
                    elif isinstance(first_val, set):
                        # Already good
                        pass
                
                print(f"✅ Loaded {len(synonyms):,} synonym mappings\n")
                return synonyms
            except Exception as e:
                print(f"⚠️  Error loading synonyms: {e}\n")
        
        return self._build_basic_synonyms()
    
    def _build_basic_synonyms(self):
        """Basic synonyms as fallback"""
        synonyms = {
            # Footwear
            'shoes': {'footwear', 'sneakers', 'boots'},
            'sneakers': {'shoes', 'trainers', 'athletic shoes'},
            'boots': {'footwear', 'shoes'},
            
            # Clothing
            'pants': {'trousers', 'slacks', 'bottoms', 'jeans'},
            'shirt': {'top', 'blouse', 'tee', 't-shirt'},
            'jacket': {'coat', 'outerwear', 'blazer'},
            
            # Electronics
            'tv': {'television', 'smart tv'},
            'phone': {'mobile', 'smartphone', 'cell phone'},
            'laptop': {'notebook', 'computer'},
            'headphones': {'earphones', 'headset', 'earbuds'},
            
            # Appliances
            'washing machine': {'washer', 'laundry machine'},
            'refrigerator': {'fridge', 'cooler'},
            'microwave': {'microwave oven'},
            
            # General
            'kids': {'children', 'childrens', 'youth', 'junior'},
            'women': {'womens', 'ladies', 'female'},
            'men': {'mens', 'male', 'gents'},
        }
        
        print(f"📚 Using {len(synonyms)} basic synonyms\n")
        return synonyms
    
    def load_data(self, filepath):
        """Load category data"""
        print("📂 Loading category data...")
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"CSV not found: {filepath}")
        
        try:
            # Read CSV - handle with/without header
            self.df = pd.read_csv(filepath, dtype=str)
            
            # Rename columns if needed
            if len(self.df.columns) >= 2:
                self.df.columns = ['category_id', 'category_path'] + list(self.df.columns[2:])
            else:
                raise ValueError("CSV must have at least 2 columns")
            
            # Clean data
            before = len(self.df)
            self.df = self.df.dropna(subset=['category_path'])
            self.df.drop_duplicates(subset=['category_path'], inplace=True)
            after = len(self.df)
            
            if before != after:
                print(f"   Cleaned: {before - after} rows removed")
            
            print(f"✅ Loaded {len(self.df):,} unique categories")
            
            # Show samples
            print(f"\n📝 Sample paths:")
            for path in self.df['category_path'].head(3):
                print(f"   • {path}")
            print()
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_auto_tags(self, json_path):
        """Load auto-tags (optional)"""
        print("📂 Loading auto-tags...")
        
        json_path = Path(json_path)
        if not json_path.exists():
            print(f"ℹ️  Not found, continuing without tags\n")
            return False
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.auto_tags = json.load(f)
            print(f"✅ Loaded tags for {len(self.auto_tags):,} categories\n")
            return True
        except Exception as e:
            print(f"⚠️  Error: {e}\n")
            return False
    
    def load_model(self):
        """Load e5-base-v2 model with GPU support"""
        print(f"🤖 Loading {self.model_name}...")
        
        # Check GPU
        if TORCH_AVAILABLE:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if self.device == "cuda":
                print(f"🔥 NVIDIA GPU detected!")
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    print(f"   GPU: {gpu_name}")
                    print(f"   VRAM: {vram_gb:.1f} GB")
                except:
                    pass
        else:
            self.device = "cpu"
        
        print(f"   Device: {self.device.upper()}")
        print(f"   (First run downloads ~500 MB)\n")
        
        try:
            self.encoder = SentenceTransformer(self.model_name, device=self.device)
            
            # Use FP16 on GPU
            if self.device == "cuda":
                self.encoder = self.encoder.half()
                print("⚡ Enabled FP16 precision for faster training\n")
            
            print("✅ Model loaded\n")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def clean_text(self, text):
        """Clean text"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def split_path(self, path):
        """Split category path"""
        if pd.isna(path):
            return []
        path = str(path).strip()
        levels = [l.strip() for l in path.split('/') if l.strip()]
        return levels
    
    def extract_terms_with_synonyms(self, text):
        """Extract terms + synonyms"""
        cleaned = self.clean_text(text)
        words = cleaned.split()
        
        all_terms = set()
        all_terms.add(cleaned)
        
        # Single words + synonyms
        for word in words:
            if len(word) > 2:
                all_terms.add(word)
                if word in self.cross_store_synonyms:
                    if isinstance(self.cross_store_synonyms[word], (list, set)):
                        syns = list(self.cross_store_synonyms[word])[:5]
                        all_terms.update(syns)
        
        # 2-word phrases + synonyms
        for i in range(len(words) - 1):
            if len(words[i]) > 2 and len(words[i+1]) > 2:
                phrase = f"{words[i]} {words[i+1]}"
                all_terms.add(phrase)
                if phrase in self.cross_store_synonyms:
                    if isinstance(self.cross_store_synonyms[phrase], (list, set)):
                        syns = list(self.cross_store_synonyms[phrase])[:5]
                        all_terms.update(syns)
        
        return list(all_terms)
    
    def analyze_paths(self):
        """Analyze category structure"""
        print("🔍 ANALYZING CATEGORY STRUCTURE")
        print("="*80)
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Analyzing"):
            cat_path = str(row['category_path'])
            levels = self.split_path(cat_path)
            
            self.path_hierarchy[cat_path] = levels
            self.path_depth[cat_path] = len(levels)
            self.max_depth = max(self.max_depth, len(levels))
        
        print(f"\n✅ Analysis complete")
        print(f"   Max depth: {self.max_depth}")
        print(f"   Synonyms: {len(self.cross_store_synonyms):,}")
        print("="*80 + "\n")
    
    def build_enhanced_text(self, row):
        """Build E5-optimized training text"""
        cat_path = str(row['category_path'])
        cat_id = str(row['category_id'])
        levels = self.path_hierarchy.get(cat_path, [])
        
        if not levels:
            return "passage: unknown"
        
        components = []
        
        # CRITICAL: E5 requires "passage:" prefix
        components.append("passage:")
        
        # 1. FINAL PRODUCT - MAXIMUM EMPHASIS (15x)
        final_product = levels[-1]
        final_clean = self.clean_text(final_product)
        components.append(' '.join([final_clean] * 15))
        
        # 2. Synonym variations (5x)
        terms = self.extract_terms_with_synonyms(final_product)
        components.append(' '.join(terms * 5))
        
        # 3. Auto-tags (if available)
        auto_tags = self.auto_tags.get(cat_id, []) or self.auto_tags.get(cat_path, [])
        if auto_tags:
            if isinstance(auto_tags, list):
                components.append(' '.join(auto_tags[:30]))
        
        # 4. Parent level (3x)
        if len(levels) >= 2:
            parent = self.clean_text(levels[-2])
            components.append(' '.join([parent] * 3))
            
            # Parent + final combined
            combined = f"{parent} {final_clean}"
            components.append(' '.join([combined] * 3))
        
        # 5. Full hierarchy (weighted)
        for i, level in enumerate(levels):
            cleaned = self.clean_text(level)
            if i == len(levels) - 1:  # Last
                components.append(cleaned)
            elif i == len(levels) - 2:  # Parent
                components.append(cleaned)
            elif i == len(levels) - 3:  # Grandparent
                components.append(' '.join([cleaned] * 2))
            else:  # Top levels
                components.append(cleaned)
        
        # 6. Synonym expansion for all levels
        for level in levels:
            terms = self.extract_terms_with_synonyms(level)
            components.append(' '.join(terms[:10]))
        
        # 7. Full path
        components.append(cat_path.lower())
        
        return ' '.join(components)
    
    def prepare_texts(self):
        """Prepare training texts"""
        print("📝 PREPARING E5-OPTIMIZED TEXTS")
        print("="*80)
        
        texts = []
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing"):
            enhanced = self.build_enhanced_text(row)
            texts.append(enhanced)
        
        print(f"\n✅ Prepared {len(texts):,} texts")
        
        # Show sample
        if texts:
            print(f"\n📝 Sample (first 200 chars):")
            print(f"   {texts[0][:200]}...")
        
        print("="*80 + "\n")
        
        return texts
    
    def encode_texts(self, texts, batch_size=32):
        """Encode with E5"""
        print("🔄 ENCODING WITH E5-BASE")
        print("="*80)
        
        # Adjust batch size for GPU
        if self.device == "cuda":
            batch_size = 64
        else:
            batch_size = 16
        
        print(f"   Batch size: {batch_size}")
        
        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        embeddings = np.array(embeddings, dtype='float32')
        
        print(f"\n✅ Encoded shape: {embeddings.shape}")
        print(f"   Dimension: {embeddings.shape[1]} (E5-Base)")
        print("="*80 + "\n")
        
        return embeddings
    
    def build_faiss_index(self):
        """Build FAISS index"""
        print("🔍 BUILDING FAISS INDEX")
        print("="*80)
        
        dimension = self.embeddings.shape[1]
        
        # Inner Product for normalized embeddings (faster than cosine)
        index = faiss.IndexFlatIP(dimension)
        index.add(self.embeddings)
        
        index_path = self.cache_dir / 'main_index.faiss'
        faiss.write_index(index, str(index_path))
        
        print(f"✅ Built index: {index.ntotal:,} vectors")
        print(f"✅ Saved: {index_path}")
        print("="*80 + "\n")
    
    def save_all_data(self):
        """Save training data"""
        print("💾 SAVING TRAINING DATA")
        print("="*80)
        
        # Embeddings
        emb_path = self.cache_dir / 'embeddings.npy'
        np.save(emb_path, self.embeddings)
        print(f"✅ {emb_path}")
        
        # Metadata
        metadata = []
        for idx, row in self.df.iterrows():
            metadata.append({
                'category_id': str(row['category_id']),
                'category_path': str(row['category_path']),
                'auto_tags': self.auto_tags.get(str(row['category_id']), []),
                'depth': self.path_depth.get(str(row['category_path']), 0),
                'levels': self.path_hierarchy.get(str(row['category_path']), [])
            })
        
        meta_path = self.cache_dir / 'metadata.pkl'
        with open(meta_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✅ {meta_path} ({len(metadata):,} entries)")
        
        # Model info
        model_info = {
            'model_name': self.model_name,
            'embedding_dim': self.embeddings.shape[1],
            'num_categories': len(self.df),
            'max_depth': self.max_depth,
            'has_auto_tags': bool(self.auto_tags),
            'device': self.device
        }
        
        info_path = self.cache_dir / 'model_info.json'
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        print(f"✅ {info_path}")
        
        print("="*80 + "\n")
    
    def train(self, csv_path, json_path=None):
        """Complete training pipeline"""
        print("\n" + "="*80)
        print("🚀 STARTING TRAINING")
        print("="*80 + "\n")
        
        # Load data
        if not self.load_data(csv_path):
            return False
        
        if json_path:
            self.load_auto_tags(json_path)
        
        # Analyze
        self.analyze_paths()
        
        # Load model
        if not self.load_model():
            return False
        
        # Prepare and encode
        texts = self.prepare_texts()
        self.embeddings = self.encode_texts(texts)
        
        # Build index
        self.build_faiss_index()
        
        # Save
        self.save_all_data()
        
        # Summary
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE!")
        print("="*80)
        print(f"\n📊 Summary:")
        print(f"   Model: {self.model_name}")
        print(f"   Device: {self.device.upper()}")
        print(f"   Categories: {len(self.df):,}")
        print(f"   Max depth: {self.max_depth}")
        print(f"   Embedding dim: {self.embeddings.shape[1]}")
        print(f"   Synonyms: {len(self.cross_store_synonyms):,}")
        print(f"\n📁 Output:")
        print(f"   {self.cache_dir}/main_index.faiss")
        print(f"   {self.cache_dir}/metadata.pkl")
        print(f"   {self.cache_dir}/embeddings.npy")
        print("="*80 + "\n")
        
        return True


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("\n❌ CSV path required")
        print("\nUsage:")
        print("   python train_fixed.py data/category_id_path_only.csv")
        print("   python train_fixed.py data/category_id_path_only.csv data/tags.json")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    json_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(csv_path).exists():
        print(f"\n❌ File not found: {csv_path}")
        sys.exit(1)
    
    trainer = FixedTrainer()
    success = trainer.train(csv_path, json_path)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()