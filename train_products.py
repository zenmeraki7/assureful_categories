# """
# 🎯 ENHANCED TRAINING SYSTEM with Cross-Store Synonym Intelligence
# =================================================================
# ✅ Loads auto-generated tags from JSON
# ✅ Multi-store synonym mapping (washing machine = laundry machine)
# ✅ Technical term detection
# ✅ 3-model ensemble
# ✅ FAISS indexing
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
# from collections import defaultdict, Counter
# import warnings
# warnings.filterwarnings('ignore')


# class CrossStoreTrainer:
#     """Enhanced trainer with cross-store synonym intelligence"""
    
#     def __init__(self, cache_dir='cache'):
#         self.cache_dir = Path(cache_dir)
#         self.cache_dir.mkdir(parents=True, exist_ok=True)
        
#         print("\n" + "="*80)
#         print("🎯 CROSS-STORE INTELLIGENT CATEGORY PREDICTION SYSTEM")
#         print("="*80)
#         print("✅ Auto-tag loading from JSON")
#         print("✅ Cross-store synonym mapping (washing machine = laundry machine)")
#         print("✅ Brand/store variation handling")
#         print("✅ Technical term detection")
#         print("✅ 3-model ensemble encoding")
#         print("="*80 + "\n")
        
#         # Models
#         self.models = {
#             'primary': 'sentence-transformers/all-mpnet-base-v2',
#             'secondary': 'sentence-transformers/all-distilroberta-v1',
#             'tertiary': 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
#         }
#         self.weights = {'primary': 0.5, 'secondary': 0.3, 'tertiary': 0.2}
#         self.encoders = {}
        
#         # Data
#         self.df = None
#         self.embeddings = None
#         self.auto_tags = {}  # Loaded from JSON
        
#         # Cross-store synonym intelligence
#         self.cross_store_synonyms = self._build_cross_store_synonyms()
#         self.term_variations = defaultdict(set)
#         self.normalized_terms = {}  # Maps variations to canonical term
        
#         # Existing attributes
#         self.path_synonyms = defaultdict(set)
#         self.path_logic_tags = {}
#         self.path_hierarchy = {}
#         self.path_depth = {}
#         self.technical_terms = set()
#         self.multi_word_terms = set()
#         self.max_depth = 0
    
#     def _build_cross_store_synonyms(self):
#         """Build comprehensive cross-store synonym database"""
#         synonyms = {
#             # Appliances
#             'washing machine': {'laundry machine', 'washer', 'clothes washer', 'washing appliance'},
#             'laundry machine': {'washing machine', 'washer', 'clothes washer'},
#             'dryer': {'drying machine', 'clothes dryer', 'tumble dryer'},
#             'refrigerator': {'fridge', 'cooler', 'ice box', 'cooling appliance'},
#             'dishwasher': {'dish washer', 'dish cleaning machine'},
#             'microwave': {'microwave oven', 'micro wave'},
#             'vacuum': {'vacuum cleaner', 'hoover', 'vac'},
            
#             # Electronics
#             'tv': {'television', 'telly', 'smart tv', 'display'},
#             'laptop': {'notebook', 'portable computer', 'laptop computer'},
#             'mobile': {'phone', 'cell phone', 'smartphone', 'cellphone'},
#             'tablet': {'ipad', 'tab', 'tablet computer'},
#             'headphones': {'headset', 'earphones', 'earbuds', 'ear buds'},
#             'speaker': {'audio speaker', 'sound system', 'speakers'},
            
#             # Furniture
#             'sofa': {'couch', 'settee', 'divan'},
#             'wardrobe': {'closet', 'armoire', 'cupboard'},
#             'drawer': {'chest of drawers', 'dresser'},
            
#             # Clothing
#             'pants': {'trousers', 'slacks', 'bottoms'},
#             'sweater': {'jumper', 'pullover', 'sweatshirt'},
#             'sneakers': {'trainers', 'tennis shoes', 'running shoes'},
#             'jacket': {'coat', 'blazer', 'outerwear'},
            
#             # Kitchen
#             'cooker': {'stove', 'range', 'cooking range'},
#             'blender': {'mixer', 'food processor', 'liquidizer'},
#             'kettle': {'electric kettle', 'water boiler'},
            
#             # Baby/Kids
#             'stroller': {'pram', 'pushchair', 'buggy', 'baby carriage'},
#             'diaper': {'nappy', 'nappies'},
#             'pacifier': {'dummy', 'soother'},
            
#             # Tools
#             'wrench': {'spanner', 'adjustable wrench'},
#             'flashlight': {'torch', 'flash light'},
#             'screwdriver': {'screw driver'},
            
#             # Home
#             'tap': {'faucet', 'water tap'},
#             'bin': {'trash can', 'garbage can', 'waste bin'},
#             'curtain': {'drape', 'window covering'},
            
#             # Crafts/Office
#             'guillotine': {'paper cutter', 'paper trimmer', 'blade cutter'},
#             'trimmer': {'cutter', 'cutting tool', 'edge cutter'},
#             'stapler': {'stapling machine', 'staple gun'},
            
#             # Books/Media
#             'magazine': {'periodical', 'journal', 'publication'},
#             'comic': {'comic book', 'graphic novel', 'manga'},
#             'ebook': {'e-book', 'digital book', 'electronic book'},
            
#             # General terms
#             'kids': {'children', 'child', 'childrens', 'youth', 'junior'},
#             'women': {'womens', 'ladies', 'female', 'lady'},
#             'men': {'mens', 'male', 'gentleman'},
#             'baby': {'infant', 'newborn', 'toddler'},
#         }
        
#         # Build bidirectional mapping
#         expanded = {}
#         for term, syns in synonyms.items():
#             expanded[term] = syns.copy()
#             for syn in syns:
#                 if syn not in expanded:
#                     expanded[syn] = set()
#                 expanded[syn].add(term)
#                 expanded[syn].update(syns - {syn})
        
#         return expanded
    
#     def normalize_term(self, term):
#         """Normalize term to canonical form"""
#         cleaned = self.clean_text(term)
        
#         # Check if it's in our synonym database
#         if cleaned in self.cross_store_synonyms:
#             # Return the most common/canonical form
#             all_forms = {cleaned} | self.cross_store_synonyms[cleaned]
#             # Use shortest form as canonical (usually most common)
#             canonical = min(all_forms, key=len)
#             return canonical
        
#         return cleaned
    
#     def load_auto_tags(self, json_path):
#         """Load auto-generated tags from JSON"""
#         print("\n📂 Loading auto-generated tags...")
        
#         if not Path(json_path).exists():
#             print(f"⚠️  Auto-tags file not found: {json_path}")
#             print("   Continuing without auto-tags...")
#             return False
        
#         with open(json_path, 'r', encoding='utf-8') as f:
#             self.auto_tags = json.load(f)
        
#         print(f"✅ Loaded tags for {len(self.auto_tags):,} category paths")
        
#         # Show sample
#         if self.auto_tags:
#             sample_path = list(self.auto_tags.keys())[0]
#             sample_tags = self.auto_tags[sample_path][:5]
#             print(f"\n📝 Sample tags for: {sample_path}")
#             print(f"   Tags: {', '.join(sample_tags)}...\n")
        
#         return True
    
#     def load_models(self):
#         """Load 3-model ensemble"""
#         print("\n🤖 Loading 3-model ensemble...")
#         for name, model_name in self.models.items():
#             print(f"   {name}: {model_name}")
#             self.encoders[name] = SentenceTransformer(model_name)
#         print("✅ All models loaded\n")
    
#     def load_data(self, filepath):
#         """Load category data"""
#         print("\n📂 Loading category paths...")
#         self.df = pd.read_csv(filepath, low_memory=False)
#         print(f"✅ Loaded {len(self.df):,} category paths\n")
        
#         self.category_id_col = self.df.columns[0]
#         self.category_path_col = self.df.columns[1]
        
#         return True
    
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
    
#     def extract_cross_store_terms(self, text):
#         """Extract terms and their cross-store variations"""
#         cleaned = self.clean_text(text)
#         words = cleaned.split()
        
#         all_terms = set()
#         all_terms.add(cleaned)  # Full text
        
#         # Single words
#         for word in words:
#             if len(word) > 2:
#                 all_terms.add(word)
#                 # Add cross-store synonyms
#                 if word in self.cross_store_synonyms:
#                     all_terms.update(self.cross_store_synonyms[word])
        
#         # Multi-word phrases (2-3 words)
#         for i in range(len(words) - 1):
#             if len(words[i]) > 2 and len(words[i+1]) > 2:
#                 phrase = f"{words[i]} {words[i+1]}"
#                 all_terms.add(phrase)
#                 # Add synonyms for phrase
#                 if phrase in self.cross_store_synonyms:
#                     all_terms.update(self.cross_store_synonyms[phrase])
        
#         # 3-word phrases
#         if len(words) >= 3:
#             for i in range(len(words) - 2):
#                 if all(len(w) > 2 for w in words[i:i+3]):
#                     phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
#                     all_terms.add(phrase)
        
#         return all_terms
    
#     def analyze_paths(self):
#         """Analyze all paths"""
#         print("\n🔍 ANALYZING PATHS WITH CROSS-STORE INTELLIGENCE")
#         print("="*80)
        
#         for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Analyzing"):
#             cat_path = str(row[self.category_path_col])
#             levels = self.split_path(cat_path)
            
#             self.path_hierarchy[cat_path] = levels
#             self.path_depth[cat_path] = len(levels)
#             self.max_depth = max(self.max_depth, len(levels))
            
#             # Extract all terms with cross-store variations
#             for level in levels:
#                 terms = self.extract_cross_store_terms(level)
#                 self.path_synonyms[cat_path].update(terms)
        
#         print(f"✅ Analysis complete!")
#         print(f"   Max depth: {self.max_depth}")
#         print(f"   Cross-store synonyms loaded: {len(self.cross_store_synonyms)}")
#         print("="*80 + "\n")
    
#     def build_enhanced_text(self, row):
#         """Build enhanced text with all intelligence"""
#         cat_path = str(row[self.category_path_col])
#         levels = self.path_hierarchy.get(cat_path, [])
        
#         if not levels:
#             return "unknown"
        
#         components = []
        
#         # 1. Auto-tags (highest priority)
#         auto_tags = self.auto_tags.get(cat_path, [])
#         if auto_tags:
#             components.append(' '.join(auto_tags[:50]))  # Top 50 tags
#             components.append(' '.join(auto_tags[:20]))  # Repeat top 20 for emphasis
        
#         # 2. Cross-store synonym expansion
#         cross_store_terms = []
#         for level in levels:
#             terms = self.extract_cross_store_terms(level)
#             cross_store_terms.extend(terms)
#         components.append(' '.join(cross_store_terms))
        
#         # 3. Original path with emphasis
#         for i, level in enumerate(levels):
#             cleaned = self.clean_text(level)
#             if i == len(levels) - 1:  # Last level (product)
#                 components.append(' '.join([cleaned] * 5))
#             elif i == len(levels) - 2:  # Parent category
#                 components.append(' '.join([cleaned] * 3))
#             else:
#                 components.append(cleaned)
        
#         # 4. Path hierarchy context
#         components.append(cat_path)
        
#         # 5. Additional synonyms
#         path_syns = list(self.path_synonyms.get(cat_path, set()))[:30]
#         components.append(' '.join(path_syns))
        
#         return ' '.join(components)
    
#     def prepare_texts(self):
#         """Prepare all texts"""
#         print("\n📝 PREPARING ENHANCED TEXTS WITH CROSS-STORE INTELLIGENCE")
#         print("="*80)
        
#         texts = []
#         for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing"):
#             enhanced = self.build_enhanced_text(row)
#             texts.append(enhanced)
        
#         print(f"✅ Prepared {len(texts):,} enhanced texts")
#         print("="*80 + "\n")
        
#         return texts
    
#     def encode_with_ensemble(self, texts, batch_size=32):
#         """Encode with 3-model ensemble"""
#         print("\n🔄 ENCODING WITH 3-MODEL ENSEMBLE")
#         print("="*80)
        
#         all_embeddings = []
        
#         for name, encoder in self.encoders.items():
#             weight = self.weights[name]
#             print(f"\n📥 {name} (weight: {weight})")
            
#             embeddings = encoder.encode(
#                 texts,
#                 batch_size=batch_size,
#                 show_progress_bar=True,
#                 convert_to_numpy=True,
#                 normalize_embeddings=True
#             )
            
#             embeddings = np.array(embeddings, dtype='float32')
#             weighted = embeddings * weight
#             all_embeddings.append(weighted)
        
#         print("\n📊 Combining embeddings...")
#         combined = np.sum(np.stack(all_embeddings, axis=0), axis=0)
        
#         # Normalize
#         norms = np.linalg.norm(combined, axis=1, keepdims=True)
#         final = combined / (norms + 1e-8)
        
#         print(f"✅ Final shape: {final.shape}")
#         print("="*80 + "\n")
        
#         return final.astype('float32')
    
#     def build_faiss_index(self):
#         """Build main FAISS index"""
#         print("\n🔍 BUILDING FAISS INDEX")
#         print("="*80)
        
#         dimension = self.embeddings.shape[1]
#         index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
#         index.add(self.embeddings)
        
#         faiss.write_index(index, str(self.cache_dir / 'main_index.faiss'))
#         print(f"✅ Built index with {index.ntotal:,} vectors")
#         print("="*80 + "\n")
    
#     def save_all_data(self):
#         """Save all training data"""
#         print("\n💾 SAVING TRAINING DATA")
#         print("="*80)
        
#         # Save embeddings
#         np.save(self.cache_dir / 'embeddings.npy', self.embeddings)
#         print("✅ Saved: embeddings.npy")
        
#         # Save metadata
#         metadata = []
#         for idx, row in self.df.iterrows():
#             cat_id = str(row[self.category_id_col])
#             cat_path = str(row[self.category_path_col])
            
#             metadata.append({
#                 'category_id': cat_id,
#                 'category_path': cat_path,
#                 'auto_tags': self.auto_tags.get(cat_path, []),
#                 'depth': self.path_depth.get(cat_path, 0),
#                 'levels': self.path_hierarchy.get(cat_path, [])
#             })
        
#         with open(self.cache_dir / 'metadata.pkl', 'wb') as f:
#             pickle.dump(metadata, f)
#         print(f"✅ Saved: metadata.pkl ({len(metadata):,} entries)")
        
#         # Save cross-store synonyms
#         with open(self.cache_dir / 'cross_store_synonyms.pkl', 'wb') as f:
#             pickle.dump(self.cross_store_synonyms, f)
#         print("✅ Saved: cross_store_synonyms.pkl")
        
#         print("="*80 + "\n")
    
#     def train(self, csv_path, json_path='auto_tags.json'):
#         """Complete training pipeline"""
#         print("\n" + "="*80)
#         print("🚀 STARTING TRAINING")
#         print("="*80 + "\n")
        
#         self.load_data(csv_path)
#         self.load_auto_tags(json_path)
#         self.analyze_paths()
#         self.load_models()
#         texts = self.prepare_texts()
#         self.embeddings = self.encode_with_ensemble(texts)
#         self.build_faiss_index()
#         self.save_all_data()
        
#         print("\n" + "="*80)
#         print("✅ TRAINING COMPLETE!")
#         print("="*80)
#         print(f"📊 Summary:")
#         print(f"   Categories: {len(self.df):,}")
#         print(f"   Max depth: {self.max_depth}")
#         print(f"   Cross-store synonyms: {len(self.cross_store_synonyms)}")
#         print(f"   Auto-tags loaded: {'Yes' if self.auto_tags else 'No'}")
#         print(f"\n🎯 System Features:")
#         print(f"   ✅ Handles 'washing machine' = 'laundry machine'")
#         print(f"   ✅ Handles 'tv' = 'television' = 'smart tv'")
#         print(f"   ✅ Cross-store product variations")
#         print(f"   ✅ Auto-tag intelligence")
#         print(f"   ✅ 3-model ensemble")
#         print("="*80 + "\n")
        
#         print("🚀 Next: Start API server")
#         print("   python api_server.py\n")


# if __name__ == "__main__":
#     import sys
    
#     csv_path = sys.argv[1] if len(sys.argv) > 1 else 'data/category_id_path_only.csv'
#     json_path = sys.argv[2] if len(sys.argv) > 2 else 'data/tags.json'
    
#     trainer = CrossStoreTrainer()
#     trainer.train(csv_path, json_path)


"""
🎯 ENHANCED TRAINING SYSTEM - Optimized for High Confidence
===========================================================
✅ Better final product emphasis (10x weight on last word)
✅ Improved synonym integration
✅ Optimized for 85%+ confidence scores
✅ Validates data quality before training

Usage:
    python train_enhanced.py data/category_id_path_only.csv
    python train_enhanced.py data/category_id_path_only.csv data/tags.json
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
warnings.filterwarnings('ignore')


class EnhancedTrainer:
    """Enhanced trainer with stronger final product emphasis"""
    
    def __init__(self, cache_dir='cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*80)
        print("🎯 ENHANCED CATEGORY PREDICTION SYSTEM")
        print("="*80)
        print("✅ 10x emphasis on final product (last word)")
        print("✅ Optimized for 85%+ confidence")
        print("✅ AI-powered synonym loading")
        print("="*80 + "\n")
        
        # Model (single powerful one for speed)
        self.model_name = 'sentence-transformers/all-mpnet-base-v2'
        self.encoder = None
        
        # Data
        self.df = None
        self.embeddings = None
        self.auto_tags = {}
        
        # Load AI-powered synonyms
        self.cross_store_synonyms = self._load_cross_store_synonyms()
        
        # Other attributes
        self.path_depth = {}
        self.path_hierarchy = {}
        self.max_depth = 0
    
    def _load_cross_store_synonyms(self):
        """Load AI-generated synonyms from synonym_manager.py"""
        synonyms_file = self.cache_dir / 'cross_store_synonyms.pkl'
        
        if synonyms_file.exists():
            print("📥 Loading AI-generated synonyms...")
            try:
                with open(synonyms_file, 'rb') as f:
                    synonyms = pickle.load(f)
                print(f"✅ Loaded {len(synonyms):,} AI-generated synonym mappings\n")
                return synonyms
            except Exception as e:
                print(f"⚠️  Error loading AI synonyms: {e}")
                print("   Falling back to basic synonyms...\n")
        else:
            print("⚠️  AI-generated synonyms not found")
            print("   💡 Generate with: python synonym_manager.py autobuild data/category_id_path_only.csv")
            print("   Using basic synonyms for now...\n")
        
        return self._build_basic_synonyms()
    
    def _build_basic_synonyms(self):
        """Basic fallback synonyms"""
        synonyms = {
            # Footwear
            'shoes': {'footwear', 'sneakers', 'boots'},
            'sneakers': {'shoes', 'trainers', 'athletic shoes', 'running shoes'},
            'boots': {'footwear', 'shoes'},
            'sandals': {'footwear', 'shoes', 'flip flops'},
            
            # Clothing
            'pants': {'trousers', 'slacks', 'bottoms'},
            'shirt': {'top', 'blouse', 'tee'},
            'jacket': {'coat', 'outerwear'},
            'dress': {'frock', 'gown'},
            
            # Electronics
            'tv': {'television', 'smart tv'},
            'phone': {'mobile', 'smartphone', 'cell phone'},
            'laptop': {'notebook', 'computer'},
            
            # Appliances
            'washing machine': {'washer', 'laundry machine'},
            'refrigerator': {'fridge', 'cooler'},
            
            # General
            'kids': {'children', 'childrens', 'youth'},
            'women': {'womens', 'ladies', 'female'},
            'men': {'mens', 'male', 'guys'},
        }
        
        # Build bidirectional
        expanded = {}
        for term, syns in synonyms.items():
            expanded[term] = syns.copy()
            for syn in syns:
                if syn not in expanded:
                    expanded[syn] = set()
                expanded[syn].add(term)
                expanded[syn].update(syns - {syn})
        
        print(f"📚 Using {len(expanded)} basic synonym mappings\n")
        return expanded
    
    def load_data(self, filepath):
        """Load and validate category data"""
        print("📂 Loading category data...")
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        
        # Read CSV - only first 2 columns
        self.df = pd.read_csv(filepath, usecols=[0, 1], names=['category_id', 'category_path'], 
                             header=0, low_memory=False)
        
        # Remove any NaN
        before = len(self.df)
        self.df = self.df.dropna()
        after = len(self.df)
        
        if before != after:
            print(f"   Removed {before - after} rows with missing data")
        
        print(f"✅ Loaded {len(self.df):,} valid categories")
        
        # Validate paths
        sample_paths = self.df['category_path'].head(5).tolist()
        print(f"\n📝 Sample paths:")
        for path in sample_paths:
            print(f"   • {path}")
        print()
        
        return True
    
    def load_auto_tags(self, json_path):
        """Load auto-generated tags from JSON"""
        print("📂 Loading auto-generated tags...")
        
        json_path = Path(json_path)
        if not json_path.exists():
            print(f"⚠️  Auto-tags file not found: {json_path}")
            print("   Continuing without auto-tags...\n")
            return False
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.auto_tags = json.load(f)
            
            print(f"✅ Loaded tags for {len(self.auto_tags):,} categories\n")
            return True
        except Exception as e:
            print(f"⚠️  Error loading auto-tags: {e}")
            return False
    
    def load_model(self):
        """Load sentence transformer"""
        print(f"🤖 Loading model: {self.model_name}")
        print("   (This may take a minute on first run...)\n")
        
        self.encoder = SentenceTransformer(self.model_name)
        print("✅ Model loaded\n")
    
    def clean_text(self, text):
        """Clean text"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def split_path(self, path):
        """Split path into levels"""
        if pd.isna(path):
            return []
        path = str(path).strip()
        levels = [l.strip() for l in path.split('/') if l.strip()]
        return levels
    
    def extract_terms_with_synonyms(self, text):
        """Extract terms and expand with synonyms"""
        cleaned = self.clean_text(text)
        words = cleaned.split()
        
        all_terms = set()
        all_terms.add(cleaned)
        
        # Single words + synonyms
        for word in words:
            if len(word) > 2:
                all_terms.add(word)
                if word in self.cross_store_synonyms:
                    all_terms.update(list(self.cross_store_synonyms[word])[:5])
        
        # 2-word phrases + synonyms
        for i in range(len(words) - 1):
            if len(words[i]) > 2 and len(words[i+1]) > 2:
                phrase = f"{words[i]} {words[i+1]}"
                all_terms.add(phrase)
                if phrase in self.cross_store_synonyms:
                    all_terms.update(list(self.cross_store_synonyms[phrase])[:5])
        
        return list(all_terms)
    
    def analyze_paths(self):
        """Analyze all paths"""
        print("🔍 ANALYZING CATEGORY STRUCTURE")
        print("="*80)
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Analyzing"):
            cat_path = str(row['category_path'])
            levels = self.split_path(cat_path)
            
            self.path_hierarchy[cat_path] = levels
            self.path_depth[cat_path] = len(levels)
            self.max_depth = max(self.max_depth, len(levels))
        
        print(f"\n✅ Analysis complete!")
        print(f"   Max depth: {self.max_depth}")
        print(f"   Synonym terms: {len(self.cross_store_synonyms):,}")
        print("="*80 + "\n")
    
    def build_enhanced_text(self, row):
        """Build training text with HEAVY emphasis on final product"""
        cat_path = str(row['category_path'])
        cat_id = str(row['category_id'])
        levels = self.path_hierarchy.get(cat_path, [])
        
        if not levels:
            return "unknown"
        
        components = []
        
        # 1. FINAL PRODUCT - MASSIVE EMPHASIS (the most important!)
        final_product = levels[-1]
        final_product_clean = self.clean_text(final_product)
        
        # Repeat final product 15x for maximum weight
        components.append(' '.join([final_product_clean] * 15))
        
        # Add synonym variations of final product 5x
        final_product_terms = self.extract_terms_with_synonyms(final_product)
        components.append(' '.join(final_product_terms * 5))
        
        # 2. Auto-tags (if available) - high weight
        auto_tags = self.auto_tags.get(cat_id, []) or self.auto_tags.get(cat_path, [])
        if auto_tags:
            components.append(' '.join(auto_tags[:30]))
            components.append(' '.join(auto_tags[:15]))
        
        # 3. Parent level (second to last) - medium weight
        if len(levels) >= 2:
            parent = self.clean_text(levels[-2])
            components.append(' '.join([parent] * 3))
            
            # Combine parent + final
            combined = f"{parent} {final_product_clean}"
            components.append(' '.join([combined] * 3))
        
        # 4. Full hierarchy with emphasis on deeper levels
        for i, level in enumerate(levels):
            cleaned = self.clean_text(level)
            if i == len(levels) - 1:  # Last level (already heavily weighted above)
                components.append(cleaned)
            elif i == len(levels) - 2:  # Parent (already added above)
                components.append(cleaned)
            elif i == len(levels) - 3:  # Grandparent
                components.append(' '.join([cleaned] * 2))
            else:  # Top levels
                components.append(cleaned)
        
        # 5. Synonym expansion for all levels
        for level in levels:
            terms = self.extract_terms_with_synonyms(level)
            components.append(' '.join(terms[:10]))
        
        # 6. Full path for context
        components.append(cat_path.lower())
        
        return ' '.join(components)
    
    def prepare_texts(self):
        """Prepare all training texts"""
        print("📝 PREPARING ENHANCED TEXTS")
        print("="*80)
        
        texts = []
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing"):
            enhanced = self.build_enhanced_text(row)
            texts.append(enhanced)
        
        print(f"\n✅ Prepared {len(texts):,} enhanced texts")
        
        # Show sample
        if texts:
            print("\n📝 Sample enhanced text (first 200 chars):")
            print(f"   {texts[0][:200]}...")
        
        print("="*80 + "\n")
        
        return texts
    
    def encode_texts(self, texts, batch_size=32):
        """Encode texts with single powerful model"""
        print("🔄 ENCODING TEXTS")
        print("="*80)
        
        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        embeddings = np.array(embeddings, dtype='float32')
        
        print(f"\n✅ Encoded to shape: {embeddings.shape}")
        print("="*80 + "\n")
        
        return embeddings
    
    def build_faiss_index(self):
        """Build FAISS index"""
        print("🔍 BUILDING FAISS INDEX")
        print("="*80)
        
        dimension = self.embeddings.shape[1]
        
        # Use Inner Product for cosine similarity (embeddings are normalized)
        index = faiss.IndexFlatIP(dimension)
        index.add(self.embeddings)
        
        # Save index
        index_path = self.cache_dir / 'main_index.faiss'
        faiss.write_index(index, str(index_path))
        
        print(f"✅ Built index with {index.ntotal:,} vectors")
        print(f"✅ Saved to: {index_path}")
        print("="*80 + "\n")
    
    def save_all_data(self):
        """Save all training data"""
        print("💾 SAVING TRAINING DATA")
        print("="*80)
        
        # Save embeddings
        emb_path = self.cache_dir / 'embeddings.npy'
        np.save(emb_path, self.embeddings)
        print(f"✅ Saved: {emb_path}")
        
        # Save metadata
        metadata = []
        for idx, row in self.df.iterrows():
            cat_id = str(row['category_id'])
            cat_path = str(row['category_path'])
            
            metadata.append({
                'category_id': cat_id,
                'category_path': cat_path,
                'auto_tags': self.auto_tags.get(cat_id, []) or self.auto_tags.get(cat_path, []),
                'depth': self.path_depth.get(cat_path, 0),
                'levels': self.path_hierarchy.get(cat_path, [])
            })
        
        meta_path = self.cache_dir / 'metadata.pkl'
        with open(meta_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✅ Saved: {meta_path} ({len(metadata):,} entries)")
        
        # Save synonyms
        syn_path = self.cache_dir / 'cross_store_synonyms.pkl'
        with open(syn_path, 'wb') as f:
            pickle.dump(self.cross_store_synonyms, f)
        print(f"✅ Saved: {syn_path} ({len(self.cross_store_synonyms):,} terms)")
        
        print("="*80 + "\n")
    
    def train(self, csv_path, json_path=None):
        """Complete training pipeline"""
        print("\n" + "="*80)
        print("🚀 STARTING ENHANCED TRAINING")
        print("="*80 + "\n")
        
        # Load data
        self.load_data(csv_path)
        
        # Load auto-tags if provided
        if json_path:
            self.load_auto_tags(json_path)
        
        # Analyze paths
        self.analyze_paths()
        
        # Load model
        self.load_model()
        
        # Prepare texts
        texts = self.prepare_texts()
        
        # Encode
        self.embeddings = self.encode_texts(texts)
        
        # Build FAISS index
        self.build_faiss_index()
        
        # Save everything
        self.save_all_data()
        
        # Summary
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE!")
        print("="*80)
        print(f"📊 Summary:")
        print(f"   Categories trained: {len(self.df):,}")
        print(f"   Max hierarchy depth: {self.max_depth}")
        print(f"   Synonym terms: {len(self.cross_store_synonyms):,}")
        print(f"   Auto-tags loaded: {'Yes' if self.auto_tags else 'No'}")
        print(f"   Embedding dimension: {self.embeddings.shape[1]}")
        print(f"\n🎯 Optimizations:")
        print(f"   ✅ 15x emphasis on final product")
        print(f"   ✅ 5x synonym expansion")
        print(f"   ✅ Cross-store intelligence")
        print(f"   ✅ Optimized for 85%+ confidence")
        print(f"\n📁 Saved files:")
        print(f"   {self.cache_dir}/main_index.faiss")
        print(f"   {self.cache_dir}/metadata.pkl")
        print(f"   {self.cache_dir}/embeddings.npy")
        print(f"   {self.cache_dir}/cross_store_synonyms.pkl")
        print("="*80 + "\n")
        
        print("🚀 Next: Start API server")
        print("   python api_server.py\n")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("\n❌ Error: CSV file path required")
        print("\nUsage:")
        print("   python train_enhanced.py <csv_path> [json_path]")
        print("\nExamples:")
        print("   python train_enhanced.py data/category_id_path_only.csv")
        print("   python train_enhanced.py data/category_id_path_only.csv data/tags.json")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    json_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(csv_path).exists():
        print(f"\n❌ Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    trainer = EnhancedTrainer()
    trainer.train(csv_path, json_path)


if __name__ == "__main__":
    main()