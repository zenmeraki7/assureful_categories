# """
# 🤖 FIXED AI-POWERED SYNONYM MANAGER
# ====================================
# ✅ Windows + NVIDIA GPU optimized
# ✅ Uses e5-base-v2 (lower memory)
# ✅ Proper error handling
# ✅ Progress tracking

# Usage:
#     python synonym_manager_fixed.py autobuild data/category_id_path_only.csv
#     python synonym_manager_fixed.py autobuild data/category_id_path_only.csv --fast
# """

# import pickle
# from pathlib import Path
# import json
# from collections import defaultdict
# from tqdm import tqdm
# import warnings
# import sys
# import os

# warnings.filterwarnings('ignore')

# # Fix CUDA issues on Windows
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# try:
#     from nltk.corpus import wordnet
#     from nltk import download as nltk_download
#     WORDNET_AVAILABLE = True
# except ImportError:
#     WORDNET_AVAILABLE = False
#     print("⚠️  NLTK not available. Install with: pip install nltk")

# try:
#     from sentence_transformers import SentenceTransformer, util
#     import torch
#     TRANSFORMERS_AVAILABLE = True
# except ImportError:
#     TRANSFORMERS_AVAILABLE = False
#     print("⚠️  SentenceTransformers not available.")
#     print("   Install with: pip install sentence-transformers torch")


# class FixedAISynonymManager:
#     """Fixed AI-powered synonym manager for Windows + NVIDIA GPU"""

#     def __init__(self, cache_dir='cache', tags_file='data/tags.json', fast_mode=False):
#         self.cache_dir = Path(cache_dir)
#         self.synonyms_file = self.cache_dir / 'cross_store_synonyms.pkl'
#         self.tags_file = Path(tags_file)
#         self.synonyms = {}
#         self.tags_data = {}
#         self.model = None
#         self.device = "cpu"
#         self.fast_mode = fast_mode
        
#         # Create cache directory
#         self.cache_dir.mkdir(parents=True, exist_ok=True)
        
#         # Load existing data
#         self.load_tags()
#         if self.synonyms_file.exists():
#             self.load_synonyms()
#         else:
#             print("📝 No existing synonyms file. Will create new one.")

#     def load_tags(self):
#         """Load domain-specific tags (optional)"""
#         if self.tags_file.exists():
#             try:
#                 with open(self.tags_file, 'r', encoding='utf-8') as f:
#                     self.tags_data = json.load(f)
#                 print(f"✅ Loaded {len(self.tags_data)} tag entries")
#                 return True
#             except Exception as e:
#                 print(f"⚠️  Could not load tags.json: {e}")
#         else:
#             print(f"ℹ️  tags.json not found (optional)")
#         return False

#     def load_synonyms(self):
#         """Load existing synonyms with format conversion"""
#         try:
#             with open(self.synonyms_file, 'rb') as f:
#                 loaded = pickle.load(f)
            
#             # Handle different formats
#             if not loaded:
#                 self.synonyms = {}
#                 return
            
#             # Check format
#             first_val = next(iter(loaded.values()))
            
#             if isinstance(first_val, list):
#                 if first_val and isinstance(first_val[0], tuple):
#                     # New format: [(syn, conf, src), ...]
#                     self.synonyms = loaded
#                     print(f"✅ Loaded {len(self.synonyms)} synonym entries (new format)")
#                 elif first_val and isinstance(first_val[0], str):
#                     # Legacy format: [syn1, syn2, ...]
#                     self.synonyms = {
#                         k: [(v, 0.8, 'legacy') for v in vals]
#                         for k, vals in loaded.items()
#                     }
#                     print(f"✅ Converted {len(self.synonyms)} legacy synonym entries")
#             elif isinstance(first_val, set):
#                 # Set format
#                 self.synonyms = {
#                     k: [(v, 0.8, 'legacy') for v in vals]
#                     for k, vals in loaded.items()
#                 }
#                 print(f"✅ Converted {len(self.synonyms)} set-based entries")
#             else:
#                 self.synonyms = {}
#                 print(f"⚠️  Unknown synonym format")
                
#         except Exception as e:
#             print(f"❌ Error loading synonyms: {e}")
#             self.synonyms = {}

#     def save_synonyms(self):
#         """Save synonyms in both formats"""
#         try:
#             # Save binary format
#             with open(self.synonyms_file, 'wb') as f:
#                 pickle.dump(self.synonyms, f)

#             # Save readable JSON
#             json_file = self.cache_dir / 'synonyms_readable.json'
#             readable = {}
#             for term, syns in self.synonyms.items():
#                 readable[term] = [
#                     {'synonym': syn, 'confidence': float(conf), 'source': src}
#                     for syn, conf, src in syns
#                 ]
            
#             with open(json_file, 'w', encoding='utf-8') as f:
#                 json.dump(readable, f, indent=2, ensure_ascii=False)

#             print(f"\n✅ Saved {len(self.synonyms)} synonym entries")
#             print(f"   📁 Binary: {self.synonyms_file}")
#             print(f"   📁 JSON: {json_file}")
#             return True
#         except Exception as e:
#             print(f"❌ Error saving synonyms: {e}")
#             return False

#     def load_transformer_model(self):
#         """Load e5-base-v2 model with GPU support"""
#         if not TRANSFORMERS_AVAILABLE:
#             print("❌ SentenceTransformers not installed!")
#             return False

#         # Check for CUDA
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
#         if self.device == "cuda":
#             print(f"🔥 NVIDIA GPU detected!")
#             try:
#                 gpu_name = torch.cuda.get_device_name(0)
#                 vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
#                 print(f"   GPU: {gpu_name}")
#                 print(f"   VRAM: {vram_gb:.1f} GB")
#             except:
#                 pass
#         else:
#             print("💻 Using CPU (slower)")

#         # Use e5-base-v2 for better memory efficiency
#         model_name = "intfloat/e5-base-v2"
#         print(f"\n🤖 Loading model: {model_name}")
        
#         try:
#             self.model = SentenceTransformer(model_name, device=self.device)
#             self.model.max_seq_length = 256
            
#             # Use FP16 on GPU for speed
#             if self.device == "cuda":
#                 self.model = self.model.half()
#                 print("⚡ Enabled FP16 precision")
            
#             print("✅ Model loaded successfully\n")
#             return True
#         except Exception as e:
#             print(f"❌ Failed to load model: {e}")
#             return False

#     def get_wordnet_synonyms(self, word, limit=10):
#         """Get WordNet synonyms"""
#         if self.fast_mode or not WORDNET_AVAILABLE:
#             return []
        
#         try:
#             # Ensure WordNet is downloaded
#             try:
#                 wordnet.synsets('test')
#             except:
#                 print("📥 Downloading WordNet data...")
#                 nltk_download('wordnet', quiet=True)
#                 nltk_download('omw-1.4', quiet=True)
            
#             synonyms = []
#             word_clean = word.lower().replace(' ', '_')
            
#             for syn in wordnet.synsets(word_clean):
#                 for lemma in syn.lemmas():
#                     synonym = lemma.name().replace('_', ' ').lower()
#                     if synonym != word.lower() and len(synonym) > 2:
#                         confidence = 0.75  # Fixed confidence for WordNet
#                         synonyms.append((synonym, confidence, 'wordnet'))
#                         if len(synonyms) >= limit:
#                             break
#                 if len(synonyms) >= limit:
#                     break
            
#             return synonyms[:limit]
#         except Exception:
#             return []

#     def get_semantic_synonyms(self, term, candidate_pool, threshold=0.70, limit=15):
#         """Get semantic synonyms using embeddings"""
#         if not self.model or not candidate_pool:
#             return []
        
#         try:
#             # E5 model requires query/passage prefixes
#             query = f"query: {term}"
#             candidates_prefixed = [f"passage: {c}" for c in candidate_pool]
            
#             # Encode query
#             term_emb = self.model.encode(
#                 query, 
#                 convert_to_tensor=True, 
#                 show_progress_bar=False
#             )

#             # Encode candidates in batches
#             batch_size = 32 if self.device == "cuda" else 8
#             all_embeddings = []
            
#             for i in range(0, len(candidates_prefixed), batch_size):
#                 batch = candidates_prefixed[i:i + batch_size]
#                 emb = self.model.encode(
#                     batch, 
#                     convert_to_tensor=True, 
#                     show_progress_bar=False
#                 )
#                 all_embeddings.append(emb)
            
#             # Concatenate all embeddings
#             candidate_embs = torch.cat(all_embeddings, dim=0)
            
#             # Calculate cosine similarity
#             scores = util.cos_sim(term_emb, candidate_embs)[0]
            
#             # Filter by threshold
#             synonyms = []
#             for candidate, score in zip(candidate_pool, scores):
#                 score_val = float(score)
#                 if score_val > threshold and candidate.lower() != term.lower():
#                     # Scale confidence between 0.6 and 0.95
#                     confidence = 0.60 + (score_val - threshold) * 0.35 / (1 - threshold)
#                     synonyms.append((candidate, confidence, 'semantic'))
            
#             # Sort by confidence
#             synonyms.sort(key=lambda x: x[1], reverse=True)
#             return synonyms[:limit]
            
#         except Exception as e:
#             print(f"⚠️  Semantic error: {e}")
#             return []

#     def auto_generate_synonyms(self, term, candidate_pool=None, 
#                               semantic_threshold=0.70, silent=False):
#         """Generate synonyms from multiple sources"""
#         all_synonyms = []
        
#         if not silent:
#             print(f"\n🔍 Finding synonyms for: '{term}'")

#         # Source 1: WordNet
#         if WORDNET_AVAILABLE and not self.fast_mode:
#             wn_syns = self.get_wordnet_synonyms(term, limit=10)
#             all_synonyms.extend(wn_syns)
        
#         # Source 2: Semantic similarity
#         if candidate_pool and self.model:
#             sem_syns = self.get_semantic_synonyms(
#                 term, candidate_pool,
#                 threshold=semantic_threshold, 
#                 limit=15
#             )
#             all_synonyms.extend(sem_syns)

#         # Deduplicate (keep highest confidence)
#         synonym_map = {}
#         for syn, conf, source in all_synonyms:
#             syn_lower = syn.lower()
#             if syn_lower not in synonym_map or conf > synonym_map[syn_lower][1]:
#                 synonym_map[syn_lower] = (syn, conf, source)
        
#         final_synonyms = sorted(
#             synonym_map.values(), 
#             key=lambda x: x[1], 
#             reverse=True
#         )
        
#         return final_synonyms

#     def add_synonym_group(self, term, synonyms_with_confidence):
#         """Add synonym group"""
#         term_lower = term.lower()
#         if term_lower not in self.synonyms:
#             self.synonyms[term_lower] = []
        
#         for syn, conf, src in synonyms_with_confidence:
#             # Check if already exists
#             if not any(s[0].lower() == syn.lower() for s in self.synonyms[term_lower]):
#                 self.synonyms[term_lower].append((syn, conf, src))

#     def extract_terms_from_categories(self, csv_path, min_frequency=2):
#         """Extract terms from category CSV"""
#         print(f"\n📂 Extracting terms from: {csv_path}")
        
#         try:
#             import pandas as pd
            
#             # Read CSV
#             df = pd.read_csv(csv_path)
            
#             # Find path column (usually second column)
#             path_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
#             paths = df[path_col].dropna().astype(str)
            
#             print(f"   Processing {len(paths):,} category paths...")
            
#             term_freq = defaultdict(int)
            
#             for path in tqdm(paths, desc="Analyzing paths"):
#                 levels = path.split('/')
                
#                 for level in levels:
#                     words = level.lower().split()
                    
#                     # Single words
#                     for word in words:
#                         if len(word) > 2 and word.isalpha():
#                             term_freq[word] += 1
                    
#                     # Two-word phrases
#                     for i in range(len(words) - 1):
#                         if len(words[i]) > 2 and len(words[i+1]) > 2:
#                             phrase = f"{words[i]} {words[i+1]}"
#                             if phrase.replace(' ', '').isalpha():
#                                 term_freq[phrase] += 1
            
#             # Filter by frequency
#             candidates = [
#                 term for term, freq in term_freq.items() 
#                 if freq >= min_frequency
#             ]
            
#             print(f"✅ Extracted {len(candidates):,} terms (min frequency: {min_frequency})")
#             return candidates, term_freq
            
#         except Exception as e:
#             print(f"❌ Error extracting terms: {e}")
#             import traceback
#             traceback.print_exc()
#             return [], {}

#     def auto_build_from_categories(self, csv_path, top_terms=1000, 
#                                    semantic_threshold=0.70):
#         """Auto-build synonym database from categories"""
#         print("\n" + "="*80)
#         print("🚀 AUTO-BUILD SYNONYM DATABASE")
#         print("="*80)

#         # Load model
#         if not self.load_transformer_model():
#             print("\n⚠️  Continuing with WordNet only (limited coverage)")

#         # Extract terms
#         all_terms, term_freq = self.extract_terms_from_categories(csv_path)
#         if not all_terms:
#             print("❌ No terms extracted")
#             return False

#         # Select top terms
#         print(f"\n🎯 Selecting top {top_terms} terms...")
#         top_frequent = sorted(
#             term_freq.items(), 
#             key=lambda x: x[1], 
#             reverse=True
#         )[:top_terms]
#         terms_to_process = [term for term, _ in top_frequent]

#         print(f"✅ Selected {len(terms_to_process)} terms")
#         print(f"📊 Top 10: {', '.join(terms_to_process[:10])}")
#         print(f"\n🔄 Generating synonyms (threshold={semantic_threshold})...\n")

#         # Process terms
#         stats = {
#             'processed': 0, 
#             'synonyms': 0, 
#             'high_conf': 0
#         }
        
#         for term in tqdm(terms_to_process, desc="Processing"):
#             # Skip if already has enough synonyms
#             if term in self.synonyms and len(self.synonyms[term]) >= 10:
#                 continue
            
#             # Generate synonyms
#             syns = self.auto_generate_synonyms(
#                 term, 
#                 candidate_pool=all_terms,
#                 semantic_threshold=semantic_threshold, 
#                 silent=True
#             )
            
#             if syns:
#                 self.add_synonym_group(term, syns)
#                 stats['processed'] += 1
#                 stats['synonyms'] += len(syns)
#                 stats['high_conf'] += sum(1 for _, c, _ in syns if c >= 0.8)

#         # Print stats
#         print(f"\n✅ Processed: {stats['processed']:,} terms")
#         print(f"✅ Total synonyms: {stats['synonyms']:,}")
#         print(f"✅ High confidence (≥0.8): {stats['high_conf']:,}")

#         # Save
#         self.save_synonyms()
        
#         print("\n🎉 AUTO-BUILD COMPLETE!\n")
#         return True


# def main():
#     """Main entry point"""
#     print("\n" + "="*80)
#     print("🤖 AI-POWERED SYNONYM MANAGER (Windows + NVIDIA GPU)")
#     print("="*80 + "\n")
    
#     # Parse arguments
#     fast_mode = '--fast' in sys.argv
    
#     if len(sys.argv) < 2:
#         print("Usage:")
#         print("  python synonym_manager_fixed.py autobuild <csv_file>")
#         print("  python synonym_manager_fixed.py autobuild <csv_file> --fast")
#         print("\nExample:")
#         print("  python synonym_manager_fixed.py autobuild data/category_id_path_only.csv")
#         return
    
#     command = sys.argv[1].lower()
    
#     if command == 'autobuild':
#         if len(sys.argv) < 3:
#             print("❌ CSV file path required")
#             return
        
#         csv_path = sys.argv[2]
        
#         if not Path(csv_path).exists():
#             print(f"❌ File not found: {csv_path}")
#             return
        
#         # Initialize manager
#         manager = FixedAISynonymManager(fast_mode=fast_mode)
        
#         # Run auto-build
#         manager.auto_build_from_categories(csv_path, top_terms=1000)
    
#     else:
#         print(f"❌ Unknown command: {command}")


# if __name__ == "__main__":
#     main()


#for cache2


"""
🤖 AI-POWERED SYNONYM MANAGER (Fixed for Windows + GPU)
========================================================
✅ Uses e5-base-v2 (768D, memory-efficient)
✅ Windows + NVIDIA GPU optimized
✅ Generates cross-store synonyms automatically

Usage:
    python synonym_manager_fixed.py autobuild data/category_id_path_only.csv
    python synonym_manager_fixed.py autobuild data/category_id_path_only.csv --fast
"""

import pickle
from pathlib import Path
import json
from collections import defaultdict
from tqdm import tqdm
import warnings
import sys
import os

warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

try:
    from nltk.corpus import wordnet
    from nltk import download as nltk_download
    WORDNET_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class SynonymManager:
    """AI-powered synonym manager"""
    
    def __init__(self, cache_dir='cache', fast_mode=False):
        self.cache_dir = Path(cache_dir)
        self.synonyms_file = self.cache_dir / 'cross_store_synonyms.pkl'
        self.synonyms = {}
        self.model = None
        self.device = "cpu"
        self.fast_mode = fast_mode
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if self.synonyms_file.exists():
            self.load_synonyms()
    
    def load_synonyms(self):
        """Load existing synonyms"""
        try:
            with open(self.synonyms_file, 'rb') as f:
                loaded = pickle.load(f)
            
            if loaded and list(loaded.values()):
                first_val = next(iter(loaded.values()))
                
                if isinstance(first_val, list) and first_val:
                    if isinstance(first_val[0], tuple):
                        self.synonyms = loaded
                    else:
                        self.synonyms = {k: [(v, 0.8, 'legacy') for v in vals] for k, vals in loaded.items()}
                elif isinstance(first_val, set):
                    self.synonyms = {k: [(v, 0.8, 'legacy') for v in vals] for k, vals in loaded.items()}
            
            print(f"✅ Loaded {len(self.synonyms):,} synonym entries")
        except Exception as e:
            print(f"❌ Error loading synonyms: {e}")
            self.synonyms = {}
    
    def save_synonyms(self):
        """Save synonyms"""
        try:
            with open(self.synonyms_file, 'wb') as f:
                pickle.dump(self.synonyms, f)
            
            json_file = self.cache_dir / 'synonyms_readable.json'
            readable = {
                term: [
                    {'synonym': syn, 'confidence': conf, 'source': src}
                    for syn, conf, src in syns
                ]
                for term, syns in self.synonyms.items()
            }
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(readable, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Saved {len(self.synonyms):,} synonym entries")
            return True
        except Exception as e:
            print(f"❌ Error saving synonyms: {e}")
            return False
    
    def load_transformer_model(self):
        """Load e5-base-v2 model"""
        if not TRANSFORMERS_AVAILABLE:
            print("❌ SentenceTransformers not installed!")
            return False
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.device == "cuda":
            print(f"🔥 NVIDIA GPU detected!")
        
        model_name = "intfloat/e5-base-v2"
        print(f"\n🤖 Loading {model_name}...")
        
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            
            if self.device == "cuda":
                self.model = self.model.half()
                print("⚡ Enabled FP16 precision")
            
            print("✅ Model loaded\n")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def get_wordnet_synonyms(self, word, limit=10):
        """Get WordNet synonyms"""
        if self.fast_mode or not WORDNET_AVAILABLE:
            return []
        
        try:
            try:
                wordnet.synsets('test')
            except:
                nltk_download('wordnet', quiet=True)
                nltk_download('omw-1.4', quiet=True)
            
            synonyms = []
            word_clean = word.lower().replace(' ', '_')
            
            for syn in wordnet.synsets(word_clean):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ').lower()
                    if synonym != word.lower() and len(synonym) > 2:
                        confidence = 0.75
                        synonyms.append((synonym, confidence, 'wordnet'))
                        if len(synonyms) >= limit:
                            break
                if len(synonyms) >= limit:
                    break
            
            return synonyms[:limit]
        except Exception:
            return []
    
    def get_semantic_synonyms(self, term, candidate_pool, threshold=0.70, limit=15):
        """Get semantic synonyms using E5"""
        if not self.model or not candidate_pool:
            return []
        
        try:
            query = f"query: {term}"
            candidates_prefixed = [f"passage: {c}" for c in candidate_pool]
            
            term_emb = self.model.encode(query, convert_to_tensor=True, show_progress_bar=False)
            
            batch_size = 32 if self.device == "cuda" else 8
            all_embeddings = []
            
            for i in range(0, len(candidates_prefixed), batch_size):
                batch = candidates_prefixed[i:i + batch_size]
                emb = self.model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
                all_embeddings.append(emb)
            
            candidate_embs = torch.cat(all_embeddings, dim=0)
            scores = util.cos_sim(term_emb, candidate_embs)[0]
            
            synonyms = []
            for candidate, score in zip(candidate_pool, scores):
                score_val = float(score)
                if score_val > threshold and candidate.lower() != term.lower():
                    confidence = 0.60 + (score_val - threshold) * 0.35 / (1 - threshold)
                    synonyms.append((candidate, confidence, 'semantic'))
            
            synonyms.sort(key=lambda x: x[1], reverse=True)
            return synonyms[:limit]
            
        except Exception as e:
            print(f"⚠️  Semantic error: {e}")
            return []
    
    def auto_generate_synonyms(self, term, candidate_pool=None, semantic_threshold=0.70, silent=False):
        """Generate synonyms from multiple sources"""
        all_synonyms = []
        
        if not silent:
            print(f"\n🔍 Finding synonyms for: '{term}'")
        
        if WORDNET_AVAILABLE and not self.fast_mode:
            wn_syns = self.get_wordnet_synonyms(term, limit=10)
            all_synonyms.extend(wn_syns)
        
        if candidate_pool and self.model:
            sem_syns = self.get_semantic_synonyms(
                term, candidate_pool,
                threshold=semantic_threshold,
                limit=15
            )
            all_synonyms.extend(sem_syns)
        
        synonym_map = {}
        for syn, conf, source in all_synonyms:
            syn_lower = syn.lower()
            if syn_lower not in synonym_map or conf > synonym_map[syn_lower][1]:
                synonym_map[syn_lower] = (syn, conf, source)
        
        final_synonyms = sorted(synonym_map.values(), key=lambda x: x[1], reverse=True)
        return final_synonyms
    
    def add_synonym_group(self, term, synonyms_with_confidence):
        """Add synonym group"""
        term_lower = term.lower()
        if term_lower not in self.synonyms:
            self.synonyms[term_lower] = []
        
        for syn, conf, src in synonyms_with_confidence:
            if not any(s[0].lower() == syn.lower() for s in self.synonyms[term_lower]):
                self.synonyms[term_lower].append((syn, conf, src))
    
    def extract_terms_from_categories(self, csv_path, min_frequency=2):
        """Extract terms from category CSV"""
        print(f"\n📂 Extracting terms from: {csv_path}")
        
        try:
            import pandas as pd
            
            df = pd.read_csv(csv_path)
            path_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            paths = df[path_col].dropna().astype(str)
            
            print(f"   Processing {len(paths):,} category paths...")
            
            term_freq = defaultdict(int)
            
            for path in tqdm(paths, desc="Analyzing paths"):
                levels = path.split('/')
                
                for level in levels:
                    words = level.lower().split()
                    
                    for word in words:
                        if len(word) > 2 and word.isalpha():
                            term_freq[word] += 1
                    
                    for i in range(len(words) - 1):
                        if len(words[i]) > 2 and len(words[i+1]) > 2:
                            phrase = f"{words[i]} {words[i+1]}"
                            if phrase.replace(' ', '').isalpha():
                                term_freq[phrase] += 1
            
            candidates = [
                term for term, freq in term_freq.items()
                if freq >= min_frequency
            ]
            
            print(f"✅ Extracted {len(candidates):,} terms (min frequency: {min_frequency})")
            return candidates, term_freq
            
        except Exception as e:
            print(f"❌ Error extracting terms: {e}")
            import traceback
            traceback.print_exc()
            return [], {}
    
    def auto_build_from_categories(self, csv_path, top_terms=1000, semantic_threshold=0.70):
        """Auto-build synonym database"""
        print("\n" + "="*80)
        print("🚀 AUTO-BUILD SYNONYM DATABASE")
        print("="*80)
        
        if not self.load_transformer_model():
            print("\n⚠️  Continuing with WordNet only")
        
        all_terms, term_freq = self.extract_terms_from_categories(csv_path)
        if not all_terms:
            print("❌ No terms extracted")
            return False
        
        print(f"\n🎯 Selecting top {top_terms} terms...")
        top_frequent = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)[:top_terms]
        terms_to_process = [term for term, _ in top_frequent]
        
        print(f"✅ Selected {len(terms_to_process)} terms")
        print(f"📊 Top 10: {', '.join(terms_to_process[:10])}")
        print(f"\n🔄 Generating synonyms (threshold={semantic_threshold})...\n")
        
        stats = {'processed': 0, 'synonyms': 0, 'high_conf': 0}
        
        for term in tqdm(terms_to_process, desc="Processing"):
            if term in self.synonyms and len(self.synonyms[term]) >= 10:
                continue
            
            syns = self.auto_generate_synonyms(
                term,
                candidate_pool=all_terms,
                semantic_threshold=semantic_threshold,
                silent=True
            )
            
            if syns:
                self.add_synonym_group(term, syns)
                stats['processed'] += 1
                stats['synonyms'] += len(syns)
                stats['high_conf'] += sum(1 for _, c, _ in syns if c >= 0.8)
        
        print(f"\n✅ Processed: {stats['processed']:,} terms")
        print(f"✅ Total synonyms: {stats['synonyms']:,}")
        print(f"✅ High confidence (≥0.8): {stats['high_conf']:,}")
        
        self.save_synonyms()
        
        print("\n🎉 AUTO-BUILD COMPLETE!\n")
        return True


def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("🤖 AI-POWERED SYNONYM MANAGER")
    print("="*80 + "\n")
    
    fast_mode = '--fast' in sys.argv
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python synonym_manager_fixed.py autobuild <csv_file>")
        print("  python synonym_manager_fixed.py autobuild <csv_file> --fast")
        print("\nExample:")
        print("  python synonym_manager_fixed.py autobuild data/category_id_path_only.csv")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'autobuild':
        if len(sys.argv) < 3:
            print("❌ CSV file path required")
            return
        
        csv_path = sys.argv[2]
        
        if not Path(csv_path).exists():
            print(f"❌ File not found: {csv_path}")
            return
        
        manager = SynonymManager(fast_mode=fast_mode)
        manager.auto_build_from_categories(csv_path, top_terms=1000)
    
    else:
        print(f"❌ Unknown command: {command}")


if __name__ == "__main__":
    main()