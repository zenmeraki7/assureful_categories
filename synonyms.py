# # """
# # 🤖 FIXED AI-POWERED SYNONYM MANAGER
# # ====================================
# # ✅ Windows + NVIDIA GPU optimized
# # ✅ Uses e5-base-v2 (lower memory)
# # ✅ Proper error handling
# # ✅ Progress tracking

# # Usage:
# #     python synonym_manager_fixed.py autobuild data/category_id_path_only.csv
# #     python synonym_manager_fixed.py autobuild data/category_id_path_only.csv --fast
# # """

# # import pickle
# # from pathlib import Path
# # import json
# # from collections import defaultdict
# # from tqdm import tqdm
# # import warnings
# # import sys
# # import os

# # warnings.filterwarnings('ignore')

# # # Fix CUDA issues on Windows
# # os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# # try:
# #     from nltk.corpus import wordnet
# #     from nltk import download as nltk_download
# #     WORDNET_AVAILABLE = True
# # except ImportError:
# #     WORDNET_AVAILABLE = False
# #     print("⚠️  NLTK not available. Install with: pip install nltk")

# # try:
# #     from sentence_transformers import SentenceTransformer, util
# #     import torch
# #     TRANSFORMERS_AVAILABLE = True
# # except ImportError:
# #     TRANSFORMERS_AVAILABLE = False
# #     print("⚠️  SentenceTransformers not available.")
# #     print("   Install with: pip install sentence-transformers torch")


# # class FixedAISynonymManager:
# #     """Fixed AI-powered synonym manager for Windows + NVIDIA GPU"""

# #     def __init__(self, cache_dir='cache', tags_file='data/tags.json', fast_mode=False):
# #         self.cache_dir = Path(cache_dir)
# #         self.synonyms_file = self.cache_dir / 'cross_store_synonyms.pkl'
# #         self.tags_file = Path(tags_file)
# #         self.synonyms = {}
# #         self.tags_data = {}
# #         self.model = None
# #         self.device = "cpu"
# #         self.fast_mode = fast_mode
        
# #         # Create cache directory
# #         self.cache_dir.mkdir(parents=True, exist_ok=True)
        
# #         # Load existing data
# #         self.load_tags()
# #         if self.synonyms_file.exists():
# #             self.load_synonyms()
# #         else:
# #             print("📝 No existing synonyms file. Will create new one.")

# #     def load_tags(self):
# #         """Load domain-specific tags (optional)"""
# #         if self.tags_file.exists():
# #             try:
# #                 with open(self.tags_file, 'r', encoding='utf-8') as f:
# #                     self.tags_data = json.load(f)
# #                 print(f"✅ Loaded {len(self.tags_data)} tag entries")
# #                 return True
# #             except Exception as e:
# #                 print(f"⚠️  Could not load tags.json: {e}")
# #         else:
# #             print(f"ℹ️  tags.json not found (optional)")
# #         return False

# #     def load_synonyms(self):
# #         """Load existing synonyms with format conversion"""
# #         try:
# #             with open(self.synonyms_file, 'rb') as f:
# #                 loaded = pickle.load(f)
            
# #             # Handle different formats
# #             if not loaded:
# #                 self.synonyms = {}
# #                 return
            
# #             # Check format
# #             first_val = next(iter(loaded.values()))
            
# #             if isinstance(first_val, list):
# #                 if first_val and isinstance(first_val[0], tuple):
# #                     # New format: [(syn, conf, src), ...]
# #                     self.synonyms = loaded
# #                     print(f"✅ Loaded {len(self.synonyms)} synonym entries (new format)")
# #                 elif first_val and isinstance(first_val[0], str):
# #                     # Legacy format: [syn1, syn2, ...]
# #                     self.synonyms = {
# #                         k: [(v, 0.8, 'legacy') for v in vals]
# #                         for k, vals in loaded.items()
# #                     }
# #                     print(f"✅ Converted {len(self.synonyms)} legacy synonym entries")
# #             elif isinstance(first_val, set):
# #                 # Set format
# #                 self.synonyms = {
# #                     k: [(v, 0.8, 'legacy') for v in vals]
# #                     for k, vals in loaded.items()
# #                 }
# #                 print(f"✅ Converted {len(self.synonyms)} set-based entries")
# #             else:
# #                 self.synonyms = {}
# #                 print(f"⚠️  Unknown synonym format")
                
# #         except Exception as e:
# #             print(f"❌ Error loading synonyms: {e}")
# #             self.synonyms = {}

# #     def save_synonyms(self):
# #         """Save synonyms in both formats"""
# #         try:
# #             # Save binary format
# #             with open(self.synonyms_file, 'wb') as f:
# #                 pickle.dump(self.synonyms, f)

# #             # Save readable JSON
# #             json_file = self.cache_dir / 'synonyms_readable.json'
# #             readable = {}
# #             for term, syns in self.synonyms.items():
# #                 readable[term] = [
# #                     {'synonym': syn, 'confidence': float(conf), 'source': src}
# #                     for syn, conf, src in syns
# #                 ]
            
# #             with open(json_file, 'w', encoding='utf-8') as f:
# #                 json.dump(readable, f, indent=2, ensure_ascii=False)

# #             print(f"\n✅ Saved {len(self.synonyms)} synonym entries")
# #             print(f"   📁 Binary: {self.synonyms_file}")
# #             print(f"   📁 JSON: {json_file}")
# #             return True
# #         except Exception as e:
# #             print(f"❌ Error saving synonyms: {e}")
# #             return False

# #     def load_transformer_model(self):
# #         """Load e5-base-v2 model with GPU support"""
# #         if not TRANSFORMERS_AVAILABLE:
# #             print("❌ SentenceTransformers not installed!")
# #             return False

# #         # Check for CUDA
# #         self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
# #         if self.device == "cuda":
# #             print(f"🔥 NVIDIA GPU detected!")
# #             try:
# #                 gpu_name = torch.cuda.get_device_name(0)
# #                 vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
# #                 print(f"   GPU: {gpu_name}")
# #                 print(f"   VRAM: {vram_gb:.1f} GB")
# #             except:
# #                 pass
# #         else:
# #             print("💻 Using CPU (slower)")

# #         # Use e5-base-v2 for better memory efficiency
# #         model_name = "intfloat/e5-base-v2"
# #         print(f"\n🤖 Loading model: {model_name}")
        
# #         try:
# #             self.model = SentenceTransformer(model_name, device=self.device)
# #             self.model.max_seq_length = 256
            
# #             # Use FP16 on GPU for speed
# #             if self.device == "cuda":
# #                 self.model = self.model.half()
# #                 print("⚡ Enabled FP16 precision")
            
# #             print("✅ Model loaded successfully\n")
# #             return True
# #         except Exception as e:
# #             print(f"❌ Failed to load model: {e}")
# #             return False

# #     def get_wordnet_synonyms(self, word, limit=10):
# #         """Get WordNet synonyms"""
# #         if self.fast_mode or not WORDNET_AVAILABLE:
# #             return []
        
# #         try:
# #             # Ensure WordNet is downloaded
# #             try:
# #                 wordnet.synsets('test')
# #             except:
# #                 print("📥 Downloading WordNet data...")
# #                 nltk_download('wordnet', quiet=True)
# #                 nltk_download('omw-1.4', quiet=True)
            
# #             synonyms = []
# #             word_clean = word.lower().replace(' ', '_')
            
# #             for syn in wordnet.synsets(word_clean):
# #                 for lemma in syn.lemmas():
# #                     synonym = lemma.name().replace('_', ' ').lower()
# #                     if synonym != word.lower() and len(synonym) > 2:
# #                         confidence = 0.75  # Fixed confidence for WordNet
# #                         synonyms.append((synonym, confidence, 'wordnet'))
# #                         if len(synonyms) >= limit:
# #                             break
# #                 if len(synonyms) >= limit:
# #                     break
            
# #             return synonyms[:limit]
# #         except Exception:
# #             return []

# #     def get_semantic_synonyms(self, term, candidate_pool, threshold=0.70, limit=15):
# #         """Get semantic synonyms using embeddings"""
# #         if not self.model or not candidate_pool:
# #             return []
        
# #         try:
# #             # E5 model requires query/passage prefixes
# #             query = f"query: {term}"
# #             candidates_prefixed = [f"passage: {c}" for c in candidate_pool]
            
# #             # Encode query
# #             term_emb = self.model.encode(
# #                 query, 
# #                 convert_to_tensor=True, 
# #                 show_progress_bar=False
# #             )

# #             # Encode candidates in batches
# #             batch_size = 32 if self.device == "cuda" else 8
# #             all_embeddings = []
            
# #             for i in range(0, len(candidates_prefixed), batch_size):
# #                 batch = candidates_prefixed[i:i + batch_size]
# #                 emb = self.model.encode(
# #                     batch, 
# #                     convert_to_tensor=True, 
# #                     show_progress_bar=False
# #                 )
# #                 all_embeddings.append(emb)
            
# #             # Concatenate all embeddings
# #             candidate_embs = torch.cat(all_embeddings, dim=0)
            
# #             # Calculate cosine similarity
# #             scores = util.cos_sim(term_emb, candidate_embs)[0]
            
# #             # Filter by threshold
# #             synonyms = []
# #             for candidate, score in zip(candidate_pool, scores):
# #                 score_val = float(score)
# #                 if score_val > threshold and candidate.lower() != term.lower():
# #                     # Scale confidence between 0.6 and 0.95
# #                     confidence = 0.60 + (score_val - threshold) * 0.35 / (1 - threshold)
# #                     synonyms.append((candidate, confidence, 'semantic'))
            
# #             # Sort by confidence
# #             synonyms.sort(key=lambda x: x[1], reverse=True)
# #             return synonyms[:limit]
            
# #         except Exception as e:
# #             print(f"⚠️  Semantic error: {e}")
# #             return []

# #     def auto_generate_synonyms(self, term, candidate_pool=None, 
# #                               semantic_threshold=0.70, silent=False):
# #         """Generate synonyms from multiple sources"""
# #         all_synonyms = []
        
# #         if not silent:
# #             print(f"\n🔍 Finding synonyms for: '{term}'")

# #         # Source 1: WordNet
# #         if WORDNET_AVAILABLE and not self.fast_mode:
# #             wn_syns = self.get_wordnet_synonyms(term, limit=10)
# #             all_synonyms.extend(wn_syns)
        
# #         # Source 2: Semantic similarity
# #         if candidate_pool and self.model:
# #             sem_syns = self.get_semantic_synonyms(
# #                 term, candidate_pool,
# #                 threshold=semantic_threshold, 
# #                 limit=15
# #             )
# #             all_synonyms.extend(sem_syns)

# #         # Deduplicate (keep highest confidence)
# #         synonym_map = {}
# #         for syn, conf, source in all_synonyms:
# #             syn_lower = syn.lower()
# #             if syn_lower not in synonym_map or conf > synonym_map[syn_lower][1]:
# #                 synonym_map[syn_lower] = (syn, conf, source)
        
# #         final_synonyms = sorted(
# #             synonym_map.values(), 
# #             key=lambda x: x[1], 
# #             reverse=True
# #         )
        
# #         return final_synonyms

# #     def add_synonym_group(self, term, synonyms_with_confidence):
# #         """Add synonym group"""
# #         term_lower = term.lower()
# #         if term_lower not in self.synonyms:
# #             self.synonyms[term_lower] = []
        
# #         for syn, conf, src in synonyms_with_confidence:
# #             # Check if already exists
# #             if not any(s[0].lower() == syn.lower() for s in self.synonyms[term_lower]):
# #                 self.synonyms[term_lower].append((syn, conf, src))

# #     def extract_terms_from_categories(self, csv_path, min_frequency=2):
# #         """Extract terms from category CSV"""
# #         print(f"\n📂 Extracting terms from: {csv_path}")
        
# #         try:
# #             import pandas as pd
            
# #             # Read CSV
# #             df = pd.read_csv(csv_path)
            
# #             # Find path column (usually second column)
# #             path_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
# #             paths = df[path_col].dropna().astype(str)
            
# #             print(f"   Processing {len(paths):,} category paths...")
            
# #             term_freq = defaultdict(int)
            
# #             for path in tqdm(paths, desc="Analyzing paths"):
# #                 levels = path.split('/')
                
# #                 for level in levels:
# #                     words = level.lower().split()
                    
# #                     # Single words
# #                     for word in words:
# #                         if len(word) > 2 and word.isalpha():
# #                             term_freq[word] += 1
                    
# #                     # Two-word phrases
# #                     for i in range(len(words) - 1):
# #                         if len(words[i]) > 2 and len(words[i+1]) > 2:
# #                             phrase = f"{words[i]} {words[i+1]}"
# #                             if phrase.replace(' ', '').isalpha():
# #                                 term_freq[phrase] += 1
            
# #             # Filter by frequency
# #             candidates = [
# #                 term for term, freq in term_freq.items() 
# #                 if freq >= min_frequency
# #             ]
            
# #             print(f"✅ Extracted {len(candidates):,} terms (min frequency: {min_frequency})")
# #             return candidates, term_freq
            
# #         except Exception as e:
# #             print(f"❌ Error extracting terms: {e}")
# #             import traceback
# #             traceback.print_exc()
# #             return [], {}

# #     def auto_build_from_categories(self, csv_path, top_terms=1000, 
# #                                    semantic_threshold=0.70):
# #         """Auto-build synonym database from categories"""
# #         print("\n" + "="*80)
# #         print("🚀 AUTO-BUILD SYNONYM DATABASE")
# #         print("="*80)

# #         # Load model
# #         if not self.load_transformer_model():
# #             print("\n⚠️  Continuing with WordNet only (limited coverage)")

# #         # Extract terms
# #         all_terms, term_freq = self.extract_terms_from_categories(csv_path)
# #         if not all_terms:
# #             print("❌ No terms extracted")
# #             return False

# #         # Select top terms
# #         print(f"\n🎯 Selecting top {top_terms} terms...")
# #         top_frequent = sorted(
# #             term_freq.items(), 
# #             key=lambda x: x[1], 
# #             reverse=True
# #         )[:top_terms]
# #         terms_to_process = [term for term, _ in top_frequent]

# #         print(f"✅ Selected {len(terms_to_process)} terms")
# #         print(f"📊 Top 10: {', '.join(terms_to_process[:10])}")
# #         print(f"\n🔄 Generating synonyms (threshold={semantic_threshold})...\n")

# #         # Process terms
# #         stats = {
# #             'processed': 0, 
# #             'synonyms': 0, 
# #             'high_conf': 0
# #         }
        
# #         for term in tqdm(terms_to_process, desc="Processing"):
# #             # Skip if already has enough synonyms
# #             if term in self.synonyms and len(self.synonyms[term]) >= 10:
# #                 continue
            
# #             # Generate synonyms
# #             syns = self.auto_generate_synonyms(
# #                 term, 
# #                 candidate_pool=all_terms,
# #                 semantic_threshold=semantic_threshold, 
# #                 silent=True
# #             )
            
# #             if syns:
# #                 self.add_synonym_group(term, syns)
# #                 stats['processed'] += 1
# #                 stats['synonyms'] += len(syns)
# #                 stats['high_conf'] += sum(1 for _, c, _ in syns if c >= 0.8)

# #         # Print stats
# #         print(f"\n✅ Processed: {stats['processed']:,} terms")
# #         print(f"✅ Total synonyms: {stats['synonyms']:,}")
# #         print(f"✅ High confidence (≥0.8): {stats['high_conf']:,}")

# #         # Save
# #         self.save_synonyms()
        
# #         print("\n🎉 AUTO-BUILD COMPLETE!\n")
# #         return True


# # def main():
# #     """Main entry point"""
# #     print("\n" + "="*80)
# #     print("🤖 AI-POWERED SYNONYM MANAGER (Windows + NVIDIA GPU)")
# #     print("="*80 + "\n")
    
# #     # Parse arguments
# #     fast_mode = '--fast' in sys.argv
    
# #     if len(sys.argv) < 2:
# #         print("Usage:")
# #         print("  python synonym_manager_fixed.py autobuild <csv_file>")
# #         print("  python synonym_manager_fixed.py autobuild <csv_file> --fast")
# #         print("\nExample:")
# #         print("  python synonym_manager_fixed.py autobuild data/category_id_path_only.csv")
# #         return
    
# #     command = sys.argv[1].lower()
    
# #     if command == 'autobuild':
# #         if len(sys.argv) < 3:
# #             print("❌ CSV file path required")
# #             return
        
# #         csv_path = sys.argv[2]
        
# #         if not Path(csv_path).exists():
# #             print(f"❌ File not found: {csv_path}")
# #             return
        
# #         # Initialize manager
# #         manager = FixedAISynonymManager(fast_mode=fast_mode)
        
# #         # Run auto-build
# #         manager.auto_build_from_categories(csv_path, top_terms=1000)
    
# #     else:
# #         print(f"❌ Unknown command: {command}")


# # if __name__ == "__main__":
# #     main()


# #for cache2


# """
# 🤖 AI-POWERED SYNONYM MANAGER (Fixed for Windows + GPU)
# ========================================================
# ✅ Uses e5-base-v2 (768D, memory-efficient)
# ✅ Windows + NVIDIA GPU optimized
# ✅ Generates cross-store synonyms automatically

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
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# try:
#     from nltk.corpus import wordnet
#     from nltk import download as nltk_download
#     WORDNET_AVAILABLE = True
# except ImportError:
#     WORDNET_AVAILABLE = False

# try:
#     from sentence_transformers import SentenceTransformer, util
#     import torch
#     TRANSFORMERS_AVAILABLE = True
# except ImportError:
#     TRANSFORMERS_AVAILABLE = False


# class SynonymManager:
#     """AI-powered synonym manager"""
    
#     def __init__(self, cache_dir='cache', fast_mode=False):
#         self.cache_dir = Path(cache_dir)
#         self.synonyms_file = self.cache_dir / 'cross_store_synonyms.pkl'
#         self.synonyms = {}
#         self.model = None
#         self.device = "cpu"
#         self.fast_mode = fast_mode
        
#         self.cache_dir.mkdir(parents=True, exist_ok=True)
        
#         if self.synonyms_file.exists():
#             self.load_synonyms()
    
#     def load_synonyms(self):
#         """Load existing synonyms"""
#         try:
#             with open(self.synonyms_file, 'rb') as f:
#                 loaded = pickle.load(f)
            
#             if loaded and list(loaded.values()):
#                 first_val = next(iter(loaded.values()))
                
#                 if isinstance(first_val, list) and first_val:
#                     if isinstance(first_val[0], tuple):
#                         self.synonyms = loaded
#                     else:
#                         self.synonyms = {k: [(v, 0.8, 'legacy') for v in vals] for k, vals in loaded.items()}
#                 elif isinstance(first_val, set):
#                     self.synonyms = {k: [(v, 0.8, 'legacy') for v in vals] for k, vals in loaded.items()}
            
#             print(f"✅ Loaded {len(self.synonyms):,} synonym entries")
#         except Exception as e:
#             print(f"❌ Error loading synonyms: {e}")
#             self.synonyms = {}
    
#     def save_synonyms(self):
#         """Save synonyms"""
#         try:
#             with open(self.synonyms_file, 'wb') as f:
#                 pickle.dump(self.synonyms, f)
            
#             json_file = self.cache_dir / 'synonyms_readable.json'
#             readable = {
#                 term: [
#                     {'synonym': syn, 'confidence': conf, 'source': src}
#                     for syn, conf, src in syns
#                 ]
#                 for term, syns in self.synonyms.items()
#             }
#             with open(json_file, 'w', encoding='utf-8') as f:
#                 json.dump(readable, f, indent=2, ensure_ascii=False)
            
#             print(f"✅ Saved {len(self.synonyms):,} synonym entries")
#             return True
#         except Exception as e:
#             print(f"❌ Error saving synonyms: {e}")
#             return False
    
#     def load_transformer_model(self):
#         """Load e5-base-v2 model"""
#         if not TRANSFORMERS_AVAILABLE:
#             print("❌ SentenceTransformers not installed!")
#             return False
        
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
#         if self.device == "cuda":
#             print(f"🔥 NVIDIA GPU detected!")
        
#         model_name = "intfloat/e5-base-v2"
#         print(f"\n🤖 Loading {model_name}...")
        
#         try:
#             self.model = SentenceTransformer(model_name, device=self.device)
            
#             if self.device == "cuda":
#                 self.model = self.model.half()
#                 print("⚡ Enabled FP16 precision")
            
#             print("✅ Model loaded\n")
#             return True
#         except Exception as e:
#             print(f"❌ Failed to load model: {e}")
#             return False
    
#     def get_wordnet_synonyms(self, word, limit=10):
#         """Get WordNet synonyms"""
#         if self.fast_mode or not WORDNET_AVAILABLE:
#             return []
        
#         try:
#             try:
#                 wordnet.synsets('test')
#             except:
#                 nltk_download('wordnet', quiet=True)
#                 nltk_download('omw-1.4', quiet=True)
            
#             synonyms = []
#             word_clean = word.lower().replace(' ', '_')
            
#             for syn in wordnet.synsets(word_clean):
#                 for lemma in syn.lemmas():
#                     synonym = lemma.name().replace('_', ' ').lower()
#                     if synonym != word.lower() and len(synonym) > 2:
#                         confidence = 0.75
#                         synonyms.append((synonym, confidence, 'wordnet'))
#                         if len(synonyms) >= limit:
#                             break
#                 if len(synonyms) >= limit:
#                     break
            
#             return synonyms[:limit]
#         except Exception:
#             return []
    
#     def get_semantic_synonyms(self, term, candidate_pool, threshold=0.70, limit=15):
#         """Get semantic synonyms using E5"""
#         if not self.model or not candidate_pool:
#             return []
        
#         try:
#             query = f"query: {term}"
#             candidates_prefixed = [f"passage: {c}" for c in candidate_pool]
            
#             term_emb = self.model.encode(query, convert_to_tensor=True, show_progress_bar=False)
            
#             batch_size = 32 if self.device == "cuda" else 8
#             all_embeddings = []
            
#             for i in range(0, len(candidates_prefixed), batch_size):
#                 batch = candidates_prefixed[i:i + batch_size]
#                 emb = self.model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
#                 all_embeddings.append(emb)
            
#             candidate_embs = torch.cat(all_embeddings, dim=0)
#             scores = util.cos_sim(term_emb, candidate_embs)[0]
            
#             synonyms = []
#             for candidate, score in zip(candidate_pool, scores):
#                 score_val = float(score)
#                 if score_val > threshold and candidate.lower() != term.lower():
#                     confidence = 0.60 + (score_val - threshold) * 0.35 / (1 - threshold)
#                     synonyms.append((candidate, confidence, 'semantic'))
            
#             synonyms.sort(key=lambda x: x[1], reverse=True)
#             return synonyms[:limit]
            
#         except Exception as e:
#             print(f"⚠️  Semantic error: {e}")
#             return []
    
#     def auto_generate_synonyms(self, term, candidate_pool=None, semantic_threshold=0.70, silent=False):
#         """Generate synonyms from multiple sources"""
#         all_synonyms = []
        
#         if not silent:
#             print(f"\n🔍 Finding synonyms for: '{term}'")
        
#         if WORDNET_AVAILABLE and not self.fast_mode:
#             wn_syns = self.get_wordnet_synonyms(term, limit=10)
#             all_synonyms.extend(wn_syns)
        
#         if candidate_pool and self.model:
#             sem_syns = self.get_semantic_synonyms(
#                 term, candidate_pool,
#                 threshold=semantic_threshold,
#                 limit=15
#             )
#             all_synonyms.extend(sem_syns)
        
#         synonym_map = {}
#         for syn, conf, source in all_synonyms:
#             syn_lower = syn.lower()
#             if syn_lower not in synonym_map or conf > synonym_map[syn_lower][1]:
#                 synonym_map[syn_lower] = (syn, conf, source)
        
#         final_synonyms = sorted(synonym_map.values(), key=lambda x: x[1], reverse=True)
#         return final_synonyms
    
#     def add_synonym_group(self, term, synonyms_with_confidence):
#         """Add synonym group"""
#         term_lower = term.lower()
#         if term_lower not in self.synonyms:
#             self.synonyms[term_lower] = []
        
#         for syn, conf, src in synonyms_with_confidence:
#             if not any(s[0].lower() == syn.lower() for s in self.synonyms[term_lower]):
#                 self.synonyms[term_lower].append((syn, conf, src))
    
#     def extract_terms_from_categories(self, csv_path, min_frequency=2):
#         """Extract terms from category CSV"""
#         print(f"\n📂 Extracting terms from: {csv_path}")
        
#         try:
#             import pandas as pd
            
#             df = pd.read_csv(csv_path)
#             path_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
#             paths = df[path_col].dropna().astype(str)
            
#             print(f"   Processing {len(paths):,} category paths...")
            
#             term_freq = defaultdict(int)
            
#             for path in tqdm(paths, desc="Analyzing paths"):
#                 levels = path.split('/')
                
#                 for level in levels:
#                     words = level.lower().split()
                    
#                     for word in words:
#                         if len(word) > 2 and word.isalpha():
#                             term_freq[word] += 1
                    
#                     for i in range(len(words) - 1):
#                         if len(words[i]) > 2 and len(words[i+1]) > 2:
#                             phrase = f"{words[i]} {words[i+1]}"
#                             if phrase.replace(' ', '').isalpha():
#                                 term_freq[phrase] += 1
            
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
    
#     def auto_build_from_categories(self, csv_path, top_terms=1000, semantic_threshold=0.70):
#         """Auto-build synonym database"""
#         print("\n" + "="*80)
#         print("🚀 AUTO-BUILD SYNONYM DATABASE")
#         print("="*80)
        
#         if not self.load_transformer_model():
#             print("\n⚠️  Continuing with WordNet only")
        
#         all_terms, term_freq = self.extract_terms_from_categories(csv_path)
#         if not all_terms:
#             print("❌ No terms extracted")
#             return False
        
#         print(f"\n🎯 Selecting top {top_terms} terms...")
#         top_frequent = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)[:top_terms]
#         terms_to_process = [term for term, _ in top_frequent]
        
#         print(f"✅ Selected {len(terms_to_process)} terms")
#         print(f"📊 Top 10: {', '.join(terms_to_process[:10])}")
#         print(f"\n🔄 Generating synonyms (threshold={semantic_threshold})...\n")
        
#         stats = {'processed': 0, 'synonyms': 0, 'high_conf': 0}
        
#         for term in tqdm(terms_to_process, desc="Processing"):
#             if term in self.synonyms and len(self.synonyms[term]) >= 10:
#                 continue
            
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
        
#         print(f"\n✅ Processed: {stats['processed']:,} terms")
#         print(f"✅ Total synonyms: {stats['synonyms']:,}")
#         print(f"✅ High confidence (≥0.8): {stats['high_conf']:,}")
        
#         self.save_synonyms()
        
#         print("\n🎉 AUTO-BUILD COMPLETE!\n")
#         return True


# def main():
#     """Main entry point"""
#     print("\n" + "="*80)
#     print("🤖 AI-POWERED SYNONYM MANAGER")
#     print("="*80 + "\n")
    
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
        
#         manager = SynonymManager(fast_mode=fast_mode)
#         manager.auto_build_from_categories(csv_path, top_terms=1000)
    
#     else:
#         print(f"❌ Unknown command: {command}")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
gradio_app.py - The "Universal" Config-Driven Hybrid Classifier

FEATURES:
1. CONFIG DRIVEN: Loads 'data/rules.json'.
2. CONFLICT RESOLUTION: "Hair Oil" (Beauty) vs "Cooking Oil" (Food).
3. SAFETY LOCKS: Prevents "Short Sleeve" -> Electronics.
4. GAMING & PROCESSOR LOGIC: Handles Laptops, CPUs, and GPUs correctly.
5. INSURANCE LOGIC: Predicts Approved/Reject based on CSV data (Straight Through vs Reject).
"""

import os
import json
import pickle
import re
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

import pandas as pd
import faiss
import gradio as gr
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# ---------------------------------------------------------
# ⚙️ CONFIGURATION
# ---------------------------------------------------------
CACHE_DIR = Path("cache")
DATA_DIR = Path("data")
CACHE_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# File Paths
RULES_PATH = DATA_DIR / "rules.json"
CSV_PATH = DATA_DIR / "categories.csv"
TAGS_PATH = DATA_DIR / "tags.json"
INDEX_PATH_E5 = CACHE_DIR / "index_e5.faiss"
INDEX_PATH_MPNET = CACHE_DIR / "index_mpnet.faiss"
METADATA_PATH = CACHE_DIR / "metadata.pkl"

# Model Names
MODEL_NAME_E5 = "intfloat/e5-base-v2"
MODEL_NAME_MPNET = "sentence-transformers/all-mpnet-base-v2"
MODEL_NAME_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2" 

# Device Detection
try:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

print(f"⚡ Running on: {DEVICE.upper()}")

# ---------------------------------------------------------
# 🌍 LOAD RULES & GLOBALS
# ---------------------------------------------------------

# Hardcoded Mappings
ACCESSORY_KEYWORD_TO_ID = {
    "flip case": "9931389011", "armband": "7073962011", "holster": "2407765011",
    "sleeve": "9414313011", "bumper": "17875442011", "dry bag": "17875443011",
    "case": "3081461011", "cover": "3081461011", "screen protector": "3081461011",
    "tempered glass": "3081461011"
}

def load_rules():
    defaults = {"ignore_words": [], "domain_keywords": {}, "critical_ids": {}}
    if not RULES_PATH.exists(): return defaults
    try:
        with open(RULES_PATH, 'r', encoding='utf-8') as f: return json.load(f)
    except Exception: return defaults

RULES_DATA = load_rules()
IGNORE_LIST = set(RULES_DATA.get("ignore_words", []))
DOMAIN_KEYWORDS = {k: set(v) for k, v in RULES_DATA.get("domain_keywords", {}).items()}
CRITICAL_IDS = RULES_DATA.get("critical_ids", {})

# Global Placeholders
retriever_e5 = None
retriever_mpnet = None
reranker = None
index_e5 = None
index_mpnet = None
metadata: List[Dict] = []
catid_to_meta: Dict[str, Dict] = {} 
tag_lookup: Dict[str, Set[str]] = {} 

# ---------------------------------------------------------
# 🛠️ HELPER FUNCTIONS
# ---------------------------------------------------------

def clean_text(text: str) -> str:
    if not text: return ""
    s = str(text).lower().strip()
    s = re.sub(r"[^\w\s\-]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()

def get_final_product_name(path: str) -> str:
    if not path: return ""
    parts = [p for p in path.split('/') if p]
    return parts[-1].strip() if parts else path.strip()

def score_to_pct(score):
    boosted_score = (float(score) + 4.5) * 1.5
    if boosted_score > 10: return 99.9
    if boosted_score < -10: return 1.0
    try:
        prob = 1 / (1 + np.exp(-boosted_score))
        return round(prob * 100, 2)
    except:
        return 0.0

def simple_stem(w): return w.rstrip('s')

def has_word(text, word_set):
    text_lower = clean_text(text)
    for w in word_set:
        if re.search(r'\b' + re.escape(w) + r'\b', text_lower): return True
    return False

# ---------------------------------------------------------
# 🧠 LOGIC ENGINES
# ---------------------------------------------------------

def check_universal_match(title, path):
    t_words = clean_text(title).split()
    t_stems = {simple_stem(w) for w in t_words}
    leaf_words = clean_text(path.split('/')[-1]).split()
    leaf_stems = {simple_stem(w) for w in leaf_words}
    ignore_stems = {simple_stem(w) for w in IGNORE_LIST}
    overlap = t_stems.intersection(leaf_stems) - ignore_stems
    if len(overlap) > 0:
        return 40.0 * len(overlap), [f"✅ Univ Match: {overlap}"]
    return 0.0, []

def check_guardrails(title, description, path):
    score = 0.0
    log = []
    p = path.lower()
    full_text = f"{title} {description}"
    
    univ_score, univ_log = check_universal_match(title, path)
    score += univ_score
    log.extend(univ_log)

    is_beauty = has_word(full_text, DOMAIN_KEYWORDS.get("beauty", set()))

    for domain, keywords in DOMAIN_KEYWORDS.items():
        if has_word(full_text, keywords):
            if domain == "food" and is_beauty: continue 

            if domain in p:
                score += 50.0
                log.append(f"✅ {domain.title()} Rule (Boost)")
            
            elif domain == "clothing" and ("apparel" in p or "garment" in p):
                score += 50.0; log.append("✅ Clothing Synonym Rule")
            elif domain == "home" and ("decor" in p or "bedding" in p):
                score += 40.0; log.append("✅ Home Synonym Rule")
            
            if domain == "clothing" and ("food" in p or "toy" in p):
                if "doll" not in clean_text(title): score -= 500.0; log.append("⛔ Not Clothing")
            if domain == "food" and ("toy" in p or "electronic" in p):
                score -= 200.0; log.append("⛔ Not Food")

    return score, log

def get_rule_match_id(title: str, description: str = "") -> Tuple[Optional[str], str]:
    """Hardcoded Logic with Processor & Laptop Support."""
    txt = clean_text(title)
    
    # --- 1. PROCESSOR & CPU LOGIC ---
    if "processor" in txt or "cpu" in txt:
        if any(w in txt for w in ["food", "kitchen", "chopper", "cup"]): return None, "AI Handle Food Processor"
        if any(w in txt for w in ["intel", "amd", "ryzen", "core", "gaming", "ghz", "threadripper"]):
             return "19390085011", "Rule: Computer Processor (CPU)"

    # --- 2. COMPUTERS & GAMING LOGIC ---
    if "laptop" in txt or "macbook" in txt:
        if any(w in txt for w in ["toy", "learning", "kid", "baby", "vtech"]): return None, "" 
        if "stand" in txt or "riser" in txt: return "3015409011", "Rule: Laptop Stand"
        if "sleeve" in txt or "case" in txt or "bag" in txt: return "172470", "Rule: Laptop Bag/Case"
        return "565108", "Rule: Laptop Computer"

    if ("gaming" in txt or "gamer" in txt) and ("pc" in txt or "desktop" in txt or "computer" in txt):
        if not any(w in txt for w in ["chair", "desk", "headset", "mouse", "keyboard"]):
            return "565108", "Rule: Gaming Desktop/PC"

    # --- 3. CHEBE & HAIR LOGIC ---
    hair_triggers = ["hair", "chebe", "karkar", "alopecia", "scalp", "locks", "tresses", "coils", "curls"]
    is_hair_product = any(k in txt for k in hair_triggers)

    if is_hair_product:
        if "mask" in txt: return "10666437011", "Rule: Hair Context -> Hair Masks"
        elif "oil" in txt: return "10666439011", "Rule: Hair Context -> Hair Oils"
        elif "shampoo" in txt: return "11057651", "Rule: Hair Context -> Shampoo"
        elif "conditioner" in txt:
            if "deep" in txt: return "17911767011", "Rule: Hair Context -> Deep Conditioner"
            return "11057251", "Rule: Hair Context -> Conditioner"
        elif any(k in txt for k in ["growth", "loss", "regrowth"]): return "11057581", "Rule: Hair Context -> Hair Growth/Loss"
        elif any(k in txt for k in ["pomade", "paste", "gel", "wax", "styling"]):
             if "gel" in txt: return "11057871", "Rule: Hair Context -> Gels"
             if "spray" in txt: return "11057891", "Rule: Hair Context -> Sprays"
             return "11057841", "Rule: Hair Context -> Styling Products"
        else: return "11057431", "Rule: Hair Context -> Hair Treatment (Generic)"

    # --- SAFETY CHECK: CLOTHING CONTEXT ---
    is_clothing_context = any(w in txt for w in ["shirt", "dress", "pant", "short", "size", "cotton", "baby", "girl", "boy", "men", "women", "apparel", "garment"])

    # --- 4. Accessories (With Safety Lock) ---
    for phrase, cid in ACCESSORY_KEYWORD_TO_ID.items():
        if is_clothing_context and phrase in ["sleeve", "case", "cover"]: continue
        if re.search(r"\b" + re.escape(phrase) + r"\b", txt): return cid, f"Rule: Accessory '{phrase}'"

    # --- 5. iPhone Books ---
    if "iphone" in txt and any(k in txt for k in ["book", "guide", "manual"]): 
        return CRITICAL_IDS.get("iphone_books", "6133978011"), "Rule: iPhone Book"
    
    # --- 6. Appliances ---
    is_laundry = False
    if "machine washable" not in txt and "hand wash" not in txt:
        if "washing machine" in txt or "clothes washer" in txt: is_laundry = True
    if is_laundry:
        if not any(a in txt for a in ["cover", "stand", "cleaner", "powder", "part"]):
            return CRITICAL_IDS.get("washing_machine", "2383576011"), "Rule: Appliance Match"

    # --- 7. Phones vs Carriers ---
    phone_kws = {"smartphone", "iphone", "galaxy"} 
    if any(b in txt for b in phone_kws):
        if "carrier" in txt or "locked" in txt or "prepaid" in txt: return CRITICAL_IDS.get("carrier_phone", "2407748011"), "Rule: Carrier Phone"
        if not any(a in txt for a in ["case", "cover", "screen", "cable", "charger"]): return CRITICAL_IDS.get("unlocked_phone", "2407749011"), "Rule: Unlocked Phone"

    return None, ""

# ---------------------------------------------------------
# 🔎 CLASSIFICATION ENGINE
# ---------------------------------------------------------

def add_candidate_manual(cand_map, meta, raw_score, source):
    cid = meta['category_id']
    if cid in cand_map:
        cand_map[cid]['sources'].add(source)
        cand_map[cid]['retrieval_score'] = max(cand_map[cid]['retrieval_score'], float(raw_score))
    else:
        cand_map[cid] = {
            'category_id': cid,
            'category_path': meta['category_path'],
            'final_product': meta['final_product'],
            'status': meta.get('status', 'Unknown'),
            'risk': meta.get('risk', ''),
            'depth': meta['depth'],
            'retrieval_score': float(raw_score),
            'confidence': float(raw_score) * 100.0,
            'sources': {source},
            'logic_log': []
        }

def add_candidate(cand_map, idx, raw_score, source):
    if idx < len(metadata):
        meta = metadata[idx]
        add_candidate_manual(cand_map, meta, raw_score, source)

def classify_product(title: str, description: str = "") -> Dict:
    full_text = f"{title} {description}".strip()
    candidates_map = {}
    
    # 1. Retrieval
    e5_emb = retriever_e5.encode(f"passage: {full_text}", convert_to_numpy=True, normalize_embeddings=True)
    dists_e5, idxs_e5 = index_e5.search(e5_emb.reshape(1, -1).astype('float32'), 30)
    for i, idx in enumerate(idxs_e5[0]):
        if idx >= 0: add_candidate(candidates_map, int(idx), float(dists_e5[0][i]), 'E5')

    mp_emb = retriever_mpnet.encode(full_text, convert_to_numpy=True, normalize_embeddings=True)
    dists_mp, idxs_mp = index_mpnet.search(mp_emb.reshape(1, -1).astype('float32'), 30)
    for i, idx in enumerate(idxs_mp[0]):
        if idx >= 0: add_candidate(candidates_map, int(idx), float(dists_mp[0][i]), 'MPNet')

    # 2. Tags
    words = clean_text(title).split()
    for i in range(len(words)):
        for j in range(i, min(i+6, len(words))):
            phrase = " ".join(words[i:j+1])
            if phrase in tag_lookup:
                for cid in tag_lookup[phrase]:
                    if cid in candidates_map:
                        candidates_map[cid]['confidence'] += 30.0
                        candidates_map[cid]['sources'].add('Tag')
                    else:
                        if cid in catid_to_meta:
                            add_candidate_manual(candidates_map, catid_to_meta[cid], 0.95, 'Tag')

    candidates = list(candidates_map.values())
    
    # 3. Guardrails
    for res in candidates:
        if len(res['sources']) > 1: res['confidence'] += 5.0
        res['confidence'] += min(8.0, res['depth'] * 1.5)
        g_score, g_log = check_guardrails(title, description, res['category_path'])
        res['confidence'] += g_score
        res['logic_log'].extend(g_log)

    candidates.sort(key=lambda x: x['confidence'], reverse=True)
    top_candidates = candidates[:30]
    
    # 4. Rerank
    if reranker and top_candidates:
        rerank_inputs = [[title, c['category_path']] for c in top_candidates]
        scores = reranker.predict(rerank_inputs)
        for i, score in enumerate(scores): 
            top_candidates[i]['rerank_score'] = float(score)
        top_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)

    # 5. Rule Override
    rule_id, rule_reason = get_rule_match_id(title, description)
    final_top_results = []
    
    if rule_id and rule_id in catid_to_meta:
        meta = catid_to_meta[rule_id]
        rule_winner = {
            'category_id': rule_id,
            'category_path': meta['category_path'],
            'final_product': meta['final_product'],
            'status': meta.get('status', 'Unknown'),
            'risk': meta.get('risk', ''),
            'rerank_score': 99.99, 
            'sources': {'Rule'},
            'logic_log': [f"🏆 {rule_reason}"]
        }
        final_top_results = [rule_winner] + [c for c in top_candidates if c['category_id'] != rule_id][:19]
    else:
        final_top_results = top_candidates[:20]

    top = final_top_results[0] if final_top_results else None
    
    return {
        'final_product': top['final_product'] if top else 'Unknown',
        'category_path': top['category_path'] if top else 'Unknown',
        'category_id': top['category_id'] if top else 'Unknown',
        'status': top['status'] if top else 'Unknown',
        'risk': top['risk'] if top else '',
        'rerank_score': top.get('rerank_score', 0.0) if top else 0.0,
        'logic_log': top.get('logic_log', []) if top else [],
        'top_results': final_top_results
    }

# ---------------------------------------------------------
# 📊 INITIALIZATION
# ---------------------------------------------------------

def determine_status(row):
    """Parses columns to decide Approved vs Rejected."""
    # Use exact column names from your CSV
    sa = str(row.get('Special Acceptance', '')).strip()
    reject = str(row.get('Reject', '')).strip()
    straight = str(row.get('Straight Through business', '')).strip()
    
    # 1. Use "Special Acceptance" if available
    if sa:
        if "straight through" in sa.lower(): return "🟢 APPROVED"
        if "reject" in sa.lower(): return "🔴 REJECT"
        if "refer" in sa.lower(): return "🟡 REFERRAL"
        return f"⚪ {sa}"
    
    # 2. Fallback Columns
    if reject.lower() == 'yes': return "🔴 REJECT"
    if straight.lower() == 'yes': return "🟢 APPROVED"
    return "⚪ REVIEW"

def build_metadata_from_csv(csv_path: Path) -> List[Dict]:
    try:
        if str(csv_path).endswith('.xlsx'): df = pd.read_excel(csv_path, dtype=str).fillna("")
        else:
            try: df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, on_bad_lines='skip', encoding='utf-8')
            except UnicodeDecodeError: df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, on_bad_lines='skip', encoding='latin1')
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return []
    rows = []
    if len(df.columns) < 2: return []
    col_id = df.columns[0]
    col_path = df.columns[1]
    
    for _, row in df.iterrows():
        cid = str(row[col_id]).strip()
        path = str(row[col_path]).strip()
        if not cid or not path: continue
        
        # Capture Insurance Data
        status_text = determine_status(row)
        risk_val = str(row.get('Assureful RR', '')).strip()
        
        rows.append({
            'category_id': cid, 
            'category_path': path, 
            'final_product': get_final_product_name(path),
            'status': status_text,
            'risk': risk_val,
            'depth': len(path.split('/')), 
            'indexed_text': f"passage: {path}"
        })
    return rows

def build_index(model, items, index_path, use_prefix=False):
    texts = [it['indexed_text'] if use_prefix else it['category_path'] for it in items]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.asarray(embeddings, dtype='float32'))
    faiss.write_index(index, str(index_path))
    return index

def initialize():
    global retriever_e5, retriever_mpnet, reranker, index_e5, index_mpnet, metadata, catid_to_meta, tag_lookup
    print(f"🚀 Initializing Engine (device={DEVICE})...")
    retriever_e5 = SentenceTransformer(MODEL_NAME_E5, device=DEVICE)
    retriever_mpnet = SentenceTransformer(MODEL_NAME_MPNET, device=DEVICE)
    try: reranker = CrossEncoder(MODEL_NAME_RERANKER, device=DEVICE)
    except: reranker = CrossEncoder(MODEL_NAME_RERANKER, device='cpu')

    if not CSV_PATH.exists(): raise FileNotFoundError("categories.csv missing!")
    print("📊 Building metadata...")
    metadata = build_metadata_from_csv(CSV_PATH)
    catid_to_meta = {m['category_id']: m for m in metadata}
    with open(METADATA_PATH, 'wb') as f: pickle.dump(metadata, f)

    if not INDEX_PATH_E5.exists():
        index_e5 = build_index(retriever_e5, metadata, INDEX_PATH_E5, True)
        index_mpnet = build_index(retriever_mpnet, metadata, INDEX_PATH_MPNET, False)
    else:
        index_e5 = faiss.read_index(str(INDEX_PATH_E5))
        index_mpnet = faiss.read_index(str(INDEX_PATH_MPNET))

    tag_lookup = {}
    if TAGS_PATH.exists():
        print("🏷️ Loading tags...")
        try:
            with open(TAGS_PATH, 'r', encoding='utf-8') as f:
                tags_data = json.load(f)
            for cat_id, tags in tags_data.items():
                if cat_id not in catid_to_meta or not isinstance(tags, list): continue
                for t in tags:
                    t_clean = clean_text(t)
                    if len(t_clean) < 3: continue
                    tag_lookup.setdefault(t_clean, set()).add(str(cat_id))
            print(f"✅ Loaded {len(tag_lookup)} tag variations.")
        except Exception as e: print(f"⚠️ Tag Load Error: {e}")
    print("✅ System Ready.")

# ---------------------------------------------------------
# 🖥️ UI HANDLERS
# ---------------------------------------------------------

def read_file_robust(path):
    try:
        if path.endswith('.xlsx') or path.endswith('.xls'): return pd.read_excel(path, dtype=str).fillna("")
        try: return pd.read_csv(path, dtype=str, encoding='utf-8').fillna("")
        except UnicodeDecodeError: return pd.read_csv(path, dtype=str, encoding='latin1').fillna("")
    except Exception as e: raise gr.Error(f"File Read Error: {e}")

def analyze_uploaded_csv(file_obj):
    if file_obj is None: return pd.DataFrame(), gr.update(choices=[]), gr.update(choices=[])
    path = file_obj.name
    df = read_file_robust(path)
    cols = list(df.columns)
    default_title = "product_name" if "product_name" in cols else cols[0]
    default_desc = "description" if "description" in cols else (cols[1] if len(cols)>1 else cols[0])
    return df.head(5), gr.update(choices=cols, value=default_title), gr.update(choices=cols, value=default_desc)

def process_batch_csv(file_obj, title_col, desc_col, limit_choice, threshold_val, progress=gr.Progress()):
    if file_obj is None: raise gr.Error("No file.")
    df = read_file_robust(file_obj.name)
    limit = int(str(limit_choice).split()[0])
    results_data = []
    
    for idx, row in progress.tqdm(df.iterrows(), total=len(df), desc="Classifying"):
        t = str(row[title_col])
        d = str(row[desc_col]) if desc_col in df.columns else ""
        res = classify_product(t, d)
        top_N = res.get('top_results', [])[:limit]
        out_row = row.to_dict()
        raw_score = res.get('rerank_score', -99.0)
        
        if raw_score < threshold_val:
            out_row["Best_Match_ID"] = "N/A"
            out_row["Best_Match_Path"] = "⛔ Below Threshold"
            out_row["Insurance_Status"] = "Unknown"
            out_row["Risk_Rating"] = ""
            out_row["Confidence_%"] = score_to_pct(raw_score)
        else:
            out_row["Best_Match_ID"] = str(res.get('category_id'))
            out_row["Best_Match_Path"] = str(res.get('category_path'))
            out_row["Insurance_Status"] = str(res.get('status'))
            out_row["Risk_Rating"] = str(res.get('risk'))
            out_row["Confidence_%"] = score_to_pct(raw_score)
            
        for i in range(limit):
            if i < len(top_N):
                item = top_N[i]
                out_row[f"Rank_{i+1}_Path"] = str(item['category_path'])
                out_row[f"Rank_{i+1}_ID"] = str(item['category_id'])
                out_row[f"Rank_{i+1}_Status"] = str(item.get('status', ''))
                out_row[f"Rank_{i+1}_Score"] = score_to_pct(item.get('rerank_score', 0))
            else:
                out_row[f"Rank_{i+1}_Path"] = ""
                out_row[f"Rank_{i+1}_ID"] = ""
                out_row[f"Rank_{i+1}_Status"] = ""
                out_row[f"Rank_{i+1}_Score"] = ""
        results_data.append(out_row)

    out_df = pd.DataFrame(results_data)
    out_df.to_csv("batch_results.csv", index=False)
    return "batch_results.csv", out_df.head(10)

def gradio_classify_ui(title, desc):
    res = classify_product(title, desc)
    top_text = ""
    for i, item in enumerate(res.get('top_results', []), 1):
        logs = " | ".join(item.get('logic_log', []))
        pct = score_to_pct(item.get('rerank_score', 0))
        status_icon = item.get('status', '')
        # THIS LINE ADDS STATUS TO DETAILS BOX
        top_text += f"{i}. {item['final_product']}\n   ID: {item['category_id']}\n   Status: {status_icon}\n   Score: {pct}%\n   Path: {item['category_path']}\n   Logic: {logs}\n\n"
    
    # UI Output Fields
    final_status = res.get('status', 'Unknown')
    return (
        str(res.get('final_product')), 
        str(res.get('category_path')), 
        str(res.get('category_id')), 
        final_status,
        f"{score_to_pct(res.get('rerank_score', 0))}%", 
        " | ".join(res.get('logic_log',[])), 
        top_text
    )

def main():
    initialize()
    with gr.Blocks(title="AI Category Classifier") as app:
        gr.Markdown("# ⚡ Universal Hybrid Classifier")
        with gr.Tabs():
            with gr.TabItem("Single Prediction"):
                with gr.Row():
                    t_in = gr.Textbox(label="Title")
                    d_in = gr.Textbox(label="Description")
                    btn = gr.Button("Classify", variant="primary")
                with gr.Row():
                    o_winner = gr.Textbox(label="Winner")
                    o_path = gr.Textbox(label="Path")
                    o_id = gr.Textbox(label="ID")
                    o_ins_status = gr.Textbox(label="Insurance Status")
                with gr.Row():
                    o_conf = gr.Textbox(label="Confidence")
                    o_logic = gr.Textbox(label="Logic")
                o_details = gr.TextArea(label="Details")
                btn.click(gradio_classify_ui, [t_in, d_in], [o_winner, o_path, o_id, o_ins_status, o_conf, o_logic, o_details])
            
            with gr.TabItem("Batch Prediction"):
                with gr.Row():
                    file_in = gr.File(label="Upload CSV/XLSX")
                    with gr.Column():
                        df_prev = gr.DataFrame(label="Preview", interactive=False)
                        c_title = gr.Dropdown(label="Title Col")
                        c_desc = gr.Dropdown(label="Desc Col")
                        c_limit = gr.Dropdown(label="Limit", choices=["1 Result", "5 Results", "10 Results"], value="5 Results")
                        c_thresh = gr.Slider(label="Threshold", minimum=-10, maximum=10, value=-5.0)
                btn_batch = gr.Button("Process Batch", variant="primary")
                with gr.Row():
                    f_out = gr.File(label="Download")
                    df_out = gr.DataFrame(label="Results")
                file_in.upload(analyze_uploaded_csv, file_in, [df_prev, c_title, c_desc])
                btn_batch.click(process_batch_csv, [file_in, c_title, c_desc, c_limit, c_thresh], [f_out, df_out])

    app.launch(server_name="127.0.0.1", server_port=7860, share=True)

if __name__ == "__main__":
    main()