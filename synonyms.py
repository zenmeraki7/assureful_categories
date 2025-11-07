# """
# 🔧 SYNONYM MANAGEMENT TOOL
# ==========================
# Add, view, and update cross-store synonyms
# """

# import pickle
# from pathlib import Path
# import json
# from collections import defaultdict


# class SynonymManager:
#     """Manage cross-store synonyms"""
    
#     def __init__(self, cache_dir='cache'):
#         self.cache_dir = Path(cache_dir)
#         self.synonyms_file = self.cache_dir / 'cross_store_synonyms.pkl'
#         self.synonyms = {}
        
#         if self.synonyms_file.exists():
#             self.load_synonyms()
#         else:
#             print("⚠️  No existing synonyms file found. Starting fresh.")
    
#     def load_synonyms(self):
#         """Load existing synonyms"""
#         try:
#             with open(self.synonyms_file, 'rb') as f:
#                 self.synonyms = pickle.load(f)
#             print(f"✅ Loaded {len(self.synonyms)} synonym entries")
#         except Exception as e:
#             print(f"❌ Error loading synonyms: {e}")
#             self.synonyms = {}
    
#     def save_synonyms(self):
#         """Save synonyms"""
#         try:
#             self.cache_dir.mkdir(exist_ok=True)
#             with open(self.synonyms_file, 'wb') as f:
#                 pickle.dump(self.synonyms, f)
#             print(f"✅ Saved {len(self.synonyms)} synonym entries")
#             return True
#         except Exception as e:
#             print(f"❌ Error saving synonyms: {e}")
#             return False
    
#     def add_synonym_group(self, term, synonyms):
#         """Add a group of synonyms"""
#         term = term.lower().strip()
#         synonyms = {s.lower().strip() for s in synonyms if s.strip()}
        
#         if term not in self.synonyms:
#             self.synonyms[term] = set()
        
#         self.synonyms[term].update(synonyms)
        
#         # Add bidirectional mappings
#         for syn in synonyms:
#             if syn not in self.synonyms:
#                 self.synonyms[syn] = set()
#             self.synonyms[syn].add(term)
#             self.synonyms[syn].update(synonyms - {syn})
        
#         print(f"✅ Added synonym group for '{term}':")
#         print(f"   Synonyms: {', '.join(sorted(self.synonyms[term]))}")
    
#     def remove_synonym(self, term, synonym_to_remove):
#         """Remove a specific synonym"""
#         term = term.lower().strip()
#         synonym_to_remove = synonym_to_remove.lower().strip()
        
#         if term in self.synonyms and synonym_to_remove in self.synonyms[term]:
#             self.synonyms[term].discard(synonym_to_remove)
#             print(f"✅ Removed '{synonym_to_remove}' from '{term}'")
#         else:
#             print(f"⚠️  Synonym not found")
    
#     def get_synonyms(self, term):
#         """Get synonyms for a term"""
#         term = term.lower().strip()
#         if term in self.synonyms:
#             return sorted(self.synonyms[term])
#         return []
    
#     def list_all(self):
#         """List all synonym groups"""
#         print("\n" + "="*80)
#         print("📚 ALL SYNONYM GROUPS")
#         print("="*80)
        
#         if not self.synonyms:
#             print("No synonyms defined yet.")
#             return
        
#         # Group by canonical term (shortest in each group)
#         seen = set()
#         for term in sorted(self.synonyms.keys()):
#             if term in seen:
#                 continue
            
#             group = {term} | self.synonyms[term]
#             seen.update(group)
            
#             canonical = min(group, key=len)
#             others = sorted(group - {canonical})
            
#             print(f"\n🔸 {canonical}")
#             print(f"   → {', '.join(others)}")
    
#     def export_to_json(self, output_file='synonyms_export.json'):
#         """Export synonyms to JSON"""
#         export_data = {k: list(v) for k, v in self.synonyms.items()}
        
#         try:
#             with open(output_file, 'w', encoding='utf-8') as f:
#                 json.dump(export_data, f, indent=2, ensure_ascii=False)
#             print(f"✅ Exported to {output_file}")
#             return True
#         except Exception as e:
#             print(f"❌ Export failed: {e}")
#             return False
    
#     def import_from_json(self, input_file):
#         """Import synonyms from JSON"""
#         try:
#             with open(input_file, 'r', encoding='utf-8') as f:
#                 import_data = json.load(f)
            
#             for term, syns in import_data.items():
#                 self.add_synonym_group(term, syns)
            
#             print(f"✅ Imported from {input_file}")
#             return True
#         except Exception as e:
#             print(f"❌ Import failed: {e}")
#             return False
    
#     def bulk_add_from_list(self, synonym_pairs):
#         """
#         Bulk add from list of tuples
#         Example: [
#             ('washing machine', ['laundry machine', 'washer']),
#             ('tv', ['television', 'smart tv'])
#         ]
#         """
#         for term, syns in synonym_pairs:
#             self.add_synonym_group(term, syns)
    
#     def search_term(self, query):
#         """Search for terms containing query"""
#         query = query.lower()
#         matches = []
        
#         for term in self.synonyms:
#             if query in term:
#                 matches.append(term)
        
#         if matches:
#             print(f"\n🔍 Found {len(matches)} matches for '{query}':")
#             for match in sorted(matches):
#                 syns = self.get_synonyms(match)
#                 print(f"   • {match} → {', '.join(syns[:3])}" + 
#                       (f" (+{len(syns)-3} more)" if len(syns) > 3 else ""))
#         else:
#             print(f"⚠️  No matches found for '{query}'")


# def interactive_mode():
#     """Interactive synonym management"""
#     manager = SynonymManager()
    
#     print("\n" + "="*80)
#     print("🔧 INTERACTIVE SYNONYM MANAGEMENT")
#     print("="*80)
#     print("\nCommands:")
#     print("  add <term> <syn1> <syn2> ...  - Add synonyms")
#     print("  get <term>                    - Get synonyms for term")
#     print("  remove <term> <synonym>       - Remove a synonym")
#     print("  list                          - List all synonyms")
#     print("  search <query>                - Search for terms")
#     print("  export [filename]             - Export to JSON")
#     print("  import <filename>             - Import from JSON")
#     print("  save                          - Save changes")
#     print("  quit                          - Exit")
#     print("="*80)
    
#     while True:
#         try:
#             cmd = input("\n>>> ").strip()
            
#             if not cmd:
#                 continue
            
#             parts = cmd.split()
#             action = parts[0].lower()
            
#             if action == 'quit' or action == 'exit':
#                 # Ask to save
#                 save = input("Save changes before exit? (y/n): ")
#                 if save.lower() == 'y':
#                     manager.save_synonyms()
#                 print("\n👋 Goodbye!")
#                 break
            
#             elif action == 'add' and len(parts) >= 3:
#                 term = parts[1]
#                 synonyms = parts[2:]
#                 manager.add_synonym_group(term, synonyms)
            
#             elif action == 'get' and len(parts) == 2:
#                 term = parts[1]
#                 syns = manager.get_synonyms(term)
#                 if syns:
#                     print(f"✅ Synonyms for '{term}': {', '.join(syns)}")
#                 else:
#                     print(f"⚠️  No synonyms found for '{term}'")
            
#             elif action == 'remove' and len(parts) == 3:
#                 term = parts[1]
#                 synonym = parts[2]
#                 manager.remove_synonym(term, synonym)
            
#             elif action == 'list':
#                 manager.list_all()
            
#             elif action == 'search' and len(parts) == 2:
#                 query = parts[1]
#                 manager.search_term(query)
            
#             elif action == 'export':
#                 filename = parts[1] if len(parts) > 1 else 'synonyms_export.json'
#                 manager.export_to_json(filename)
            
#             elif action == 'import' and len(parts) == 2:
#                 filename = parts[1]
#                 manager.import_from_json(filename)
            
#             elif action == 'save':
#                 manager.save_synonyms()
            
#             else:
#                 print("⚠️  Unknown command or invalid arguments")
        
#         except KeyboardInterrupt:
#             print("\n\n👋 Goodbye!")
#             break
#         except Exception as e:
#             print(f"❌ Error: {e}")


# def quick_add_common_synonyms():
#     """Quickly add common cross-store synonyms"""
#     manager = SynonymManager()
    
#     print("\n" + "="*80)
#     print("🚀 ADDING COMMON CROSS-STORE SYNONYMS")
#     print("="*80)
    
#     common_synonyms = [
#         # Appliances
#         ('washing machine', ['laundry machine', 'washer', 'clothes washer']),
#         ('refrigerator', ['fridge', 'cooler', 'ice box']),
#         ('dishwasher', ['dish washer', 'dish cleaning machine']),
#         ('microwave', ['microwave oven', 'micro wave']),
#         ('vacuum', ['vacuum cleaner', 'hoover']),
#         ('dryer', ['drying machine', 'clothes dryer', 'tumble dryer']),
        
#         # Electronics
#         ('tv', ['television', 'smart tv', 'telly']),
#         ('mobile', ['phone', 'smartphone', 'cell phone', 'cellphone']),
#         ('laptop', ['notebook', 'portable computer']),
#         ('tablet', ['ipad', 'tab']),
#         ('headphones', ['headset', 'earphones', 'earbuds']),
        
#         # Furniture
#         ('sofa', ['couch', 'settee']),
#         ('wardrobe', ['closet', 'armoire']),
        
#         # Clothing (US vs UK)
#         ('pants', ['trousers', 'slacks']),
#         ('sweater', ['jumper', 'pullover']),
#         ('sneakers', ['trainers', 'tennis shoes']),
        
#         # Baby
#         ('stroller', ['pram', 'pushchair', 'buggy']),
#         ('diaper', ['nappy']),
#         ('pacifier', ['dummy', 'soother']),
        
#         # General
#         ('kids', ['children', 'childrens', 'youth']),
#         ('women', ['womens', 'ladies', 'female']),
#         ('men', ['mens', 'male']),
#     ]
    
#     manager.bulk_add_from_list(common_synonyms)
    
#     print("\n✅ Added all common synonyms")
    
#     # Save
#     if manager.save_synonyms():
#         print("\n🎉 Ready to use! Retrain your model to apply changes:")
#         print("   python train_enhanced_system.py data/category_id_path_only.csv")


# def main():
#     """Main entry point"""
#     print("\n" + "="*80)
#     print("🔧 SYNONYM MANAGEMENT TOOL")
#     print("="*80)
#     print("\nWhat would you like to do?")
#     print("  1. Interactive mode (add/edit synonyms)")
#     print("  2. Quick-add common synonyms")
#     print("  3. List all synonyms")
#     print("  4. Export synonyms to JSON")
#     print("  5. Exit")
#     print("="*80)
    
#     try:
#         choice = input("\nChoice (1-5): ").strip()
        
#         if choice == '1':
#             interactive_mode()
#         elif choice == '2':
#             quick_add_common_synonyms()
#         elif choice == '3':
#             manager = SynonymManager()
#             manager.list_all()
#         elif choice == '4':
#             manager = SynonymManager()
#             filename = input("Export filename (default: synonyms_export.json): ").strip()
#             if not filename:
#                 filename = 'synonyms_export.json'
#             manager.export_to_json(filename)
#         elif choice == '5':
#             print("\n👋 Goodbye!")
#         else:
#             print("⚠️  Invalid choice")
    
#     except KeyboardInterrupt:
#         print("\n\n👋 Goodbye!")


# if __name__ == "__main__":
#     main()





"""
🤖 AI-POWERED SYNONYM MANAGEMENT TOOL
======================================
Uses pretrained models to auto-generate synonyms:
- WordNet for linguistic synonyms
- SentenceTransformers for semantic similarity
- Auto-clustering from category data

Usage:
    # Auto-build from category CSV (RECOMMENDED)
    python synonym_manager.py autobuild data/category_id_path_only.csv
    
    # Interactive mode
    python synonym_manager.py
    
    # Direct command line
    python synonym_manager.py --mode autobuild --csv data/category_id_path_only.csv
"""

import pickle
from pathlib import Path
import json
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import warnings
import sys
warnings.filterwarnings('ignore')

# Try to import NLP libraries
try:
    from nltk.corpus import wordnet
    from nltk import download as nltk_download
    WORDNET_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False
    print("⚠️  NLTK not available. Install with: pip install nltk")

try:
    from sentence_transformers import SentenceTransformer, util
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  SentenceTransformers not available. Install with: pip install sentence-transformers")


class AISynonymManager:
    """AI-powered synonym manager using pretrained models"""
    
    def __init__(self, cache_dir='cache'):
        self.cache_dir = Path(cache_dir)
        self.synonyms_file = self.cache_dir / 'cross_store_synonyms.pkl'
        self.synonyms = {}
        self.model = None
        
        # Create cache directory if doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing synonyms if available
        if self.synonyms_file.exists():
            self.load_synonyms()
        else:
            print("📝 No existing synonyms file. Will create new one.")
    
    def load_synonyms(self):
        """Load existing synonyms"""
        try:
            with open(self.synonyms_file, 'rb') as f:
                self.synonyms = pickle.load(f)
            print(f"✅ Loaded {len(self.synonyms)} synonym entries")
        except Exception as e:
            print(f"❌ Error loading synonyms: {e}")
            self.synonyms = {}
    
    def save_synonyms(self):
        """Save synonyms"""
        try:
            self.cache_dir.mkdir(exist_ok=True)
            with open(self.synonyms_file, 'wb') as f:
                pickle.dump(self.synonyms, f)
            print(f"✅ Saved {len(self.synonyms)} synonym entries to: {self.synonyms_file}")
            return True
        except Exception as e:
            print(f"❌ Error saving synonyms: {e}")
            return False
    
    def load_transformer_model(self, model_name='all-MiniLM-L6-v2'):
        """Load SentenceTransformer model (lightweight version for speed)"""
        if not TRANSFORMERS_AVAILABLE:
            print("❌ SentenceTransformers not installed!")
            print("   Install with: pip install sentence-transformers")
            return False
        
        print(f"\n🤖 Loading AI model: {model_name}")
        print("   (This may take a minute on first run...)")
        try:
            self.model = SentenceTransformer(model_name)
            print("✅ Model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def get_wordnet_synonyms(self, word, limit=10):
        """Get synonyms using WordNet"""
        if not WORDNET_AVAILABLE:
            return []
        
        try:
            # Download WordNet if needed
            try:
                wordnet.synsets('test')
            except:
                print("📥 Downloading WordNet data (one-time)...")
                nltk_download('wordnet', quiet=True)
                nltk_download('omw-1.4', quiet=True)
            
            synonyms = set()
            word_clean = word.lower().replace(' ', '_')
            
            for syn in wordnet.synsets(word_clean):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ').lower()
                    if synonym != word.lower() and len(synonym) > 2:
                        synonyms.add(synonym)
                        
                        if len(synonyms) >= limit:
                            return list(synonyms)[:limit]
            
            return list(synonyms)[:limit]
        
        except Exception as e:
            # Silently skip errors
            return []
    
    def get_semantic_synonyms(self, term, candidate_pool, threshold=0.75, limit=10):
        """Get synonyms using semantic similarity"""
        if not self.model or not candidate_pool:
            return []
        
        try:
            # Encode query and candidates
            term_emb = self.model.encode(term, convert_to_tensor=True, show_progress_bar=False)
            candidate_embs = self.model.encode(candidate_pool, convert_to_tensor=True, show_progress_bar=False)
            
            # Calculate cosine similarity
            scores = util.cos_sim(term_emb, candidate_embs)[0]
            
            # Get high-scoring candidates
            synonyms = []
            for candidate, score in zip(candidate_pool, scores):
                if score > threshold and candidate.lower() != term.lower():
                    synonyms.append((candidate, float(score)))
            
            # Sort by score and return top matches
            synonyms.sort(key=lambda x: x[1], reverse=True)
            return [syn for syn, score in synonyms[:limit]]
        
        except Exception as e:
            print(f"⚠️  Semantic similarity error: {e}")
            return []
    
    def auto_generate_synonyms(self, term, candidate_pool=None, use_wordnet=True, 
                              use_semantic=True, semantic_threshold=0.75, silent=False):
        """
        Auto-generate synonyms using AI models
        
        Args:
            term: The term to find synonyms for
            candidate_pool: List of potential synonyms to check against (for semantic)
            use_wordnet: Use WordNet for linguistic synonyms
            use_semantic: Use semantic similarity (requires candidate_pool)
            semantic_threshold: Minimum similarity score (0-1)
            silent: Don't print progress
        
        Returns:
            List of discovered synonyms
        """
        all_synonyms = set()
        
        if not silent:
            print(f"\n🔍 Finding synonyms for: '{term}'")
        
        # Method 1: WordNet
        if use_wordnet and WORDNET_AVAILABLE:
            if not silent:
                print("   📚 Checking WordNet...")
            wn_syns = self.get_wordnet_synonyms(term, limit=15)
            if wn_syns:
                all_synonyms.update(wn_syns)
                if not silent:
                    print(f"      Found {len(wn_syns)}: {', '.join(wn_syns[:5])}" + 
                          (f" (+{len(wn_syns)-5} more)" if len(wn_syns) > 5 else ""))
        
        # Method 2: Semantic similarity
        if use_semantic and candidate_pool and self.model:
            if not silent:
                print("   🤖 Checking semantic similarity...")
            sem_syns = self.get_semantic_synonyms(term, candidate_pool, 
                                                  threshold=semantic_threshold, limit=15)
            if sem_syns:
                all_synonyms.update(sem_syns)
                if not silent:
                    print(f"      Found {len(sem_syns)}: {', '.join(sem_syns[:5])}" + 
                          (f" (+{len(sem_syns)-5} more)" if len(sem_syns) > 5 else ""))
        
        result = list(all_synonyms)
        if not silent:
            print(f"   ✅ Total unique synonyms: {len(result)}")
        
        return result
    
    def extract_terms_from_categories(self, csv_path, min_frequency=2):
        """Extract common terms from category CSV to build candidate pool"""
        print(f"\n📂 Extracting terms from: {csv_path}")
        
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            
            # Get category paths (second column)
            paths = df.iloc[:, 1].dropna().astype(str)
            
            # Extract all terms
            term_freq = defaultdict(int)
            
            for path in tqdm(paths, desc="Processing paths"):
                # Split by / and clean
                levels = path.split('/')
                for level in levels:
                    # Extract words
                    words = level.lower().split()
                    
                    # Single words
                    for word in words:
                        if len(word) > 2 and word.isalpha():
                            term_freq[word] += 1
                    
                    # 2-word phrases
                    for i in range(len(words) - 1):
                        if len(words[i]) > 2 and len(words[i+1]) > 2:
                            phrase = f"{words[i]} {words[i+1]}"
                            if phrase.replace(' ', '').isalpha():
                                term_freq[phrase] += 1
                    
                    # 3-word phrases
                    for i in range(len(words) - 2):
                        if all(len(w) > 2 for w in words[i:i+3]):
                            phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
                            if phrase.replace(' ', '').isalpha():
                                term_freq[phrase] += 1
            
            # Filter by frequency
            candidates = [term for term, freq in term_freq.items() 
                         if freq >= min_frequency]
            
            print(f"✅ Extracted {len(candidates):,} terms (frequency >= {min_frequency})")
            
            return candidates, term_freq
        
        except Exception as e:
            print(f"❌ Error extracting terms: {e}")
            import traceback
            traceback.print_exc()
            return [], {}
    
    def auto_build_from_categories(self, csv_path, top_terms=500, 
                                   semantic_threshold=0.75, batch_size=50):
        """
        Automatically build synonym database from category CSV
        
        Args:
            csv_path: Path to category CSV
            top_terms: Number of most frequent terms to process
            semantic_threshold: Minimum similarity for synonyms (0.75 = 75% similar)
            batch_size: Process this many terms at a time
        """
        print("\n" + "="*80)
        print("🤖 AUTO-BUILDING SYNONYM DATABASE FROM CATEGORIES")
        print("="*80)
        
        # Step 1: Load transformer model
        if not self.load_transformer_model():
            print("\n⚠️  Continuing with WordNet only (limited coverage)")
            print("   For best results: pip install sentence-transformers")
        
        # Step 2: Extract terms from categories
        all_terms, term_freq = self.extract_terms_from_categories(csv_path)
        
        if not all_terms:
            print("❌ No terms extracted")
            return False
        
        # Step 3: Get most common terms
        print(f"\n🎯 Selecting top {top_terms} terms to process...")
        
        # Sort by frequency
        top_frequent = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)[:top_terms]
        terms_to_process = [term for term, _ in top_frequent]
        
        print(f"✅ Selected {len(terms_to_process)} terms")
        print(f"\n📊 Top 10 terms: {', '.join(terms_to_process[:10])}")
        
        # Step 4: Generate synonyms for each term
        print(f"\n🔄 Generating synonyms...")
        print(f"   Semantic threshold: {semantic_threshold} (higher = stricter)")
        print(f"   This will take a few minutes...\n")
        
        new_synonyms_count = 0
        processed_count = 0
        
        for i in tqdm(range(0, len(terms_to_process), batch_size), desc="Processing batches"):
            batch = terms_to_process[i:i+batch_size]
            
            for term in batch:
                # Skip if already has many synonyms
                if term in self.synonyms and len(self.synonyms[term]) >= 10:
                    continue
                
                # Generate synonyms
                syns = self.auto_generate_synonyms(
                    term, 
                    candidate_pool=all_terms,
                    use_wordnet=WORDNET_AVAILABLE,
                    use_semantic=self.model is not None,
                    semantic_threshold=semantic_threshold,
                    silent=True
                )
                
                if syns:
                    self.add_synonym_group(term, syns, silent=True)
                    new_synonyms_count += len(syns)
                    processed_count += 1
        
        print(f"\n✅ Processed {processed_count:,} terms")
        print(f"✅ Generated {new_synonyms_count:,} new synonym relationships")
        
        # Step 5: Save
        print("\n💾 Saving synonym database...")
        self.save_synonyms()
        
        # Show statistics
        print("\n" + "="*80)
        print("🎉 AUTO-BUILD COMPLETE!")
        print("="*80)
        print(f"📊 Statistics:")
        print(f"   Total terms with synonyms: {len(self.synonyms):,}")
        print(f"   Total synonym relationships: {new_synonyms_count:,}")
        print(f"   Saved to: {self.synonyms_file}")
        
        # Show samples
        print(f"\n📝 Sample synonym groups:")
        sample_count = 0
        for term in sorted(self.synonyms.keys()):
            if sample_count >= 10:
                break
            syns = list(self.synonyms[term])[:3]
            if syns:
                print(f"   • {term} → {', '.join(syns)}" + 
                      (f" (+{len(self.synonyms[term])-3} more)" if len(self.synonyms[term]) > 3 else ""))
                sample_count += 1
        
        print("\n🚀 Next steps:")
        print("   1. Review synonyms: python synonym_manager.py (option 3)")
        print("   2. Retrain model: python train.py data/category_id_path_only.csv")
        print("   3. Start API: python api_server.py")
        print("="*80 + "\n")
        
        return True
    
    def add_synonym_group(self, term, synonyms, silent=False):
        """Add a group of synonyms with bidirectional mapping"""
        term = term.lower().strip()
        synonyms = {s.lower().strip() for s in synonyms if s.strip() and s.lower().strip() != term}
        
        if not synonyms:
            return
        
        if term not in self.synonyms:
            self.synonyms[term] = set()
        
        self.synonyms[term].update(synonyms)
        
        # Add bidirectional mappings
        for syn in synonyms:
            if syn not in self.synonyms:
                self.synonyms[syn] = set()
            self.synonyms[syn].add(term)
            self.synonyms[syn].update(synonyms - {syn})
        
        if not silent:
            print(f"✅ Added synonym group for '{term}':")
            print(f"   Synonyms: {', '.join(sorted(list(self.synonyms[term])[:10]))}" +
                  (f" (+{len(self.synonyms[term])-10} more)" if len(self.synonyms[term]) > 10 else ""))
    
    def get_synonyms(self, term):
        """Get synonyms for a term"""
        term = term.lower().strip()
        if term in self.synonyms:
            return sorted(self.synonyms[term])
        return []
    
    def list_all(self, limit=None):
        """List all synonym groups"""
        print("\n" + "="*80)
        print("📚 SYNONYM GROUPS")
        print("="*80)
        
        if not self.synonyms:
            print("No synonyms defined yet.")
            print("\n💡 Run: python synonym_manager.py autobuild data/category_id_path_only.csv")
            return
        
        # Group by canonical term
        seen = set()
        groups = []
        
        for term in sorted(self.synonyms.keys()):
            if term in seen:
                continue
            
            group = {term} | self.synonyms[term]
            seen.update(group)
            
            canonical = min(group, key=len)
            others = sorted(group - {canonical})
            
            groups.append((canonical, others))
        
        # Display
        display_count = limit if limit else len(groups)
        for i, (canonical, others) in enumerate(groups[:display_count], 1):
            print(f"\n{i}. {canonical}")
            print(f"   → {', '.join(others[:10])}" + 
                  (f" (+{len(others)-10} more)" if len(others) > 10 else ""))
        
        if limit and len(groups) > limit:
            print(f"\n... and {len(groups) - limit} more groups")
        
        print(f"\nTotal: {len(groups)} synonym groups")
        print("="*80)
    
    def search_term(self, query):
        """Search for terms containing query"""
        query = query.lower()
        matches = []
        
        for term in self.synonyms:
            if query in term:
                matches.append(term)
        
        if matches:
            print(f"\n🔍 Found {len(matches)} matches for '{query}':")
            for match in sorted(matches)[:20]:
                syns = self.get_synonyms(match)
                print(f"   • {match} → {', '.join(syns[:3])}" + 
                      (f" (+{len(syns)-3} more)" if len(syns) > 3 else ""))
            if len(matches) > 20:
                print(f"   ... and {len(matches)-20} more")
        else:
            print(f"⚠️  No matches found for '{query}'")
    
    def export_to_json(self, output_file='synonyms_export.json'):
        """Export synonyms to JSON"""
        export_data = {k: list(v) for k, v in self.synonyms.items()}
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            print(f"✅ Exported {len(export_data)} groups to: {output_file}")
            return True
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return False


def interactive_mode():
    """Interactive AI-powered synonym management"""
    manager = AISynonymManager()
    
    print("\n" + "="*80)
    print("🤖 AI-POWERED SYNONYM MANAGEMENT")
    print("="*80)
    print("\nCommands:")
    print("  autobuild <csv_path>          - Auto-build from category CSV (RECOMMENDED)")
    print("  auto <term>                   - Auto-generate synonyms for a term")
    print("  add <term> <syn1> <syn2> ...  - Manually add synonyms")
    print("  get <term>                    - Get synonyms for term")
    print("  list [limit]                  - List all synonyms (optional limit)")
    print("  search <query>                - Search for terms")
    print("  export [filename]             - Export to JSON")
    print("  save                          - Save changes")
    print("  quit                          - Exit")
    print("="*80)
    
    while True:
        try:
            cmd = input("\n>>> ").strip()
            
            if not cmd:
                continue
            
            parts = cmd.split()
            action = parts[0].lower()
            
            if action in ['quit', 'exit']:
                print("\n👋 Goodbye!")
                break
            
            elif action == 'autobuild' and len(parts) >= 2:
                csv_path = parts[1]
                manager.auto_build_from_categories(csv_path)
            
            elif action == 'auto' and len(parts) >= 2:
                term = ' '.join(parts[1:])
                
                # Load model if needed
                if not manager.model:
                    manager.load_transformer_model()
                
                if manager.model or WORDNET_AVAILABLE:
                    syns = manager.auto_generate_synonyms(term, candidate_pool=None)
                    if syns:
                        manager.add_synonym_group(term, syns)
                else:
                    print("❌ No AI models available")
            
            elif action == 'add' and len(parts) >= 3:
                term = parts[1]
                synonyms = parts[2:]
                manager.add_synonym_group(term, synonyms)
            
            elif action == 'get' and len(parts) >= 2:
                term = ' '.join(parts[1:])
                syns = manager.get_synonyms(term)
                if syns:
                    print(f"✅ Synonyms for '{term}': {', '.join(syns)}")
                else:
                    print(f"⚠️  No synonyms found for '{term}'")
            
            elif action == 'list':
                limit = int(parts[1]) if len(parts) > 1 else None
                manager.list_all(limit=limit)
            
            elif action == 'search' and len(parts) >= 2:
                query = ' '.join(parts[1:])
                manager.search_term(query)
            
            elif action == 'export':
                filename = parts[1] if len(parts) > 1 else 'synonyms_export.json'
                manager.export_to_json(filename)
            
            elif action == 'save':
                manager.save_synonyms()
            
            else:
                print("⚠️  Unknown command or invalid arguments")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("🤖 AI-POWERED SYNONYM MANAGEMENT TOOL")
    print("="*80)
    
    # Check for command line arguments
    if len(sys.argv) >= 3 and sys.argv[1] == 'autobuild':
        # Direct autobuild mode
        csv_path = sys.argv[2]
        manager = AISynonymManager()
        manager.auto_build_from_categories(csv_path)
        return
    
    print("\nWhat would you like to do?")
    print("  1. Auto-build from category CSV (RECOMMENDED) ⭐")
    print("  2. Interactive mode (advanced)")
    print("  3. List current synonyms")
    print("  4. Export synonyms to JSON")
    print("  5. Exit")
    print("="*80)
    
    try:
        choice = input("\nChoice (1-5): ").strip()
        
        if choice == '1':
            csv_path = input("Enter path to category CSV (default: data/category_id_path_only.csv): ").strip()
            if not csv_path:
                csv_path = 'data/category_id_path_only.csv'
            
            manager = AISynonymManager()
            manager.auto_build_from_categories(csv_path)
        
        elif choice == '2':
            interactive_mode()
        
        elif choice == '3':
            manager = AISynonymManager()
            limit = input("Show how many groups? (press Enter for all): ").strip()
            limit = int(limit) if limit else None
            manager.list_all(limit=limit)
        
        elif choice == '4':
            manager = AISynonymManager()
            filename = input("Export filename (default: synonyms_export.json): ").strip()
            if not filename:
                filename = 'synonyms_export.json'
            manager.export_to_json(filename)
        
        elif choice == '5':
            print("\n👋 Goodbye!")
        
        else:
            print("⚠️  Invalid choice")
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()