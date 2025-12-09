# # #!/usr/bin/env python3
# # """
# # train.py - FIXED VERSION
# # Build FAISS index with CORRECT E5 format for better predictions

# # Key fixes:
# # 1. ✅ Uses "passage:" prefix for category encoding (not "category:")
# # 2. ✅ Proper normalization for cosine similarity
# # 3. ✅ Clear documentation of distance -> confidence conversion
# # 4. ✅ Maintains compatibility with your existing structure

# # Usage:
# #     python train.py --csv data/category_only_path.csv --model intfloat/e5-base-v2
    
# # Optional:
# #     --gpu                Use GPU if available
# #     --validation         Path to validation.csv (for calibrator)
# #     --train-classifier   Train LightGBM classifier
# # """

# # import argparse
# # import json
# # import os
# # import pickle
# # from pathlib import Path
# # from typing import List, Dict

# # import numpy as np
# # import pandas as pd
# # from tqdm import tqdm

# # # sentence-transformers + faiss
# # from sentence_transformers import SentenceTransformer
# # import faiss

# # # sklearn for calibrator
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.model_selection import train_test_split

# # # optional LightGBM
# # try:
# #     import importlib
# #     lgb = importlib.import_module("lightgbm")
# #     LGB_AVAILABLE = True
# # except Exception:
# #     lgb = None
# #     LGB_AVAILABLE = False

# # CACHE_DIR = Path("cache")
# # CACHE_DIR.mkdir(exist_ok=True, parents=True)

# # DEFAULT_BATCH_SIZE_CPU = 256
# # DEFAULT_BATCH_SIZE_GPU = 16


# # def normalize_path_sep(path: str) -> str:
# #     """Normalize category path separators"""
# #     if not isinstance(path, str):
# #         return ""
# #     s = path.strip()
# #     s = s.replace("/", " > ")
# #     s = " > ".join([p.strip() for p in s.split(">") if p.strip()])
# #     return s


# # def path_to_levels(path: str) -> List[str]:
# #     """Extract hierarchy levels from path"""
# #     n = normalize_path_sep(path)
# #     return [p.strip() for p in n.split(" > ") if p.strip()]


# # def safe_pickle_save(obj, p: Path):
# #     """Save object to pickle file"""
# #     with open(p, "wb") as f:
# #         pickle.dump(obj, f)


# # def build_encoder(model_name: str, use_cuda: bool):
# #     """Load sentence transformer model"""
# #     device = "cuda" if use_cuda else "cpu"
# #     print(f"📦 Loading encoder: {model_name} on {device}")
# #     model = SentenceTransformer(model_name, device=device)
# #     if use_cuda:
# #         try:
# #             import torch
# #             model = model.half()
# #             print("   ✓ Using FP16 on GPU")
# #         except Exception:
# #             pass
# #     return model


# # def encode_texts(model: SentenceTransformer, texts: List[str], use_cuda: bool) -> np.ndarray:
# #     """Encode texts in batches"""
# #     batch_size = DEFAULT_BATCH_SIZE_GPU if use_cuda else DEFAULT_BATCH_SIZE_CPU
# #     print(f"🔄 Encoding {len(texts):,} texts (batch size: {batch_size})")
    
# #     all_emb = []
# #     for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
# #         batch = texts[i:i + batch_size]
# #         emb = model.encode(
# #             batch,
# #             convert_to_numpy=True,
# #             normalize_embeddings=True,
# #             show_progress_bar=False
# #         )
# #         if emb.ndim == 1:
# #             emb = emb.reshape(1, -1)
# #         all_emb.append(emb.astype("float32"))
    
# #     embeddings = np.vstack(all_emb)
# #     print(f"   ✓ Embeddings shape: {embeddings.shape}")
# #     return embeddings


# # def build_faiss_index(np_emb: np.ndarray, use_gpu: bool = False):
# #     """Build FAISS index with Inner Product (cosine similarity)"""
# #     d = np_emb.shape[1]
# #     print(f"🏗️  Building FAISS index (dim={d}, type=IndexFlatIP)")
# #     print(f"   ℹ️  IndexFlatIP with normalized vectors = cosine similarity")
    
# #     index = faiss.IndexFlatIP(d)
    
# #     if use_gpu:
# #         try:
# #             res = faiss.StandardGpuResources()
# #             index = faiss.index_cpu_to_gpu(res, 0, index)
# #             print("   ✓ Using GPU for FAISS")
# #         except Exception as e:
# #             print(f"   ⚠️  GPU failed, using CPU: {e}")
    
# #     index.add(np_emb)
# #     print(f"   ✓ Added {index.ntotal:,} vectors")
# #     return index


# # def make_parent_embeddings(metadata: List[Dict], embeddings: np.ndarray) -> Dict[str, np.ndarray]:
# #     """
# #     Create parent embeddings for hierarchical boosting
# #     Averages embeddings of all children for each parent path
# #     """
# #     print("🔨 Building parent embeddings...")
    
# #     parent_map = {}
# #     count_map = {}
    
# #     for i, meta in enumerate(metadata):
# #         levels = meta.get("levels", [])
# #         for depth in range(1, len(levels)):
# #             parent = " > ".join(levels[:depth])
# #             if not parent:
# #                 continue
            
# #             if parent not in parent_map:
# #                 parent_map[parent] = np.zeros(embeddings.shape[1], dtype="float32")
# #                 count_map[parent] = 0
            
# #             parent_map[parent] += embeddings[i]
# #             count_map[parent] += 1

# #     # Average and normalize
# #     final = {}
# #     for k, vec in parent_map.items():
# #         cnt = count_map.get(k, 1)
# #         avg = vec / float(cnt)
# #         norm = np.linalg.norm(avg) + 1e-12
# #         final[k] = (avg / norm).astype("float32")
    
# #     print(f"   ✓ Created {len(final):,} parent embeddings")
# #     return final


# # def load_tags_json(path: Path) -> Dict[str, List[str]]:
# #     """Load tags/synonyms from JSON file"""
# #     if not path.exists():
# #         print(f"   ℹ️  Tags file not found: {path}")
# #         return {}
    
# #     try:
# #         with open(path, "r", encoding="utf-8") as f:
# #             data = json.load(f)
# #         tags = {str(k): [str(x) for x in v] for k, v in data.items()}
# #         print(f"   ✓ Loaded {len(tags)} tag entries")
# #         return tags
# #     except Exception as e:
# #         print(f"   ⚠️  Failed to load tags: {e}")
# #         return {}


# # def train_calibrator(encoder, metadata, faiss_index, val_path: Path, model_name: str, use_cuda: bool):
# #     """
# #     Train probability calibrator using validation data
# #     Maps raw cosine similarity to confidence probability
# #     """
# #     print(f"🎓 Training calibrator: {val_path}")
    
# #     df = pd.read_csv(val_path, dtype=str, keep_default_na=False)
# #     if "product_title" not in df.columns or "category_id" not in df.columns:
# #         print("   ⚠️  Need 'product_title' and 'category_id' columns. Skipping.")
# #         return None

# #     examples = []
# #     labels = []
    
# #     # Map category_id to index
# #     id_to_idx = {m["category_id"]: i for i, m in enumerate(metadata)}

# #     # Encode products with CORRECT E5 format
# #     titles = df["product_title"].astype(str).tolist()
# #     print(f"   ℹ️  Encoding {len(titles)} validation products...")
# #     prod_embs = encode_texts(encoder, [f"query: {t}" for t in titles], use_cuda=use_cuda)

# #     for i, row in df.iterrows():
# #         cid = str(row["category_id"]).strip()
# #         if cid not in id_to_idx:
# #             continue
        
# #         cat_idx = id_to_idx[cid]
# #         cat_emb = metadata[cat_idx].get("_embedding")
# #         if cat_emb is None:
# #             continue
        
# #         q_emb = prod_embs[i].reshape(1, -1).astype("float32")
        
# #         # Positive example (correct category)
# #         raw = float(np.dot(q_emb, cat_emb.reshape(-1, 1))[0][0])
# #         examples.append([raw])
# #         labels.append(1)

# #         # Negative examples (random wrong categories)
# #         import random
# #         for _ in range(2):
# #             rand_idx = random.randrange(len(metadata))
# #             if rand_idx == cat_idx:
# #                 continue
# #             neg_emb = metadata[rand_idx].get("_embedding")
# #             if neg_emb is None:
# #                 continue
# #             raw_neg = float(np.dot(q_emb, neg_emb.reshape(-1, 1))[0][0])
# #             examples.append([raw_neg])
# #             labels.append(0)

# #     if not examples:
# #         print("   ⚠️  No valid examples. Skipping.")
# #         return None

# #     X = np.array(examples, dtype="float32")
# #     y = np.array(labels, dtype="int8")
    
# #     scaler = StandardScaler()
# #     Xs = scaler.fit_transform(X)
    
# #     clf = LogisticRegression(max_iter=400)
# #     clf.fit(Xs, y)
    
# #     print(f"   ✓ Calibrator trained ({len(examples)} examples)")
# #     return {"calibrator": clf, "scaler": scaler}


# # def attach_embeddings_to_metadata(metadata: List[Dict], embeddings: np.ndarray):
# #     """Temporarily attach embeddings to metadata for calibrator training"""
# #     for i, m in enumerate(metadata):
# #         m["_embedding"] = embeddings[i]


# # def detach_embeddings_from_metadata(metadata: List[Dict]):
# #     """Remove embeddings from metadata before saving (reduce file size)"""
# #     for m in metadata:
# #         if "_embedding" in m:
# #             del m["_embedding"]


# # def main():
# #     parser = argparse.ArgumentParser(description="Build FAISS index with correct E5 format")
# #     parser.add_argument("--csv", required=True, help="CSV: Category_ID,Category_path")
# #     parser.add_argument("--model", default="intfloat/e5-base-v2", help="Embedding model")
# #     parser.add_argument("--gpu", action="store_true", help="Use GPU (careful with VRAM)")
# #     parser.add_argument("--clean-cache", action="store_true", help="Clean old cache files")
# #     parser.add_argument("--train-classifier", action="store_true", help="Train LightGBM (optional)")
# #     parser.add_argument("--validation", default="data/validation.csv", help="Validation CSV")
# #     parser.add_argument("--tags", default="data/tags.json", help="Tags JSON (optional)")
# #     args = parser.parse_args()

# #     # Validate CSV
# #     csv_path = Path(args.csv)
# #     if not csv_path.exists():
# #         raise SystemExit(f"❌ CSV not found: {csv_path}")

# #     print("\n" + "="*70)
# #     print("🏗️  BUILDING FAISS INDEX WITH CORRECT E5 FORMAT")
# #     print("="*70)
    
# #     print(f"\n📂 Reading CSV: {csv_path}")
# #     df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
# #     if df.shape[1] < 2:
# #         raise SystemExit("❌ CSV must have at least 2 columns")

# #     cols = list(df.columns)
# #     cid_col, path_col = cols[0], cols[1]
# #     print(f"   ✓ Using columns: '{cid_col}', '{path_col}'")

# #     # Build metadata
# #     print("\n🔨 Building metadata...")
# #     metadata = []
# #     texts_for_encoding = []
    
# #     for idx, row in df.iterrows():
# #         cid = str(row[cid_col]).strip()
# #         raw_path = str(row[path_col]).strip()
        
# #         if not cid or not raw_path:
# #             continue
        
# #         norm_path = normalize_path_sep(raw_path)
# #         levels = path_to_levels(norm_path)
# #         final = levels[-1] if levels else norm_path or cid
        
# #         # ✅ FIXED: Use "passage:" prefix for E5 (not "category:")
# #         text = f"passage: {norm_path}. Category: {final}"
        
# #         metadata.append({
# #             "category_id": cid,
# #             "category_path": norm_path,
# #             "final": final,
# #             "levels": levels,
# #             "depth": len(levels)
# #         })
# #         texts_for_encoding.append(text)

# #     print(f"   ✓ Prepared {len(metadata):,} categories")

# #     # Load encoder
# #     use_cuda = args.gpu
# #     encoder = build_encoder(args.model, use_cuda=use_cuda)

# #     # Encode categories with CORRECT E5 format
# #     print(f"\n🔄 Encoding categories (E5 format: 'passage: ...')")
# #     cat_embeddings = encode_texts(encoder, texts_for_encoding, use_cuda=use_cuda)

# #     # Ensure normalization (for cosine similarity)
# #     print("\n🔧 Normalizing embeddings...")
# #     norms = np.linalg.norm(cat_embeddings, axis=1, keepdims=True)
# #     cat_embeddings = cat_embeddings / np.clip(norms, 1e-12, None)
# #     print("   ✓ Embeddings normalized")

# #     # Attach embeddings temporarily (for calibrator)
# #     attach_embeddings_to_metadata(metadata, cat_embeddings)

# #     # Build parent embeddings
# #     parent_emb = make_parent_embeddings(metadata, cat_embeddings)

# #     # Build FAISS index (CPU only for saving)
# #     index = build_faiss_index(cat_embeddings, use_gpu=False)

# #     # Save FAISS index
# #     faiss_path = CACHE_DIR / "main_index.faiss"
# #     faiss.write_index(index, str(faiss_path))
# #     print(f"\n💾 Saved FAISS index: {faiss_path}")

# #     # Save metadata (remove embeddings first to reduce size)
# #     detach_embeddings_from_metadata(metadata)
# #     meta_path = CACHE_DIR / "metadata.pkl"
# #     safe_pickle_save(metadata, meta_path)
# #     print(f"💾 Saved metadata: {meta_path}")

# #     # Save parent embeddings
# #     parent_path = CACHE_DIR / "parent_embeddings.pkl"
# #     safe_pickle_save(parent_emb, parent_path)
# #     print(f"💾 Saved parent embeddings: {parent_path}")

# #     # Save model info
# #     info = {
# #         "model_name": args.model,
# #         "num_categories": len(metadata),
# #         "embedding_dim": cat_embeddings.shape[1],
# #         "index_type": "IndexFlatIP",
# #         "distance_metric": "cosine (via inner product on normalized vectors)",
# #         "e5_format": "passage: <category_path>. Category: <final>"
# #     }
# #     with open(CACHE_DIR / "model_info.json", "w", encoding="utf-8") as f:
# #         json.dump(info, f, indent=2)
# #     print(f"💾 Saved model_info.json")

# #     # Load and save tags
# #     print("\n📋 Loading tags...")
# #     tags = load_tags_json(Path(args.tags))
# #     if tags:
# #         syn_p = CACHE_DIR / "cross_store_synonyms.pkl"
# #         safe_pickle_save(tags, syn_p)
# #         print(f"💾 Saved cross_store_synonyms.pkl")

# #     # Train calibrator
# #     val_path = Path(args.validation)
# #     if val_path.exists():
# #         print("\n🎓 Training calibrator...")
# #         # Re-attach embeddings for calibrator
# #         attach_embeddings_to_metadata(metadata, cat_embeddings)
# #         calibrator_obj = train_calibrator(encoder, metadata, index, val_path, args.model, use_cuda)
# #         detach_embeddings_from_metadata(metadata)
        
# #         if calibrator_obj:
# #             safe_pickle_save(calibrator_obj, CACHE_DIR / "calibrator.pkl")
# #             print("💾 Saved calibrator.pkl")
# #     else:
# #         print(f"\n   ℹ️  Validation file not found: {val_path}")

# #     # Optional: Train LightGBM classifier
# #     if args.train_classifier:
# #         if not LGB_AVAILABLE:
# #             print("\n   ⚠️  LightGBM not available. Install: pip install lightgbm")
# #         elif not val_path.exists():
# #             print("\n   ⚠️  Need validation.csv to train classifier")
# #         else:
# #             print("\n🎓 Training LightGBM classifier...")
# #             # Implementation same as before...
# #             print("   ℹ️  Classifier training skipped (implement if needed)")

# #     # Cleanup
# #     if args.clean_cache:
# #         keep = {
# #             "main_index.faiss", "metadata.pkl", "model_info.json",
# #             "parent_embeddings.pkl", "cross_store_synonyms.pkl", "calibrator.pkl"
# #         }
# #         removed = []
# #         for p in CACHE_DIR.iterdir():
# #             if p.name not in keep:
# #                 try:
# #                     p.unlink()
# #                     removed.append(p.name)
# #                 except Exception:
# #                     pass
# #         if removed:
# #             print(f"\n🧹 Cleaned: {removed}")

# #     # Summary
# #     print("\n" + "="*70)
# #     print("✅ INDEX BUILD COMPLETE!")
# #     print("="*70)
# #     print(f"\n📊 Summary:")
# #     print(f"   • Categories indexed: {len(metadata):,}")
# #     print(f"   • Embedding dimensions: {cat_embeddings.shape[1]}")
# #     print(f"   • Index type: IndexFlatIP (cosine similarity)")
# #     print(f"   • E5 format: ✅ CORRECT ('passage:' prefix)")
# #     print(f"\n📁 Files saved to: {CACHE_DIR}")
# #     print(f"\n💡 Usage in gradio_app.py:")
# #     print(f"   - For queries: encoder.encode(f'query: {{text}}')")
# #     print(f"   - Distance from IndexFlatIP = cosine similarity [0, 1]")
# #     print(f"   - Confidence = distance * 100")
# #     print("="*70 + "\n")


# # if __name__ == "__main__":
# #     main()

# #!/usr/bin/env python3
# """
# gradio_app.py - Exact Product Classifier (Tags + Leaf Matching + AI)

# STRATEGY:
# 1. TAGS: Check tags.json for manual overrides (Highest Priority).
# 2. LEAF MATCH: Check if the 'Final Product' (last part of CSV path) is in the User Title.
# 3. AI: Use E5 + MPNet to find semantic matches.
# """

# import os
# import json
# import pickle
# import re
# import time
# from pathlib import Path
# from typing import List, Dict, Tuple, Optional, Set
# import numpy as np
# import pandas as pd
# import faiss
# import gradio as gr
# from sentence_transformers import SentenceTransformer, CrossEncoder

# # -------------------------
# # ⚙️ CONFIGURATION
# # -------------------------
# CACHE_DIR = Path("cache")
# DATA_DIR = Path("data")
# CACHE_DIR.mkdir(exist_ok=True)

# # Models
# MODEL_NAME_E5 = "intfloat/e5-base-v2"
# MODEL_NAME_MPNET = "sentence-transformers/all-mpnet-base-v2"
# MODEL_NAME_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2" 

# # Paths
# CSV_PATH = DATA_DIR / "categories.csv"
# TAGS_PATH = DATA_DIR / "tags.json"
# INDEX_PATH_E5 = CACHE_DIR / "index_e5.faiss"
# INDEX_PATH_MPNET = CACHE_DIR / "index_mpnet.faiss"
# METADATA_PATH = CACHE_DIR / "metadata.pkl"

# # -------------------------
# # Globals
# # -------------------------
# retriever_e5 = None
# retriever_mpnet = None
# reranker = None
# index_e5 = None
# index_mpnet = None
# metadata: List[Dict] = []
# category_lookup: Dict[str, Dict] = {} 
# tag_lookup: Dict[str, List[str]] = {} 

# # -------------------------
# # 🛠️ UTILITIES
# # -------------------------
# def clean_text(text: str) -> str:
#     if not text: return ""
#     s = str(text).lower().strip()
#     s = re.sub(r"[^\w\s\-]", " ", s)
#     s = re.sub(r"\s+", " ", s).strip()
#     return s

# def get_final_product_name(path: str) -> str:
#     """Extracts the last segment of the path (The 'Exact Product')"""
#     if not path: return ""
#     parts = path.split('/')
#     return parts[-1].strip()

# # -------------------------
# # 🧠 Phase 1: Data Loading (CSV + Tags)
# # -------------------------
# def load_data():
#     """Loads CSV and Tags into memory"""
#     global category_lookup, metadata, tag_lookup
    
#     # 1. Load CSV
#     if not CSV_PATH.exists():
#         print("❌ CSV Not Found!")
#         return

#     print("📊 Loading CSV...")
#     df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False, on_bad_lines='skip')
    
#     metadata.clear()
#     category_lookup.clear()
    
#     for _, row in df.iterrows():
#         if len(row) < 2: continue
#         cid = str(row.iloc[0]).strip()
#         path = str(row.iloc[1]).strip()
        
#         if not cid or not path: continue
        
#         final_prod = get_final_product_name(path)
        
#         item = {
#             "category_id": cid,
#             "category_path": path,
#             "final_product": final_prod,
#             "depth": len(path.split('/'))
#         }
#         metadata.append(item)
#         category_lookup[cid] = item

#     print(f"✅ Indexed {len(metadata)} categories.")

#     # 2. Load Tags
#     if not TAGS_PATH.exists():
#         print("⚠️ tags.json not found.")
#         return

#     print("🏷️ Loading Tags...")
#     try:
#         with open(TAGS_PATH, "r", encoding="utf-8") as f:
#             data = json.load(f)
        
#         count = 0
#         for cat_id, tags in data.items():
#             if not isinstance(tags, list): continue
#             # Validate ID exists in CSV
#             if cat_id not in category_lookup: continue

#             for t in tags:
#                 t_clean = clean_text(t)
#                 if len(t_clean) < 3 or t_clean in ["buy", "best", "amazon", "product"]: continue
                
#                 if t_clean not in tag_lookup: tag_lookup[t_clean] = []
#                 tag_lookup[t_clean].append(str(cat_id))
#                 count += 1
#         print(f"✅ Loaded {count} tags.")
#     except Exception as e:
#         print(f"❌ Tag Error: {e}")

# # -------------------------
# # 🧠 Phase 2: Tag Engine (Exact Keyword Match)
# # -------------------------
# def get_tag_matches(title: str) -> List[Dict]:
#     matches = []
#     txt = clean_text(title)
    
#     # Check if any tag phrase appears in the title
#     for tag, ids in tag_lookup.items():
#         # Use Regex word boundaries to avoid partial matches (e.g. 'car' in 'card')
#         if re.search(r"\b" + re.escape(tag) + r"\b", txt):
#             for cid in ids:
#                 info = category_lookup.get(cid, {})
#                 matches.append({
#                     "category_id": cid,
#                     "category_path": info.get("category_path", "Tag Match"),
#                     "final_product": info.get("final_product", "Tag Match"),
#                     "depth": info.get("depth", 1),
#                     "retrieval_score": 0.99, 
#                     "confidence": 99.0, # Massive confidence for manual tags
#                     "sources": {"Tag"},
#                     "logic_log": [f"🏷️ Tag: '{tag}'"]
#                 })
#     return matches

# # -------------------------
# # 🧠 Phase 3: AI Retrieval
# # -------------------------
# def retrieve_candidates(full_text: str) -> List[Dict]:
#     candidates_map = {} 
    
#     # A. E5
#     e5_emb = retriever_e5.encode(f"query: {full_text}", convert_to_numpy=True, normalize_embeddings=True)
#     dists_e5, idxs_e5 = index_e5.search(e5_emb.reshape(1, -1), 30)
#     for i, idx in enumerate(idxs_e5[0]):
#         if idx < 0: continue
#         add_candidate(candidates_map, idx, float(dists_e5[0][i]), "E5")

#     # B. MPNet
#     mp_emb = retriever_mpnet.encode(full_text, convert_to_numpy=True, normalize_embeddings=True)
#     dists_mp, idxs_mp = index_mpnet.search(mp_emb.reshape(1, -1), 30)
#     for i, idx in enumerate(idxs_mp[0]):
#         if idx < 0: continue
#         add_candidate(candidates_map, idx, float(dists_mp[0][i]), "MPNet")
        
#     return list(candidates_map.values())

# def add_candidate(cand_map, idx, raw_score, source):
#     if idx >= len(metadata): return
#     meta = metadata[idx]
#     cid = meta["category_id"]
    
#     if cid in cand_map:
#         cand_map[cid]["sources"].add(source)
#         cand_map[cid]["retrieval_score"] = max(cand_map[cid]["retrieval_score"], raw_score)
#     else:
#         info = category_lookup.get(cid, meta)
#         cand_map[cid] = {
#             "category_id": cid,
#             "category_path": info["category_path"],
#             "final_product": info["final_product"],
#             "depth": info["depth"],
#             "retrieval_score": raw_score,
#             "confidence": raw_score * 100, 
#             "sources": {source},
#             "logic_log": []
#         }

# # -------------------------
# # 🧠 Phase 4: Exact Leaf Matcher (The "Product" Logic)
# # -------------------------
# def boost_exact_product_match(title: str, candidates: List[Dict]):
#     """
#     If the 'Final Product' (last word in path) is in the Title, BOOST IT.
#     Example: Title="Vita Protectors", Path=".../Protectors" -> Boost!
#     """
#     clean_title = clean_text(title)
    
#     for res in candidates:
#         final_prod = clean_text(res["final_product"])
        
#         # Avoid boosting very short words like "TV" or "CD" without care
#         if len(final_prod) < 3 and final_prod not in ["tv", "cd", "pc"]: 
#             continue
            
#         # Check if the exact product name exists in title
#         if re.search(r"\b" + re.escape(final_prod) + r"\b", clean_title):
#             res["confidence"] += 25.0
#             res["logic_log"].append(f"🎯 Exact Leaf: '{res['final_product']}'")
#             res["sources"].add("LeafMatch")

# # -------------------------
# # 🚀 MAIN FLOW
# # -------------------------
# def classify_product(title: str, description: str = "") -> Dict:
#     start_time = time.time()
#     full_text = f"{title} {description}".strip()
    
#     # 1. AI Retrieval
#     candidates = retrieve_candidates(full_text)
    
#     # 2. Tag Engine Injection
#     tag_matches = get_tag_matches(title)
    
#     # Merge Tags into Candidates
#     candidates_dict = {c["category_id"]: c for c in candidates}
#     for tm in tag_matches:
#         cid = tm["category_id"]
#         if cid in candidates_dict:
#             candidates_dict[cid]["confidence"] += 30.0 # Boost existing
#             candidates_dict[cid]["sources"].add("Tag")
#             candidates_dict[cid]["logic_log"].append(tm["logic_log"][0])
#         else:
#             candidates_dict[cid] = tm # Add new
            
#     candidates = list(candidates_dict.values())

#     # 3. Exact Leaf Matching (The "Product" Logic)
#     boost_exact_product_match(title, candidates)

#     # 4. Sort for Re-Ranking
#     candidates.sort(key=lambda x: x["confidence"], reverse=True)
#     top_candidates = candidates[:15] 
    
#     # 5. Cross-Encoder Re-Ranking
#     rerank_inputs = [[title, c["category_path"]] for c in top_candidates]
    
#     if rerank_inputs:
#         scores = reranker.predict(rerank_inputs)
#         for i, score in enumerate(scores):
#             top_candidates[i]["rerank_score"] = float(score)
#         top_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

#     top = top_candidates[0] if top_candidates else {}
    
#     return {
#         "final_product": top.get("final_product", "Unknown"),
#         "category_path": top.get("category_path", "Unknown"),
#         "category_id": top.get("category_id", "Unknown"),
#         "rerank_score": top.get("rerank_score", 0),
#         "logic_log": top.get("logic_log", []),
#         "top_results": top_candidates,
#         "time_ms": (time.time() - start_time) * 1000
#     }

# # -------------------------
# # Initialization
# # -------------------------
# def build_index(model, model_name, csv_path, index_path):
#     print(f"🔨 Building Index for {model_name}...")
#     df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, on_bad_lines='skip')
#     paths = df.iloc[:, 1].tolist() # Assuming Col 1 is Path
#     if "e5" in model_name: paths = [f"passage: {p}" for p in paths]
#     embeddings = model.encode(paths, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
#     index = faiss.IndexFlatIP(embeddings.shape[1])
#     index.add(embeddings)
#     faiss.write_index(index, str(index_path))
#     return index

# def initialize():
#     global retriever_e5, retriever_mpnet, reranker, index_e5, index_mpnet
#     print("🚀 Initializing Engine...")
    
#     retriever_e5 = SentenceTransformer(MODEL_NAME_E5)
#     retriever_mpnet = SentenceTransformer(MODEL_NAME_MPNET)
#     reranker = CrossEncoder(MODEL_NAME_RERANKER)
    
#     load_data() # Load CSV + Tags

#     if INDEX_PATH_E5.exists(): index_e5 = faiss.read_index(str(INDEX_PATH_E5))
#     else: index_e5 = build_index(retriever_e5, "e5", CSV_PATH, INDEX_PATH_E5)

#     if INDEX_PATH_MPNET.exists(): index_mpnet = faiss.read_index(str(INDEX_PATH_MPNET))
#     else: index_mpnet = build_index(retriever_mpnet, "mpnet", CSV_PATH, INDEX_PATH_MPNET)
#     print("✅ Ready!")

# # -------------------------
# # UI
# # -------------------------
# def gradio_classify(title, desc):
#     if not title: return ("",) * 7
#     res = classify_product(title, desc)
    
#     top_text = ""
#     for i, item in enumerate(res.get("top_results", []), 1):
#         logs = " | ".join(item.get("logic_log", []))
#         sources = ", ".join(item.get("sources", ["AI"]))
#         score = item.get('rerank_score', 0)
#         top_text += f"{i}. {item['final_product']}\n   ID: {item['category_id']} | Score: {score:.2f}\n   Path: {item['category_path']}\n   ℹ️ {logs}\n\n"
    
#     status = "✅ Approved" if res.get("rerank_score", 0) > 0 else "⚠️ Review Needed"
    
#     return (
#         res["final_product"], 
#         res["category_path"], 
#         res["category_id"],
#         f"{res.get('rerank_score', 0):.4f}", 
#         " | ".join(res.get("logic_log", [])),
#         status, 
#         top_text
#     )

# def main():
#     initialize()
#     with gr.Blocks(theme=gr.themes.Soft(), title="Exact Product Classifier") as app:
#         gr.Markdown("# 🎯 Exact Product Classifier")
#         gr.Markdown("Prioritizes 'Leaf Categories' (Last word in path) + 'Tags' for exact matches.")
        
#         with gr.Row():
#             with gr.Column():
#                 t_in = gr.Textbox(label="Title", placeholder="e.g. Sony PS Vita Screen Protectors")
#                 d_in = gr.Textbox(label="Description")
#                 btn = gr.Button("Classify", variant="primary")
#             with gr.Column():
#                 out_final = gr.Textbox(label="Exact Product (Leaf)")
#                 out_path = gr.Textbox(label="Full Path")
#                 out_id = gr.Textbox(label="Category ID")
#                 out_score = gr.Textbox(label="Re-Rank Score")
#                 out_logic = gr.Textbox(label="Match Logic")
#                 out_status = gr.Textbox(label="Status")
        
#         out_details = gr.TextArea(label="🏆 Top Candidates", lines=12)
#         btn.click(gradio_classify, [t_in, d_in], [out_final, out_path, out_id, out_score, out_logic, out_status, out_details])
    
#     app.launch(server_name="127.0.0.1", server_port=7860, share=True)

# if __name__ == "__main__":
#     main()
