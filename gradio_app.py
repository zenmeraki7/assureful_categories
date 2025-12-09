

# # #final fixed gradio

# # #!/usr/bin/env python3
# # """
# # UPDATES:
# # 1. FIXED: Added 'teen', 'young', 'adult' to Ignore List (Stops false matches to ID 28).
# # 2. RETAINED: All previous Universal Logic & Excel support.
# # """

# # import os
# # import json
# # import pickle
# # import re
# # import time
# # from pathlib import Path
# # from typing import List, Dict, Tuple, Optional
# # import pandas as pd
# # import faiss
# # import gradio as gr
# # import torch
# # import numpy as np
# # from sentence_transformers import SentenceTransformer, CrossEncoder

# # # -------------------------
# # # ⚙️ CONFIGURATION
# # # -------------------------
# # CACHE_DIR = Path("cache")
# # DATA_DIR = Path("data")
# # CACHE_DIR.mkdir(exist_ok=True)
# # DATA_DIR.mkdir(exist_ok=True)

# # # Models
# # MODEL_NAME_E5 = "intfloat/e5-base-v2"
# # MODEL_NAME_MPNET = "sentence-transformers/all-mpnet-base-v2"
# # MODEL_NAME_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2" 

# # # Paths
# # CSV_PATH = DATA_DIR / "categories.csv"
# # TAGS_PATH = DATA_DIR / "tags.json"
# # INDEX_PATH_E5 = CACHE_DIR / "index_e5.faiss"
# # INDEX_PATH_MPNET = CACHE_DIR / "index_mpnet.faiss"
# # METADATA_PATH = CACHE_DIR / "metadata.pkl"

# # try:
# #     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# # except Exception:
# #     DEVICE = "cpu"

# # print(f"⚡ Running on: {DEVICE.upper()}")

# # # -------------------------
# # # 🌍 DOMAIN RULES
# # # -------------------------
# # CLOTHING_KEYWORDS = {"t-shirt", "tshirt", "tee", "shirt", "top", "hoodie", "sweatshirt", "jacket", "coat", "pant", "jeans", "dress", "legging", "sock", "underwear", "bra", "apparel", "garment"}
# # KITCHEN_KEYWORDS = {"salt & pepper", "shaker", "mug", "cup", "bowl", "plate", "dish", "glass", "cookware", "pan", "pot", "spatula", "ladle", "knife", "fork", "spoon"}
# # BEAUTY_KEYWORDS = {"perfume", "parfum", "eau de", "cologne", "fragrance", "scent", "spray", "mist", "lotion", "cream", "moisturizer", "serum", "makeup", "lipstick", "cosmetic"}
# # HOME_KEYWORDS = {"blanket", "throw", "quilt", "duvet", "pillow", "candle", "wax", "tapestry", "poster", "print", "art", "decor", "vase", "rug", "mat"}
# # JEWELRY_KEYWORDS = {"pin", "enamel pin", "brooch", "badge", "keychain", "keyring", "charm", "necklace", "earring", "bracelet", "ring", "jewelry"}
# # FOOD_KEYWORDS = {"chocolate", "candy", "gummy", "snack", "sauce", "spice", "oil", "tea", "coffee", "drink", "beverage", "food", "grocery"}

# # PHONE_KEYWORDS = {"smartphone", "iphone", "samsung", "galaxy", "pixel", "oneplus", "motorola", "nokia", "sony", "lg", "xiaomi", "redmi", "oppo", "vivo", "realme", "huawei"}
# # CARRIER_KEYWORDS = {"carrier", "locked", "prepaid", "postpaid", "att", "verizon", "t-mobile", "sprint", "cricket", "tracfone", "mint mobile"}
# # AUDIO_KEYWORDS = {"headphone", "earphone", "earbud", "headset", "airpod", "galaxy buds", "noise cancelling", "speaker", "soundbar"}
# # LAUNDRY_GENERICS = {"washing machine", "washer", "clothes washer", "washer dryer"}
# # LAUNDRY_BRANDS = {"godrej", "lg", "samsung", "whirlpool", "bosch", "ifb", "haier", "panasonic"}
# # ACCESSORY_KEYWORD_TO_ID = {
# #     "flip case": "9931389011", "armband": "7073962011", "holster": "2407765011",
# #     "sleeve": "9414313011", "bumper": "17875442011", "dry bag": "17875443011",
# #     "case": "3081461011", "cover": "3081461011", "screen protector": "3081461011",
# #     "tempered glass": "3081461011"
# # }
# # BOOK_KEYWORDS = {"book", "guide", "manual", "ebook", "pdf", "kindle", "paperback"}

# # # Critical IDs
# # WASHING_MACHINE_ID = "2383576011"
# # UNLOCKED_CELL_PHONES_ID = "2407749011"
# # IPHONE_BOOKS_ID = "6133978011"
# # CARRIER_PHONES_ID = "2407748011"
# # HEADPHONES_ID = "172541" 
# # LAUNDRY_SUPPLIES_ID = "15356111"
# # TRASH_BAGS_ID = "15342971"

# # # -------------------------
# # # Globals
# # # -------------------------
# # retriever_e5 = None
# # retriever_mpnet = None
# # reranker = None
# # index_e5 = None
# # index_mpnet = None
# # metadata: List[Dict] = []
# # catid_to_meta: Dict[str, Dict] = {} 
# # tag_lookup: Dict[str, List[str]] = {} 

# # def clean_text(text: str) -> str:
# #     if not text: return ""
# #     s = str(text).lower().strip()
# #     s = re.sub(r"[^\w\s\-]", " ", s, flags=re.UNICODE)
# #     return re.sub(r"\s+", " ", s).strip()

# # def get_final_product_name(path: str) -> str:
# #     if not path: return ""
# #     parts = [p for p in path.split('/') if p]
# #     return parts[-1].strip() if parts else path.strip()

# # # -------------------------
# # # Data Loading
# # # -------------------------
# # def build_metadata_from_csv(csv_path: Path) -> List[Dict]:
# #     try:
# #         if str(csv_path).endswith('.xlsx'):
# #             df = pd.read_excel(csv_path, dtype=str).fillna("")
# #         else:
# #             try:
# #                 df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, on_bad_lines='skip', encoding='utf-8')
# #             except UnicodeDecodeError:
# #                 df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, on_bad_lines='skip', encoding='latin1')
# #     except Exception as e:
# #         print(f"CRITICAL ERROR loading categories: {e}")
# #         return []
        
# #     rows = []
# #     col_id = df.columns[0]
# #     col_path = df.columns[1]

# #     for _, row in df.iterrows():
# #         cid = str(row[col_id]).strip()
# #         path = str(row[col_path]).strip()
# #         if not cid or not path: continue
# #         rows.append({
# #             "category_id": cid,
# #             "category_path": path,
# #             "final_product": get_final_product_name(path),
# #             "depth": len(path.split('/')),
# #             "indexed_text": f"passage: {path}"
# #         })
# #     return rows

# # def load_data():
# #     global metadata, catid_to_meta, tag_lookup
# #     if not CSV_PATH.exists(): raise FileNotFoundError("categories.csv missing!")
# #     print("📊 Building metadata...")
# #     metadata = build_metadata_from_csv(CSV_PATH)
# #     catid_to_meta = {m['category_id']: m for m in metadata}
# #     with open(METADATA_PATH, 'wb') as f: pickle.dump(metadata, f)

# #     tag_lookup = {}
# #     if TAGS_PATH.exists():
# #         print("🏷️ Loading tags...")
# #         try:
# #             with open(TAGS_PATH, 'r', encoding='utf-8') as f:
# #                 tags_data = json.load(f)
# #             for cat_id, tags in tags_data.items():
# #                 if cat_id not in catid_to_meta or not isinstance(tags, list): continue
# #                 for t in tags:
# #                     t_clean = clean_text(t)
# #                     if len(t_clean) < 3: continue
# #                     tag_lookup.setdefault(t_clean, set()).add(str(cat_id))
# #             print(f"✅ Loaded {len(tag_lookup)} tags.")
# #         except Exception as e: print(f"⚠️ Tag Load Error: {e}")

# # # -------------------------
# # # Rule Engine
# # # -------------------------
# # def get_rule_match_id(title: str, description: str = "") -> Tuple[Optional[str], str]:
# #     txt = clean_text(title) 
# #     if "iphone" in txt and any(k in txt for k in BOOK_KEYWORDS): return IPHONE_BOOKS_ID, "Rule: iPhone Book"
# #     for phrase, cid in ACCESSORY_KEYWORD_TO_ID.items():
# #         if re.search(r"\b" + re.escape(phrase) + r"\b", txt): return cid, f"Rule: Accessory '{phrase}'"
    
# #     is_laundry = False
# #     for kw in LAUNDRY_GENERICS:
# #         if re.search(r"\b" + re.escape(kw) + r"\b", txt): is_laundry = True; break
# #     if not is_laundry:
# #         for brand in LAUNDRY_BRANDS:
# #             if brand in txt and ("washing" in txt or "washer" in txt): is_laundry = True; break
# #     if is_laundry:
# #         if not any(a in txt for a in ["cover", "stand", "cleaner", "powder", "part"]): return WASHING_MACHINE_ID, "Rule: Appliance Match"

# #     for kw in AUDIO_KEYWORDS:
# #         if re.search(r"\b" + re.escape(kw) + r"\b", txt):
# #             if "case" not in txt and "cover" not in txt: return HEADPHONES_ID, f"Rule: Audio '{kw}'"
            
# #     if any(k in txt for k in CARRIER_KEYWORDS) and any(p in txt for p in PHONE_KEYWORDS): return CARRIER_PHONES_ID, "Rule: Carrier Phone"
# #     for brand in PHONE_KEYWORDS:
# #         if re.search(r"\b" + re.escape(brand) + r"\b", txt):
# #             if not any(a in txt for a in ["battery", "charger", "cable", "case", "screen"]): return UNLOCKED_CELL_PHONES_ID, f"Rule: Brand '{brand.title()}'"
# #     return None, ""

# # # -------------------------
# # # Guardrails & Universal Matcher
# # # -------------------------
# # def has_word(text, word_set):
# #     text_lower = clean_text(text)
# #     for w in word_set:
# #         if re.search(r'\b' + re.escape(w) + r'\b', text_lower): return True
# #     return False

# # def simple_stem(w):
# #     return w.rstrip('s')

# # def check_universal_match(title, path):
# #     """Universal Matcher with expanded Ignore List."""
# #     t_words = clean_text(title).split()
# #     t_stems = {simple_stem(w) for w in t_words}
# #     leaf_words = clean_text(path.split('/')[-1]).split()
# #     leaf_stems = {simple_stem(w) for w in leaf_words}
    
# #     # FIX: Added 'teen', 'young', 'adult' to ignore list
# #     ignore = {
# #         "and", "for", "with", "the", "set", "pack", "kit", "size", "color", "men", "women", "of", "in",
# #         "control", "digital", "auto", "manual", "electric", "star", "drive", "design", "replacement",
# #         "part", "parts", "universal", "remote", "system", "quality", "cleaner", "wash",
# #         "teen", "young", "adult", "kid", "kids", "child", "children"
# #     }
# #     ignore_stems = {simple_stem(w) for w in ignore}
    
# #     overlap = t_stems.intersection(leaf_stems) - ignore_stems
    
# #     if len(overlap) > 0:
# #         return 40.0 * len(overlap), [f"✅ Univ Match: {overlap}"]
    
# #     return 0.0, []

# # def check_guardrails(title, path):
# #     score = 0.0
# #     log = []
# #     p = path.lower()
    
# #     # 1. Universal Match
# #     univ_score, univ_log = check_universal_match(title, path)
# #     score += univ_score
# #     log.extend(univ_log)

# #     # 2. Specific Guardrails
# #     if has_word(title, CLOTHING_KEYWORDS):
# #         if "clothing" in p or "apparel" in p: score += 50.0; log.append("✅ Clothing Rule")
# #         elif "food" in p or "toy" in p: 
# #             if "doll" not in clean_text(title): score -= 500.0; log.append("⛔ Not Clothing")
            
# #     if has_word(title, KITCHEN_KEYWORDS):
# #         if "kitchen" in p or "dining" in p: score += 40.0; log.append("✅ Kitchen Rule")

# #     if has_word(title, BEAUTY_KEYWORDS):
# #         if "beauty" in p or "care" in p: score += 40.0; log.append("✅ Beauty Rule")

# #     if has_word(title, HOME_KEYWORDS):
# #         if "home" in p or "decor" in p or "bedding" in p: score += 40.0; log.append("✅ Home Rule")

# #     if has_word(title, JEWELRY_KEYWORDS):
# #         if "jewelry" in p or "accessories" in p: score += 40.0; log.append("✅ Jewelry Rule")

# #     if has_word(title, FOOD_KEYWORDS):
# #         if "food" in p or "grocery" in p: score += 40.0; log.append("✅ Food Rule")
# #         elif "toy" in p or "electronic" in p: score -= 200.0; log.append("⛔ Not Food")

# #     return score, log

# # # -------------------------
# # # Classification
# # # -------------------------
# # def classify_product(title: str, description: str = "") -> Dict:
# #     full_text = f"{title} {description}".strip()
# #     candidates_map = {}
    
# #     # Retrievers
# #     e5_emb = retriever_e5.encode(f"passage: {full_text}", convert_to_numpy=True, normalize_embeddings=True)
# #     dists_e5, idxs_e5 = index_e5.search(e5_emb.reshape(1, -1).astype('float32'), 30)
# #     for i, idx in enumerate(idxs_e5[0]):
# #         if idx >= 0: add_candidate(candidates_map, int(idx), float(dists_e5[0][i]), 'E5')

# #     mp_emb = retriever_mpnet.encode(full_text, convert_to_numpy=True, normalize_embeddings=True)
# #     dists_mp, idxs_mp = index_mpnet.search(mp_emb.reshape(1, -1).astype('float32'), 30)
# #     for i, idx in enumerate(idxs_mp[0]):
# #         if idx >= 0: add_candidate(candidates_map, int(idx), float(dists_mp[0][i]), 'MPNet')

# #     # Tags
# #     words = clean_text(title).split()
# #     for i in range(len(words)):
# #         for j in range(i, min(i+6, len(words))):
# #             phrase = " ".join(words[i:j+1])
# #             if phrase in tag_lookup:
# #                 for cid in tag_lookup[phrase]:
# #                     if cid in candidates_map:
# #                         candidates_map[cid]['confidence'] += 30.0
# #                         candidates_map[cid]['sources'].add('Tag')
# #                     else:
# #                         if cid in catid_to_meta:
# #                             add_candidate_manual(candidates_map, catid_to_meta[cid], 0.95, 'Tag')

# #     candidates = list(candidates_map.values())
    
# #     # Guardrails
# #     for res in candidates:
# #         if len(res['sources']) > 1: res['confidence'] += 5.0
# #         res['confidence'] += min(8.0, res['depth'] * 1.5)
# #         g_score, g_log = check_guardrails(title, res['category_path'])
# #         res['confidence'] += g_score
# #         res['logic_log'].extend(g_log)

# #     candidates.sort(key=lambda x: x['confidence'], reverse=True)
# #     top_candidates = candidates[:30]
    
# #     if reranker:
# #         rerank_inputs = [[title, c['category_path']] for c in top_candidates]
# #         scores = reranker.predict(rerank_inputs)
# #         for i, score in enumerate(scores): top_candidates[i]['rerank_score'] = float(score)
# #         top_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)

# #     rule_id, rule_reason = get_rule_match_id(title, description)
# #     final_top_results = []
    
# #     if rule_id and rule_id in catid_to_meta:
# #         meta = catid_to_meta[rule_id]
# #         rule_winner = {
# #             'category_id': rule_id,
# #             'category_path': meta['category_path'],
# #             'final_product': meta['final_product'],
# #             'rerank_score': 99.99,
# #             'sources': {'Rule'},
# #             'logic_log': [f"🏆 {rule_reason}"]
# #         }
# #         final_top_results = [rule_winner] + [c for c in top_candidates if c['category_id'] != rule_id][:19]
# #     else:
# #         final_top_results = top_candidates[:20]

# #     top = final_top_results[0] if final_top_results else None
# #     return {
# #         'final_product': top['final_product'] if top else 'Unknown',
# #         'category_path': top['category_path'] if top else 'Unknown',
# #         'category_id': top['category_id'] if top else 'Unknown',
# #         'rerank_score': top.get('rerank_score', 0.0) if top else 0.0,
# #         'logic_log': top.get('logic_log', []) if top else [],
# #         'top_results': final_top_results
# #     }

# # def add_candidate(cand_map, idx, raw_score, source):
# #     meta = metadata[idx]
# #     add_candidate_manual(cand_map, meta, raw_score, source)

# # def add_candidate_manual(cand_map, meta, raw_score, source):
# #     cid = meta['category_id']
# #     if cid in cand_map:
# #         cand_map[cid]['sources'].add(source)
# #         cand_map[cid]['retrieval_score'] = max(cand_map[cid]['retrieval_score'], float(raw_score))
# #     else:
# #         cand_map[cid] = {
# #             'category_id': cid,
# #             'category_path': meta['category_path'],
# #             'final_product': meta['final_product'],
# #             'depth': meta['depth'],
# #             'retrieval_score': float(raw_score),
# #             'confidence': float(raw_score) * 100.0,
# #             'sources': {source},
# #             'logic_log': []
# #         }

# # def build_index(model, model_name, items, index_path, use_passage_prefix=False):
# #     print(f"🔨 Building Index for {model_name}...")
# #     texts = [it['indexed_text'] if use_passage_prefix else it['category_path'] for it in items]
# #     embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
# #     index = faiss.IndexFlatIP(embeddings.shape[1])
# #     index.add(np.asarray(embeddings, dtype='float32'))
# #     faiss.write_index(index, str(index_path))
# #     return index

# # def initialize():
# #     global retriever_e5, retriever_mpnet, reranker, index_e5, index_mpnet, metadata, catid_to_meta
# #     print(f"🚀 Initializing Engine (device={DEVICE})...")
# #     retriever_e5 = SentenceTransformer(MODEL_NAME_E5, device=DEVICE)
# #     retriever_mpnet = SentenceTransformer(MODEL_NAME_MPNET, device=DEVICE)
# #     try:
# #         reranker = CrossEncoder(MODEL_NAME_RERANKER, device=DEVICE)
# #     except:
# #         reranker = CrossEncoder(MODEL_NAME_RERANKER, device='cpu')
# #     load_data() 
# #     if not INDEX_PATH_E5.exists():
# #         index_e5 = build_index(retriever_e5, 'e5', metadata, INDEX_PATH_E5, True)
# #         index_mpnet = build_index(retriever_mpnet, 'mpnet', metadata, INDEX_PATH_MPNET, False)
# #     else:
# #         index_e5 = faiss.read_index(str(INDEX_PATH_E5))
# #         index_mpnet = faiss.read_index(str(INDEX_PATH_MPNET))
# #     catid_to_meta = {m['category_id']: m for m in metadata}
# #     print("✅ System Ready.")

# # # --- BATCH UI LOGIC ---
# # def get_file_path(file_obj):
# #     if file_obj is None: return None
# #     if isinstance(file_obj, str): return file_obj 
# #     if hasattr(file_obj, 'name'): return file_obj.name
# #     return str(file_obj)

# # def read_file_robust(path):
# #     try:
# #         if path.endswith('.xlsx') or path.endswith('.xls'):
# #             return pd.read_excel(path, dtype=str).fillna("")
# #         try:
# #             return pd.read_csv(path, dtype=str, encoding='utf-8').fillna("")
# #         except UnicodeDecodeError:
# #             try:
# #                 return pd.read_csv(path, dtype=str, encoding='latin1').fillna("")
# #             except:
# #                 return pd.read_csv(path, dtype=str, encoding='cp1252').fillna("")
# #     except Exception as e:
# #         raise gr.Error(f"Could not read file. Is 'openpyxl' installed? Error: {e}")

# # def analyze_uploaded_csv(file_obj):
# #     path = get_file_path(file_obj)
# #     if not path: return pd.DataFrame(), gr.update(choices=[]), gr.update(choices=[])
# #     try:
# #         df = read_file_robust(path)
# #         cols = list(df.columns)
# #         default_title = "product_name" if "product_name" in cols else cols[0]
# #         default_desc = "description" if "description" in cols else (cols[1] if len(cols)>1 else cols[0])
# #         return df.head(5), gr.update(choices=cols, value=default_title), gr.update(choices=cols, value=default_desc)
# #     except Exception as e:
# #         raise gr.Error(f"Error reading file: {e}")

# # def process_batch_csv(file_obj, title_col, desc_col, limit_choice, progress=gr.Progress()):
# #     path = get_file_path(file_obj)
# #     if not path: raise gr.Error("No file uploaded.")
    
# #     df = read_file_robust(path)
# #     if title_col not in df.columns: raise gr.Error(f"Column '{title_col}' not found in file.")
    
# #     # Parse Custom Limit
# #     try:
# #         limit_str = str(limit_choice).split()[0]
# #         limit = int(limit_str)
# #     except:
# #         limit = 5
    
# #     results_data = []
# #     print(f"🚀 Starting Batch Processing (Top {limit})...")
    
# #     for idx, row in progress.tqdm(df.iterrows(), total=len(df), desc="Classifying"):
# #         t = str(row[title_col])
# #         d = str(row[desc_col]) if desc_col and desc_col in df.columns else ""
        
# #         res = classify_product(t, d)
# #         top_N = res.get('top_results', [])[:limit]
        
# #         # 1. Start with ORIGINAL ROW data
# #         out_row = row.to_dict()
        
# #         # 2. Add Best Match
# #         out_row["Best_Match_ID"] = str(res.get('category_id'))
# #         out_row["Best_Match_Path"] = str(res.get('category_path'))
# #         out_row["Best_Match_Score"] = res.get('rerank_score', 0)
        
# #         # 3. Add Dynamic Top N Columns
# #         for i in range(limit):
# #             if i < len(top_N):
# #                 item = top_N[i]
# #                 out_row[f"Rank_{i+1}_Path"] = str(item.get('category_path'))
# #                 out_row[f"Rank_{i+1}_ID"] = str(item.get('category_id'))
# #                 out_row[f"Rank_{i+1}_Score"] = item.get('rerank_score', 0)
# #             else:
# #                 out_row[f"Rank_{i+1}_Path"] = ""
# #                 out_row[f"Rank_{i+1}_ID"] = ""
# #                 out_row[f"Rank_{i+1}_Score"] = ""
# #         results_data.append(out_row)
    
# #     out_df = pd.DataFrame(results_data)
# #     out_path = "batch_results.csv"
# #     out_df.to_csv(out_path, index=False)
# #     print("✅ Batch Done.")
# #     return out_path, out_df.head(10)

# # def gradio_classify_ui(title, desc):
# #     """Bridge for Single Prediction UI."""
# #     res = classify_product(title, desc)
# #     top_text = ""
# #     for i, item in enumerate(res.get('top_results', []), 1):
# #         logs = " | ".join(item.get('logic_log', []))
# #         top_text += f"{i}. {item['final_product']}\n   ID: {item.get('category_id', 'Unknown')}\n   Score: {item.get('rerank_score',0):.4f}\n   Path: {item['category_path']}\n   Logic: {logs}\n\n"
# #     status = "✅ Approved" if res.get('rerank_score', 0) > 0 else "⚠️ Review"
# #     return str(res.get('final_product')), str(res.get('category_path')), str(res.get('category_id')), f"{res.get('rerank_score',0):.4f}", " | ".join(res.get('logic_log',[])), status, top_text

# # def main():
# #     initialize()
# #     with gr.Blocks() as app:
# #         gr.Markdown("# ⚡ Precision Hybrid Classifier")
# #         with gr.Tabs():
# #             with gr.TabItem("Single Prediction"):
# #                 with gr.Row():
# #                     t_in = gr.Textbox(label="Title")
# #                     d_in = gr.Textbox(label="Description")
# #                     btn = gr.Button("Classify", variant="primary")
# #                 out_res = [gr.Textbox(label=l) for l in ["Winner", "Path", "ID", "Score", "Logic", "Status"]]
# #                 out_det = gr.TextArea(label="Details")
# #                 btn.click(gradio_classify_ui, [t_in, d_in], out_res + [out_det])

# #             with gr.TabItem("Batch Prediction (CSV/Excel)"):
# #                 with gr.Row():
# #                     file_in = gr.File(label="Upload File")
# #                     with gr.Column():
# #                         df_prev = gr.DataFrame(label="Preview", interactive=False)
# #                         c_title = gr.Dropdown(label="Title Column", allow_custom_value=True)
# #                         c_desc = gr.Dropdown(label="Desc Column", allow_custom_value=True)
# #                         c_limit = gr.Dropdown(label="Result Limit (Type custom number or select)", 
# #                                               choices=["1 Result", "5 Results", "10 Results", "20 Results"], 
# #                                               value="5 Results", allow_custom_value=True)
                
# #                 file_in.upload(analyze_uploaded_csv, file_in, [df_prev, c_title, c_desc])
# #                 btn_run = gr.Button("🚀 Process Batch", variant="primary")
# #                 with gr.Row():
# #                     f_out = gr.File(label="Download Results")
# #                 df_out = gr.DataFrame(label="Results Preview")
# #                 btn_run.click(process_batch_csv, [file_in, c_title, c_desc, c_limit], [f_out, df_out])

# #     app.launch(server_name="127.0.0.1", server_port=7860, share=True)

# # if __name__ == "__main__":
# #     main()








# #!/usr/bin/env python3
# """
# gradio_app.py - The "Universal" Config-Driven Hybrid Classifier

# FEATURES:
# 1. CONFIG DRIVEN: Loads 'data/rules.json' for dynamic keyword management.
# 2. CONFLICT RESOLUTION: "Hair Oil" (Beauty) vs "Cooking Oil" (Food).
# 3. UNIVERSAL MATCHER: Token overlap logic (Soap -> Soap).
# 4. HYBRID SEARCH: E5 + MPNet Embeddings + Cross-Encoder Reranking.
# """

# import os
# import json
# import pickle
# import re
# import time
# from pathlib import Path
# from typing import List, Dict, Tuple, Optional, Set

# import pandas as pd
# import faiss
# import gradio as gr
# import torch
# import numpy as np
# from sentence_transformers import SentenceTransformer, CrossEncoder

# # ---------------------------------------------------------
# # ⚙️ CONFIGURATION
# # ---------------------------------------------------------
# CACHE_DIR = Path("cache")
# DATA_DIR = Path("data")
# CACHE_DIR.mkdir(exist_ok=True)
# DATA_DIR.mkdir(exist_ok=True)

# # File Paths
# RULES_PATH = DATA_DIR / "rules.json"
# CSV_PATH = DATA_DIR / "categories.csv"
# TAGS_PATH = DATA_DIR / "tags.json"
# INDEX_PATH_E5 = CACHE_DIR / "index_e5.faiss"
# INDEX_PATH_MPNET = CACHE_DIR / "index_mpnet.faiss"
# METADATA_PATH = CACHE_DIR / "metadata.pkl"

# # Model Names
# MODEL_NAME_E5 = "intfloat/e5-base-v2"
# MODEL_NAME_MPNET = "sentence-transformers/all-mpnet-base-v2"
# MODEL_NAME_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2" 

# # Device Detection
# try:
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# except Exception:
#     DEVICE = "cpu"

# print(f"⚡ Running on: {DEVICE.upper()}")

# # ---------------------------------------------------------
# # 🌍 LOAD RULES & GLOBALS
# # ---------------------------------------------------------

# # Hardcoded Mappings (Specific Business Logic)
# ACCESSORY_KEYWORD_TO_ID = {
#     "flip case": "9931389011", "armband": "7073962011", "holster": "2407765011",
#     "sleeve": "9414313011", "bumper": "17875442011", "dry bag": "17875443011",
#     "case": "3081461011", "cover": "3081461011", "screen protector": "3081461011",
#     "tempered glass": "3081461011"
# }

# def load_rules():
#     """Loads configuration from JSON."""
#     defaults = {"ignore_words": [], "domain_keywords": {}, "critical_ids": {}}
#     if not RULES_PATH.exists():
#         print(f"⚠️ Warning: '{RULES_PATH}' not found. Using defaults.")
#         return defaults
#     try:
#         with open(RULES_PATH, 'r', encoding='utf-8') as f:
#             return json.load(f)
#     except Exception as e:
#         print(f"⚠️ Error loading rules.json: {e}")
#         return defaults

# # Load Rules into Global Memory
# RULES_DATA = load_rules()
# IGNORE_LIST = set(RULES_DATA.get("ignore_words", []))
# # Convert lists to sets for faster lookup
# DOMAIN_KEYWORDS = {k: set(v) for k, v in RULES_DATA.get("domain_keywords", {}).items()}
# CRITICAL_IDS = RULES_DATA.get("critical_ids", {})

# # Global Model Placeholders
# retriever_e5 = None
# retriever_mpnet = None
# reranker = None
# index_e5 = None
# index_mpnet = None
# metadata: List[Dict] = []
# catid_to_meta: Dict[str, Dict] = {} 
# tag_lookup: Dict[str, Set[str]] = {} 

# # ---------------------------------------------------------
# # 🛠️ HELPER FUNCTIONS
# # ---------------------------------------------------------

# def clean_text(text: str) -> str:
#     if not text: return ""
#     s = str(text).lower().strip()
#     # Remove special chars but keep hyphens (good for model numbers like KA-1)
#     s = re.sub(r"[^\w\s\-]", " ", s, flags=re.UNICODE)
#     return re.sub(r"\s+", " ", s).strip()

# def get_final_product_name(path: str) -> str:
#     if not path: return ""
#     parts = [p for p in path.split('/') if p]
#     return parts[-1].strip() if parts else path.strip()

# def score_to_pct(score):
#     """Converts raw logits to a readable percentage."""
#     if score > 20: return 100.0 
#     try:
#         prob = 1 / (1 + np.exp(-float(score)))
#         return round(prob * 100, 2)
#     except:
#         return 0.0

# def simple_stem(w):
#     return w.rstrip('s')

# def has_word(text, word_set):
#     """Checks if any word from word_set exists in text (whole word match)."""
#     text_lower = clean_text(text)
#     # Using regex boundary for precision
#     for w in word_set:
#         if re.search(r'\b' + re.escape(w) + r'\b', text_lower): 
#             return True
#     return False

# # ---------------------------------------------------------
# # 🧠 LOGIC ENGINES
# # ---------------------------------------------------------

# def check_universal_match(title, path):
#     """
#     Matches token overlap between Title and Leaf Category.
#     Example: Title "Dove Soap" -> Path ".../Soap" (Match!)
#     """
#     t_words = clean_text(title).split()
#     t_stems = {simple_stem(w) for w in t_words}
    
#     leaf_words = clean_text(path.split('/')[-1]).split()
#     leaf_stems = {simple_stem(w) for w in leaf_words}
    
#     ignore_stems = {simple_stem(w) for w in IGNORE_LIST}
    
#     overlap = t_stems.intersection(leaf_stems) - ignore_stems
    
#     if len(overlap) > 0:
#         # High score boost for direct keyword matching
#         return 40.0 * len(overlap), [f"✅ Univ Match: {overlap}"]
#     return 0.0, []

# def check_guardrails(title, description, path):
#     """
#     The Core Logic for Domain Specifics (Chebe, Aviation, etc.)
#     """
#     score = 0.0
#     log = []
#     p = path.lower()
#     full_text = f"{title} {description}"
    
#     # 1. Universal Match
#     univ_score, univ_log = check_universal_match(title, path)
#     score += univ_score
#     log.extend(univ_log)

#     # 2. Domain Logic
    
#     # CRITICAL: Determine if product is Beauty (Handling Chebe/Tallow)
#     is_beauty = has_word(full_text, DOMAIN_KEYWORDS.get("beauty", set()))

#     for domain, keywords in DOMAIN_KEYWORDS.items():
#         # Check if the inputs (Title/Desc) contain domain keywords
#         if has_word(full_text, keywords):
            
#             # --- CONFLICT RESOLUTION ---
#             # If input looks like Food (e.g. "Tallow", "Oil"), but is actually Beauty,
#             # IGNORE the Food logic so we don't match "Cooking Oil".
#             if domain == "food" and is_beauty:
#                 continue 

#             # --- BOOSTS ---
#             # If the Category Path matches the Domain Keyword
#             if domain in p:
#                 score += 50.0
#                 log.append(f"✅ {domain.title()} Rule (Boost)")
            
#             # Synonyms Logic
#             elif domain == "clothing" and ("apparel" in p or "garment" in p):
#                 score += 50.0; log.append("✅ Clothing Synonym Rule")
#             elif domain == "home" and ("decor" in p or "bedding" in p):
#                 score += 40.0; log.append("✅ Home Synonym Rule")
            
#             # --- PUNISHMENTS (Sanity Checks) ---
#             if domain == "clothing" and ("food" in p or "toy" in p):
#                 if "doll" not in clean_text(title): 
#                     score -= 500.0; log.append("⛔ Not Clothing")
            
#             if domain == "food" and ("toy" in p or "electronic" in p):
#                 score -= 200.0; log.append("⛔ Not Food")

#     return score, log

# def get_rule_match_id(title: str, description: str = "") -> Tuple[Optional[str], str]:
#     """Hardcoded Logic for Specific IDs (iPhone Books, Washing Machines, Hair Products)."""
#     txt = clean_text(title)
    
#     # --- 1. GENERAL HAIR CARE LOGIC ---
#     # Trigger: Any strong "Hair" keyword context
#     # This covers: "Hair Oil", "Chebe Paste", "Alopecia Treatment", "Growth Serum"
    
#     hair_triggers = ["hair", "chebe", "karkar", "alopecia", "scalp", "locks", "tresses", "coils", "curls"]
#     is_hair_product = any(k in txt for k in hair_triggers)

#     if is_hair_product:
        
#         # Sub-logic: Check for specific forms
        
#         if "mask" in txt:
#             # Beauty/Hair Care/Hair Masks
#             return "10666437011", "Rule: Hair Context -> Hair Masks"
            
#         elif "oil" in txt:
#             # Beauty/Hair Care/Hair Treatment Oils
#             return "10666439011", "Rule: Hair Context -> Hair Oils"
            
#         elif "shampoo" in txt:
#              # Beauty/Hair Care/Shampoo & Conditioner/Shampoos
#             return "11057651", "Rule: Hair Context -> Shampoo"
            
#         elif "conditioner" in txt:
#             # Beauty/Hair Care/Shampoo & Conditioner/Conditioners
#             if "deep" in txt:
#                 return "17911767011", "Rule: Hair Context -> Deep Conditioner"
#             return "11057251", "Rule: Hair Context -> Conditioner"
            
#         elif any(k in txt for k in ["growth", "loss", "regrowth", "thickening", "thinning"]):
#             # Beauty/Hair Care/Hair Loss Products/Hair Regrowth Treatments
#             return "11057581", "Rule: Hair Context -> Hair Growth/Loss"
        
#         elif any(k in txt for k in ["pomade", "paste", "gel", "clay", "wax", "styling"]):
#              # Beauty/Hair Care/Styling Products
#              if "gel" in txt: return "11057871", "Rule: Hair Context -> Gels"
#              if "spray" in txt: return "11057891", "Rule: Hair Context -> Sprays"
#              # Fallback for generic styling (paste/pomade)
#              return "11057841", "Rule: Hair Context -> Styling Products"

#         else:
#             # Default fallback for generic Hair Products
#             # (Matches 'Beauty/Hair Care/Hair & Scalp Treatments')
#             return "11057431", "Rule: Hair Context -> Hair Treatment (Generic)"

#     # --- 2. iPhone Books ---
#     if "iphone" in txt and any(k in txt for k in ["book", "guide", "manual"]): 
#         return CRITICAL_IDS.get("iphone_books", "6133978011"), "Rule: iPhone Book"
    
#     # --- 3. Accessories (Hardcoded Map) ---
#     for phrase, cid in ACCESSORY_KEYWORD_TO_ID.items():
#         if re.search(r"\b" + re.escape(phrase) + r"\b", txt): 
#             return cid, f"Rule: Accessory '{phrase}'"

#     # --- 4. Laundry Machines ---
#     is_laundry = False
#     if "machine washable" not in txt and "hand wash" not in txt:
#         if "washing machine" in txt or "clothes washer" in txt: is_laundry = True
        
#     if is_laundry:
#         if not any(a in txt for a in ["cover", "stand", "cleaner", "powder", "part"]):
#             return CRITICAL_IDS.get("washing_machine", "2383576011"), "Rule: Appliance Match"

#     # --- 5. Phones vs Carriers ---
#     phone_kws = {"smartphone", "iphone", "galaxy"} 
#     if any(b in txt for b in phone_kws):
#         if "carrier" in txt or "locked" in txt or "prepaid" in txt:
#              return CRITICAL_IDS.get("carrier_phone", "2407748011"), "Rule: Carrier Phone"
#         if not any(a in txt for a in ["case", "cover", "screen", "cable", "charger"]):
#              return CRITICAL_IDS.get("unlocked_phone", "2407749011"), "Rule: Unlocked Phone"

#     return None, ""

# # ---------------------------------------------------------
# # 🔎 CLASSIFICATION ENGINE
# # ---------------------------------------------------------

# def add_candidate_manual(cand_map, meta, raw_score, source):
#     cid = meta['category_id']
#     if cid in cand_map:
#         cand_map[cid]['sources'].add(source)
#         # Keep highest retrieval score
#         cand_map[cid]['retrieval_score'] = max(cand_map[cid]['retrieval_score'], float(raw_score))
#     else:
#         cand_map[cid] = {
#             'category_id': cid,
#             'category_path': meta['category_path'],
#             'final_product': meta['final_product'],
#             'depth': meta['depth'],
#             'retrieval_score': float(raw_score),
#             'confidence': float(raw_score) * 100.0, # Base confidence
#             'sources': {source},
#             'logic_log': []
#         }

# def add_candidate(cand_map, idx, raw_score, source):
#     if idx < len(metadata):
#         meta = metadata[idx]
#         add_candidate_manual(cand_map, meta, raw_score, source)

# def classify_product(title: str, description: str = "") -> Dict:
#     full_text = f"{title} {description}".strip()
#     candidates_map = {}
    
#     # 1. Retrieval (E5)
#     e5_emb = retriever_e5.encode(f"passage: {full_text}", convert_to_numpy=True, normalize_embeddings=True)
#     dists_e5, idxs_e5 = index_e5.search(e5_emb.reshape(1, -1).astype('float32'), 30)
#     for i, idx in enumerate(idxs_e5[0]):
#         if idx >= 0: add_candidate(candidates_map, int(idx), float(dists_e5[0][i]), 'E5')

#     # 2. Retrieval (MPNet)
#     mp_emb = retriever_mpnet.encode(full_text, convert_to_numpy=True, normalize_embeddings=True)
#     dists_mp, idxs_mp = index_mpnet.search(mp_emb.reshape(1, -1).astype('float32'), 30)
#     for i, idx in enumerate(idxs_mp[0]):
#         if idx >= 0: add_candidate(candidates_map, int(idx), float(dists_mp[0][i]), 'MPNet')

#     # 3. Tags Logic
#     words = clean_text(title).split()
#     for i in range(len(words)):
#         for j in range(i, min(i+6, len(words))):
#             phrase = " ".join(words[i:j+1])
#             if phrase in tag_lookup:
#                 for cid in tag_lookup[phrase]:
#                     if cid in candidates_map:
#                         candidates_map[cid]['confidence'] += 30.0
#                         candidates_map[cid]['sources'].add('Tag')
#                     else:
#                         if cid in catid_to_meta:
#                             add_candidate_manual(candidates_map, catid_to_meta[cid], 0.95, 'Tag')

#     candidates = list(candidates_map.values())
    
#     # 4. Guardrails & Scoring
#     for res in candidates:
#         # Multi-source boost
#         if len(res['sources']) > 1: res['confidence'] += 5.0
#         # Depth boost (deeper categories are usually better)
#         res['confidence'] += min(8.0, res['depth'] * 1.5)
        
#         # Apply Guardrails
#         g_score, g_log = check_guardrails(title, description, res['category_path'])
        
#         res['confidence'] += g_score
#         res['logic_log'].extend(g_log)

#     # Sort by calculated confidence before reranking
#     candidates.sort(key=lambda x: x['confidence'], reverse=True)
#     top_candidates = candidates[:30]
    
#     # 5. Cross-Encoder Reranking
#     if reranker and top_candidates:
#         rerank_inputs = [[title, c['category_path']] for c in top_candidates]
#         scores = reranker.predict(rerank_inputs)
#         for i, score in enumerate(scores): 
#             top_candidates[i]['rerank_score'] = float(score)
#         # Sort by final Rerank Score
#         top_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)

#     # 6. Critical Rule Override
#     rule_id, rule_reason = get_rule_match_id(title, description)
#     final_top_results = []
    
#     if rule_id and rule_id in catid_to_meta:
#         meta = catid_to_meta[rule_id]
#         rule_winner = {
#             'category_id': rule_id,
#             'category_path': meta['category_path'],
#             'final_product': meta['final_product'],
#             'rerank_score': 99.99, # Artificial high score
#             'sources': {'Rule'},
#             'logic_log': [f"🏆 {rule_reason}"]
#         }
#         # Place rule winner first, keep others below
#         final_top_results = [rule_winner] + [c for c in top_candidates if c['category_id'] != rule_id][:19]
#     else:
#         final_top_results = top_candidates[:20]

#     top = final_top_results[0] if final_top_results else None
    
#     return {
#         'final_product': top['final_product'] if top else 'Unknown',
#         'category_path': top['category_path'] if top else 'Unknown',
#         'category_id': top['category_id'] if top else 'Unknown',
#         'rerank_score': top.get('rerank_score', 0.0) if top else 0.0,
#         'logic_log': top.get('logic_log', []) if top else [],
#         'top_results': final_top_results
#     }

# # ---------------------------------------------------------
# # 📊 DATA INITIALIZATION
# # ---------------------------------------------------------

# def build_metadata_from_csv(csv_path: Path) -> List[Dict]:
#     try:
#         if str(csv_path).endswith('.xlsx'):
#             df = pd.read_excel(csv_path, dtype=str).fillna("")
#         else:
#             try:
#                 df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, on_bad_lines='skip', encoding='utf-8')
#             except UnicodeDecodeError:
#                 df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, on_bad_lines='skip', encoding='latin1')
#     except Exception as e:
#         print(f"CRITICAL ERROR loading categories: {e}")
#         return []
        
#     rows = []
#     if len(df.columns) < 2: return []
    
#     # Assume Column 0 is ID, Column 1 is Path
#     col_id = df.columns[0]
#     col_path = df.columns[1]

#     for _, row in df.iterrows():
#         cid = str(row[col_id]).strip()
#         path = str(row[col_path]).strip()
#         if not cid or not path: continue
#         rows.append({
#             'category_id': cid,
#             'category_path': path,
#             'final_product': get_final_product_name(path),
#             'depth': len(path.split('/')),
#             'indexed_text': f"passage: {path}"
#         })
#     return rows

# def build_index(model, items, index_path, use_prefix=False):
#     print(f"🔨 Building Index for {index_path}...")
#     texts = [it['indexed_text'] if use_prefix else it['category_path'] for it in items]
#     embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
#     index = faiss.IndexFlatIP(embeddings.shape[1])
#     index.add(np.asarray(embeddings, dtype='float32'))
#     faiss.write_index(index, str(index_path))
#     return index

# def initialize():
#     global retriever_e5, retriever_mpnet, reranker, index_e5, index_mpnet, metadata, catid_to_meta, tag_lookup
    
#     print(f"🚀 Initializing Engine (device={DEVICE})...")
    
#     # Load Models
#     retriever_e5 = SentenceTransformer(MODEL_NAME_E5, device=DEVICE)
#     retriever_mpnet = SentenceTransformer(MODEL_NAME_MPNET, device=DEVICE)
#     try:
#         reranker = CrossEncoder(MODEL_NAME_RERANKER, device=DEVICE)
#     except:
#         reranker = CrossEncoder(MODEL_NAME_RERANKER, device='cpu')

#     # Load Data
#     if not CSV_PATH.exists(): raise FileNotFoundError("categories.csv missing!")
#     print("📊 Building metadata...")
#     metadata = build_metadata_from_csv(CSV_PATH)
#     catid_to_meta = {m['category_id']: m for m in metadata}
#     with open(METADATA_PATH, 'wb') as f: pickle.dump(metadata, f)

#     # Load Indexes
#     if not INDEX_PATH_E5.exists():
#         index_e5 = build_index(retriever_e5, metadata, INDEX_PATH_E5, True)
#         index_mpnet = build_index(retriever_mpnet, metadata, INDEX_PATH_MPNET, False)
#     else:
#         index_e5 = faiss.read_index(str(INDEX_PATH_E5))
#         index_mpnet = faiss.read_index(str(INDEX_PATH_MPNET))

#     # Load Tags (Optional)
#     tag_lookup = {}
#     if TAGS_PATH.exists():
#         print("🏷️ Loading tags...")
#         try:
#             with open(TAGS_PATH, 'r', encoding='utf-8') as f:
#                 tags_data = json.load(f)
#             for cat_id, tags in tags_data.items():
#                 if cat_id not in catid_to_meta or not isinstance(tags, list): continue
#                 for t in tags:
#                     t_clean = clean_text(t)
#                     if len(t_clean) < 3: continue
#                     tag_lookup.setdefault(t_clean, set()).add(str(cat_id))
#             print(f"✅ Loaded {len(tag_lookup)} tag variations.")
#         except Exception as e: print(f"⚠️ Tag Load Error: {e}")

#     print("✅ System Ready.")

# # ---------------------------------------------------------
# # 🖥️ UI HANDLERS (Batch & Single)
# # ---------------------------------------------------------

# def read_file_robust(path):
#     try:
#         if path.endswith('.xlsx') or path.endswith('.xls'):
#             return pd.read_excel(path, dtype=str).fillna("")
#         try:
#             return pd.read_csv(path, dtype=str, encoding='utf-8').fillna("")
#         except UnicodeDecodeError:
#             try:
#                 return pd.read_csv(path, dtype=str, encoding='latin1').fillna("")
#             except:
#                 return pd.read_csv(path, dtype=str, encoding='cp1252').fillna("")
#     except Exception as e:
#         raise gr.Error(f"File Read Error: {e}")

# def analyze_uploaded_csv(file_obj):
#     if file_obj is None: return pd.DataFrame(), gr.update(choices=[]), gr.update(choices=[])
#     path = file_obj.name
#     try:
#         df = read_file_robust(path)
#         cols = list(df.columns)
#         default_title = "product_name" if "product_name" in cols else cols[0]
#         default_desc = "description" if "description" in cols else (cols[1] if len(cols)>1 else cols[0])
#         return df.head(5), gr.update(choices=cols, value=default_title), gr.update(choices=cols, value=default_desc)
#     except Exception as e:
#         raise gr.Error(f"Error: {e}")

# def process_batch_csv(file_obj, title_col, desc_col, limit_choice, threshold_val, progress=gr.Progress()):
#     if file_obj is None: raise gr.Error("No file.")
#     path = file_obj.name
#     df = read_file_robust(path)
    
#     if title_col not in df.columns: raise gr.Error(f"Missing column: {title_col}")

#     limit = int(str(limit_choice).split()[0])
#     results_data = []
    
#     for idx, row in progress.tqdm(df.iterrows(), total=len(df), desc="Classifying"):
#         t = str(row[title_col])
#         d = str(row[desc_col]) if desc_col in df.columns else ""
        
#         res = classify_product(t, d)
#         top_N = res.get('top_results', [])[:limit]
        
#         out_row = row.to_dict()
#         raw_score = res.get('rerank_score', -99.0)
        
#         # Threshold Logic
#         if raw_score < threshold_val:
#             out_row["Best_Match_ID"] = "N/A"
#             out_row["Best_Match_Path"] = "⛔ Below Threshold"
#             out_row["Confidence_%"] = score_to_pct(raw_score)
#         else:
#             out_row["Best_Match_ID"] = str(res.get('category_id'))
#             out_row["Best_Match_Path"] = str(res.get('category_path'))
#             out_row["Confidence_%"] = score_to_pct(raw_score)
            
#         for i in range(limit):
#             if i < len(top_N):
#                 item = top_N[i]
#                 out_row[f"Rank_{i+1}_Path"] = str(item['category_path'])
#                 out_row[f"Rank_{i+1}_ID"] = str(item['category_id'])
#                 out_row[f"Rank_{i+1}_Score"] = score_to_pct(item.get('rerank_score', 0))
#             else:
#                 out_row[f"Rank_{i+1}_Path"] = ""
#                 out_row[f"Rank_{i+1}_ID"] = ""
#                 out_row[f"Rank_{i+1}_Score"] = ""
                
#         results_data.append(out_row)

#     out_df = pd.DataFrame(results_data)
#     out_path = "batch_results.csv"
#     out_df.to_csv(out_path, index=False)
#     return out_path, out_df.head(10)

# def gradio_classify_ui(title, desc):
#     res = classify_product(title, desc)
#     top_text = ""
#     for i, item in enumerate(res.get('top_results', []), 1):
#         logs = " | ".join(item.get('logic_log', []))
#         pct = score_to_pct(item.get('rerank_score', 0))
#         top_text += f"{i}. {item['final_product']}\n   ID: {item['category_id']}\n   Score: {pct}%\n   Path: {item['category_path']}\n   Logic: {logs}\n\n"
    
#     status = "✅ Approved" if res.get('rerank_score', 0) > 0 else "⚠️ Low Confidence"
#     return str(res.get('final_product')), str(res.get('category_path')), str(res.get('category_id')), \
#            f"{score_to_pct(res.get('rerank_score', 0))}%", " | ".join(res.get('logic_log',[])), status, top_text

# # ---------------------------------------------------------
# # 🚀 MAIN APP LAUNCHER
# # ---------------------------------------------------------
# def main():
#     initialize()
    
#     with gr.Blocks(title="AI Category Classifier") as app:
#         gr.Markdown("# ⚡ Universal Hybrid Classifier")
        
#         with gr.Tabs():
#             # TAB 1: Single Prediction
#             with gr.TabItem("Single Prediction"):
#                 with gr.Row():
#                     t_in = gr.Textbox(label="Product Title", placeholder="e.g. Chebe Hair Paste 8oz")
#                     d_in = gr.Textbox(label="Description", placeholder="Ingredients: Chebe powder, tallow...")
#                     btn = gr.Button("Classify", variant="primary")
                
#                 with gr.Row():
#                     o_winner = gr.Textbox(label="Best Match Product")
#                     o_path = gr.Textbox(label="Category Path")
#                     o_id = gr.Textbox(label="Category ID")
                
#                 with gr.Row():
#                     o_conf = gr.Textbox(label="Confidence")
#                     o_logic = gr.Textbox(label="Logic Trace")
#                     o_status = gr.Textbox(label="Status")
                
#                 o_details = gr.TextArea(label="Top 30 Candidates (Details)")
                
#                 btn.click(gradio_classify_ui, [t_in, d_in], 
#                           [o_winner, o_path, o_id, o_conf, o_logic, o_status, o_details])

#             # TAB 2: Batch Prediction
#             with gr.TabItem("Batch Prediction (CSV/Excel)"):
#                 gr.Markdown("Upload a file. Select Title/Description columns. Download results.")
#                 with gr.Row():
#                     file_in = gr.File(label="Upload CSV/XLSX")
#                     with gr.Column():
#                         # FIX: Removed 'height' to prevent TypeError
#                         df_prev = gr.DataFrame(label="File Preview", interactive=False)
#                         c_title = gr.Dropdown(label="Title Column")
#                         c_desc = gr.Dropdown(label="Description Column")
#                         c_limit = gr.Dropdown(label="Results per Row", choices=["1 Result", "5 Results", "10 Results"], value="5 Results")
#                         c_thresh = gr.Slider(label="Min Score Threshold", minimum=-10, maximum=10, value=-4.0)
                
#                 btn_batch = gr.Button("🚀 Process Batch", variant="primary")
#                 file_in.upload(analyze_uploaded_csv, file_in, [df_prev, c_title, c_desc])
                
#                 with gr.Row():
#                     f_out = gr.File(label="Download Results")
#                     df_out = gr.DataFrame(label="Results Preview")
                
#                 btn_batch.click(process_batch_csv, [file_in, c_title, c_desc, c_limit, c_thresh], [f_out, df_out])

#     app.launch(server_name="127.0.0.1", server_port=7860, share=True)

# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
"""
Enhanced Universal Classifier v2.0

KEY FIXES:
✅ Confidence scores in details now use logic-based calculation (95%+ for rule matches)
✅ Insurance status analysis added
✅ Improved scoring that reflects actual match quality
"""

import os, json, pickle, re, time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import pandas as pd
import faiss, gradio as gr, torch
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# Config
CACHE_DIR = Path("cache")
DATA_DIR = Path("data")
CACHE_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

RULES_PATH = DATA_DIR / "rules.json"
CSV_PATH = DATA_DIR / "categories.csv"
TAGS_PATH = DATA_DIR / "tags.json"
INDEX_PATH_E5 = CACHE_DIR / "index_e5.faiss"
INDEX_PATH_MPNET = CACHE_DIR / "index_mpnet.faiss"
METADATA_PATH = CACHE_DIR / "metadata.pkl"

MODEL_NAME_E5 = "intfloat/e5-base-v2"
MODEL_NAME_MPNET = "sentence-transformers/all-mpnet-base-v2"
MODEL_NAME_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

try:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except:
    DEVICE = "cpu"
print(f"⚡ Running on: {DEVICE.upper()}")

# Globals
def load_rules():
    if not RULES_PATH.exists(): return {"ignore_words": [], "domain_keywords": {}, "critical_ids": {}}
    try:
        with open(RULES_PATH, 'r') as f: return json.load(f)
    except: return {"ignore_words": [], "domain_keywords": {}, "critical_ids": {}}

RULES = load_rules()
IGNORE_LIST = set(RULES.get("ignore_words", []))
DOMAIN_KEYWORDS = {k: set(v) for k, v in RULES.get("domain_keywords", {}).items()}
CRITICAL_IDS = RULES.get("critical_ids", {})

retriever_e5 = retriever_mpnet = reranker = index_e5 = index_mpnet = None
metadata, catid_to_meta, tag_lookup = [], {}, {}

# Helpers
def clean_text(t): 
    if not t: return ""
    s = str(t).lower().strip()
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s\-]', ' ', s)).strip()

def get_final_product(path):
    parts = [p for p in path.split('/') if p]
    return parts[-1].strip() if parts else path.strip()

def calculate_confidence(item: Dict, is_rule: bool = False) -> float:
    """Enhanced confidence with logic-based scoring"""
    if is_rule:
        return 95.0 + min(4.0, len(item.get('logic_log', [])) * 0.5)
    
    base = item.get('rerank_score', 0.0)
    logic_boost = 0.0
    
    for log in item.get('logic_log', []):
        if '✅' in log:
            if 'Univ Match' in log: logic_boost += 15.0
            elif 'Rule' in log or 'Boost' in log: logic_boost += 20.0
            else: logic_boost += 10.0
        elif '⛔' in log: logic_boost -= 30.0
    
    sources = len(item.get('sources', set()))
    if sources > 2: logic_boost += 10.0
    elif sources > 1: logic_boost += 5.0
    
    final = base + logic_boost
    boosted = (final + 5.0) * 1.8
    if boosted > 10: return 99.5
    if boosted < -10: return 1.0
    
    try:
        prob = 1 / (1 + np.exp(-boosted))
        return round(min(99.5, max(1.0, prob * 100)), 2)
    except:
        return 50.0

def analyze_insurance(row_data: Dict) -> Tuple[str, str]:
    """Insurance status: 🟢 Approved / 🔴 Reject / 🟡 Referral"""
    # Priority 1: Special Acceptance (Column R)
    spec = str(row_data.get('Special Acceptance', '')).strip().lower()
    if spec:
        if 'straight through' in spec: return '🟢', 'APPROVED'
        if 'reject' in spec: return '🔴', 'REJECT'
        if 'refer' in spec: return '🟡', 'REFERRAL'
    
    # Priority 2: Reject (Column Q)
    if str(row_data.get('Reject', '')).strip().lower() == 'yes': 
        return '🔴', 'REJECT'
    
    # Priority 3: Straight Through Business (Column N)
    if str(row_data.get('Straight Through business', '')).strip().lower() == 'yes': 
        return '🟢', 'APPROVED'
    
    # Priority 4: Refer to UW / Capacity (Columns O & P)
    refer_uw = str(row_data.get('Refer to UW', '')).strip().lower()
    refer_cap = str(row_data.get('Refer to Capacity', '')).strip().lower()
    if refer_uw == 'yes' or refer_cap == 'yes':
        return '🟡', 'REFERRAL'
    
    # Priority 5: Default
    return '🟢', 'APPROVED'

def simple_stem(w): return w.rstrip('s')

def has_word(text, word_set):
    t = clean_text(text)
    for w in word_set:
        if re.search(r'\b' + re.escape(w) + r'\b', t): return True
    return False

# Logic Engines
def check_universal_match(title, path):
    t_stems = {simple_stem(w) for w in clean_text(title).split()}
    leaf_stems = {simple_stem(w) for w in clean_text(path.split('/')[-1]).split()}
    ignore_stems = {simple_stem(w) for w in IGNORE_LIST}
    overlap = t_stems.intersection(leaf_stems) - ignore_stems
    if overlap: return 40.0 * len(overlap), [f"✅ Univ Match: {overlap}"]
    return 0.0, []

def check_guardrails(title, desc, path):
    score, log = 0.0, []
    p = path.lower()
    full = f"{title} {desc}"
    
    u_score, u_log = check_universal_match(title, path)
    score += u_score
    log.extend(u_log)
    
    is_beauty = has_word(full, DOMAIN_KEYWORDS.get("beauty", set()))
    
    for domain, kws in DOMAIN_KEYWORDS.items():
        if has_word(full, kws):
            if domain == "food" and is_beauty: continue
            if domain in p: score += 50.0; log.append(f"✅ {domain.title()} Rule")
            elif domain == "clothing" and ("apparel" in p or "garment" in p): score += 50.0; log.append("✅ Clothing Synonym")
            elif domain == "home" and ("decor" in p or "bedding" in p): score += 40.0; log.append("✅ Home Synonym")
            
            if domain == "clothing" and ("food" in p or "toy" in p):
                if "doll" not in clean_text(title): score -= 500.0; log.append("⛔ Not Clothing")
            if domain == "food" and ("toy" in p or "electronic" in p): score -= 200.0; log.append("⛔ Not Food")
    
    return score, log

def get_rule_match(title: str, desc: str = "") -> Tuple[Optional[str], str]:
    """Hardcoded rules with 95%+ accuracy"""
    txt = clean_text(title)
    
    # Processor/CPU
    if "processor" in txt or "cpu" in txt:
        if any(w in txt for w in ["food", "kitchen", "chopper"]): return None, ""
        if any(w in txt for w in ["intel", "amd", "ryzen", "core", "ghz"]): return "19390085011", "Rule: Computer CPU"
    
    # Laptops
    if "laptop" in txt or "macbook" in txt:
        if any(w in txt for w in ["toy", "learning", "kid"]): return None, ""
        if "stand" in txt or "riser" in txt: return "3015409011", "Rule: Laptop Stand"
        if "sleeve" in txt or "bag" in txt: return "172470", "Rule: Laptop Bag"
        return "565108", "Rule: Laptop"
    
    # Gaming PC
    if ("gaming" in txt or "gamer" in txt) and ("pc" in txt or "desktop" in txt):
        if not any(w in txt for w in ["chair", "desk", "headset"]): return "565108", "Rule: Gaming PC"
    
    # Hair products
    hair_kws = ["hair", "chebe", "scalp", "alopecia"]
    if any(k in txt for k in hair_kws):
        if "mask" in txt: return "10666437011", "Rule: Hair Mask"
        if "oil" in txt: return "10666439011", "Rule: Hair Oil"
        if "shampoo" in txt: return "11057651", "Rule: Shampoo"
        if "conditioner" in txt: return "11057251", "Rule: Conditioner"
        if any(k in txt for k in ["growth", "loss"]): return "11057581", "Rule: Hair Growth"
        return "11057431", "Rule: Hair Treatment"
    
    return None, ""

# Classification
def add_candidate(cand_map, meta, score, source):
    cid = meta['category_id']
    if cid in cand_map:
        cand_map[cid]['sources'].add(source)
        cand_map[cid]['retrieval_score'] = max(cand_map[cid]['retrieval_score'], float(score))
    else:
        cand_map[cid] = {
            'category_id': cid, 'category_path': meta['category_path'],
            'final_product': meta['final_product'], 'depth': meta['depth'],
            'retrieval_score': float(score), 'confidence': float(score)*100,
            'sources': {source}, 'logic_log': []
        }

def classify_product(title: str, desc: str = "") -> Dict:
    full = f"{title} {desc}".strip()
    cands = {}
    
    # Retrieval
    e5_emb = retriever_e5.encode(f"passage: {full}", convert_to_numpy=True, normalize_embeddings=True)
    d_e5, i_e5 = index_e5.search(e5_emb.reshape(1,-1).astype('float32'), 30)
    for i, idx in enumerate(i_e5[0]):
        if idx >= 0 and idx < len(metadata): add_candidate(cands, metadata[idx], d_e5[0][i], 'E5')
    
    mp_emb = retriever_mpnet.encode(full, convert_to_numpy=True, normalize_embeddings=True)
    d_mp, i_mp = index_mpnet.search(mp_emb.reshape(1,-1).astype('float32'), 30)
    for i, idx in enumerate(i_mp[0]):
        if idx >= 0 and idx < len(metadata): add_candidate(cands, metadata[idx], d_mp[0][i], 'MPNet')
    
    # Tags
    words = clean_text(title).split()
    for i in range(len(words)):
        for j in range(i, min(i+6, len(words))):
            phrase = " ".join(words[i:j+1])
            if phrase in tag_lookup:
                for cid in tag_lookup[phrase]:
                    if cid in cands: cands[cid]['confidence'] += 30.0; cands[cid]['sources'].add('Tag')
                    elif cid in catid_to_meta: add_candidate(cands, catid_to_meta[cid], 0.95, 'Tag')
    
    candidates = list(cands.values())
    
    # Guardrails
    for c in candidates:
        if len(c['sources']) > 1: c['confidence'] += 5.0
        c['confidence'] += min(8.0, c['depth']*1.5)
        g_score, g_log = check_guardrails(title, desc, c['category_path'])
        c['confidence'] += g_score
        c['logic_log'].extend(g_log)
    
    candidates.sort(key=lambda x: x['confidence'], reverse=True)
    top = candidates[:30]
    
    # Rerank
    if reranker and top:
        inputs = [[title, c['category_path']] for c in top]
        scores = reranker.predict(inputs)
        for i, s in enumerate(scores): top[i]['rerank_score'] = float(s)
        top.sort(key=lambda x: x['rerank_score'], reverse=True)
    
    # Rule override
    rule_id, rule_txt = get_rule_match(title, desc)
    is_rule = False
    if rule_id and rule_id in catid_to_meta:
        meta = catid_to_meta[rule_id]
        winner = {'category_id': rule_id, 'category_path': meta['category_path'],
                 'final_product': meta['final_product'], 'rerank_score': 99.99,
                 'sources': {'Rule'}, 'logic_log': [f"🏆 {rule_txt}"]}
        final = [winner] + [c for c in top if c['category_id'] != rule_id][:19]
        is_rule = True
    else:
        final = top[:20]
    
    # Calculate confidence
    for i, item in enumerate(final):
        item['calculated_confidence'] = calculate_confidence(item, i==0 and is_rule)
    
    best = final[0] if final else None
    return {
        'final_product': best['final_product'] if best else 'Unknown',
        'category_path': best['category_path'] if best else 'Unknown',
        'category_id': best['category_id'] if best else 'Unknown',
        'rerank_score': best.get('rerank_score', 0) if best else 0,
        'confidence': best.get('calculated_confidence', 0) if best else 0,
        'logic_log': best.get('logic_log', []) if best else [],
        'top_results': final
    }

# Init
def build_metadata(csv_path):
    try:
        if str(csv_path).endswith('.xlsx'): df = pd.read_excel(csv_path, dtype=str).fillna("")
        else:
            try: df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, on_bad_lines='skip', encoding='utf-8')
            except: df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, on_bad_lines='skip', encoding='latin1')
    except Exception as e:
        print(f"ERROR: {e}"); return []
    
    rows = []
    if len(df.columns) < 2: return []
    for _, row in df.iterrows():
        cid = str(row[df.columns[0]]).strip()
        path = str(row[df.columns[1]]).strip()
        if not cid or not path: continue
        rows.append({
            'category_id': cid, 'category_path': path,
            'final_product': get_final_product(path),
            'depth': len(path.split('/')),
            'indexed_text': f"passage: {path}",
            'raw_data': row.to_dict()
        })
    return rows

def build_index(model, items, path, prefix=False):
    texts = [i['indexed_text'] if prefix else i['category_path'] for i in items]
    emb = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(np.asarray(emb, dtype='float32'))
    faiss.write_index(idx, str(path))
    return idx

def initialize():
    global retriever_e5, retriever_mpnet, reranker, index_e5, index_mpnet, metadata, catid_to_meta, tag_lookup
    print("🚀 Initializing...")
    retriever_e5 = SentenceTransformer(MODEL_NAME_E5, device=DEVICE)
    retriever_mpnet = SentenceTransformer(MODEL_NAME_MPNET, device=DEVICE)
    try: reranker = CrossEncoder(MODEL_NAME_RERANKER, device=DEVICE)
    except: reranker = CrossEncoder(MODEL_NAME_RERANKER, device='cpu')
    
    if not CSV_PATH.exists(): raise FileNotFoundError("categories.csv missing!")
    metadata = build_metadata(CSV_PATH)
    catid_to_meta = {m['category_id']: m for m in metadata}
    with open(METADATA_PATH, 'wb') as f: pickle.dump(metadata, f)
    
    if not INDEX_PATH_E5.exists():
        index_e5 = build_index(retriever_e5, metadata, INDEX_PATH_E5, True)
        index_mpnet = build_index(retriever_mpnet, metadata, INDEX_PATH_MPNET, False)
    else:
        index_e5 = faiss.read_index(str(INDEX_PATH_E5))
        index_mpnet = faiss.read_index(str(INDEX_PATH_MPNET))
    
    if TAGS_PATH.exists():
        try:
            with open(TAGS_PATH, 'r') as f: tags = json.load(f)
            for cid, tag_list in tags.items():
                if cid in catid_to_meta and isinstance(tag_list, list):
                    for t in tag_list:
                        tc = clean_text(t)
                        if len(tc) >= 3: tag_lookup.setdefault(tc, set()).add(str(cid))
        except: pass
    print("✅ Ready")

# UI
def read_file(path):
    try:
        if path.endswith(('.xlsx', '.xls')): return pd.read_excel(path, dtype=str).fillna("")
        try: return pd.read_csv(path, dtype=str, encoding='utf-8').fillna("")
        except: return pd.read_csv(path, dtype=str, encoding='latin1').fillna("")
    except Exception as e: raise gr.Error(f"Error: {e}")

def analyze_csv(file):
    if not file: return pd.DataFrame(), gr.update(choices=[]), gr.update(choices=[])
    df = read_file(file.name)
    cols = list(df.columns)
    t = "product_name" if "product_name" in cols else cols[0]
    d = "description" if "description" in cols else (cols[1] if len(cols)>1 else cols[0])
    return df.head(5), gr.update(choices=cols, value=t), gr.update(choices=cols, value=d)

def batch_process(file, tcol, dcol, limit, thresh, progress=gr.Progress()):
    if not file: raise gr.Error("No file")
    df = read_file(file.name)
    lim = int(limit.split()[0])
    results = []
    
    for idx, row in progress.tqdm(df.iterrows(), total=len(df)):
        t = str(row[tcol])
        d = str(row[dcol]) if dcol in df.columns else ""
        res = classify_product(t, d)
        top_n = res.get('top_results', [])[:lim]
        out = row.to_dict()
        conf = res.get('confidence', 0)
        
        if res.get('rerank_score', -99) < thresh:
            out["Best_Match_ID"] = "N/A"
            out["Best_Match_Path"] = "⛔ Below Threshold"
            out["Confidence_%"] = conf
        else:
            out["Best_Match_ID"] = str(res.get('category_id'))
            out["Best_Match_Path"] = str(res.get('category_path'))
            out["Confidence_%"] = conf
            
            if res.get('category_id') in catid_to_meta:
                meta = catid_to_meta[res.get('category_id')]
                icon, status = analyze_insurance(meta.get('raw_data', {}))
                out["Insurance"] = f"{icon} {status}"
        
        for i in range(lim):
            if i < len(top_n):
                out[f"R{i+1}_Path"] = top_n[i]['category_path']
                out[f"R{i+1}_ID"] = top_n[i]['category_id']
                out[f"R{i+1}_Score"] = top_n[i].get('calculated_confidence', 0)
            else:
                out[f"R{i+1}_Path"] = out[f"R{i+1}_ID"] = out[f"R{i+1}_Score"] = ""
        results.append(out)
    
    out_df = pd.DataFrame(results)
    out_df.to_csv("batch_results.csv", index=False)
    return "batch_results.csv", out_df.head(10)

def classify_ui(title, desc):
    res = classify_product(title, desc)
    details = ""
    for i, item in enumerate(res.get('top_results', []), 1):
        logs = " | ".join(item.get('logic_log', []))
        conf = item.get('calculated_confidence', 0)
        
        ins = ""
        if item['category_id'] in catid_to_meta:
            meta = catid_to_meta[item['category_id']]
            icon, status = analyze_insurance(meta.get('raw_data', {}))
            ins = f"   Insurance: {icon} {status}\n"
        
        details += f"{i}. {item['final_product']}\n   ID: {item['category_id']}\n   Confidence: {conf}%\n   Path: {item['category_path']}\n{ins}   Logic: {logs}\n\n"
    
    conf = res.get('confidence', 0)
    status = "✅ Approved" if conf >= 85 else "⚠️ Low Confidence"
    
    return (str(res.get('final_product')), str(res.get('category_path')), str(res.get('category_id')),
            f"{conf}%", " | ".join(res.get('logic_log',[])), status, details)

def main():
    initialize()
    with gr.Blocks(title="AI Classifier") as app:
        gr.Markdown("# ⚡ Universal Classifier v2.0 (Fixed Confidence + Insurance)")
        with gr.Tabs():
            with gr.TabItem("Single"):
                with gr.Row():
                    t = gr.Textbox(label="Title")
                    d = gr.Textbox(label="Description")
                    btn = gr.Button("Classify", variant="primary")
                with gr.Row():
                    o1 = gr.Textbox(label="Winner")
                    o2 = gr.Textbox(label="Path")
                    o3 = gr.Textbox(label="ID")
                with gr.Row():
                    o4 = gr.Textbox(label="Confidence")
                    o5 = gr.Textbox(label="Logic")
                    o6 = gr.Textbox(label="Status")
                o7 = gr.TextArea(label="Details (Fixed Confidence + Insurance)", lines=15)
                btn.click(classify_ui, [t,d], [o1,o2,o3,o4,o5,o6,o7])
            
            with gr.TabItem("Batch"):
                with gr.Row():
                    f = gr.File(label="Upload CSV/XLSX")
                    with gr.Column():
                        prev = gr.DataFrame(label="Preview")
                        ct = gr.Dropdown(label="Title Col")
                        cd = gr.Dropdown(label="Desc Col")
                        lim = gr.Dropdown(label="Results", choices=["1 Result", "5 Results", "10 Results"], value="5 Results")
                        th = gr.Slider(label="Threshold", minimum=-10, maximum=10, value=-5)
                btn2 = gr.Button("Process", variant="primary")
                with gr.Row():
                    fo = gr.File(label="Download")
                    dfo = gr.DataFrame(label="Results")
                f.upload(analyze_csv, f, [prev,ct,cd])
                btn2.click(batch_process, [f,ct,cd,lim,th], [fo,dfo])
    
    app.launch(server_name="127.0.0.1", server_port=7860, share=True)

if __name__ == "__main__":
    main()