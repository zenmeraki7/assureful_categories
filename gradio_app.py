# # #!/usr/bin/env python3
# # """
# # gradio_app.py - The "Precision" Hybrid Classifier (Batch Edition)

# # UPDATES:
# # 1. Added "Batch CSV" Tab.
# # 2. Processes uploaded CSVs and outputs Top 5 predictions.
# # 3. Maintains all previous fixes (Keyboard, Laundry, Auto).
# # """

# # import os
# # import json
# # import pickle
# # import re
# # import time
# # from pathlib import Path
# # from typing import List, Dict, Tuple, Optional, Set

# # import numpy as np
# # import pandas as pd
# # import faiss
# # import gradio as gr
# # import torch

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

# # # Device Selection
# # try:
# #     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# # except Exception:
# #     DEVICE = "cpu"

# # print(f"⚡ Running on: {DEVICE.upper()}")

# # # -------------------------
# # # 🌍 DOMAIN RULES & KEYWORDS
# # # -------------------------

# # # 1. PHONES
# # PHONE_KEYWORDS = {
# #     "smartphone", "mobile phone", "cell phone", "android", "handset", "telephone",
# #     "iphone", "apple", "samsung", "galaxy", "pixel", "google", "oneplus",
# #     "motorola", "moto", "nokia", "sony", "xperia", "lg", "htc", "asus", "rog phone",
# #     "xiaomi", "redmi", "mi phone", "poco", "oppo", "vivo", "realme", "iqoo",
# #     "huawei", "honor", "zte", "nubia", "redmagic", "tecno", "infinix", "itel", 
# #     "tcl", "alcatel", "blu", "cat phone", "caterpillar", "kyocera", "sonim", 
# #     "agm", "blackview", "cubot", "doogee", "oukitel", "ulefone", "umidigi", "unihertz",
# #     "sharp", "aquos", "fairphone", "nothing", "micromax", "lava", "jio phone"
# # }

# # # 2. CARRIERS
# # CARRIER_KEYWORDS = {
# #     "carrier", "locked", "sim locked", "contract", "prepaid", "postpaid",
# #     "att", "at&t", "verizon", "t-mobile", "sprint", "us cellular",
# #     "cricket", "metro", "metropcs", "boost mobile", "tracfone", "straight talk", 
# #     "total wireless", "simple mobile", "mint mobile", "google fi", "consumer cellular",
# #     "visible", "xfinity mobile", "spectrum mobile", "vodafone", "orange", "o2", 
# #     "ee", "three", "rogers", "bell", "telus", "jio", "airtel", "vi", "bsnl"
# # }

# # # 3. AUDIO
# # AUDIO_KEYWORDS = {
# #     "headphone", "headphones", "earphone", "earphones", "earbud", "earbuds", 
# #     "headset", "tws", "airpod", "airpods", "galaxy buds", "pixel buds", 
# #     "noise cancelling", "anc", "bluetooth speaker", "soundbar", "home theater"
# # }

# # # 4. LAUNDRY (Hardware)
# # LAUNDRY_BRANDS = {
# #     "godrej", "lg", "samsung", "whirlpool", "bosch", "ifb", "haier", "panasonic", "midea"
# # }
# # LAUNDRY_GENERICS = {
# #     "washing machine", "washer", "clothes washer", "washer dryer", "fully automatic", "semi automatic"
# # }

# # # 5. KEYBOARDS (Disambiguation Lists)
# # COMP_KEYBOARD_BRANDS = {"hp", "dell", "logitech", "razer", "corsair", "lenovo", "asus", "acer", "microsoft", "keychron", "rk royal", "zebronics"}
# # COMP_KEYBOARD_TERMS = {"usb", "wired", "wireless", "bluetooth", "mouse", "combo", "mechanical", "membrane", "gaming", "qwerty", "typing", "pc", "laptop", "computer", "chiclet"}

# # # 5. ACCESSORIES
# # ACCESSORY_KEYWORD_TO_ID = {
# #     "flip case": "9931389011", "armband": "7073962011", "holster": "2407765011",
# #     "sleeve": "9414313011", "bumper": "17875442011", "dry bag": "17875443011",
# #     "case": "3081461011", "cover": "3081461011", "back cover": "3081461011",
# #     "screen protector": "3081461011", "tempered glass": "3081461011", "glass guard": "3081461011",
# #     "mobile broadband": "2407750011"
# # }

# # BOOK_KEYWORDS = {"book", "guide", "manual", "ebook", "pdf", "kindle", "paperback"}

# # # ⚠️ CRITICAL CATEGORY IDs
# # WASHING_MACHINE_ID = "2383576011" 
# # UNLOCKED_CELL_PHONES_ID = "2407749011"
# # IPHONE_BOOKS_ID = "6133978011"
# # CARRIER_PHONES_ID = "2407748011"
# # HEADPHONES_ID = "172541" 
# # LAUNDRY_SUPPLIES_ID = "15356111"
# # TRASH_BAGS_ID = "15342971"

# # # -------------------------
# # # Globals & Utilities
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
# #     s = re.sub(r"\s+", " ", s).strip()
# #     return s

# # def get_final_product_name(path: str) -> str:
# #     if not path: return ""
# #     parts = [p for p in path.split('/') if p]
# #     return parts[-1].strip() if parts else path.strip()

# # # -------------------------
# # # Phase 1: Data Loading
# # # -------------------------
# # def build_metadata_from_csv(csv_path: Path) -> List[Dict]:
# #     df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, on_bad_lines='skip')
# #     rows = []
# #     for _, row in df.iterrows():
# #         if len(row) < 2: continue
# #         cid = str(row.iloc[0]).strip()
# #         path = str(row.iloc[1]).strip()
# #         if not cid or not path: continue
        
# #         final_prod = get_final_product_name(path)
# #         indexed_text = f"passage: {path}"
        
# #         rows.append({
# #             "category_id": cid,
# #             "category_path": path,
# #             "final_product": final_prod,
# #             "depth": len(path.split('/')),
# #             "indexed_text": indexed_text
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
# #         # Blacklist logic
# #         TAG_BLACKLIST = {
# #             "appliance", "appliances", "drum", "drums", "barrel", 
# #             "wash", "washing", "machine", "machines", "fan", "fans", "pump", "pumps", 
# #             "stand", "stands", "rack", "mount", "case", "cover", "bag", 
# #             "light", "lamp", "bulb", "remote", "switch", "wire", "cable",
# #             "automatic", "fully automatic", "semi automatic", "manual", 
# #             "front load", "top load", "white", "black", "silver"
# #         }
# #         try:
# #             with open(TAGS_PATH, 'r', encoding='utf-8') as f:
# #                 tags_data = json.load(f)
# #             count = 0
# #             for cat_id, tags in tags_data.items():
# #                 if cat_id not in catid_to_meta: continue
# #                 if not isinstance(tags, list): continue
# #                 for t in tags:
# #                     t_clean = clean_text(t)
# #                     if len(t_clean) < 3 or t_clean in ["buy", "best", "amazon", "product"]: continue
# #                     if t_clean in TAG_BLACKLIST: continue
# #                     tag_lookup.setdefault(t_clean, set()).add(str(cat_id))
# #                     count += 1
# #             tag_lookup = {k: list(v) for k, v in tag_lookup.items()}
# #             print(f"✅ Loaded {count} tags.")
# #         except Exception as e: print(f"⚠️ Tag Load Error: {e}")

# # # -------------------------
# # # Phase 2: Rule Engine
# # # -------------------------
# # def get_rule_match_id(title: str, description: str = "") -> Tuple[Optional[str], str]:
# #     txt = clean_text(title) 
    
# #     if "iphone" in txt and any(k in txt for k in BOOK_KEYWORDS):
# #         return IPHONE_BOOKS_ID, "Rule: iPhone Book"
    
# #     for phrase, cid in ACCESSORY_KEYWORD_TO_ID.items():
# #         if re.search(r"\b" + re.escape(phrase) + r"\b", txt):
# #             return cid, f"Rule: Accessory '{phrase}'"

# #     for kw in ["trash bag", "garbage bag", "bin liner"]:
# #         if kw in txt: return TRASH_BAGS_ID, f"Rule: Supply '{kw}'"
# #     for kw in ["detergent", "washing powder", "fabric softener", "liquid detergent"]:
# #         if kw in txt: return LAUNDRY_SUPPLIES_ID, f"Rule: Supply '{kw}'"

# #     is_laundry_machine = False
# #     for kw in LAUNDRY_GENERICS:
# #         if re.search(r"\b" + re.escape(kw) + r"\b", txt):
# #             is_laundry_machine = True
# #             break
            
# #     if not is_laundry_machine:
# #         for brand in LAUNDRY_BRANDS:
# #             if brand in txt and ("washing" in txt or "washer" in txt):
# #                 is_laundry_machine = True
# #                 break

# #     if is_laundry_machine:
# #         exclusion_list = ["cover", "stand", "cleaner", "powder", "liquid", "detergent", "mat", "pipe", "part", "basket", "bag", "bin"]
# #         if not any(a in txt for a in exclusion_list):
# #             return WASHING_MACHINE_ID, "Rule: Appliance Match (Laundry)"

# #     for kw in AUDIO_KEYWORDS:
# #         if re.search(r"\b" + re.escape(kw) + r"\b", txt):
# #             if "case" not in txt and "cover" not in txt:
# #                 return HEADPHONES_ID, f"Rule: Audio '{kw}'"

# #     if any(k in txt for k in CARRIER_KEYWORDS) and any(p in txt for p in PHONE_KEYWORDS):
# #         return CARRIER_PHONES_ID, "Rule: Carrier Phone"

# #     for brand in PHONE_KEYWORDS:
# #         if re.search(r"\b" + re.escape(brand) + r"\b", txt):
# #             exclusion = ["battery", "charger", "cable", "adapter", "lcd", "display", "screen",
# #                          "car", "motorcycle", "bulb", "light", "lamp", "watch", "band",
# #                          "headphone", "earphone", "buds", "speaker", "soundbar"]
# #             if not any(a in txt for a in exclusion):
# #                 return UNLOCKED_CELL_PHONES_ID, f"Rule: Brand '{brand.title()}'"
# #     return None, ""

# # # -------------------------
# # # Phase 3: Guardrails
# # # -------------------------
# # def check_keyboard_ambiguity(title, path):
# #     t = clean_text(title)
# #     p = path.lower()
# #     is_comp_kb = any(b in t for b in COMP_KEYBOARD_BRANDS) or any(k in t for k in COMP_KEYBOARD_TERMS)
# #     if "keyboard" in t:
# #         if is_comp_kb and "musical instrument" in p: return -500.0
# #         if is_comp_kb and ("computer" in p or "electronics" in p or "accessory" in p): return 50.0
# #     return 0.0

# # def check_grocery_guardrail(title: str, path: str) -> float:
# #     txt = clean_text(title)
# #     path_lower = path.lower()
# #     grocery_words = {"vedaka", "presto", "happy belly", "solimo", "organic", "gluten free", "food", "snack", "dal", "rice", "trash bag"}
# #     grocery_units = [r"\d+\s?oz", r"\d+\s?lb", r"\d+\s?kg", r"\d+\s?gm", r"\d+\s?liter"]
# #     is_food = any(w in txt for w in grocery_words) or any(re.search(p, txt) for p in grocery_units)
# #     if "amazon fresh" in path_lower or "grocery" in path_lower: return 100.0 if is_food else -200.0
# #     if "electronics" in path_lower: return -200.0 if is_food else 0.0
# #     return 0.0

# # def check_accessory_penalty(title, path):
# #     title, path = title.lower(), path.lower()
# #     for sep in [r"\bfor\b", r"\bfits\b", r"\bcompatible with\b"]:
# #         match = re.search(sep, title)
# #         if match:
# #             target = title[match.end():].strip()
# #             if target in path and "accessory" not in path and "case" not in path:
# #                 return -25.0
# #     return 0.0

# # def check_automotive_guardrail(title, path):
# #     t = clean_text(title)
# #     p = path.lower()
# #     if "automotive" in p:
# #         laundry_signals = ["washing machine", "clothes washer", "laundry", "detergent"]
# #         if any(x in t for x in laundry_signals): return -500.0
# #         if "godrej" in t and "drum" in t: return -500.0
# #     return 0.0

# # # -------------------------
# # # Phase 4: Classification
# # # -------------------------
# # def classify_product(title: str, description: str = "") -> Dict:
# #     start_time = time.time()
# #     full_text = f"{title} {description}".strip()
    
# #     candidates_map = {}
    
# #     # E5
# #     e5_emb = retriever_e5.encode(f"passage: {full_text}", convert_to_numpy=True, normalize_embeddings=True)
# #     dists_e5, idxs_e5 = index_e5.search(e5_emb.reshape(1, -1).astype('float32'), 30)
# #     for i, idx in enumerate(idxs_e5[0]):
# #         if idx >= 0: add_candidate(candidates_map, int(idx), float(dists_e5[0][i]), 'E5')

# #     # MPNet
# #     mp_emb = retriever_mpnet.encode(full_text, convert_to_numpy=True, normalize_embeddings=True)
# #     dists_mp, idxs_mp = index_mpnet.search(mp_emb.reshape(1, -1).astype('float32'), 30)
# #     for i, idx in enumerate(idxs_mp[0]):
# #         if idx >= 0: add_candidate(candidates_map, int(idx), float(dists_mp[0][i]), 'MPNet')

# #     # Tag Engine
# #     words = clean_text(title).split()
# #     for i in range(len(words)):
# #         for j in range(i, min(i+6, len(words))):
# #             phrase = " ".join(words[i:j+1])
# #             if phrase in tag_lookup:
# #                 for cid in tag_lookup[phrase]:
# #                     if cid in candidates_map:
# #                         candidates_map[cid]['confidence'] += 30.0
# #                         candidates_map[cid]['sources'].add('Tag')
# #                         candidates_map[cid]['logic_log'].append(f"Tag: '{phrase}'")
# #                     else:
# #                         meta = catid_to_meta[cid]
# #                         add_candidate_manual(candidates_map, meta, 0.95, 'Tag', f"Tag: '{phrase}'")

# #     candidates = list(candidates_map.values())

# #     # Guardrails
# #     for res in candidates:
# #         if len(res['sources']) > 1: res['confidence'] += 5.0
# #         res['confidence'] += min(8.0, res['depth'] * 1.5)
        
# #         pen = check_accessory_penalty(title, res['category_path'])
# #         res['confidence'] += pen
# #         if pen < 0: res['logic_log'].append(f"Acc Pen {pen}")
        
# #         g_score = check_grocery_guardrail(title, res['category_path'])
# #         res['confidence'] += g_score
# #         if g_score < -50: res['logic_log'].append("⛔ Bad Dept (Food)")

# #         auto_pen = check_automotive_guardrail(title, res['category_path'])
# #         res['confidence'] += auto_pen
# #         if auto_pen < -100: res['logic_log'].append("⛔ Bad Dept (Auto)")
        
# #         kb_score = check_keyboard_ambiguity(title, res['category_path'])
# #         res['confidence'] += kb_score
# #         if kb_score < -100: res['logic_log'].append("⛔ Not Music KB")
# #         if kb_score > 20: res['logic_log'].append("✅ Comp KB Confirmed")

# #     # Re-Rank
# #     candidates.sort(key=lambda x: x['confidence'], reverse=True)
# #     top_candidates = candidates[:15]
    
# #     if reranker:
# #         rerank_inputs = [[title, c['category_path']] for c in top_candidates]
# #         scores = reranker.predict(rerank_inputs)
# #         for i, score in enumerate(scores):
# #             top_candidates[i]['rerank_score'] = float(score)
# #         top_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)

# #     # Rule Override
# #     rule_id, rule_reason = get_rule_match_id(title, description)
# #     final_top_results = []
    
# #     if rule_id:
# #         rule_winner = None
# #         others = []
# #         for cand in top_candidates:
# #             if cand['category_id'] == rule_id:
# #                 rule_winner = cand
# #             else:
# #                 others.append(cand)
        
# #         if not rule_winner and rule_id in catid_to_meta:
# #             meta = catid_to_meta[rule_id]
# #             rule_winner = {
# #                 'category_id': rule_id,
# #                 'category_path': meta['category_path'],
# #                 'final_product': meta['final_product'],
# #                 'rerank_score': 10.0,
# #                 'sources': {'Rule'},
# #                 'logic_log': []
# #             }
            
# #         if rule_winner:
# #             rule_winner['rerank_score'] = 99.99
# #             rule_winner['logic_log'].insert(0, f"🏆 {rule_reason}")
# #             final_top_results = [rule_winner] + others[:4]
# #     else:
# #         final_top_results = top_candidates[:5]

# #     top = final_top_results[0] if final_top_results else None
    
# #     return {
# #         'final_product': top['final_product'] if top else 'Unknown',
# #         'category_path': top['category_path'] if top else 'Unknown',
# #         'category_id': top['category_id'] if top else 'Unknown',
# #         'rerank_score': top.get('rerank_score', 0.0) if top else 0.0,
# #         'logic_log': top.get('logic_log', []) if top else [],
# #         'top_results': final_top_results,
# #         'time_ms': (time.time() - start_time) * 1000
# #     }

# # # -------------------------
# # # Initialization
# # # -------------------------
# # def add_candidate(cand_map, idx, raw_score, source):
# #     meta = metadata[idx]
# #     add_candidate_manual(cand_map, meta, raw_score, source)

# # def add_candidate_manual(cand_map, meta, raw_score, source, log=None):
# #     cid = meta['category_id']
# #     if cid in cand_map:
# #         cand_map[cid]['sources'].add(source)
# #         cand_map[cid]['retrieval_score'] = max(cand_map[cid]['retrieval_score'], float(raw_score))
# #         if log: cand_map[cid]['logic_log'].append(log)
# #     else:
# #         cand_map[cid] = {
# #             'category_id': cid,
# #             'category_path': meta['category_path'],
# #             'final_product': meta['final_product'],
# #             'depth': meta['depth'],
# #             'retrieval_score': float(raw_score),
# #             'confidence': float(raw_score) * 100.0,
# #             'sources': {source},
# #             'logic_log': [log] if log else []
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

# #     if INDEX_PATH_E5.exists() and METADATA_PATH.exists():
# #         try:
# #             with open(METADATA_PATH, 'rb') as f:
# #                 loaded_meta = pickle.load(f)
# #             if len(loaded_meta) == len(metadata):
# #                 metadata = loaded_meta
# #                 index_e5 = faiss.read_index(str(INDEX_PATH_E5))
# #                 index_mpnet = faiss.read_index(str(INDEX_PATH_MPNET))
# #                 print("✅ Loaded Indexes.")
# #             else:
# #                 raise ValueError("Metadata mismatch")
# #         except Exception:
# #             print("⚠️ Rebuilding indexes...")
# #             index_e5 = build_index(retriever_e5, 'e5', metadata, INDEX_PATH_E5, True)
# #             index_mpnet = build_index(retriever_mpnet, 'mpnet', metadata, INDEX_PATH_MPNET, False)
# #     else:
# #         index_e5 = build_index(retriever_e5, 'e5', metadata, INDEX_PATH_E5, True)
# #         index_mpnet = build_index(retriever_mpnet, 'mpnet', metadata, INDEX_PATH_MPNET, False)

# #     catid_to_meta = {m['category_id']: m for m in metadata}
# #     print("✅ System Ready.")

# # # -------------------------
# # # UI & BATCH PROCESSING
# # # -------------------------
# # def gradio_classify(title, desc):
# #     if not title: return ("",) * 7
# #     try:
# #         res = classify_product(title, desc)
# #         top_text = ""
# #         for i, item in enumerate(res.get('top_results', []), 1):
# #             score = item.get('rerank_score', 0.0)
# #             score_disp = "MAX (Rule)" if score > 90 else f"{score:.4f}"
# #             logs = " | ".join(item.get('logic_log', []))
# #             top_text += f"{i}. {item['final_product']}\n   ID: {item['category_id']} | Score: {score_disp}\n   Path: {item['category_path']}\n   ℹ️ {logs}\n\n"
        
# #         status = "✅ Approved" if float(res.get('rerank_score', 0)) > 0 else "⚠️ Review Needed"
# #         return (
# #             str(res.get('final_product')), str(res.get('category_path')), str(res.get('category_id')),
# #             f"{float(res.get('rerank_score', 0)):.4f}", " | ".join(res.get('logic_log', [])),
# #             status, top_text
# #         )
# #     except Exception as e:
# #         print(f"Error: {e}")
# #         return ("Error",) * 7

# # def process_batch_csv(file_obj):
# #     """
# #     Reads uploaded CSV, classifies each row, and returns a new CSV with top 5 results.
# #     """
# #     if file_obj is None:
# #         return None
    
# #     try:
# #         print("📂 Processing Batch CSV...")
# #         df = pd.read_csv(file_obj.name, dtype=str).fillna("")
        
# #         # Identify Title/Description columns
# #         cols = [c.lower() for c in df.columns]
# #         title_col = df.columns[0] # Default to 1st col
# #         desc_col = df.columns[1] if len(df.columns) > 1 else None # Default to 2nd col

# #         if 'title' in cols: title_col = df.columns[cols.index('title')]
# #         if 'description' in cols: desc_col = df.columns[cols.index('description')]
        
# #         results_data = []
        
# #         for idx, row in df.iterrows():
# #             t = str(row[title_col])
# #             d = str(row[desc_col]) if desc_col else ""
            
# #             # Run Classifier
# #             res = classify_product(t, d)
# #             top_5 = res.get('top_results', [])
            
# #             # Build Row
# #             out_row = {
# #                 "Original_Title": t,
# #                 "Original_Description": d,
# #                 "Best_Match_ID": res.get('category_id'),
# #                 "Best_Match_Path": res.get('category_path'),
# #                 "Best_Match_Score": res.get('rerank_score')
# #             }
            
# #             # Add Top 5 Details
# #             for i in range(5):
# #                 if i < len(top_5):
# #                     item = top_5[i]
# #                     out_row[f"Rank_{i+1}_Path"] = item['category_path']
# #                     out_row[f"Rank_{i+1}_ID"] = item['category_id']
# #                     out_row[f"Rank_{i+1}_Score"] = item.get('rerank_score', 0)
# #                     out_row[f"Rank_{i+1}_Logic"] = " | ".join(item.get('logic_log', []))
# #                 else:
# #                     out_row[f"Rank_{i+1}_Path"] = ""
# #                     out_row[f"Rank_{i+1}_ID"] = ""
# #                     out_row[f"Rank_{i+1}_Score"] = ""
# #                     out_row[f"Rank_{i+1}_Logic"] = ""
            
# #             results_data.append(out_row)
        
# #         # Save to File
# #         out_df = pd.DataFrame(results_data)
# #         out_path = "batch_results.csv"
# #         out_df.to_csv(out_path, index=False)
# #         print(f"✅ Batch processing done. Saved to {out_path}")
# #         return out_path
    
# #     except Exception as e:
# #         print(f"❌ Batch Error: {e}")
# #         return None

# # def main():
# #     initialize()
# #     with gr.Blocks() as app:
# #         gr.Markdown("# ⚡ Precision Hybrid Classifier")
        
# #         with gr.Tabs():
# #             # TAB 1: SINGLE PREDICTION
# #             with gr.TabItem("Single Prediction"):
# #                 with gr.Row():
# #                     with gr.Column():
# #                         t_in = gr.Textbox(label="Title")
# #                         d_in = gr.Textbox(label="Description")
# #                         btn = gr.Button("Classify", variant="primary")
# #                     with gr.Column():
# #                         out_final = gr.Textbox(label="Winner")
# #                         out_path = gr.Textbox(label="Path")
# #                         out_id = gr.Textbox(label="ID")
# #                         out_score = gr.Textbox(label="Score")
# #                         out_logic = gr.Textbox(label="Logic")
# #                         out_status = gr.Textbox(label="Status")
# #                 out_details = gr.TextArea(label="Top 15 Details", lines=10)
# #                 btn.click(gradio_classify, [t_in, d_in], [out_final, out_path, out_id, out_score, out_logic, out_status, out_details])

# #             # TAB 2: BATCH CSV PROCESSING
# #             with gr.TabItem("Batch Prediction (CSV)"):
# #                 gr.Markdown("Upload a CSV file containing `Title` and `Description` columns. The system will generate a CSV with the **Top 5 Predictions** for every row.")
# #                 file_in = gr.File(label="Upload CSV")
# #                 btn_batch = gr.Button("Process Batch", variant="primary")
# #                 file_out = gr.File(label="Download Results")
                
# #                 btn_batch.click(process_batch_csv, inputs=file_in, outputs=file_out)

# #     app.launch(server_name="127.0.0.1", server_port=7860, share=True)

# # if __name__ == "__main__":
# #     main()

# #!/usr/bin/env python3
# """
# gradio_app.py - The "Universal" Hybrid Classifier (Soap & Everything Else Fixed)

# UPDATES:
# 1. ADDED: 'check_universal_match' -> Matches ANY word in Title to Leaf Category.
#    (This fixes "Soap", "Hammer", "Laptop" automatically).
# 2. RETAINED: All previous Clothing/Kitchen guardrails for ambiguous items.
# """

# import os
# import json
# import pickle
# import re
# import time
# from pathlib import Path
# from typing import List, Dict, Tuple, Optional
# import pandas as pd
# import faiss
# import gradio as gr
# import torch
# import numpy as np
# from sentence_transformers import SentenceTransformer, CrossEncoder

# # -------------------------
# # ⚙️ CONFIGURATION
# # -------------------------
# CACHE_DIR = Path("cache")
# DATA_DIR = Path("data")
# CACHE_DIR.mkdir(exist_ok=True)
# DATA_DIR.mkdir(exist_ok=True)

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

# try:
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# except Exception:
#     DEVICE = "cpu"

# print(f"⚡ Running on: {DEVICE.upper()}")

# # -------------------------
# # 🌍 DOMAIN RULES (SPECIFIC)
# # -------------------------
# # These handle TRICKY items (ambiguous). 
# # For everything else (Soap, Electronics), the 'Universal Match' handles it.
# CLOTHING_KEYWORDS = {"t-shirt", "tshirt", "tee", "shirt", "top", "hoodie", "sweatshirt", "jacket", "coat", "pant", "jeans", "dress", "legging", "sock", "underwear", "bra", "apparel", "garment"}
# KITCHEN_KEYWORDS = {"salt & pepper", "shaker", "mug", "cup", "bowl", "plate", "dish", "glass", "cookware", "pan", "pot", "spatula", "ladle", "knife", "fork", "spoon"}
# PHONE_KEYWORDS = {"smartphone", "iphone", "samsung", "galaxy", "pixel", "oneplus", "motorola", "nokia", "sony", "lg", "xiaomi", "redmi", "oppo", "vivo", "realme", "huawei"}
# CARRIER_KEYWORDS = {"carrier", "locked", "prepaid", "postpaid", "att", "verizon", "t-mobile", "sprint", "cricket", "tracfone", "mint mobile"}
# AUDIO_KEYWORDS = {"headphone", "earphone", "earbud", "headset", "airpod", "galaxy buds", "noise cancelling", "speaker", "soundbar"}
# LAUNDRY_GENERICS = {"washing machine", "washer", "clothes washer", "washer dryer"}
# LAUNDRY_BRANDS = {"godrej", "lg", "samsung", "whirlpool", "bosch", "ifb", "haier", "panasonic"}
# ACCESSORY_KEYWORD_TO_ID = {
#     "flip case": "9931389011", "armband": "7073962011", "holster": "2407765011",
#     "sleeve": "9414313011", "bumper": "17875442011", "dry bag": "17875443011",
#     "case": "3081461011", "cover": "3081461011", "screen protector": "3081461011",
#     "tempered glass": "3081461011"
# }
# BOOK_KEYWORDS = {"book", "guide", "manual", "ebook", "pdf", "kindle", "paperback"}

# # Critical IDs
# WASHING_MACHINE_ID = "2383576011"
# UNLOCKED_CELL_PHONES_ID = "2407749011"
# IPHONE_BOOKS_ID = "6133978011"
# CARRIER_PHONES_ID = "2407748011"
# HEADPHONES_ID = "172541" 
# LAUNDRY_SUPPLIES_ID = "15356111"
# TRASH_BAGS_ID = "15342971"

# # -------------------------
# # Globals
# # -------------------------
# retriever_e5 = None
# retriever_mpnet = None
# reranker = None
# index_e5 = None
# index_mpnet = None
# metadata: List[Dict] = []
# catid_to_meta: Dict[str, Dict] = {} 
# tag_lookup: Dict[str, List[str]] = {} 

# def clean_text(text: str) -> str:
#     if not text: return ""
#     s = str(text).lower().strip()
#     s = re.sub(r"[^\w\s\-]", " ", s, flags=re.UNICODE)
#     return re.sub(r"\s+", " ", s).strip()

# def get_final_product_name(path: str) -> str:
#     if not path: return ""
#     parts = [p for p in path.split('/') if p]
#     return parts[-1].strip() if parts else path.strip()

# # -------------------------
# # Data Loading
# # -------------------------
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
#     col_id = df.columns[0]
#     col_path = df.columns[1] if len(df.columns) > 1 else df.columns[0]

#     for _, row in df.iterrows():
#         cid = str(row[col_id]).strip()
#         path = str(row[col_path]).strip()
#         if not cid or not path: continue
#         rows.append({
#             "category_id": cid,
#             "category_path": path,
#             "final_product": get_final_product_name(path),
#             "depth": len(path.split('/')),
#             "indexed_text": f"passage: {path}"
#         })
#     return rows

# def load_data():
#     global metadata, catid_to_meta, tag_lookup
#     if not CSV_PATH.exists(): raise FileNotFoundError("categories.csv missing!")
#     print("📊 Building metadata...")
#     metadata = build_metadata_from_csv(CSV_PATH)
#     catid_to_meta = {m['category_id']: m for m in metadata}
#     with open(METADATA_PATH, 'wb') as f: pickle.dump(metadata, f)

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
#             print(f"✅ Loaded {len(tag_lookup)} tags.")
#         except Exception as e: print(f"⚠️ Tag Load Error: {e}")

# # -------------------------
# # Rule Engine
# # -------------------------
# def get_rule_match_id(title: str, description: str = "") -> Tuple[Optional[str], str]:
#     txt = clean_text(title) 
#     if "iphone" in txt and any(k in txt for k in BOOK_KEYWORDS): return IPHONE_BOOKS_ID, "Rule: iPhone Book"
#     for phrase, cid in ACCESSORY_KEYWORD_TO_ID.items():
#         if re.search(r"\b" + re.escape(phrase) + r"\b", txt): return cid, f"Rule: Accessory '{phrase}'"
    
#     is_laundry = False
#     for kw in LAUNDRY_GENERICS:
#         if re.search(r"\b" + re.escape(kw) + r"\b", txt): is_laundry = True; break
#     if not is_laundry:
#         for brand in LAUNDRY_BRANDS:
#             if brand in txt and ("washing" in txt or "washer" in txt): is_laundry = True; break
#     if is_laundry:
#         if not any(a in txt for a in ["cover", "stand", "cleaner", "powder", "part"]): return WASHING_MACHINE_ID, "Rule: Appliance Match"

#     for kw in AUDIO_KEYWORDS:
#         if re.search(r"\b" + re.escape(kw) + r"\b", txt):
#             if "case" not in txt and "cover" not in txt: return HEADPHONES_ID, f"Rule: Audio '{kw}'"
            
#     if any(k in txt for k in CARRIER_KEYWORDS) and any(p in txt for p in PHONE_KEYWORDS): return CARRIER_PHONES_ID, "Rule: Carrier Phone"
#     for brand in PHONE_KEYWORDS:
#         if re.search(r"\b" + re.escape(brand) + r"\b", txt):
#             if not any(a in txt for a in ["battery", "charger", "cable", "case", "screen"]): return UNLOCKED_CELL_PHONES_ID, f"Rule: Brand '{brand.title()}'"
#     return None, ""

# # -------------------------
# # Guardrails & Universal Matcher
# # -------------------------
# def has_word(text, word_set):
#     text_lower = clean_text(text)
#     for w in word_set:
#         if re.search(r'\b' + re.escape(w) + r'\b', text_lower): return True
#     return False

# def check_universal_match(title, path):
#     """
#     UNIVERSAL LOGIC: 
#     If a word in the Title (e.g. 'Soap') exists in the Category Leaf (e.g. 'Bar Soap'),
#     give it a huge boost. This works for ALL products.
#     """
#     t_words = set(clean_text(title).split())
#     # Extract the last part of the path (Leaf)
#     leaf_words = set(clean_text(path.split('/')[-1]).split())
    
#     # Common words to ignore
#     ignore = {"and", "for", "with", "the", "set", "pack", "kit", "size", "color", "men", "women"}
    
#     # Find overlap
#     overlap = t_words.intersection(leaf_words) - ignore
    
#     if len(overlap) > 0:
#         # Boost score: 40 points per matching word
#         return 40.0 * len(overlap), [f"✅ Universal Match: {overlap}"]
    
#     return 0.0, []

# def check_guardrails(title, path):
#     score = 0.0
#     log = []
#     p = path.lower()
    
#     # Universal Match (The Magic Logic)
#     univ_score, univ_log = check_universal_match(title, path)
#     score += univ_score
#     log.extend(univ_log)

#     # Specific Guardrails (For tricky items)
#     if has_word(title, CLOTHING_KEYWORDS):
#         if "clothing" in p or "apparel" in p: score += 50.0; log.append("✅ Clothing Rule")
#         elif "food" in p or "toy" in p: 
#             if "doll" not in clean_text(title): score -= 500.0; log.append("⛔ Not Clothing")
            
#     if has_word(title, KITCHEN_KEYWORDS):
#         if "kitchen" in p or "dining" in p: score += 40.0; log.append("✅ Kitchen Rule")
        
#     return score, log

# # -------------------------
# # Classification
# # -------------------------
# def classify_product(title: str, description: str = "") -> Dict:
#     full_text = f"{title} {description}".strip()
#     candidates_map = {}
    
#     # Retrievers
#     e5_emb = retriever_e5.encode(f"passage: {full_text}", convert_to_numpy=True, normalize_embeddings=True)
#     dists_e5, idxs_e5 = index_e5.search(e5_emb.reshape(1, -1).astype('float32'), 30)
#     for i, idx in enumerate(idxs_e5[0]):
#         if idx >= 0: add_candidate(candidates_map, int(idx), float(dists_e5[0][i]), 'E5')

#     mp_emb = retriever_mpnet.encode(full_text, convert_to_numpy=True, normalize_embeddings=True)
#     dists_mp, idxs_mp = index_mpnet.search(mp_emb.reshape(1, -1).astype('float32'), 30)
#     for i, idx in enumerate(idxs_mp[0]):
#         if idx >= 0: add_candidate(candidates_map, int(idx), float(dists_mp[0][i]), 'MPNet')

#     # Tags
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
    
#     # Apply Universal & Specific Guardrails
#     for res in candidates:
#         if len(res['sources']) > 1: res['confidence'] += 5.0
#         res['confidence'] += min(8.0, res['depth'] * 1.5)
#         g_score, g_log = check_guardrails(title, res['category_path'])
#         res['confidence'] += g_score
#         res['logic_log'].extend(g_log)

#     candidates.sort(key=lambda x: x['confidence'], reverse=True)
#     top_candidates = candidates[:30]
    
#     if reranker:
#         rerank_inputs = [[title, c['category_path']] for c in top_candidates]
#         scores = reranker.predict(rerank_inputs)
#         for i, score in enumerate(scores): top_candidates[i]['rerank_score'] = float(score)
#         top_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)

#     rule_id, rule_reason = get_rule_match_id(title, description)
#     final_top_results = []
    
#     if rule_id and rule_id in catid_to_meta:
#         meta = catid_to_meta[rule_id]
#         rule_winner = {
#             'category_id': rule_id,
#             'category_path': meta['category_path'],
#             'final_product': meta['final_product'],
#             'rerank_score': 99.99,
#             'sources': {'Rule'},
#             'logic_log': [f"🏆 {rule_reason}"]
#         }
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

# def add_candidate(cand_map, idx, raw_score, source):
#     meta = metadata[idx]
#     add_candidate_manual(cand_map, meta, raw_score, source)

# def add_candidate_manual(cand_map, meta, raw_score, source):
#     cid = meta['category_id']
#     if cid in cand_map:
#         cand_map[cid]['sources'].add(source)
#         cand_map[cid]['retrieval_score'] = max(cand_map[cid]['retrieval_score'], float(raw_score))
#     else:
#         cand_map[cid] = {
#             'category_id': cid,
#             'category_path': meta['category_path'],
#             'final_product': meta['final_product'],
#             'depth': meta['depth'],
#             'retrieval_score': float(raw_score),
#             'confidence': float(raw_score) * 100.0,
#             'sources': {source},
#             'logic_log': []
#         }

# def build_index(model, model_name, items, index_path, use_passage_prefix=False):
#     print(f"🔨 Building Index for {model_name}...")
#     texts = [it['indexed_text'] if use_passage_prefix else it['category_path'] for it in items]
#     embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
#     index = faiss.IndexFlatIP(embeddings.shape[1])
#     index.add(np.asarray(embeddings, dtype='float32'))
#     faiss.write_index(index, str(index_path))
#     return index

# def initialize():
#     global retriever_e5, retriever_mpnet, reranker, index_e5, index_mpnet, metadata, catid_to_meta
#     print(f"🚀 Initializing Engine (device={DEVICE})...")
#     retriever_e5 = SentenceTransformer(MODEL_NAME_E5, device=DEVICE)
#     retriever_mpnet = SentenceTransformer(MODEL_NAME_MPNET, device=DEVICE)
#     try:
#         reranker = CrossEncoder(MODEL_NAME_RERANKER, device=DEVICE)
#     except:
#         reranker = CrossEncoder(MODEL_NAME_RERANKER, device='cpu')
#     load_data() 
#     if not INDEX_PATH_E5.exists():
#         index_e5 = build_index(retriever_e5, 'e5', metadata, INDEX_PATH_E5, True)
#         index_mpnet = build_index(retriever_mpnet, 'mpnet', metadata, INDEX_PATH_MPNET, False)
#     else:
#         index_e5 = faiss.read_index(str(INDEX_PATH_E5))
#         index_mpnet = faiss.read_index(str(INDEX_PATH_MPNET))
#     catid_to_meta = {m['category_id']: m for m in metadata}
#     print("✅ System Ready.")

# # --- BATCH UI LOGIC ---
# def get_file_path(file_obj):
#     if file_obj is None: return None
#     if isinstance(file_obj, str): return file_obj 
#     if hasattr(file_obj, 'name'): return file_obj.name
#     return str(file_obj)

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
#         raise gr.Error(f"Could not read file. Is 'openpyxl' installed? Error: {e}")

# def analyze_uploaded_csv(file_obj):
#     path = get_file_path(file_obj)
#     if not path: return pd.DataFrame(), gr.update(choices=[]), gr.update(choices=[])
#     try:
#         df = read_file_robust(path)
#         cols = list(df.columns)
#         default_title = "product_name" if "product_name" in cols else cols[0]
#         default_desc = "description" if "description" in cols else (cols[1] if len(cols)>1 else cols[0])
#         return df.head(5), gr.update(choices=cols, value=default_title), gr.update(choices=cols, value=default_desc)
#     except Exception as e:
#         raise gr.Error(f"Error reading file: {e}")

# def process_batch_csv(file_obj, title_col, desc_col, limit_choice, progress=gr.Progress()):
#     path = get_file_path(file_obj)
#     if not path: raise gr.Error("No file uploaded.")
    
#     df = read_file_robust(path)
#     if title_col not in df.columns: raise gr.Error(f"Column '{title_col}' not found in file.")
    
#     # Parse Custom Limit
#     try:
#         limit_str = str(limit_choice).split()[0]
#         limit = int(limit_str)
#     except:
#         limit = 5
    
#     results_data = []
#     print(f"🚀 Starting Batch Processing (Top {limit})...")
    
#     for idx, row in progress.tqdm(df.iterrows(), total=len(df), desc="Classifying"):
#         t = str(row[title_col])
#         d = str(row[desc_col]) if desc_col and desc_col in df.columns else ""
        
#         res = classify_product(t, d)
#         top_N = res.get('top_results', [])[:limit]
        
#         # 1. Start with ORIGINAL ROW data
#         out_row = row.to_dict()
        
#         # 2. Add Best Match
#         out_row["Best_Match_ID"] = str(res.get('category_id'))
#         out_row["Best_Match_Path"] = str(res.get('category_path'))
#         out_row["Best_Match_Score"] = res.get('rerank_score', 0)
        
#         # 3. Add Dynamic Top N Columns
#         for i in range(limit):
#             if i < len(top_N):
#                 item = top_N[i]
#                 out_row[f"Rank_{i+1}_Path"] = str(item.get('category_path'))
#                 out_row[f"Rank_{i+1}_ID"] = str(item.get('category_id'))
#                 out_row[f"Rank_{i+1}_Score"] = item.get('rerank_score', 0)
#             else:
#                 out_row[f"Rank_{i+1}_Path"] = ""
#                 out_row[f"Rank_{i+1}_ID"] = ""
#                 out_row[f"Rank_{i+1}_Score"] = ""
#         results_data.append(out_row)
    
#     out_df = pd.DataFrame(results_data)
#     out_path = "batch_results.csv"
#     out_df.to_csv(out_path, index=False)
#     print("✅ Batch Done.")
#     return out_path, out_df.head(10)

# def gradio_classify_ui(title, desc):
#     """Bridge for Single Prediction UI."""
#     res = classify_product(title, desc)
#     top_text = ""
#     for i, item in enumerate(res.get('top_results', []), 1):
#         logs = " | ".join(item.get('logic_log', []))
#         top_text += f"{i}. {item['final_product']}\n   Score: {item.get('rerank_score',0):.4f}\n   Path: {item['category_path']}\n   Logic: {logs}\n\n"
#     status = "✅ Approved" if res.get('rerank_score', 0) > 0 else "⚠️ Review"
#     return str(res.get('final_product')), str(res.get('category_path')), str(res.get('category_id')), f"{res.get('rerank_score',0):.4f}", " | ".join(res.get('logic_log',[])), status, top_text

# def main():
#     initialize()
#     with gr.Blocks() as app:
#         gr.Markdown("# ⚡ Precision Hybrid Classifier")
#         with gr.Tabs():
#             with gr.TabItem("Single Prediction"):
#                 with gr.Row():
#                     t_in = gr.Textbox(label="Title")
#                     d_in = gr.Textbox(label="Description")
#                     btn = gr.Button("Classify", variant="primary")
#                 out_res = [gr.Textbox(label=l) for l in ["Winner", "Path", "ID", "Score", "Logic", "Status"]]
#                 out_det = gr.TextArea(label="Details")
#                 btn.click(gradio_classify_ui, [t_in, d_in], out_res + [out_det])

#             with gr.TabItem("Batch Prediction (CSV/Excel)"):
#                 with gr.Row():
#                     file_in = gr.File(label="Upload File")
#                     with gr.Column():
#                         df_prev = gr.DataFrame(label="Preview", interactive=False)
#                         c_title = gr.Dropdown(label="Title Column", allow_custom_value=True)
#                         c_desc = gr.Dropdown(label="Desc Column", allow_custom_value=True)
#                         c_limit = gr.Dropdown(label="Result Limit (Type custom number or select)", 
#                                               choices=["1 Result", "5 Results", "10 Results", "20 Results"], 
#                                               value="5 Results", allow_custom_value=True)
                
#                 file_in.upload(analyze_uploaded_csv, file_in, [df_prev, c_title, c_desc])
#                 btn_run = gr.Button("🚀 Process Batch", variant="primary")
#                 with gr.Row():
#                     f_out = gr.File(label="Download Results")
#                 df_out = gr.DataFrame(label="Results Preview")
#                 btn_run.click(process_batch_csv, [file_in, c_title, c_desc, c_limit], [f_out, df_out])

#     app.launch(server_name="127.0.0.1", server_port=7860, share=True)

# if __name__ == "__main__":
#     main()



#final fixed gradio

#!/usr/bin/env python3
"""
UPDATES:
1. FIXED: Added 'teen', 'young', 'adult' to Ignore List (Stops false matches to ID 28).
2. RETAINED: All previous Universal Logic & Excel support.
"""

import os
import json
import pickle
import re
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import faiss
import gradio as gr
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# -------------------------
# ⚙️ CONFIGURATION
# -------------------------
CACHE_DIR = Path("cache")
DATA_DIR = Path("data")
CACHE_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Models
MODEL_NAME_E5 = "intfloat/e5-base-v2"
MODEL_NAME_MPNET = "sentence-transformers/all-mpnet-base-v2"
MODEL_NAME_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2" 

# Paths
CSV_PATH = DATA_DIR / "categories.csv"
TAGS_PATH = DATA_DIR / "tags.json"
INDEX_PATH_E5 = CACHE_DIR / "index_e5.faiss"
INDEX_PATH_MPNET = CACHE_DIR / "index_mpnet.faiss"
METADATA_PATH = CACHE_DIR / "metadata.pkl"

try:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

print(f"⚡ Running on: {DEVICE.upper()}")

# -------------------------
# 🌍 DOMAIN RULES
# -------------------------
CLOTHING_KEYWORDS = {"t-shirt", "tshirt", "tee", "shirt", "top", "hoodie", "sweatshirt", "jacket", "coat", "pant", "jeans", "dress", "legging", "sock", "underwear", "bra", "apparel", "garment"}
KITCHEN_KEYWORDS = {"salt & pepper", "shaker", "mug", "cup", "bowl", "plate", "dish", "glass", "cookware", "pan", "pot", "spatula", "ladle", "knife", "fork", "spoon"}
BEAUTY_KEYWORDS = {"perfume", "parfum", "eau de", "cologne", "fragrance", "scent", "spray", "mist", "lotion", "cream", "moisturizer", "serum", "makeup", "lipstick", "cosmetic"}
HOME_KEYWORDS = {"blanket", "throw", "quilt", "duvet", "pillow", "candle", "wax", "tapestry", "poster", "print", "art", "decor", "vase", "rug", "mat"}
JEWELRY_KEYWORDS = {"pin", "enamel pin", "brooch", "badge", "keychain", "keyring", "charm", "necklace", "earring", "bracelet", "ring", "jewelry"}
FOOD_KEYWORDS = {"chocolate", "candy", "gummy", "snack", "sauce", "spice", "oil", "tea", "coffee", "drink", "beverage", "food", "grocery"}

PHONE_KEYWORDS = {"smartphone", "iphone", "samsung", "galaxy", "pixel", "oneplus", "motorola", "nokia", "sony", "lg", "xiaomi", "redmi", "oppo", "vivo", "realme", "huawei"}
CARRIER_KEYWORDS = {"carrier", "locked", "prepaid", "postpaid", "att", "verizon", "t-mobile", "sprint", "cricket", "tracfone", "mint mobile"}
AUDIO_KEYWORDS = {"headphone", "earphone", "earbud", "headset", "airpod", "galaxy buds", "noise cancelling", "speaker", "soundbar"}
LAUNDRY_GENERICS = {"washing machine", "washer", "clothes washer", "washer dryer"}
LAUNDRY_BRANDS = {"godrej", "lg", "samsung", "whirlpool", "bosch", "ifb", "haier", "panasonic"}
ACCESSORY_KEYWORD_TO_ID = {
    "flip case": "9931389011", "armband": "7073962011", "holster": "2407765011",
    "sleeve": "9414313011", "bumper": "17875442011", "dry bag": "17875443011",
    "case": "3081461011", "cover": "3081461011", "screen protector": "3081461011",
    "tempered glass": "3081461011"
}
BOOK_KEYWORDS = {"book", "guide", "manual", "ebook", "pdf", "kindle", "paperback"}

# Critical IDs
WASHING_MACHINE_ID = "2383576011"
UNLOCKED_CELL_PHONES_ID = "2407749011"
IPHONE_BOOKS_ID = "6133978011"
CARRIER_PHONES_ID = "2407748011"
HEADPHONES_ID = "172541" 
LAUNDRY_SUPPLIES_ID = "15356111"
TRASH_BAGS_ID = "15342971"

# -------------------------
# Globals
# -------------------------
retriever_e5 = None
retriever_mpnet = None
reranker = None
index_e5 = None
index_mpnet = None
metadata: List[Dict] = []
catid_to_meta: Dict[str, Dict] = {} 
tag_lookup: Dict[str, List[str]] = {} 

def clean_text(text: str) -> str:
    if not text: return ""
    s = str(text).lower().strip()
    s = re.sub(r"[^\w\s\-]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()

def get_final_product_name(path: str) -> str:
    if not path: return ""
    parts = [p for p in path.split('/') if p]
    return parts[-1].strip() if parts else path.strip()

# -------------------------
# Data Loading
# -------------------------
def build_metadata_from_csv(csv_path: Path) -> List[Dict]:
    try:
        if str(csv_path).endswith('.xlsx'):
            df = pd.read_excel(csv_path, dtype=str).fillna("")
        else:
            try:
                df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, on_bad_lines='skip', encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, on_bad_lines='skip', encoding='latin1')
    except Exception as e:
        print(f"CRITICAL ERROR loading categories: {e}")
        return []
        
    rows = []
    col_id = df.columns[0]
    col_path = df.columns[1]

    for _, row in df.iterrows():
        cid = str(row[col_id]).strip()
        path = str(row[col_path]).strip()
        if not cid or not path: continue
        rows.append({
            "category_id": cid,
            "category_path": path,
            "final_product": get_final_product_name(path),
            "depth": len(path.split('/')),
            "indexed_text": f"passage: {path}"
        })
    return rows

def load_data():
    global metadata, catid_to_meta, tag_lookup
    if not CSV_PATH.exists(): raise FileNotFoundError("categories.csv missing!")
    print("📊 Building metadata...")
    metadata = build_metadata_from_csv(CSV_PATH)
    catid_to_meta = {m['category_id']: m for m in metadata}
    with open(METADATA_PATH, 'wb') as f: pickle.dump(metadata, f)

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
            print(f"✅ Loaded {len(tag_lookup)} tags.")
        except Exception as e: print(f"⚠️ Tag Load Error: {e}")

# -------------------------
# Rule Engine
# -------------------------
def get_rule_match_id(title: str, description: str = "") -> Tuple[Optional[str], str]:
    txt = clean_text(title) 
    if "iphone" in txt and any(k in txt for k in BOOK_KEYWORDS): return IPHONE_BOOKS_ID, "Rule: iPhone Book"
    for phrase, cid in ACCESSORY_KEYWORD_TO_ID.items():
        if re.search(r"\b" + re.escape(phrase) + r"\b", txt): return cid, f"Rule: Accessory '{phrase}'"
    
    is_laundry = False
    for kw in LAUNDRY_GENERICS:
        if re.search(r"\b" + re.escape(kw) + r"\b", txt): is_laundry = True; break
    if not is_laundry:
        for brand in LAUNDRY_BRANDS:
            if brand in txt and ("washing" in txt or "washer" in txt): is_laundry = True; break
    if is_laundry:
        if not any(a in txt for a in ["cover", "stand", "cleaner", "powder", "part"]): return WASHING_MACHINE_ID, "Rule: Appliance Match"

    for kw in AUDIO_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", txt):
            if "case" not in txt and "cover" not in txt: return HEADPHONES_ID, f"Rule: Audio '{kw}'"
            
    if any(k in txt for k in CARRIER_KEYWORDS) and any(p in txt for p in PHONE_KEYWORDS): return CARRIER_PHONES_ID, "Rule: Carrier Phone"
    for brand in PHONE_KEYWORDS:
        if re.search(r"\b" + re.escape(brand) + r"\b", txt):
            if not any(a in txt for a in ["battery", "charger", "cable", "case", "screen"]): return UNLOCKED_CELL_PHONES_ID, f"Rule: Brand '{brand.title()}'"
    return None, ""

# -------------------------
# Guardrails & Universal Matcher
# -------------------------
def has_word(text, word_set):
    text_lower = clean_text(text)
    for w in word_set:
        if re.search(r'\b' + re.escape(w) + r'\b', text_lower): return True
    return False

def simple_stem(w):
    return w.rstrip('s')

def check_universal_match(title, path):
    """Universal Matcher with expanded Ignore List."""
    t_words = clean_text(title).split()
    t_stems = {simple_stem(w) for w in t_words}
    leaf_words = clean_text(path.split('/')[-1]).split()
    leaf_stems = {simple_stem(w) for w in leaf_words}
    
    # FIX: Added 'teen', 'young', 'adult' to ignore list
    ignore = {
        "and", "for", "with", "the", "set", "pack", "kit", "size", "color", "men", "women", "of", "in",
        "control", "digital", "auto", "manual", "electric", "star", "drive", "design", "replacement",
        "part", "parts", "universal", "remote", "system", "quality", "cleaner", "wash",
        "teen", "young", "adult", "kid", "kids", "child", "children"
    }
    ignore_stems = {simple_stem(w) for w in ignore}
    
    overlap = t_stems.intersection(leaf_stems) - ignore_stems
    
    if len(overlap) > 0:
        return 40.0 * len(overlap), [f"✅ Univ Match: {overlap}"]
    
    return 0.0, []

def check_guardrails(title, path):
    score = 0.0
    log = []
    p = path.lower()
    
    # 1. Universal Match
    univ_score, univ_log = check_universal_match(title, path)
    score += univ_score
    log.extend(univ_log)

    # 2. Specific Guardrails
    if has_word(title, CLOTHING_KEYWORDS):
        if "clothing" in p or "apparel" in p: score += 50.0; log.append("✅ Clothing Rule")
        elif "food" in p or "toy" in p: 
            if "doll" not in clean_text(title): score -= 500.0; log.append("⛔ Not Clothing")
            
    if has_word(title, KITCHEN_KEYWORDS):
        if "kitchen" in p or "dining" in p: score += 40.0; log.append("✅ Kitchen Rule")

    if has_word(title, BEAUTY_KEYWORDS):
        if "beauty" in p or "care" in p: score += 40.0; log.append("✅ Beauty Rule")

    if has_word(title, HOME_KEYWORDS):
        if "home" in p or "decor" in p or "bedding" in p: score += 40.0; log.append("✅ Home Rule")

    if has_word(title, JEWELRY_KEYWORDS):
        if "jewelry" in p or "accessories" in p: score += 40.0; log.append("✅ Jewelry Rule")

    if has_word(title, FOOD_KEYWORDS):
        if "food" in p or "grocery" in p: score += 40.0; log.append("✅ Food Rule")
        elif "toy" in p or "electronic" in p: score -= 200.0; log.append("⛔ Not Food")

    return score, log

# -------------------------
# Classification
# -------------------------
def classify_product(title: str, description: str = "") -> Dict:
    full_text = f"{title} {description}".strip()
    candidates_map = {}
    
    # Retrievers
    e5_emb = retriever_e5.encode(f"passage: {full_text}", convert_to_numpy=True, normalize_embeddings=True)
    dists_e5, idxs_e5 = index_e5.search(e5_emb.reshape(1, -1).astype('float32'), 30)
    for i, idx in enumerate(idxs_e5[0]):
        if idx >= 0: add_candidate(candidates_map, int(idx), float(dists_e5[0][i]), 'E5')

    mp_emb = retriever_mpnet.encode(full_text, convert_to_numpy=True, normalize_embeddings=True)
    dists_mp, idxs_mp = index_mpnet.search(mp_emb.reshape(1, -1).astype('float32'), 30)
    for i, idx in enumerate(idxs_mp[0]):
        if idx >= 0: add_candidate(candidates_map, int(idx), float(dists_mp[0][i]), 'MPNet')

    # Tags
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
    
    # Guardrails
    for res in candidates:
        if len(res['sources']) > 1: res['confidence'] += 5.0
        res['confidence'] += min(8.0, res['depth'] * 1.5)
        g_score, g_log = check_guardrails(title, res['category_path'])
        res['confidence'] += g_score
        res['logic_log'].extend(g_log)

    candidates.sort(key=lambda x: x['confidence'], reverse=True)
    top_candidates = candidates[:30]
    
    if reranker:
        rerank_inputs = [[title, c['category_path']] for c in top_candidates]
        scores = reranker.predict(rerank_inputs)
        for i, score in enumerate(scores): top_candidates[i]['rerank_score'] = float(score)
        top_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)

    rule_id, rule_reason = get_rule_match_id(title, description)
    final_top_results = []
    
    if rule_id and rule_id in catid_to_meta:
        meta = catid_to_meta[rule_id]
        rule_winner = {
            'category_id': rule_id,
            'category_path': meta['category_path'],
            'final_product': meta['final_product'],
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
        'rerank_score': top.get('rerank_score', 0.0) if top else 0.0,
        'logic_log': top.get('logic_log', []) if top else [],
        'top_results': final_top_results
    }

def add_candidate(cand_map, idx, raw_score, source):
    meta = metadata[idx]
    add_candidate_manual(cand_map, meta, raw_score, source)

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
            'depth': meta['depth'],
            'retrieval_score': float(raw_score),
            'confidence': float(raw_score) * 100.0,
            'sources': {source},
            'logic_log': []
        }

def build_index(model, model_name, items, index_path, use_passage_prefix=False):
    print(f"🔨 Building Index for {model_name}...")
    texts = [it['indexed_text'] if use_passage_prefix else it['category_path'] for it in items]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.asarray(embeddings, dtype='float32'))
    faiss.write_index(index, str(index_path))
    return index

def initialize():
    global retriever_e5, retriever_mpnet, reranker, index_e5, index_mpnet, metadata, catid_to_meta
    print(f"🚀 Initializing Engine (device={DEVICE})...")
    retriever_e5 = SentenceTransformer(MODEL_NAME_E5, device=DEVICE)
    retriever_mpnet = SentenceTransformer(MODEL_NAME_MPNET, device=DEVICE)
    try:
        reranker = CrossEncoder(MODEL_NAME_RERANKER, device=DEVICE)
    except:
        reranker = CrossEncoder(MODEL_NAME_RERANKER, device='cpu')
    load_data() 
    if not INDEX_PATH_E5.exists():
        index_e5 = build_index(retriever_e5, 'e5', metadata, INDEX_PATH_E5, True)
        index_mpnet = build_index(retriever_mpnet, 'mpnet', metadata, INDEX_PATH_MPNET, False)
    else:
        index_e5 = faiss.read_index(str(INDEX_PATH_E5))
        index_mpnet = faiss.read_index(str(INDEX_PATH_MPNET))
    catid_to_meta = {m['category_id']: m for m in metadata}
    print("✅ System Ready.")

# --- BATCH UI LOGIC ---
def get_file_path(file_obj):
    if file_obj is None: return None
    if isinstance(file_obj, str): return file_obj 
    if hasattr(file_obj, 'name'): return file_obj.name
    return str(file_obj)

def read_file_robust(path):
    try:
        if path.endswith('.xlsx') or path.endswith('.xls'):
            return pd.read_excel(path, dtype=str).fillna("")
        try:
            return pd.read_csv(path, dtype=str, encoding='utf-8').fillna("")
        except UnicodeDecodeError:
            try:
                return pd.read_csv(path, dtype=str, encoding='latin1').fillna("")
            except:
                return pd.read_csv(path, dtype=str, encoding='cp1252').fillna("")
    except Exception as e:
        raise gr.Error(f"Could not read file. Is 'openpyxl' installed? Error: {e}")

def analyze_uploaded_csv(file_obj):
    path = get_file_path(file_obj)
    if not path: return pd.DataFrame(), gr.update(choices=[]), gr.update(choices=[])
    try:
        df = read_file_robust(path)
        cols = list(df.columns)
        default_title = "product_name" if "product_name" in cols else cols[0]
        default_desc = "description" if "description" in cols else (cols[1] if len(cols)>1 else cols[0])
        return df.head(5), gr.update(choices=cols, value=default_title), gr.update(choices=cols, value=default_desc)
    except Exception as e:
        raise gr.Error(f"Error reading file: {e}")

def process_batch_csv(file_obj, title_col, desc_col, limit_choice, progress=gr.Progress()):
    path = get_file_path(file_obj)
    if not path: raise gr.Error("No file uploaded.")
    
    df = read_file_robust(path)
    if title_col not in df.columns: raise gr.Error(f"Column '{title_col}' not found in file.")
    
    # Parse Custom Limit
    try:
        limit_str = str(limit_choice).split()[0]
        limit = int(limit_str)
    except:
        limit = 5
    
    results_data = []
    print(f"🚀 Starting Batch Processing (Top {limit})...")
    
    for idx, row in progress.tqdm(df.iterrows(), total=len(df), desc="Classifying"):
        t = str(row[title_col])
        d = str(row[desc_col]) if desc_col and desc_col in df.columns else ""
        
        res = classify_product(t, d)
        top_N = res.get('top_results', [])[:limit]
        
        # 1. Start with ORIGINAL ROW data
        out_row = row.to_dict()
        
        # 2. Add Best Match
        out_row["Best_Match_ID"] = str(res.get('category_id'))
        out_row["Best_Match_Path"] = str(res.get('category_path'))
        out_row["Best_Match_Score"] = res.get('rerank_score', 0)
        
        # 3. Add Dynamic Top N Columns
        for i in range(limit):
            if i < len(top_N):
                item = top_N[i]
                out_row[f"Rank_{i+1}_Path"] = str(item.get('category_path'))
                out_row[f"Rank_{i+1}_ID"] = str(item.get('category_id'))
                out_row[f"Rank_{i+1}_Score"] = item.get('rerank_score', 0)
            else:
                out_row[f"Rank_{i+1}_Path"] = ""
                out_row[f"Rank_{i+1}_ID"] = ""
                out_row[f"Rank_{i+1}_Score"] = ""
        results_data.append(out_row)
    
    out_df = pd.DataFrame(results_data)
    out_path = "batch_results.csv"
    out_df.to_csv(out_path, index=False)
    print("✅ Batch Done.")
    return out_path, out_df.head(10)

def gradio_classify_ui(title, desc):
    """Bridge for Single Prediction UI."""
    res = classify_product(title, desc)
    top_text = ""
    for i, item in enumerate(res.get('top_results', []), 1):
        logs = " | ".join(item.get('logic_log', []))
        top_text += f"{i}. {item['final_product']}\n   ID: {item.get('category_id', 'Unknown')}\n   Score: {item.get('rerank_score',0):.4f}\n   Path: {item['category_path']}\n   Logic: {logs}\n\n"
    status = "✅ Approved" if res.get('rerank_score', 0) > 0 else "⚠️ Review"
    return str(res.get('final_product')), str(res.get('category_path')), str(res.get('category_id')), f"{res.get('rerank_score',0):.4f}", " | ".join(res.get('logic_log',[])), status, top_text

def main():
    initialize()
    with gr.Blocks() as app:
        gr.Markdown("# ⚡ Precision Hybrid Classifier")
        with gr.Tabs():
            with gr.TabItem("Single Prediction"):
                with gr.Row():
                    t_in = gr.Textbox(label="Title")
                    d_in = gr.Textbox(label="Description")
                    btn = gr.Button("Classify", variant="primary")
                out_res = [gr.Textbox(label=l) for l in ["Winner", "Path", "ID", "Score", "Logic", "Status"]]
                out_det = gr.TextArea(label="Details")
                btn.click(gradio_classify_ui, [t_in, d_in], out_res + [out_det])

            with gr.TabItem("Batch Prediction (CSV/Excel)"):
                with gr.Row():
                    file_in = gr.File(label="Upload File")
                    with gr.Column():
                        df_prev = gr.DataFrame(label="Preview", interactive=False)
                        c_title = gr.Dropdown(label="Title Column", allow_custom_value=True)
                        c_desc = gr.Dropdown(label="Desc Column", allow_custom_value=True)
                        c_limit = gr.Dropdown(label="Result Limit (Type custom number or select)", 
                                              choices=["1 Result", "5 Results", "10 Results", "20 Results"], 
                                              value="5 Results", allow_custom_value=True)
                
                file_in.upload(analyze_uploaded_csv, file_in, [df_prev, c_title, c_desc])
                btn_run = gr.Button("🚀 Process Batch", variant="primary")
                with gr.Row():
                    f_out = gr.File(label="Download Results")
                df_out = gr.DataFrame(label="Results Preview")
                btn_run.click(process_batch_csv, [file_in, c_title, c_desc, c_limit], [f_out, df_out])

    app.launch(server_name="127.0.0.1", server_port=7860, share=True)

if __name__ == "__main__":
    main()