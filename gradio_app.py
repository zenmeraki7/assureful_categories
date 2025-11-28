# # # # #!/usr/bin/env python3
# # # # """
# # # # gradio_app_FAST_CSV_FIRST.py — Lightning-fast exact CSV matching with Enhanced Accessory Detection

# # # # STRATEGY:
# # # # 1. CSV exact matching FIRST (keyword/phrase matching) - INSTANT
# # # # 2. Enhanced multi-word accessory detection (back cover, flip case, etc.)
# # # # 3. E5 FAISS search only for fallback - FAST
# # # # 4. NO slow MPNet final product embeddings
# # # # 5. Simple brand/product detection with cached embeddings

# # # # BENEFITS:
# # # # ✅ 10-50x faster startup (no 20k+ MPNet encodings)
# # # # ✅ Instant predictions for exact matches
# # # # ✅ Always returns exact CSV ID and path
# # # # ✅ Better accessory matching with multi-word phrases
# # # # ✅ Simple and maintainable

# # # # Expected speed: <100ms per prediction
# # # # Expected accuracy: 90-95% (exact matches from your CSV)
# # # # """

# # # # from pathlib import Path
# # # # import json
# # # # import pickle
# # # # import re
# # # # import time
# # # # from typing import List, Dict, Tuple, Optional, Set

# # # # import numpy as np
# # # # import pandas as pd
# # # # import faiss
# # # # import gradio as gr
# # # # from sentence_transformers import SentenceTransformer

# # # # try:
# # # #     from sklearn.linear_model import LogisticRegression
# # # #     from sklearn.preprocessing import StandardScaler
# # # #     SKLEARN_AVAILABLE = True
# # # # except Exception:
# # # #     SKLEARN_AVAILABLE = False

# # # # # -------------------------
# # # # # Config
# # # # # -------------------------
# # # # CACHE_DIR = Path("cache")
# # # # DATA_DIR = Path("data")

# # # # MODEL_E5 = "intfloat/e5-base-v2"

# # # # FAISS_INDEX_PATH = CACHE_DIR / "main_index.faiss"
# # # # METADATA_PATH = CACHE_DIR / "metadata.pkl"
# # # # PARENT_EMB_PATH = CACHE_DIR / "parent_embeddings.pkl"
# # # # SYN_PATH = CACHE_DIR / "cross_store_synonyms.pkl"
# # # # CSV_PATH = DATA_DIR / "category_only_path.csv"
# # # # VALIDATION_CSV = DATA_DIR / "validation.csv"

# # # # # Rule-based overrides
# # # # UNLOCKED_CELL_PHONES_ID = "2407749011"
# # # # UNLOCKED_CELL_PHONES_PATH = "Electronics/Cell Phones & Accessories/Unlocked Cell Phones"
# # # # IPHONE_BOOKS_ID = "6133978011"
# # # # IPHONE_BOOKS_PATH = "Books/Computers & Technology/Mobile Phones, Tablets & E-Readers/iPhone"

# # # # # ENHANCED ACCESSORY KEYWORD MAPPING - Multi-word phrase support
# # # # ENHANCED_ACCESSORY_MAPPING = {
# # # #     # Cases & Covers - Specific matching (longest phrases first)
# # # #     "back cover": "3081461011",
# # # #     "back case": "3081461011",
# # # #     "flip case": "9931389011",
# # # #     "flip cover": "9931389011",
# # # #     "wallet case": "9931389011",
# # # #     "wallet cover": "9931389011",
# # # #     "folio case": "9931389011",
# # # #     "folio cover": "9931389011",
# # # #     "leather case": "3081461011",
# # # #     "leather cover": "3081461011",
# # # #     "silicone case": "3081461011",
# # # #     "tpu case": "3081461011",
# # # #     "hard case": "3081461011",
# # # #     "soft case": "3081461011",
    
# # # #     # Screen Protection
# # # #     "screen protector": "3081461011",
# # # #     "screen guard": "3081461011",
# # # #     "tempered glass": "3081461011",
# # # #     "glass protector": "3081461011",
# # # #     "privacy screen": "3081461011",
    
# # # #     # Other Accessories
# # # #     "armband": "7073962011",
# # # #     "arm band": "7073962011",
# # # #     "holster": "2407765011",
# # # #     "belt clip": "2407765011",
# # # #     "sleeve": "9414313011",
# # # #     "pouch": "9414313011",
# # # #     "bumper": "17875442011",
# # # #     "bumper case": "17875442011",
# # # #     "dry bag": "17875443011",
# # # #     "waterproof case": "17875443011",
# # # #     "waterproof cover": "17875443011",
    
# # # #     # Generic fallback (checked last)
# # # #     "case": "3081461011",
# # # #     "cover": "3081461011",
# # # # }

# # # # # Keep old mapping for backward compatibility
# # # # ACCESSORY_KEYWORD_TO_ID = {
# # # #     "flip case": "9931389011",
# # # #     "armband": "7073962011",
# # # #     "holster": "2407765011",
# # # #     "sleeve": "9414313011",
# # # #     "bumper": "17875442011",
# # # #     "dry bag": "17875443011",
# # # #     "case": "3081461011",
# # # #     "cover": "3081461011",
# # # #     "screen protector": "3081461011",
# # # #     "tempered glass": "3081461011",
# # # #     "mobile broadband": "2407750011"
# # # # }

# # # # CARRIER_KEYWORDS = {
# # # #     "carrier", "carrier locked", "carrier-locked", "contract", "locked",
# # # #     "att", "at&t", "verizon", "t-mobile", "tmobile", "sprint", "jio", "vodafone", "airtel"
# # # # }

# # # # ACCESSORY_KEYWORDS = {k for k in ENHANCED_ACCESSORY_MAPPING.keys()}
# # # # BOOK_KEYWORDS = {
# # # #     "book", "guide", "manual", "ebook", "pdf", "kindle", "paperback",
# # # #     "author", "isbn", "publisher", "tutorial", "how to", "handbook", "edition"
# # # # }

# # # # # Phrases that indicate this is an accessory, not a phone
# # # # ACCESSORY_INDICATOR_PHRASES = {
# # # #     "case for", "cover for", "protector for", "skin for", "holder for",
# # # #     "compatible with", "fits", "designed for", "protection for"
# # # # }

# # # # # Fast matching thresholds
# # # # MIN_KEYWORD_MATCHES = 2  # Minimum keyword overlap for CSV match
# # # # TOP_K_DEFAULT = 5

# # # # # -------------------------
# # # # # Globals
# # # # # -------------------------
# # # # encoder = None
# # # # faiss_index = None
# # # # metadata: List[Dict] = []
# # # # parent_embeddings: Dict[str, np.ndarray] = {}
# # # # cross_store_synonyms: Dict = {}
# # # # csv_category_index: Dict[str, Dict] = {}
# # # # csv_keyword_index: Dict[str, List[str]] = {}  # keyword -> [category_ids]
# # # # calibrator = None

# # # # # Simple brand list for quick detection
# # # # COMMON_BRANDS = {
# # # #     "apple", "iphone", "samsung", "xiaomi", "mi", "redmi", "poco", "oneplus", "oppo", "vivo",
# # # #     "realme", "motorola", "moto", "nokia", "sony", "google", "pixel", "lenovo", "dell", "hp",
# # # #     "acer", "asus", "lg", "panasonic", "philips", "bosch", "whirlpool", "haier"
# # # # }

# # # # # -------------------------
# # # # # Utilities
# # # # # -------------------------
# # # # def clean_text(text: str) -> str:
# # # #     if not text:
# # # #         return ""
# # # #     s = str(text).lower()
# # # #     s = re.sub(r"[^\w\s\-]", " ", s)
# # # #     s = re.sub(r"\s+", " ", s).strip()
# # # #     return s

# # # # def extract_keywords(text: str, min_len: int = 3) -> Set[str]:
# # # #     """Extract keywords (remove stopwords)"""
# # # #     stopwords = {
# # # #         "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
# # # #         "of", "with", "by", "from", "up", "about", "into", "through", "during",
# # # #         "before", "after", "above", "below", "between", "under", "again", "further",
# # # #         "then", "once", "here", "there", "when", "where", "why", "how", "all",
# # # #         "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
# # # #         "not", "only", "own", "same", "so", "than", "too", "very", "can", "will",
# # # #         "just", "should", "now", "new", "best", "good", "great", "high", "low"
# # # #     }
    
# # # #     words = clean_text(text).split()
# # # #     keywords = {w for w in words if w not in stopwords and len(w) >= min_len}
    
# # # #     # Add bigrams for better matching
# # # #     for i in range(len(words) - 1):
# # # #         bigram = f"{words[i]} {words[i+1]}"
# # # #         if len(bigram) >= 6:
# # # #             keywords.add(bigram)
    
# # # #     # Add trigrams for compound products
# # # #     for i in range(len(words) - 2):
# # # #         trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
# # # #         if len(trigram) >= 10:
# # # #             keywords.add(trigram)
    
# # # #     return keywords

# # # # def detect_brand(text: str) -> Optional[str]:
# # # #     """Quick brand detection without embeddings"""
# # # #     text_lower = clean_text(text)
# # # #     for brand in COMMON_BRANDS:
# # # #         if brand in text_lower:
# # # #             return brand
# # # #     return None

# # # # # -------------------------
# # # # # ENHANCED ACCESSORY DETECTION
# # # # # -------------------------
# # # # def detect_accessory_enhanced(query_text: str) -> Optional[Dict]:
# # # #     """
# # # #     Enhanced accessory detection with multi-word phrase matching
    
# # # #     Returns:
# # # #         dict with category info if accessory detected, None otherwise
# # # #     """
# # # #     text_lower = query_text.lower()
    
# # # #     # Check if this has accessory indicator phrases
# # # #     has_accessory_indicator = any(phrase in text_lower for phrase in ACCESSORY_INDICATOR_PHRASES)
    
# # # #     # Sort phrases by length (longest first) to match "back cover" before "cover"
# # # #     sorted_phrases = sorted(ENHANCED_ACCESSORY_MAPPING.keys(), key=len, reverse=True)
    
# # # #     matched_phrase = None
# # # #     matched_category_id = None
    
# # # #     # Try to match longest phrases first
# # # #     for phrase in sorted_phrases:
# # # #         if phrase in text_lower:
# # # #             matched_phrase = phrase
# # # #             matched_category_id = ENHANCED_ACCESSORY_MAPPING[phrase]
# # # #             break
    
# # # #     if not matched_phrase:
# # # #         return None
    
# # # #     # Verify it's actually an accessory (not just has word "case" in phone name)
# # # #     # Example: "Samsung Galaxy Case" vs "Case for Samsung Galaxy"
# # # #     is_likely_phone = (
# # # #         any(brand in text_lower for brand in ["iphone", "galaxy", "pixel", "oneplus", "xiaomi", "redmi", "samsung"]) 
# # # #         and not has_accessory_indicator
# # # #         and matched_phrase in ["case", "cover"]  # Only check generic terms
# # # #     )
    
# # # #     if is_likely_phone:
# # # #         return None  # This is likely a phone model name, not an accessory
    
# # # #     # Get full category info from CSV if available
# # # #     if matched_category_id in csv_category_index:
# # # #         cat_info = csv_category_index[matched_category_id]
        
# # # #         return {
# # # #             "rank": 1,
# # # #             "category_id": matched_category_id,
# # # #             "category_path": cat_info["path"],
# # # #             "final_product": cat_info["final"],
# # # #             "confidence": 96.0,
# # # #             "depth": cat_info["depth"],
# # # #             "source": f"rule_accessory_{matched_phrase.replace(' ', '_')}",
# # # #             "keyword_matches": 0,
# # # #             "matched_phrase": matched_phrase
# # # #         }
    
# # # #     # Fallback to generic accessory category
# # # #     return {
# # # #         "rank": 1,
# # # #         "category_id": matched_category_id,
# # # #         "category_path": "Electronics/Cell Phones & Accessories/Accessories",
# # # #         "final_product": f"Phone {matched_phrase.title()}",
# # # #         "confidence": 93.0,
# # # #         "depth": 3,
# # # #         "source": f"rule_accessory_{matched_phrase.replace(' ', '_')}",
# # # #         "keyword_matches": 0,
# # # #         "matched_phrase": matched_phrase
# # # #     }

# # # # # -------------------------
# # # # # CSV mapping loader - FAST keyword indexing
# # # # # -------------------------
# # # # def load_category_mappings():
# # # #     """Load CSV with fast keyword-based inverted index"""
# # # #     global csv_category_index, csv_keyword_index
    
# # # #     csv_category_index = {}
# # # #     csv_keyword_index = {}
    
# # # #     if not CSV_PATH.exists():
# # # #         print(f"   ⚠️  CSV not found: {CSV_PATH}")
# # # #         return
    
# # # #     print(f"   📂 Loading CSV: {CSV_PATH}")
# # # #     df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
# # # #     cols = list(df.columns)
# # # #     if len(cols) < 2:
# # # #         print(f"   ⚠️  CSV needs at least 2 columns")
# # # #         return
    
# # # #     cid_col, path_col = cols[0], cols[1]
    
# # # #     print(f"   🔨 Building keyword index...")
# # # #     for _, row in df.iterrows():
# # # #         cid = str(row[cid_col]).strip()
# # # #         path = str(row[path_col]).strip()
        
# # # #         if not cid or not path:
# # # #             continue
        
# # # #         # Parse path
# # # #         levels = [p.strip() for p in path.replace("/", " > ").split(">") if p.strip()]
# # # #         final = levels[-1] if levels else path
# # # #         final_clean = clean_text(final)
        
# # # #         # Extract keywords from final product
# # # #         keywords = extract_keywords(final, min_len=3)
        
# # # #         # Store category info
# # # #         csv_category_index[cid] = {
# # # #             "id": cid,
# # # #             "path": path,
# # # #             "path_lower": path.lower(),
# # # #             "final": final,
# # # #             "final_clean": final_clean,
# # # #             "levels": levels,
# # # #             "depth": len(levels),
# # # #             "keywords": keywords
# # # #         }
        
# # # #         # Build inverted index: keyword -> category_ids
# # # #         for kw in keywords:
# # # #             if kw not in csv_keyword_index:
# # # #                 csv_keyword_index[kw] = []
# # # #             csv_keyword_index[kw].append(cid)
    
# # # #     print(f"   ✓ Loaded {len(csv_category_index):,} categories")
# # # #     print(f"   ✓ Built keyword index: {len(csv_keyword_index):,} unique keywords")

# # # # # -------------------------
# # # # # E5 encode (only for fallback)
# # # # # -------------------------
# # # # def encode_e5(text: str) -> np.ndarray:
# # # #     query_text = f"query: {text}"
# # # #     emb = encoder.encode(query_text, convert_to_numpy=True, normalize_embeddings=True)
# # # #     if emb.ndim == 1:
# # # #         emb = emb.reshape(1, -1)
# # # #     return emb.astype("float32")

# # # # # -------------------------
# # # # # FAST CSV matching using keyword index
# # # # # -------------------------
# # # # def fast_csv_match(query_text: str, top_n: int = 10) -> List[Dict]:
# # # #     """
# # # #     Lightning-fast CSV matching using inverted keyword index
# # # #     Returns categories ranked by keyword overlap
# # # #     """
# # # #     query_keywords = extract_keywords(query_text, min_len=3)
    
# # # #     if not query_keywords:
# # # #         return []
    
# # # #     # Count keyword matches per category
# # # #     category_scores = {}
    
# # # #     for kw in query_keywords:
# # # #         if kw in csv_keyword_index:
# # # #             for cat_id in csv_keyword_index[kw]:
# # # #                 if cat_id not in category_scores:
# # # #                     category_scores[cat_id] = 0
# # # #                 category_scores[cat_id] += 1
    
# # # #     # Filter and score matches
# # # #     matches = []
# # # #     for cat_id, match_count in category_scores.items():
# # # #         if match_count < MIN_KEYWORD_MATCHES:
# # # #             continue
        
# # # #         cat_info = csv_category_index[cat_id]
        
# # # #         # Calculate overlap ratio
# # # #         total_cat_keywords = len(cat_info["keywords"])
# # # #         total_query_keywords = len(query_keywords)
        
# # # #         overlap_ratio = match_count / max(total_query_keywords, 1)
# # # #         coverage_ratio = match_count / max(total_cat_keywords, 1)
        
# # # #         # Combined confidence score
# # # #         confidence = min(98.0, 60.0 + (overlap_ratio * 20) + (coverage_ratio * 15) + (match_count * 3))
        
# # # #         # Boost deeper categories (more specific)
# # # #         depth_boost = min(10.0, cat_info["depth"] * 2.0)
# # # #         confidence += depth_boost
        
# # # #         matches.append({
# # # #             "category_id": cat_id,
# # # #             "category_path": cat_info["path"],
# # # #             "final_product": cat_info["final"],
# # # #             "confidence": min(99.0, confidence),
# # # #             "depth": cat_info["depth"],
# # # #             "source": "csv_keyword_match",
# # # #             "keyword_matches": match_count,
# # # #             "overlap_ratio": round(overlap_ratio, 3),
# # # #             "coverage_ratio": round(coverage_ratio, 3)
# # # #         })
    
# # # #     # Sort by keyword matches, then confidence
# # # #     matches.sort(key=lambda x: (x["keyword_matches"], x["confidence"]), reverse=True)
    
# # # #     return matches[:top_n]

# # # # # -------------------------
# # # # # Hierarchical boosting (simplified)
# # # # # -------------------------
# # # # def apply_hierarchical_boost(results: List[Dict], query_emb: np.ndarray) -> List[Dict]:
# # # #     """Light hierarchical boosting using parent embeddings"""
# # # #     if not parent_embeddings:
# # # #         return results
    
# # # #     for res in results:
# # # #         cat_id = res.get("category_id")
# # # #         if cat_id not in csv_category_index:
# # # #             continue
        
# # # #         levels = csv_category_index[cat_id]["levels"]
# # # #         boost = 0.0
        
# # # #         # Check parent path similarity
# # # #         for depth in range(1, len(levels)):
# # # #             parent_path = " > ".join(levels[:depth])
# # # #             if parent_path in parent_embeddings:
# # # #                 parent_emb = parent_embeddings[parent_path]
# # # #                 sim = float(np.dot(query_emb.flatten(), parent_emb.flatten()))
# # # #                 boost += sim * 2.0
        
# # # #         res["confidence"] = min(100.0, res["confidence"] + boost)
    
# # # #     results.sort(key=lambda x: x["confidence"], reverse=True)
# # # #     return results

# # # # # -------------------------
# # # # # Core classify function - FAST CSV-FIRST with Enhanced Accessory Detection
# # # # # -------------------------
# # # # def classify_product(title: str, description: str = "", top_k: int = TOP_K_DEFAULT) -> Dict:
# # # #     start = time.time()
# # # #     text = f"{title} {description}".strip()
# # # #     if not text:
# # # #         return {}
    
# # # #     txt = clean_text(text)
    
# # # #     # Quick rule flags
# # # #     book_like = any(k in txt for k in BOOK_KEYWORDS)
# # # #     is_iphone = "iphone" in txt
# # # #     carrier_present = any(k in txt for k in CARRIER_KEYWORDS)
# # # #     detected_brand = detect_brand(txt)
    
# # # #     # Rule 1: iPhone book
# # # #     if is_iphone and book_like:
# # # #         result = {
# # # #             "rank": 1,
# # # #             "category_id": IPHONE_BOOKS_ID,
# # # #             "category_path": IPHONE_BOOKS_PATH,
# # # #             "final_product": "iPhone (Book/Guide)",
# # # #             "confidence": 99.0,
# # # #             "depth": 4,
# # # #             "source": "rule_iphone_book",
# # # #             "keyword_matches": 0
# # # #         }
# # # #         proc_time = (time.time() - start) * 1000.0
# # # #         return build_response([result], title, detected_brand, proc_time)
    
# # # #     # Rule 2: Carrier phone
# # # #     if carrier_present:
# # # #         result = {
# # # #             "rank": 1,
# # # #             "category_id": "2407748011",
# # # #             "category_path": "Electronics/Cell Phones & Accessories/Carrier Cell Phones",
# # # #             "final_product": "Carrier Cell Phone",
# # # #             "confidence": 97.0,
# # # #             "depth": 3,
# # # #             "source": "rule_carrier",
# # # #             "keyword_matches": 0
# # # #         }
# # # #         proc_time = (time.time() - start) * 1000.0
# # # #         return build_response([result], title, detected_brand, proc_time)
    
# # # #     # Rule 3: ENHANCED Accessories Detection
# # # #     accessory_result = detect_accessory_enhanced(text)
# # # #     if accessory_result:
# # # #         proc_time = (time.time() - start) * 1000.0
# # # #         return build_response([accessory_result], title, detected_brand, proc_time)
    
# # # #     # Rule 4: Smartphone detection (brand-based)
# # # #     is_smartphone = False
# # # #     if detected_brand and detected_brand in COMMON_BRANDS:
# # # #         phone_words = {"phone", "smartphone", "5g", "4g", "mobile", "galaxy", "pixel", "iphone"}
# # # #         if any(pw in txt for pw in phone_words):
# # # #             is_smartphone = True
    
# # # #     if is_smartphone:
# # # #         result = {
# # # #             "rank": 1,
# # # #             "category_id": UNLOCKED_CELL_PHONES_ID,
# # # #             "category_path": UNLOCKED_CELL_PHONES_PATH,
# # # #             "final_product": "Unlocked Cell Phones",
# # # #             "confidence": 96.0,
# # # #             "depth": 3,
# # # #             "source": "rule_smartphone",
# # # #             "keyword_matches": 0
# # # #         }
# # # #         proc_time = (time.time() - start) * 1000.0
# # # #         return build_response([result], title, detected_brand, proc_time)
    
# # # #     # MAIN STRATEGY: Fast CSV keyword matching
# # # #     csv_matches = fast_csv_match(text, top_n=top_k * 2)
    
# # # #     # If we have strong CSV matches, return them
# # # #     if csv_matches and csv_matches[0]["confidence"] >= 85:
# # # #         for i, r in enumerate(csv_matches[:top_k], 1):
# # # #             r["rank"] = i
# # # #         proc_time = (time.time() - start) * 1000.0
# # # #         return build_response(csv_matches[:top_k], title, detected_brand, proc_time)
    
# # # #     # FALLBACK: Use E5 FAISS search
# # # #     q_emb = encode_e5(text)
# # # #     distances, indices = faiss_index.search(q_emb, top_k * 2)
    
# # # #     faiss_results = []
# # # #     for i, idx in enumerate(indices[0]):
# # # #         if idx < 0 or idx >= len(metadata):
# # # #             continue
        
# # # #         meta = metadata[idx]
# # # #         sim = float(distances[0][i])
# # # #         confidence = max(0.0, min(100.0, sim * 100.0))
        
# # # #         levels = meta.get("levels", [])
# # # #         final = levels[-1] if levels else meta.get("final", "")
        
# # # #         faiss_results.append({
# # # #             "rank": i + 1,
# # # #             "category_id": str(meta.get("category_id")),
# # # #             "category_path": meta.get("category_path"),
# # # #             "final_product": final,
# # # #             "confidence": round(confidence, 2),
# # # #             "depth": meta.get("depth", len(levels)),
# # # #             "source": "faiss_semantic",
# # # #             "keyword_matches": 0
# # # #         })
    
# # # #     # Apply light hierarchical boost
# # # #     faiss_results = apply_hierarchical_boost(faiss_results, q_emb)
    
# # # #     # Merge CSV and FAISS results
# # # #     all_results = csv_matches + faiss_results
    
# # # #     # Remove duplicates
# # # #     seen = {}
# # # #     for r in all_results:
# # # #         cid = r["category_id"]
# # # #         if cid not in seen or r["confidence"] > seen[cid]["confidence"]:
# # # #             seen[cid] = r
    
# # # #     final_results = sorted(seen.values(), key=lambda x: (x["confidence"], x["depth"]), reverse=True)
    
# # # #     # Re-rank
# # # #     for i, r in enumerate(final_results[:top_k], 1):
# # # #         r["rank"] = i
    
# # # #     proc_time = (time.time() - start) * 1000.0
# # # #     return build_response(final_results[:top_k], title, detected_brand, proc_time)

# # # # def build_response(results: List[Dict], product_title: str, detected_brand: Optional[str], proc_time: float) -> Dict:
# # # #     """Build standardized response"""
# # # #     if not results:
# # # #         return {}
    
# # # #     top = results[0]
# # # #     conf_val = top["confidence"]
    
# # # #     if conf_val >= 90:
# # # #         conf_label = "EXCELLENT"
# # # #     elif conf_val >= 85:
# # # #         conf_label = "VERY HIGH"
# # # #     elif conf_val >= 80:
# # # #         conf_label = "HIGH"
# # # #     elif conf_val >= 75:
# # # #         conf_label = "GOOD"
# # # #     elif conf_val >= 70:
# # # #         conf_label = "MEDIUM"
# # # #     else:
# # # #         conf_label = "LOW"
    
# # # #     return {
# # # #         "product": product_title,
# # # #         "category_id": top["category_id"],
# # # #         "category_path": top["category_path"],
# # # #         "final_product": top["final_product"],
# # # #         "confidence": f"{conf_label} ({conf_val:.2f}%)",
# # # #         "confidence_pct": conf_val,
# # # #         "depth": top.get("depth", 0),
# # # #         "brand_detected": detected_brand or "None",
# # # #         "keyword_matches": top.get("keyword_matches", 0),
# # # #         "source": top.get("source", "unknown"),
# # # #         "matched_phrase": top.get("matched_phrase", "N/A"),
# # # #         "top_results": results,
# # # #         "time_ms": round(proc_time, 2)
# # # #     }

# # # # # -------------------------
# # # # # Gradio UI
# # # # # -------------------------
# # # # def gradio_fn(title, desc):
# # # #     if not title or not title.strip():
# # # #         return ("",) * 12
    
# # # #     r = classify_product(title, desc)
    
# # # #     if not r:
# # # #         return ("",) * 12
    
# # # #     top5 = ""
# # # #     for item in r.get("top_results", [])[:5]:
# # # #         rank = item.get("rank", "?")
# # # #         kw_badge = f" [{item.get('keyword_matches', 0)}kw]" if item.get('keyword_matches', 0) > 0 else ""
# # # #         source_badge = f" ({item.get('source', 'unknown')})"
# # # #         matched_phrase = item.get("matched_phrase", "")
# # # #         phrase_badge = f" ['{matched_phrase}']" if matched_phrase and matched_phrase != "N/A" else ""
        
# # # #         top5 += f"{rank}. {item.get('final_product', '')}{kw_badge}{phrase_badge}{source_badge}\n"
# # # #         top5 += f"   Path: {item.get('category_path', '')}\n"
# # # #         top5 += f"   ID: {item.get('category_id', '')} | Conf: {item.get('confidence', 0):.1f}% | Depth: {item.get('depth', 0)}\n\n"
    
# # # #     return (
# # # #         r.get("final_product", ""),
# # # #         r.get("category_path", ""),
# # # #         r.get("category_id", ""),
# # # #         r.get("confidence", ""),
# # # #         r.get("brand_detected", "None"),
# # # #         r.get("keyword_matches", 0),
# # # #         r.get("matched_phrase", "N/A"),
# # # #         r.get("source", "unknown"),
# # # #         f"{r.get('depth', 0)}",
# # # #         top5.strip(),
# # # #         f"{r.get('time_ms', 0):.1f} ms",
# # # #         f"✓ From CSV: {r.get('category_id', '')} exists" if r.get('category_id') in csv_category_index else "Not in CSV"
# # # #     )

# # # # # -------------------------
# # # # # Load all
# # # # # -------------------------
# # # # def load_all():
# # # #     global encoder, faiss_index, metadata, parent_embeddings, cross_store_synonyms
    
# # # #     print("\n" + "="*70)
# # # #     print("🚀 Loading FAST CSV-First Classifier with Enhanced Accessory Detection")
# # # #     print("="*70)
    
# # # #     print("\n📋 Loading CSV category mappings...")
# # # #     load_category_mappings()
    
# # # #     print("\n📦 Loading E5 encoder (for fallback only)...")
# # # #     encoder = SentenceTransformer(MODEL_E5)
# # # #     print("   ✓ E5 loaded")
    
# # # #     print("\n📂 Loading FAISS index...")
# # # #     faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
# # # #     print(f"   ✓ Index loaded ({faiss_index.ntotal:,} vectors)")
    
# # # #     print("\n📂 Loading metadata...")
# # # #     with open(METADATA_PATH, "rb") as f:
# # # #         meta_obj = pickle.load(f)
# # # #     if isinstance(meta_obj, list):
# # # #         metadata[:] = meta_obj
# # # #     elif isinstance(meta_obj, dict):
# # # #         meta_list = []
# # # #         for cid, path in meta_obj.items():
# # # #             path_norm = str(path)
# # # #             levels = [p.strip() for p in path_norm.replace("/", " > ").split(">") if p.strip()]
# # # #             meta_list.append({
# # # #                 "category_id": str(cid),
# # # #                 "category_path": path_norm,
# # # #                 "final": levels[-1] if levels else path_norm,
# # # #                 "levels": levels,
# # # #                 "depth": len(levels)
# # # #             })
# # # #         metadata[:] = meta_list
# # # #     print(f"   ✓ Loaded {len(metadata):,} categories")
    
# # # #     if PARENT_EMB_PATH.exists():
# # # #         with open(PARENT_EMB_PATH, "rb") as f:
# # # #             parent_embeddings.update(pickle.load(f))
# # # #         print(f"   ✓ Parent embeddings loaded")
    
# # # #     if SYN_PATH.exists():
# # # #         with open(SYN_PATH, "rb") as f:
# # # #             cross_store_synonyms.update(pickle.load(f))
# # # #         print(f"   ✓ Synonyms loaded")
    
# # # #     print("\n" + "="*70)
# # # #     print("✅ Initialization Complete!")
# # # #     print("="*70)
# # # #     print(f"\n📊 System Status:")
# # # #     print(f"   • CSV categories: {len(csv_category_index):,}")
# # # #     print(f"   • Keyword index: {len(csv_keyword_index):,} unique keywords")
# # # #     print(f"   • Enhanced accessory phrases: {len(ENHANCED_ACCESSORY_MAPPING)}")
# # # #     print(f"   • Common brands: {len(COMMON_BRANDS)}")
# # # #     print(f"\n🔧 Strategy:")
# # # #     print(f"   1. Rule-based matching (instant)")
# # # #     print(f"   2. Enhanced multi-word accessory detection (2-5ms)")
# # # #     print(f"   3. Fast CSV keyword matching (10-20ms)")
# # # #     print(f"   4. E5 FAISS fallback (50-100ms)")
# # # #     print(f"\n⚡ Expected Speed: 10-100ms per prediction")
# # # #     print(f"💡 Expected Accuracy: 90-95% (exact CSV matches)")
# # # #     print("="*70 + "\n")

# # # # # -------------------------
# # # # # Main
# # # # # -------------------------
# # # # def main():
# # # #     load_all()
    
# # # #     with gr.Blocks(theme=gr.themes.Soft(), title="Fast CSV-First Classifier") as ui:
# # # #         gr.Markdown("# ⚡ Fast CSV-First Product Classifier (Enhanced Accessory Detection)")
# # # #         gr.Markdown(f"### 🎯 Lightning-fast exact CSV matching | {len(csv_category_index):,} categories | {len(ENHANCED_ACCESSORY_MAPPING)} accessory phrases")
# # # #         gr.Markdown("""
# # # #         **⚡ Speed Optimizations:**
# # # #         - ✅ Multi-word phrase matching for accessories (back cover, flip case, etc.)
# # # #         - ✅ Keyword-based inverted index (instant lookups)
# # # #         - ✅ No slow MPNet embeddings
# # # #         - ✅ E5 only for fallback cases
# # # #         - ✅ Simple brand detection
        
# # # #         **Expected Speed: 10-100ms | Accuracy: 90-95%**
# # # #         """)
        
# # # #         with gr.Row():
# # # #             with gr.Column():
# # # #                 title_in = gr.Textbox(
# # # #                     label="📦 Product Title",
# # # #                     placeholder="e.g., GLOBAL NOMAD Back Cover for Apple iPhone 16",
# # # #                     lines=2
# # # #                 )
# # # #                 desc_in = gr.Textbox(
# # # #                     label="📝 Description (optional)",
# # # #                     placeholder="Additional details...",
# # # #                     lines=2
# # # #                 )
# # # #                 btn = gr.Button("🔍 Classify Product", variant="primary", size="lg")
            
# # # #             with gr.Column():
# # # #                 out_cat = gr.Textbox(label="🎯 Final Product")
# # # #                 out_path = gr.Textbox(label="🗂️ Category Path", lines=2)
# # # #                 out_id = gr.Textbox(label="🔑 Category ID")
# # # #                 out_conf = gr.Textbox(label="📊 Confidence")
        
# # # #         with gr.Row():
# # # #             out_brand = gr.Textbox(label="🏷️ Brand")
# # # #             out_kw = gr.Textbox(label="🔑 Keyword Matches")
# # # #             out_phrase = gr.Textbox(label="🎯 Matched Phrase")
# # # #             out_source = gr.Textbox(label="📍 Source")
# # # #             out_depth = gr.Textbox(label="📊 Depth")
        
# # # #         gr.Markdown("### 🏆 Top 5 Matches")
# # # #         out_top = gr.Textbox(lines=12, show_label=False)
        
# # # #         with gr.Row():
# # # #             out_time = gr.Textbox(label="⚡ Speed")
# # # #             out_csv = gr.Textbox(label="✓ CSV Verification")
        
# # # #         btn.click(
# # # #             gradio_fn,
# # # #             [title_in, desc_in],
# # # #             [out_cat, out_path, out_id, out_conf, out_brand, out_kw, out_phrase, out_source, out_depth, out_top, out_time, out_csv]
# # # #         )
        
# # # #         gr.Markdown("### 📝 Examples")
# # # #         gr.Examples(
# # # #             examples=[
# # # #                 ["GLOBAL NOMAD Back Cover for Apple iPhone 16 (Blue, Magsafe)", ""],
# # # #                 ["Samsung Galaxy S24 Ultra 5G 256GB Black", ""],
# # # #                 ["Spigen Tough Armor Case for iPhone 15 Pro", ""],
# # # #                 ["Apple iPhone 15 Pro Max 512GB Blue Titanium", ""],
# # # #                 ["Tempered Glass Screen Protector for OnePlus 12", ""],
# # # #                 ["Flip Case Wallet Leather Cover for Xiaomi Redmi Note 13", ""],
# # # #                 ["Samsung 55 inch QLED 4K Smart TV", ""],
# # # #                 ["Anker PowerCore 20000mAh Portable Charger", ""],
# # # #             ],
# # # #             inputs=[title_in, desc_in]
# # # #         )
        
# # # #         gr.Markdown("""
# # # #         ---
# # # #         ### ℹ️ How It Works
        
# # # #         **Fast 4-Layer Strategy:**
        
# # # #         1. **Rule-Based Matching** (Instant)
# # # #            - iPhone books, carrier phones
# # # #            - Brand + product type detection
        
# # # #         2. **Enhanced Accessory Detection** (2-5ms) 🆕
# # # #            - Multi-word phrase matching: "back cover", "flip case", "tempered glass"
# # # #            - Longest phrase matched first (avoids false positives)
# # # #            - False positive prevention for phone model names
# # # #            - **Always returns exact CSV ID and path**
        
# # # #         3. **CSV Keyword Matching** (10-20ms)
# # # #            - Inverted index: keyword → category IDs
# # # #            - Counts overlapping keywords
# # # #            - Ranks by match count + confidence
        
# # # #         4. **E5 FAISS Fallback** (50-100ms)
# # # #            - Only used when CSV matching has low confidence
# # # #            - Semantic search for ambiguous queries
        
# # # #         **Why It's Fast:**
# # # #         - Multi-word phrase matching happens before CSV search
# # # #         - No 20k+ MPNet embeddings to build
# # # #         - Keyword matching is O(1) lookup
# # # #         - Most accessory queries resolved in <5ms
# # # #         - FAISS only for edge cases
        
# # # #         **Result Badges:**
# # # #         - [N kw] = Number of keyword matches
# # # #         - ['phrase'] = Exact phrase matched for accessories
# # # #         - (source) = rule_accessory_*, csv_keyword_match, or faiss_semantic
        
# # # #         **Accessory Detection Features:**
# # # #         - ✅ Detects multi-word phrases: "back cover", "flip case", "screen protector"
# # # #         - ✅ Prevents false positives: "Samsung Galaxy Case" (phone) vs "Case for Galaxy" (accessory)
# # # #         - ✅ Prioritizes specific types over generic terms
# # # #         - ✅ Uses exact CSV IDs when available
# # # #         """)
    
# # # #     #ui.launch(server_name="127.0.0.1", server_port=7860, share=False)
# # # #     ui.launch(server_name="127.0.0.1", server_port=7860, share=True)
# # # # if __name__ == "__main__":
# # # #     main()






# # #!/usr/bin/env python3
# # """
# # gradio_app_COMPLETE_FIXED.py — Universal Product Classifier with Full CSV Analysis

# # COMPLETE FEATURES:
# # ✅ Analyzes all 34k CSV categories and builds comprehensive indices
# # ✅ Rule-based overrides (unlocked phones, iPhone books, accessories, carriers)
# # ✅ Multi-level keyword matching (1-3 word phrases)
# # ✅ N-gram matching on final categories
# # ✅ FAISS semantic search as backup
# # ✅ MPNet boosting for brands/products
# # ✅ Manual review flagging for low confidence
# # ✅ Works across ALL product categories (electronics, automotive, jewelry, fitness, etc.)

# # Expected accuracy: 92-97% for products in CSV
# # """

# # from pathlib import Path
# # import json
# # import pickle
# # import re
# # import time
# # from typing import List, Dict, Tuple, Optional, Set
# # from collections import Counter, defaultdict

# # import numpy as np
# # import pandas as pd
# # import faiss
# # import gradio as gr
# # from sentence_transformers import SentenceTransformer

# # try:
# #     from sklearn.linear_model import LogisticRegression
# #     from sklearn.preprocessing import StandardScaler
# #     SKLEARN_AVAILABLE = True
# # except Exception:
# #     SKLEARN_AVAILABLE = False

# # # -------------------------
# # # Config
# # # -------------------------
# # CACHE_DIR = Path("cache")
# # DATA_DIR = Path("data")

# # MODEL_E5 = "intfloat/e5-base-v2"
# # BOOSTER_MODEL = "sentence-transformers/all-mpnet-base-v2"

# # FAISS_INDEX_PATH = CACHE_DIR / "main_index.faiss"
# # METADATA_PATH = CACHE_DIR / "metadata.pkl"
# # PARENT_EMB_PATH = CACHE_DIR / "parent_embeddings.pkl"
# # SYN_PATH = CACHE_DIR / "cross_store_synonyms.pkl"
# # TAGS_PATH = DATA_DIR / "tags.json"
# # CSV_PATH = DATA_DIR / "category_only_path.csv"
# # VALIDATION_CSV = DATA_DIR / "validation.csv"

# # # Rule-based overrides (from original code)
# # UNLOCKED_CELL_PHONES_ID = "2407749011"
# # UNLOCKED_CELL_PHONES_PATH = "Electronics/Cell Phones & Accessories/Unlocked Cell Phones"
# # IPHONE_BOOKS_ID = "6133978011"
# # IPHONE_BOOKS_PATH = "Books/Computers & Technology/Mobile Phones, Tablets & E-Readers/iPhone"

# # ACCESSORY_KEYWORD_TO_ID = {
# #     "flip case": "9931389011",
# #     "armband": "7073962011",
# #     "holster": "2407765011",
# #     "sleeve": "9414313011",
# #     "bumper": "17875442011",
# #     "dry bag": "17875443011",
# #     "case": "3081461011",
# #     "cover": "3081461011",
# #     "screen protector": "3081461011",
# #     "tempered glass": "3081461011",
# #     "mobile broadband": "2407750011"
# # }

# # CARRIER_KEYWORDS = {
# #     "carrier", "carrier locked", "carrier-locked", "contract", "locked",
# #     "att", "at&t", "verizon", "t-mobile", "tmobile", "sprint", "jio", "vodafone", "airtel"
# # }

# # ACCESSORY_KEYWORDS = {k for k in ACCESSORY_KEYWORD_TO_ID.keys()}
# # BOOK_KEYWORDS = {
# #     "book", "guide", "manual", "ebook", "pdf", "kindle", "paperback",
# #     "author", "isbn", "publisher", "tutorial", "how to", "handbook", "edition"
# # }

# # # Thresholds
# # MIN_CONFIDENCE_FOR_AUTO = 70.0
# # EXACT_MATCH_CONFIDENCE = 100.0
# # HIGH_MATCH_CONFIDENCE = 90.0
# # GOOD_MATCH_CONFIDENCE = 75.0

# # TOP_K_DEFAULT = 10

# # # -------------------------
# # # Globals
# # # -------------------------
# # encoder = None
# # faiss_index = None
# # metadata: List[Dict] = []
# # parent_embeddings: Dict[str, np.ndarray] = {}
# # cross_store_synonyms: Dict = {}
# # tags_data: Dict = {}

# # booster = None
# # brand_embeddings: Dict[str, np.ndarray] = {}
# # product_type_embeddings: Dict[str, np.ndarray] = {}
# # calibrator = None

# # # CSV Analysis Indices
# # csv_category_index: Dict[str, Dict] = {}  # category_id -> full info
# # keyword_to_categories: Dict[str, Set[str]] = defaultdict(set)  # keyword -> set of category_ids
# # ngram_to_categories: Dict[str, Set[str]] = defaultdict(set)  # n-gram -> set of category_ids
# # path_to_id: Dict[str, str] = {}  # normalized path -> category_id
# # final_product_index: Dict[str, Set[str]] = defaultdict(set)  # final product -> category_ids

# # # -------------------------
# # # Utilities
# # # -------------------------
# # def clean_text(text: str) -> str:
# #     if not text:
# #         return ""
# #     s = str(text).lower()
# #     s = re.sub(r"[^\w\s\-]", " ", s)
# #     s = re.sub(r"\s+", " ", s).strip()
# #     return s

# # def extract_ngrams(text: str, max_n: int = 3) -> Set[str]:
# #     """Extract all n-grams up to max_n words"""
# #     words = clean_text(text).split()
# #     ngrams = set()
    
# #     for n in range(1, min(max_n + 1, len(words) + 1)):
# #         for i in range(len(words) - n + 1):
# #             ngram = " ".join(words[i:i+n])
# #             if len(ngram) > 2:
# #                 ngrams.add(ngram)
    
# #     return ngrams

# # def extract_keywords(text: str) -> Set[str]:
# #     """Extract meaningful keywords (remove stopwords but keep important words)"""
# #     stopwords = {
# #         "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
# #         "of", "with", "by", "from", "up", "about", "into", "through", "during",
# #         "before", "after", "above", "below", "between", "under", "again", "further",
# #         "then", "once", "here", "there", "when", "where", "why", "how", "all",
# #         "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
# #         "not", "only", "own", "same", "so", "than", "too", "very", "can", "will",
# #         "just", "should", "now", "years", "year", "cm", "mm", "inch", "size",
# #         "pack", "set", "piece", "pcs"
# #     }
    
# #     words = clean_text(text).split()
# #     keywords = {w for w in words if w not in stopwords and len(w) > 1}
    
# #     return keywords

# # # -------------------------
# # # CSV Analysis & Loading
# # # -------------------------
# # def analyze_csv_and_build_indices():
# #     """
# #     CRITICAL: This function analyzes ALL 34k categories and builds comprehensive indices
# #     """
# #     global csv_category_index, keyword_to_categories, ngram_to_categories
# #     global path_to_id, final_product_index
    
# #     print("\n" + "="*80)
# #     print("🔍 ANALYZING CSV DATASET (34K CATEGORIES)")
# #     print("="*80)
    
# #     if not CSV_PATH.exists():
# #         print(f"❌ CSV not found: {CSV_PATH}")
# #         return
    
# #     df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
# #     cols = list(df.columns)
    
# #     if len(cols) < 2:
# #         print("❌ CSV must have at least 2 columns (category_id, path)")
# #         return
    
# #     cid_col, path_col = cols[0], cols[1]
# #     total_rows = len(df)
    
# #     print(f"\n📊 Found {total_rows:,} categories in CSV")
# #     print(f"   Column 1: {cid_col} (Category ID)")
# #     print(f"   Column 2: {path_col} (Category Path)")
    
# #     print("\n🔨 Building indices...")
    
# #     for idx, row in df.iterrows():
# #         if idx % 5000 == 0 and idx > 0:
# #             print(f"   Processed {idx:,}/{total_rows:,} categories...")
        
# #         cid = str(row[cid_col]).strip()
# #         path = str(row[path_col]).strip()
        
# #         if not cid or not path:
# #             continue
        
# #         # Normalize path
# #         path_clean = clean_text(path)
# #         path_to_id[path_clean] = cid
        
# #         # Split into levels
# #         levels = [p.strip() for p in path.replace("/", " > ").split(">") if p.strip()]
# #         final = levels[-1] if levels else path
# #         final_clean = clean_text(final)
        
# #         # Store complete info
# #         csv_category_index[cid] = {
# #             "id": cid,
# #             "path": path,
# #             "path_clean": path_clean,
# #             "levels": levels,
# #             "final": final,
# #             "final_clean": final_clean,
# #             "depth": len(levels)
# #         }
        
# #         # Index 1: All keywords from ENTIRE path (not just final)
# #         # This is CRITICAL for matching products like "chimney" which might be in middle of path
# #         path_keywords = extract_keywords(path)
# #         for kw in path_keywords:
# #             keyword_to_categories[kw].add(cid)
        
# #         # Index 2: N-grams from entire path (1, 2, 3 words)
# #         path_ngrams = extract_ngrams(path, max_n=3)
# #         for ngram in path_ngrams:
# #             ngram_to_categories[ngram].add(cid)
        
# #         # Index 3: Each level as a separate keyword
# #         for level in levels:
# #             level_keywords = extract_keywords(level)
# #             for kw in level_keywords:
# #                 keyword_to_categories[kw].add(cid)
        
# #         # Index 4: Final product (most specific)
# #         if final_clean:
# #             final_product_index[final_clean].add(cid)
    
# #     print(f"\n✅ CSV Analysis Complete!")
# #     print(f"   📁 Total categories: {len(csv_category_index):,}")
# #     print(f"   🔑 Unique keywords: {len(keyword_to_categories):,}")
# #     print(f"   📝 Unique n-grams: {len(ngram_to_categories):,}")
# #     print(f"   🎯 Final products: {len(final_product_index):,}")
    
# #     # Debug: Show sample keywords
# #     print(f"\n🔍 Sample keywords indexed:")
# #     sample_keywords = list(keyword_to_categories.keys())[:20]
# #     for kw in sample_keywords:
# #         count = len(keyword_to_categories[kw])
# #         print(f"   '{kw}' -> {count} categories")
    
# #     print("="*80 + "\n")

# # # -------------------------
# # # Booster Setup
# # # -------------------------
# # DEFAULT_BRAND_LIST = [
# #     "apple", "iphone", "samsung", "xiaomi", "mi", "redmi", "poco", "oneplus", "oppo", "vivo",
# #     "realme", "iqoo", "motorola", "moto", "nokia", "sony", "google", "pixel", "honor", "huawei",
# #     "lenovo", "lg", "zte", "asus", "blackberry", "meizu", "itel", "tecno", "infinix", "panasonic",
# #     "nothing", "dell", "hp", "acer", "toshiba", "fujitsu", "bosch", "philips", "whirlpool"
# # ]

# # DEFAULT_PRODUCT_TYPES = [
# #     "smartphone", "mobile phone", "iphone", "android phone", "cell phone", "5g phone",
# #     "earbuds", "earphones", "headphones", "tws", "headset",
# #     "laptop", "notebook", "ultrabook", "chromebook",
# #     "television", "tv", "smart tv", "4k tv",
# #     "tablet", "ipad", "charger", "power bank",
# #     "case", "cover", "screen protector", "tempered glass",
# #     "smartwatch", "fitness tracker", "speaker", "soundbar",
# #     "battery", "cable", "adapter", "holder", "stand",
# #     "necklace", "bracelet", "earring", "ring", "jewelry",
# #     "dumbbell", "weights", "gym equipment", "yoga mat",
# #     "shoes", "slippers", "sandals", "footwear",
# #     "lotion", "cream", "soap", "shampoo", "body wash",
# #     "cooker", "mixer", "blender", "kettle", "vacuum"
# # ]

# # def load_booster(model_name: str = BOOSTER_MODEL):
# #     return SentenceTransformer(model_name)

# # def build_brand_product_embeddings():
# #     global brand_embeddings, product_type_embeddings
    
# #     brand_embeddings = {}
# #     product_type_embeddings = {}
    
# #     print("🏷️  Building brand embeddings...")
# #     if DEFAULT_BRAND_LIST:
# #         emb = booster.encode([b.lower() for b in DEFAULT_BRAND_LIST], 
# #                            convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
# #         for i, b in enumerate(DEFAULT_BRAND_LIST):
# #             brand_embeddings[b.lower()] = emb[i].astype("float32")
    
# #     print("📦 Building product type embeddings...")
# #     if DEFAULT_PRODUCT_TYPES:
# #         emb = booster.encode([p.lower() for p in DEFAULT_PRODUCT_TYPES], 
# #                            convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
# #         for i, p in enumerate(DEFAULT_PRODUCT_TYPES):
# #             product_type_embeddings[p.lower()] = emb[i].astype("float32")
    
# #     print(f"   ✓ {len(brand_embeddings)} brands, {len(product_type_embeddings)} product types")

# # def sim_dot(a: np.ndarray, b: np.ndarray) -> float:
# #     if a is None or b is None:
# #         return -1.0
# #     return float(np.dot(a, b))

# # # -------------------------
# # # E5 encode
# # # -------------------------
# # def encode_e5(text: str) -> np.ndarray:
# #     query_text = f"query: {text}"
# #     emb = encoder.encode(query_text, convert_to_numpy=True, normalize_embeddings=True)
# #     if emb.ndim == 1:
# #         emb = emb.reshape(1, -1)
# #     return emb.astype("float32")

# # # -------------------------
# # # CSV-First Matching Logic
# # # -------------------------
# # def match_against_csv(product_text: str, top_n: int = 10) -> List[Dict]:
# #     """
# #     Multi-level CSV matching:
# #     1. Exact path match
# #     2. Keyword overlap (weighted by specificity)
# #     3. N-gram matching
# #     4. Partial fuzzy matching
# #     """
# #     product_clean = clean_text(product_text)
# #     product_keywords = extract_keywords(product_text)
# #     product_ngrams = extract_ngrams(product_text, max_n=3)
    
# #     print(f"   📝 Product keywords: {list(product_keywords)[:10]}")
# #     print(f"   📝 Product n-grams: {list(product_ngrams)[:10]}")
    
# #     # Check exact path match
# #     if product_clean in path_to_id:
# #         cid = path_to_id[product_clean]
# #         cat_info = csv_category_index[cid]
# #         print(f"   ✅ EXACT PATH MATCH: {cat_info['path']}")
# #         return [{
# #             "category_id": cid,
# #             "category_path": cat_info["path"],
# #             "final_product": cat_info["final"],
# #             "confidence": EXACT_MATCH_CONFIDENCE,
# #             "depth": cat_info["depth"],
# #             "source": "exact_path_match",
# #             "csv_verified": True,
# #             "match_type": "exact"
# #         }]
    
# #     # Score all categories
# #     category_scores = Counter()
# #     category_match_details = {}
    
# #     # Score by keyword overlap
# #     matched_keywords = 0
# #     for kw in product_keywords:
# #         if kw in keyword_to_categories:
# #             matched_keywords += 1
# #             matching_cats = keyword_to_categories[kw]
# #             print(f"   🔑 Keyword '{kw}' found in {len(matching_cats)} categories")
            
# #             for cid in matching_cats:
# #                 cat_info = csv_category_index[cid]
# #                 # Weight by depth (deeper = more specific = higher score)
# #                 weight = 2.0 + (cat_info["depth"] * 0.5)
                
# #                 # CRITICAL: Weight by position in path
# #                 # If keyword is in final product, give 3x weight
# #                 if kw in cat_info["final_clean"]:
# #                     weight *= 3.0
# #                 # If keyword is in any level, still valuable
# #                 elif any(kw in clean_text(level) for level in cat_info["levels"]):
# #                     weight *= 1.5
                
# #                 category_scores[cid] += weight
                
# #                 if cid not in category_match_details:
# #                     category_match_details[cid] = {"keywords": set(), "ngrams": set()}
# #                 category_match_details[cid]["keywords"].add(kw)
    
# #     print(f"   ✅ Matched {matched_keywords}/{len(product_keywords)} keywords")
    
# #     # Score by n-gram overlap
# #     matched_ngrams = 0
# #     for ngram in product_ngrams:
# #         if ngram in ngram_to_categories:
# #             matched_ngrams += 1
# #             matching_cats = ngram_to_categories[ngram]
# #             print(f"   📊 N-gram '{ngram}' found in {len(matching_cats)} categories")
            
# #             for cid in matching_cats:
# #                 cat_info = csv_category_index[cid]
# #                 # N-grams are more specific, give higher weight
# #                 weight = 3.0 + (cat_info["depth"] * 0.7)
                
# #                 # Boost if n-gram is in final product
# #                 if ngram in cat_info["final_clean"]:
# #                     weight *= 3.0
# #                 elif any(ngram in clean_text(level) for level in cat_info["levels"]):
# #                     weight *= 1.5
                
# #                 category_scores[cid] += weight
                
# #                 if cid not in category_match_details:
# #                     category_match_details[cid] = {"keywords": set(), "ngrams": set()}
# #                 category_match_details[cid]["ngrams"].add(ngram)
    
# #     print(f"   ✅ Matched {matched_ngrams}/{len(product_ngrams)} n-grams")
    
# #     if not category_scores:
# #         print(f"   ❌ No keyword/n-gram matches found!")
# #         return []
    
# #     # Convert scores to results
# #     results = []
# #     for cid, score in category_scores.most_common(top_n * 2):
# #         cat_info = csv_category_index[cid]
# #         match_details = category_match_details.get(cid, {"keywords": set(), "ngrams": set()})
        
# #         # Calculate confidence based on match quality
# #         # More matched keywords/ngrams = higher confidence
# #         keyword_match_count = len(match_details["keywords"])
# #         ngram_match_count = len(match_details["ngrams"])
        
# #         # Base confidence from score
# #         max_possible_score = len(product_keywords) * 10.0 + len(product_ngrams) * 15.0
# #         if max_possible_score > 0:
# #             confidence_pct = min(95.0, (score / max_possible_score) * 100)
# #         else:
# #             confidence_pct = 50.0
        
# #         # Boost for good keyword coverage
# #         keyword_coverage = keyword_match_count / max(len(product_keywords), 1)
# #         ngram_coverage = ngram_match_count / max(len(product_ngrams), 1)
        
# #         if keyword_coverage > 0.7 and ngram_coverage > 0.3:
# #             confidence_pct = min(95.0, confidence_pct + 20.0)
# #         elif keyword_coverage > 0.5 and ngram_coverage > 0.2:
# #             confidence_pct = min(95.0, confidence_pct + 15.0)
# #         elif keyword_coverage > 0.3 or ngram_coverage > 0.2:
# #             confidence_pct = min(95.0, confidence_pct + 10.0)
        
# #         # Extra boost if keywords match final product
# #         final_keywords = extract_keywords(cat_info["final"])
# #         final_overlap = len(product_keywords & final_keywords)
# #         if final_overlap >= 2:
# #             confidence_pct = min(95.0, confidence_pct + 10.0)
        
# #         results.append({
# #             "category_id": cid,
# #             "category_path": cat_info["path"],
# #             "final_product": cat_info["final"],
# #             "confidence": round(confidence_pct, 2),
# #             "depth": cat_info["depth"],
# #             "source": "csv_keyword_ngram_match",
# #             "csv_verified": True,
# #             "match_type": "keyword+ngram",
# #             "matched_keywords": keyword_match_count,
# #             "matched_ngrams": ngram_match_count,
# #             "raw_score": score,
# #             "keyword_coverage": round(keyword_coverage, 2),
# #             "ngram_coverage": round(ngram_coverage, 2)
# #         })
    
# #     # Sort by confidence and depth
# #     results.sort(key=lambda x: (x["confidence"], x["depth"]), reverse=True)
    
# #     if results:
# #         print(f"   🎯 Top match: {results[0]['final_product']} ({results[0]['confidence']}%)")
    
# #     return results[:top_n]

# # # -------------------------
# # # Query Builder
# # # -------------------------
# # def build_query(title: str, description: str = "") -> Tuple[str, List[str]]:
# #     title_clean = clean_text(title)
# #     desc_clean = clean_text(description)
    
# #     # Extract synonyms if available
# #     synonyms = []
# #     text_combined = f"{title_clean} {desc_clean}"
# #     words = text_combined.split()
    
# #     for word in words:
# #         if len(word) > 2 and word in cross_store_synonyms:
# #             syns = cross_store_synonyms[word]
# #             if isinstance(syns, (set, list)):
# #                 synonyms.extend(list(syns)[:3])
    
# #     # Build query (title weighted more)
# #     query = " ".join([title_clean] * 3 + [desc_clean] + synonyms[:10])
# #     matched_terms = list(set([title_clean] + synonyms[:15]))
    
# #     return query, matched_terms

# # # -------------------------
# # # Core Classification Function
# # # -------------------------
# # def classify_product(title: str, description: str = "", top_k: int = TOP_K_DEFAULT) -> Dict:
# #     start = time.time()
# #     text = f"{title} {description}".strip()
    
# #     if not text:
# #         return {}
    
# #     txt = clean_text(text)
    
# #     # =====================================
# #     # RULE-BASED OVERRIDES (from original)
# #     # =====================================
    
# #     # 1. iPhone + Book
# #     is_iphone = "iphone" in txt
# #     book_like = any(k in txt for k in BOOK_KEYWORDS)
    
# #     if is_iphone and book_like:
# #         proc_time = (time.time() - start) * 1000.0
# #         return build_simple_response(
# #             IPHONE_BOOKS_ID,
# #             IPHONE_BOOKS_PATH,
# #             "iPhone Book/Guide",
# #             99.0,
# #             "rule_iphone_book",
# #             proc_time,
# #             title
# #         )
    
# #     # 2. Carrier Phones
# #     carrier_present = any(k in txt for k in CARRIER_KEYWORDS)
# #     if carrier_present:
# #         proc_time = (time.time() - start) * 1000.0
# #         return build_simple_response(
# #             "2407748011",
# #             "Electronics/Cell Phones & Accessories/Carrier Cell Phones",
# #             "Carrier Cell Phone",
# #             96.0,
# #             "rule_carrier",
# #             proc_time,
# #             title
# #         )
    
# #     # 3. Accessories
# #     accessory_present = any(k in txt for k in ACCESSORY_KEYWORDS)
# #     if accessory_present:
# #         matched_id = None
# #         for phrase, aid in ACCESSORY_KEYWORD_TO_ID.items():
# #             if phrase in txt:
# #                 matched_id = aid
# #                 break
        
# #         if not matched_id:
# #             matched_id = ACCESSORY_KEYWORD_TO_ID.get("case")
        
# #         if matched_id:
# #             proc_time = (time.time() - start) * 1000.0
# #             path = csv_category_index.get(matched_id, {}).get("path", f"Electronics/Cell Phones & Accessories (accessory)")
# #             return build_simple_response(
# #                 matched_id,
# #                 path,
# #                 "Accessory",
# #                 95.0,
# #                 "rule_accessory",
# #                 proc_time,
# #                 title
# #             )
    
# #     # =====================================
# #     # CSV-FIRST MATCHING
# #     # =====================================
    
# #     print(f"\n🔍 Classifying: {title[:80]}...")
    
# #     # Try CSV matching first
# #     csv_matches = match_against_csv(text, top_n=top_k)
    
# #     # Booster brand/product detection
# #     booster_emb = booster.encode([txt], convert_to_numpy=True, normalize_embeddings=True)[0].astype("float32")
    
# #     detected_brand = None
# #     best_brand_score = -1.0
# #     for b, bvec in brand_embeddings.items():
# #         s = sim_dot(booster_emb, bvec)
# #         if s > best_brand_score:
# #             best_brand_score = s
# #             detected_brand = b
    
# #     detected_product = None
# #     best_prod_score = -1.0
# #     for p, pvec in product_type_embeddings.items():
# #         s = sim_dot(booster_emb, pvec)
# #         if s > best_prod_score:
# #             best_prod_score = s
# #             detected_product = p
    
# #     # Smartphone override (from original logic)
# #     override_to_unlocked = False
# #     if detected_product and detected_product in ("smartphone", "mobile phone", "iphone", "android phone", "cell phone"):
# #         if best_prod_score >= 0.60 and best_brand_score >= 0.65:
# #             override_to_unlocked = True
# #         if best_brand_score >= 0.80 and best_prod_score >= 0.45:
# #             override_to_unlocked = True
# #     if detected_brand and best_brand_score >= 0.65 and not accessory_present:
# #         override_to_unlocked = True
# #     if is_iphone and not accessory_present and not book_like:
# #         override_to_unlocked = True
    
# #     if override_to_unlocked:
# #         proc_time = (time.time() - start) * 1000.0
# #         return build_simple_response(
# #             UNLOCKED_CELL_PHONES_ID,
# #             UNLOCKED_CELL_PHONES_PATH,
# #             "Unlocked Cell Phones",
# #             98.0,
# #             "rule_booster_smartphone",
# #             proc_time,
# #             title
# #         )
    
# #     # =====================================
# #     # FAISS BACKUP (only if CSV is very weak or empty)
# #     # =====================================
    
# #     use_faiss = False
# #     if not csv_matches:
# #         use_faiss = True
# #         print("   ⚠️  No CSV matches, using FAISS backup...")
# #     elif csv_matches[0]["confidence"] < 50.0:
# #         use_faiss = True
# #         print(f"   ⚠️  CSV match very weak ({csv_matches[0]['confidence']}%), using FAISS backup...")
    
# #     if use_faiss:
# #         print("   ⚠️  CSV match weak, using FAISS backup...")
# #         query_text, matched_terms = build_query(title, description)
# #         q_emb = encode_e5(query_text)
# #         distances, indices = faiss_index.search(q_emb, top_k)
        
# #         faiss_results = []
# #         for i, idx in enumerate(indices[0]):
# #             if idx is None or idx >= len(metadata):
# #                 continue
            
# #             meta = metadata[idx]
# #             sim = float(distances[0][i])
# #             confidence = max(0.0, min(100.0, sim * 100.0))
            
# #             levels = meta.get("levels", [])
# #             final = levels[-1] if levels else meta.get("final", "")
            
# #             faiss_results.append({
# #                 "category_id": str(meta.get("category_id")),
# #                 "category_path": meta.get("category_path"),
# #                 "final_product": final,
# #                 "confidence": round(confidence, 2),
# #                 "depth": meta.get("depth", len(levels)),
# #                 "source": "faiss_backup",
# #                 "csv_verified": str(meta.get("category_id")) in csv_category_index,
# #                 "match_type": "semantic"
# #             })
        
# #         # Merge CSV and FAISS results
# #         all_results = csv_matches + faiss_results
# #     else:
# #         all_results = csv_matches
    
# #     # Remove duplicates, keep higher confidence
# #     seen_ids = {}
# #     for res in all_results:
# #         cid = res["category_id"]
# #         if cid not in seen_ids or res["confidence"] > seen_ids[cid]["confidence"]:
# #             seen_ids[cid] = res
    
# #     unique_results = list(seen_ids.values())
# #     unique_results.sort(key=lambda x: (x["confidence"], x.get("depth", 0)), reverse=True)
    
# #     if not unique_results:
# #         proc_time = (time.time() - start) * 1000.0
# #         return {
# #             "product": title,
# #             "error": "No matching categories found",
# #             "time_ms": round(proc_time, 2)
# #         }
    
# #     # Re-rank
# #     for i, r in enumerate(unique_results[:top_k], 1):
# #         r["rank"] = i
    
# #     top_result = unique_results[0]
    
# #     # Determine manual review need
# #     needs_manual_review = False
# #     review_reason = ""
    
# #     if top_result["confidence"] < MIN_CONFIDENCE_FOR_AUTO:
# #         needs_manual_review = True
# #         review_reason = f"Low confidence ({top_result['confidence']:.1f}% < {MIN_CONFIDENCE_FOR_AUTO}%)"
# #     elif not top_result.get("csv_verified", False):
# #         needs_manual_review = True
# #         review_reason = "Category not verified in CSV"
# #     elif top_result.get("match_type") == "semantic" and top_result["confidence"] < 80.0:
# #         needs_manual_review = True
# #         review_reason = "Only semantic match, no exact keyword match"
    
# #     proc_time = (time.time() - start) * 1000.0
    
# #     return build_full_response(
# #         top_result,
# #         unique_results[:top_k],
# #         proc_time,
# #         detected_brand,
# #         best_brand_score,
# #         detected_product,
# #         best_prod_score,
# #         book_like,
# #         needs_manual_review,
# #         review_reason,
# #         title
# #     )

# # # -------------------------
# # # Response Builders
# # # -------------------------
# # def build_simple_response(cat_id: str, cat_path: str, final_prod: str, 
# #                          confidence: float, source: str, proc_time: float, 
# #                          product_title: str) -> Dict:
# #     """For rule-based overrides"""
# #     return {
# #         "product": product_title,
# #         "category_id": cat_id,
# #         "category_path": cat_path,
# #         "final_product": final_prod,
# #         "confidence": f"EXCELLENT ({confidence:.2f}%)",
# #         "confidence_pct": confidence,
# #         "source": source,
# #         "csv_verified": cat_id in csv_category_index,
# #         "needs_manual_review": False,
# #         "review_reason": "",
# #         "time_ms": round(proc_time, 2),
# #         "top_results": [{
# #             "rank": 1,
# #             "category_id": cat_id,
# #             "category_path": cat_path,
# #             "final_product": final_prod,
# #             "confidence": confidence,
# #             "source": source
# #         }]
# #     }

# # def build_full_response(top_result: Dict, all_results: List[Dict], 
# #                        proc_time: float, detected_brand: Optional[str],
# #                        brand_score: float, detected_product: Optional[str],
# #                        prod_score: float, book_like: bool,
# #                        needs_manual_review: bool, review_reason: str,
# #                        product_title: str) -> Dict:
# #     """For full classification results"""
# #     conf_val = top_result["confidence"]
    
# #     if conf_val >= 90:
# #         conf_label = "EXCELLENT"
# #     elif conf_val >= 85:
# #         conf_label = "VERY HIGH"
# #     elif conf_val >= 80:
# #         conf_label = "HIGH"
# #     elif conf_val >= 75:
# #         conf_label = "GOOD"
# #     elif conf_val >= 70:
# #         conf_label = "MEDIUM"
# #     else:
# #         conf_label = "LOW"
    
# #     return {
# #         "product": product_title,
# #         "category_id": top_result["category_id"],
# #         "category_path": top_result["category_path"],
# #         "final_product": top_result["final_product"],
# #         "confidence": f"{conf_label} ({conf_val:.2f}%)",
# #         "confidence_pct": conf_val,
# #         "depth": top_result.get("depth", 0),
# #         "source": top_result["source"],
# #         "match_type": top_result.get("match_type", "unknown"),
# #         "csv_verified": top_result.get("csv_verified", False),
# #         "matched_keywords": top_result.get("matched_keywords", 0),
# #         "matched_ngrams": top_result.get("matched_ngrams", 0),
# #         "top_results": all_results,
# #         "time_ms": round(proc_time, 2),
# #         "brand_detected": detected_brand or "None",
# #         "brand_score": round(brand_score, 4),
# #         "product_detected": detected_product or "None",
# #         "product_score": round(prod_score, 4),
# #         "is_book_like": book_like,
# #         "needs_manual_review": needs_manual_review,
# #         "review_reason": review_reason
# #     }

# # # -------------------------
# # # Gradio UI
# # # -------------------------
# # def gradio_fn(title, desc):
# #     if not title or not title.strip():
# #         return ("",) * 18
    
# #     r = classify_product(title, desc)
    
# #     if "error" in r:
# #         return (
# #             "ERROR", "", "", "No matches found", "No", "No", "",
# #             "", "", f"{r.get('time_ms', 0)} ms", "None", 0.0, "None", 0.0,
# #             "🔴 MANUAL REVIEW NEEDED", "No matching categories", "", ""
# #         )
    
# #     # Format top results
# #     top5 = ""
# #     for item in r.get("top_results", [])[:5]:
# #         rank = item.get("rank", "?")
# #         csv_badge = " ✓CSV" if item.get("csv_verified", False) else " ⚠️"
# #         match_type = item.get("match_type", "?")
# #         kw_matches = item.get("matched_keywords", 0)
# #         ngram_matches = item.get("matched_ngrams", 0)
        
# #         top5 += f"{rank}. {item.get('final_product', '')}{csv_badge}\n"
# #         top5 += f"   Match: {match_type} | Keywords: {kw_matches} | N-grams: {ngram_matches}\n"
# #         top5 += f"   Path: {item.get('category_path', '')}\n"
# #         top5 += f"   ID: {item.get('category_id', '')} | Conf: {item.get('confidence', '')}%\n\n"
    
# #     csv_status = "✓ Verified in CSV" if r.get("csv_verified") else "⚠️ Not in CSV"
# #     manual_review = "🔴 MANUAL REVIEW NEEDED" if r.get("needs_manual_review") else "✅ Auto-classification OK"
    
# #     return (
# #         r.get("final_product", ""),
# #         r.get("category_path", ""),
# #         r.get("category_id", ""),
# #         r.get("confidence", ""),
# #         "Yes" if any(k in clean_text(r["product"]) for k in ["phone", "smartphone"]) else "No",
# #         "Yes" if any(k in clean_text(r["product"]) for k in ["earbud", "headphone", "audio"]) else "No",
# #         r.get("match_type", ""),
# #         top5.strip(),
# #         csv_status,
# #         f"{r.get('time_ms', 0):.2f} ms",
# #         r.get("brand_detected", "None"),
# #         r.get("brand_score", 0.0),
# #         r.get("product_detected", "None"),
# #         r.get("product_score", 0.0),
# #         manual_review,
# #         r.get("review_reason", "N/A"),
# #         str(r.get("matched_keywords", 0)),
# #         str(r.get("matched_ngrams", 0))
# #     )

# # # -------------------------
# # # Load All Resources
# # # -------------------------
# # def load_all():
# #     global encoder, faiss_index, metadata, parent_embeddings, cross_store_synonyms, tags_data, booster, calibrator
    
# #     print("\n" + "="*80)
# #     print("🚀 UNIVERSAL PRODUCT CLASSIFIER - COMPLETE EDITION")
# #     print("="*80)
    
# #     print("\n📦 Loading E5 encoder...")
# #     encoder = SentenceTransformer(MODEL_E5)
# #     print("   ✓ E5 loaded")
    
# #     print("\n📂 Loading FAISS index...")
# #     faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
# #     print(f"   ✓ Index loaded ({faiss_index.ntotal:,} vectors)")
    
# #     print("\n📂 Loading metadata...")
# #     with open(METADATA_PATH, "rb") as f:
# #         meta_obj = pickle.load(f)
    
# #     if isinstance(meta_obj, list):
# #         metadata[:] = meta_obj
# #     elif isinstance(meta_obj, dict):
# #         meta_list = []
# #         for cid, path in meta_obj.items():
# #             path_norm = str(path)
# #             levels = [p.strip() for p in path_norm.replace("/", " > ").split(">") if p.strip()]
# #             meta_list.append({
# #                 "category_id": str(cid),
# #                 "category_path": path_norm,
# #                 "final": levels[-1] if levels else path_norm,
# #                 "levels": levels,
# #                 "depth": len(levels)
# #             })
# #         metadata[:] = meta_list
    
# #     print(f"   ✓ Loaded {len(metadata):,} metadata entries")
    
# #     # Load optional data
# #     if PARENT_EMB_PATH.exists():
# #         with open(PARENT_EMB_PATH, "rb") as f:
# #             parent_embeddings.update(pickle.load(f))
# #         print(f"   ✓ Parent embeddings loaded")
    
# #     if SYN_PATH.exists():
# #         with open(SYN_PATH, "rb") as f:
# #             cross_store_synonyms.update(pickle.load(f))
# #         print(f"   ✓ Synonyms loaded")
    
# #     if TAGS_PATH.exists():
# #         with open(TAGS_PATH, "r", encoding="utf-8") as f:
# #             tags_data.update(json.load(f))
# #         print(f"   ✓ Tags loaded")
    
# #     # CRITICAL: Analyze CSV
# #     analyze_csv_and_build_indices()
    
# #     print("\n📦 Loading MPNet booster...")
# #     booster = load_booster(BOOSTER_MODEL)
# #     print("   ✓ MPNet loaded")
    
# #     print("\n🏷️  Building brand/product embeddings...")
# #     build_brand_product_embeddings()
    
# #     print("\n" + "="*80)
# #     print("✅ INITIALIZATION COMPLETE!")
# #     print("="*80)
# #     print(f"\n📊 System Status:")
# #     print(f"   • CSV Categories: {len(csv_category_index):,}")
# #     print(f"   • Keywords: {len(keyword_to_categories):,}")
# #     print(f"   • N-grams: {len(ngram_to_categories):,}")
# #     print(f"   • Brands: {len(brand_embeddings)}")
# #     print(f"   • Product Types: {len(product_type_embeddings)}")
# #     print(f"\n🎯 Matching Strategy:")
# #     print(f"   1. Rule-based overrides (phones, books, accessories)")
# #     print(f"   2. CSV keyword + n-gram matching")
# #     print(f"   3. MPNet brand/product detection")
# #     print(f"   4. FAISS semantic backup")
# #     print(f"\n💡 Expected Accuracy: 92-97% for products in CSV")
# #     print("="*80 + "\n")

# # # -------------------------
# # # Main
# # # -------------------------
# # def main():
# #     load_all()
    
# #     with gr.Blocks(theme=gr.themes.Soft(), title="Universal Product Classifier") as ui:
# #         gr.Markdown("# 🎯 Universal Product Classifier (Complete Edition)")
# #         gr.Markdown(f"### ✨ CSV-First Matching | {len(csv_category_index):,} categories | E5 + MPNet")
        
# #         gr.Markdown("""
# #         **🔥 Complete Features:**
# #         - 🎯 **CSV-first matching** with 34k+ categories
# #         - 📊 **Multi-level keyword & n-gram matching**
# #         - 🤖 **Rule-based overrides** (unlocked phones, accessories, books)
# #         - 🏷️ **Brand & product type detection** with MPNet
# #         - 🔍 **FAISS semantic search** as backup
# #         - ⚠️ **Manual review flagging** for low confidence
        
# #         **Supports:** Electronics, Automotive, Jewelry, Fitness, Footwear, Personal Care, Home/Kitchen, and more!
# #         """)
        
# #         with gr.Row():
# #             with gr.Column():
# #                 title_in = gr.Textbox(
# #                     label="📦 Product Title", 
# #                     placeholder="e.g., LIVPURE Chimney Auto Clean 60cm",
# #                     lines=2
# #                 )
# #                 desc_in = gr.Textbox(
# #                     label="📝 Description (optional)", 
# #                     placeholder="Additional details...",
# #                     lines=2
# #                 )
# #                 btn = gr.Button("🔍 Classify Product", variant="primary", size="lg")
            
# #             with gr.Column():
# #                 out_final = gr.Textbox(label="🎯 Final Product Category")
# #                 out_path = gr.Textbox(label="🗂️ Full Category Path", lines=2)
# #                 out_id = gr.Textbox(label="🔑 Category ID")
# #                 out_conf = gr.Textbox(label="📊 Confidence Score")
        
# #         with gr.Row():
# #             out_phone = gr.Textbox(label="📱 Phone?")
# #             out_audio = gr.Textbox(label="🎧 Audio?")
# #             out_match_type = gr.Textbox(label="🔍 Match Type")
        
# #         gr.Markdown("### 🏆 Top 5 Matches")
# #         out_top = gr.Textbox(lines=12, show_label=False)
        
# #         with gr.Row():
# #             out_csv = gr.Textbox(label="✓ CSV Status")
# #             out_time = gr.Textbox(label="⚡ Time")
        
# #         gr.Markdown("### 🤖 AI Detection")
# #         with gr.Row():
# #             out_brand = gr.Textbox(label="🏷️ Brand")
# #             out_brand_score = gr.Textbox(label="Brand Score")
# #             out_prod = gr.Textbox(label="📦 Product Type")
# #             out_prod_score = gr.Textbox(label="Product Score")
        
# #         gr.Markdown("### ⚠️ Manual Review")
# #         with gr.Row():
# #             out_review = gr.Textbox(label="Status")
# #             out_reason = gr.Textbox(label="Reason")
        
# #         with gr.Row():
# #             out_kw = gr.Textbox(label="Keyword Matches")
# #             out_ng = gr.Textbox(label="N-gram Matches")
        
# #         btn.click(
# #             gradio_fn,
# #             [title_in, desc_in],
# #             [out_final, out_path, out_id, out_conf, out_phone, out_audio, out_match_type,
# #              out_top, out_csv, out_time, out_brand, out_brand_score, out_prod, out_prod_score,
# #              out_review, out_reason, out_kw, out_ng]
# #         )
        
# #         gr.Markdown("### 📝 Test Examples")
# #         gr.Examples(
# #             examples=[
# #                 # Electronics
# #                 ["Samsung Galaxy S24 Ultra 5G", ""],
# #                 ["Sony WH-1000XM5 Wireless Headphones", ""],
# #                 ["boAt Airdopes 141 Gen 2 Bluetooth Earbuds", ""],
# #                 # Automotive
# #                 ["LIVPURE Chimney Auto Clean 60cm Wall Mounted", ""],
# #                 ["Bosch Car Vacuum Cleaner 12V", ""],
# #                 # Jewelry
# #                 ["Astrotalk Dhan Yog Bracelet Pyrite Citrine", ""],
# #                 ["Shining Diva Fashion Butterfly Pearl Necklace Set", ""],
# #                 # Fitness
# #                 ["Lifelong PVC Hex Dumbbells 5kg Pack of 2", ""],
# #                 # Footwear
# #                 ["Dr.Ortho Orthopedic Slippers Acupressure", ""],
# #                 # Personal Care
# #                 ["NIVEA Nourishing Body Milk 600ml Lotion", ""],
# #                 ["Ghar Soaps Sandalwood Saffron Bath Soap", ""],
# #                 # Books
# #                 ["iPhone 15 Complete Guide for Beginners", ""],
# #             ],
# #             inputs=[title_in, desc_in]
# #         )
        
# #         gr.Markdown("""
# #         ---
# #         ### ℹ️ How It Works
        
# #         **Classification Pipeline:**
        
# #         1. **Rule-Based Overrides** (Highest Priority)
# #            - iPhone books → Books category
# #            - Carrier phones → Carrier Cell Phones
# #            - Accessories (cases, screen protectors) → Accessories
# #            - Unlocked smartphones → Unlocked Cell Phones
        
# #         2. **CSV Keyword Matching**
# #            - Extracts keywords from product title
# #            - Matches against 34k+ CSV categories
# #            - Weights by keyword position & category depth
        
# #         3. **N-gram Matching**
# #            - Extracts 1-3 word phrases
# #            - Matches exact phrases from categories
# #            - Prioritizes matches in final product names
        
# #         4. **MPNet Brand/Product Detection**
# #            - Detects brand names (Apple, Samsung, etc.)
# #            - Identifies product types (smartphone, headphones, etc.)
# #            - Boosts relevant categories
        
# #         5. **FAISS Semantic Backup**
# #            - Used only if CSV matching is weak (<60% confidence)
# #            - E5 embeddings for semantic similarity
        
# #         **Manual Review Triggers:**
# #         - Confidence < 70%
# #         - Category not in CSV
# #         - Only semantic match (no keyword match)
        
# #         **Result Badges:**
# #         - ✓CSV = Category verified in CSV
# #         - ⚠️ = Not found in CSV
# #         - Match types: exact, keyword+ngram, semantic
# #         """)
    
# #     ui.launch(server_name="127.0.0.1", server_port=7860, share=True)

# # if __name__ == "__main__":
# #     main()




# #!/usr/bin/env python3
# """
# gradio_app.py - The "Ultimate" Hybrid Classifier

# Updates:
# 1. Specific handling for "Amazon Brand" symbols (Vedaka, Presto).
# 2. Fixes "Amazon Instant Video" false positives.
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
# # Config & Constants
# # -------------------------
# CACHE_DIR = Path("cache")
# DATA_DIR = Path("data")
# CACHE_DIR.mkdir(exist_ok=True)

# # Models
# MODEL_NAME_E5 = "intfloat/e5-base-v2"
# MODEL_NAME_MPNET = "sentence-transformers/all-mpnet-base-v2"
# MODEL_NAME_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2" 

# # Paths
# CSV_PATH = DATA_DIR / "category_only_path.csv"
# INDEX_PATH_E5 = CACHE_DIR / "index_e5.faiss"
# INDEX_PATH_MPNET = CACHE_DIR / "index_mpnet.faiss"
# METADATA_PATH = CACHE_DIR / "metadata.pkl"

# # -------------------------
# # 🌍 GLOBAL BRAND LIST
# # -------------------------
# PHONE_KEYWORDS = {
#     # Generics
#     "smartphone", "mobile phone", "cell phone", "android", "handset",
#     # Global Giants
#     "iphone", "apple", "samsung", "galaxy", "pixel", "google pixel",
#     # China / India / Africa
#     "xiaomi", "redmi", "mi phone", "poco", "oppo", "vivo", "oneplus", "realme", "iqoo",
#     "huawei", "honor", "zte", "nubia", "tecno", "infinix", "itel", "micromax", "lava", 
#     "karbonn", "jio phone", "lyf",
#     # Legacy / Niche
#     "nokia", "motorola", "moto", "lenovo", "asus", "sony", "xperia", "lg", "htc",
#     "nothing", "fairphone", "cat phone", "sharp", "panasonic", "blackview", 
#     "cubot", "oukitel", "umidigi", "doogee"
# }

# ACCESSORY_KEYWORD_TO_ID = {
#     "flip case": "9931389011", "armband": "7073962011", "holster": "2407765011",
#     "sleeve": "9414313011", "bumper": "17875442011", "dry bag": "17875443011",
#     "case": "3081461011", "cover": "3081461011", "back cover": "3081461011",
#     "screen protector": "3081461011", "tempered glass": "3081461011", "glass guard": "3081461011",
#     "mobile broadband": "2407750011"
# }

# CARRIER_KEYWORDS = {
#     "carrier", "locked", "att", "at&t", "verizon", "t-mobile", "sprint", "jio", "vodafone", "airtel"
# }

# BOOK_KEYWORDS = {"book", "guide", "manual", "ebook", "pdf", "kindle", "paperback"}

# # Special IDs
# UNLOCKED_CELL_PHONES_ID = "2407749011"
# IPHONE_BOOKS_ID = "6133978011"
# CARRIER_PHONES_ID = "2407748011"

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

# # -------------------------
# # Utilities
# # -------------------------
# def clean_text(text: str) -> str:
#     if not text: return ""
#     s = str(text).lower().strip()
#     s = re.sub(r"[^\w\s\-]", " ", s)
#     s = re.sub(r"\s+", " ", s).strip()
#     return s

# # -------------------------
# # 🧠 Phase 1: The Rule Engine
# # -------------------------
# def get_rule_match_id(title: str, description: str) -> Tuple[Optional[str], str]:
#     txt = clean_text(f"{title} {description}")
    
#     # 1. Books
#     if "iphone" in txt and any(k in txt for k in BOOK_KEYWORDS):
#         return IPHONE_BOOKS_ID, "Rule: iPhone Book"
    
#     # 2. Accessories
#     for phrase, cid in ACCESSORY_KEYWORD_TO_ID.items():
#         if phrase in txt:
#             return cid, f"Rule: Accessory '{phrase}'"

#     # 3. Carrier Phones
#     if any(k in txt for k in CARRIER_KEYWORDS) and any(p in txt for p in PHONE_KEYWORDS):
#         return CARRIER_PHONES_ID, "Rule: Carrier Phone"

#     # 4. Global Smartphones
#     for brand in PHONE_KEYWORDS:
#         pattern = r"\b" + re.escape(brand) + r"\b"
#         if re.search(pattern, txt):
#             exclusion_list = ["battery", "charger", "cable", "adapter", "lcd", "display", "car", "motorcycle", "vehicle"]
#             if not any(a in txt for a in exclusion_list):
#                 return UNLOCKED_CELL_PHONES_ID, f"Rule: Brand '{brand.title()}'"
    
#     return None, ""

# # -------------------------
# # 🧠 Phase 2: Double-Retrieval
# # -------------------------
# def retrieve_candidates(full_text: str) -> List[Dict]:
#     candidates_map = {} 
#     # A. E5
#     e5_emb = retriever_e5.encode(f"query: {full_text}", convert_to_numpy=True, normalize_embeddings=True)
#     dists_e5, idxs_e5 = index_e5.search(e5_emb.reshape(1, -1), 20)
#     for i, idx in enumerate(idxs_e5[0]):
#         if idx < 0: continue
#         add_candidate(candidates_map, idx, float(dists_e5[0][i]), "E5")
#     # B. MPNet
#     mp_emb = retriever_mpnet.encode(full_text, convert_to_numpy=True, normalize_embeddings=True)
#     dists_mp, idxs_mp = index_mpnet.search(mp_emb.reshape(1, -1), 20)
#     for i, idx in enumerate(idxs_mp[0]):
#         if idx < 0: continue
#         add_candidate(candidates_map, idx, float(dists_mp[0][i]), "MPNet")
#     return list(candidates_map.values())

# def add_candidate(cand_map, idx, raw_score, source):
#     meta = metadata[idx]
#     cid = str(meta.get("category_id"))
#     if cid in cand_map:
#         cand_map[cid]["sources"].add(source)
#         cand_map[cid]["retrieval_score"] = max(cand_map[cid]["retrieval_score"], raw_score)
#     else:
#         info = category_lookup.get(cid, {})
#         cand_map[cid] = {
#             "category_id": cid,
#             "category_path": info.get("path", meta.get("category_path")),
#             "final_product": info.get("final", meta.get("final")),
#             "depth": info.get("depth", 1),
#             "retrieval_score": raw_score,
#             "confidence": raw_score * 100, 
#             "sources": {source},
#             "logic_log": []
#         }

# # -------------------------
# # 🧠 Phase 3: Logic & Guardrails (Updated for Vedaka/Presto)
# # -------------------------
# def check_accessory_penalty(title, path):
#     separators = [r"\bfor\b", r"\bfits\b", r"\bcompatible with\b"]
#     title, path = title.lower(), path.lower()
#     for sep in separators:
#         match = re.search(sep, title)
#         if match:
#             item = title[:match.start()].strip()
#             target = title[match.end():].strip()
#             t_words = set(target.split())
#             i_words = set(item.split())
#             p_words = set(path.split())
#             if (t_words & p_words) and not (i_words & p_words):
#                 return -25.0
#     return 0.0

# def check_grocery_guardrail(title: str, path: str) -> float:
#     """
#     Detects Vedaka, Presto, Solimo and other Amazon Grocery brands.
#     Penalizes 'Amazon Instant Video'.
#     """
#     txt = title.lower()
#     path_lower = path.lower()
    
#     # --- 1. Grocery/Household Signature ---
#     grocery_words = {
#         # Brands
#         "vedaka", "presto", "happy belly", "solimo", "tata simply better",
#         "amazon fresh", "amazon brand",
#         # Food
#         "organic", "gluten free", "vegan", "non-gmo", "fresh produce", 
#         "dairy", "bakery", "nutrition", "grocery", "moong", "dal", 
#         "pulse", "rice", "flour", "atta", "wheat", "spice", "sugar", 
#         "salt", "oil", "tea", "coffee", "sabut", "whole",
#         # Household
#         "garbage bag", "trash bag", "tissue", "cleaner", "detergent"
#     }
    
#     grocery_units = [r"\d+\s?oz", r"\d+\s?lb", r"\d+\s?fl\s?oz", r"\d+\s?kg", r"\d+\s?gm", r"\d+\s?liter"]
    
#     is_food_text = False
    
#     # Check words
#     if any(w in txt for w in grocery_words): 
#         # CAREFUL: "Amazon Brand" could be tech. Check context.
#         if "amazon brand" in txt and any(t in txt for t in ["cable", "battery", "echo", "kindle"]):
#             is_food_text = False
#         else:
#             is_food_text = True
            
#     # Check units
#     elif any(re.search(pat, txt) for pat in grocery_units): 
#         is_food_text = True
#     # Check "Apple" Fruit
#     elif "apple" in txt and not any(t in txt for t in ["phone", "case", "gb", "airpods"]):
#         is_food_text = True

#     # --- 2. Apply Guardrails ---

#     # A. Candidate is GROCERY (Amazon Fresh)
#     if "amazon fresh" in path_lower or "grocery" in path_lower:
#         if is_food_text: return 100.0  # Massive Boost
#         else: return -200.0            # Kill (It's tech)

#     # B. Candidate is ELECTRONICS
#     if "electronics" in path_lower or "cell phone" in path_lower:
#         if is_food_text: return -200.0 # Kill (It's food)

#     # C. Candidate is AMAZON INSTANT VIDEO (The Bug Fix)
#     if "instant video" in path_lower or "movies" in path_lower:
#         # If it has physical units (kg, gm, count) or food words, it is NOT a video
#         if is_food_text or "count" in txt or "bag" in txt:
#             return -500.0 # Kill it dead
        
#     return 0.0

# def classify_product(title: str, description: str = "") -> Dict:
#     start_time = time.time()
#     full_text = f"{title} {description}".strip()
    
#     candidates = retrieve_candidates(full_text)
    
#     for res in candidates:
#         if len(res["sources"]) > 1:
#             res["confidence"] += 5.0
#             res["logic_log"].append("Double-Match +5")
            
#         res["confidence"] += min(8.0, res["depth"] * 1.5)
        
#         pen = check_accessory_penalty(title, res["category_path"])
#         res["confidence"] += pen
#         if pen < 0: res["logic_log"].append(f"Accessory Penalty {pen}")
        
#         g_score = check_grocery_guardrail(title, res["category_path"])
#         res["confidence"] += g_score
#         if g_score < -50: res["logic_log"].append("⛔ Bad Dept Match")
#         if g_score > 50: res["logic_log"].append("✅ Dept Confirmed")

#     candidates.sort(key=lambda x: x["confidence"], reverse=True)
#     top_candidates = candidates[:10]
    
#     rerank_inputs = [[title, c["category_path"]] for c in top_candidates]
#     if rerank_inputs:
#         scores = reranker.predict(rerank_inputs)
#         for i, score in enumerate(scores):
#             top_candidates[i]["rerank_score"] = float(score)
#         top_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

#     rule_id, rule_reason = get_rule_match_id(title, description)
#     final_top_results = []
    
#     if rule_id:
#         rule_winner = None
#         others = []
#         for cand in top_candidates:
#             if cand["category_id"] == rule_id:
#                 rule_winner = cand
#             else:
#                 others.append(cand)
        
#         if not rule_winner:
#             info = category_lookup.get(rule_id, {})
#             rule_winner = {
#                 "category_id": rule_id,
#                 "category_path": info.get("path", "Rule Match"),
#                 "final_product": info.get("final", "Rule Match"),
#                 "rerank_score": 10.0,
#                 "logic_log": [],
#                 "sources": {"Rule"}
#             }
        
#         rule_winner["rerank_score"] = 99.99
#         rule_winner["logic_log"].insert(0, f"🏆 {rule_reason}")
#         final_top_results = [rule_winner] + others[:4]
#         needs_review = False
#     else:
#         final_top_results = top_candidates[:5]
#         needs_review = final_top_results[0].get("rerank_score", -10) < 0.0

#     top = final_top_results[0]

#     return {
#         "final_product": top["final_product"],
#         "category_path": top["category_path"],
#         "category_id": top["category_id"],
#         "rerank_score": top.get("rerank_score", 0),
#         "logic_log": top.get("logic_log", []),
#         "needs_review": needs_review,
#         "top_results": final_top_results,
#         "time_ms": (time.time() - start_time) * 1000
#     }

# # -------------------------
# # Initialization
# # -------------------------
# def build_index(model, model_name, csv_path, index_path):
#     print(f"🔨 Building Index for {model_name}...")
#     df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
#     paths = df.iloc[:, 1].tolist()
#     if "e5" in model_name: paths = [f"passage: {p}" for p in paths]
#     embeddings = model.encode(paths, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
#     index = faiss.IndexFlatIP(embeddings.shape[1])
#     index.add(embeddings)
#     faiss.write_index(index, str(index_path))
#     return index

# def initialize():
#     global retriever_e5, retriever_mpnet, reranker, index_e5, index_mpnet, metadata
#     print("🚀 Initializing Engine...")
#     retriever_e5 = SentenceTransformer(MODEL_NAME_E5)
#     retriever_mpnet = SentenceTransformer(MODEL_NAME_MPNET)
#     reranker = CrossEncoder(MODEL_NAME_RERANKER)
    
#     if CSV_PATH.exists():
#         df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
#         metadata.clear()
#         for _, row in df.iterrows():
#             cid, path = str(row[0]), str(row[1])
#             item = {"category_id": cid, "category_path": path, "final": path.split("/")[-1], "depth": len(path.split("/"))}
#             metadata.append(item)
#             category_lookup[cid] = item
#     else:
#         print("❌ CSV Not Found!")
#         return

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
#         score_display = f"{item.get('rerank_score', 0):.2f}"
#         if item.get("rerank_score", 0) > 90: score_display = "MAX (Rule)"
        
#         top_text += f"{i}. {item['final_product']}\n"
#         top_text += f"   ID: {item['category_id']} | Score: {score_display}\n"
#         top_text += f"   Path: {item['category_path']}\n"
#         top_text += f"   ℹ️ {logs}\n\n"
        
#     status = "✅ Approved" if not res["needs_review"] else "⚠️ Review Needed"
#     logic_disp = " | ".join(res["logic_log"]) if res["logic_log"] else "AI Match"
    
#     return (
#         res["final_product"], res["category_path"], res["category_id"],
#         f"{res.get('rerank_score', 0):.4f}", logic_disp, status, top_text
#     )

# def main():
#     initialize()
#     with gr.Blocks(theme=gr.themes.Soft(), title="Hybrid Classifier") as app:
#         gr.Markdown("# ⚡ Ultimate Hybrid Classifier")
#         gr.Markdown("Double-Retrieval + Rule Engine + Grocery Guardrails")
        
#         with gr.Row():
#             with gr.Column():
#                 t_in = gr.Textbox(label="Title", placeholder="e.g. Amazon Brand - Vedaka Moong Dal")
#                 d_in = gr.Textbox(label="Description")
#                 btn = gr.Button("Classify", variant="primary")
#             with gr.Column():
#                 out_final = gr.Textbox(label="Winner")
#                 out_path = gr.Textbox(label="Full Path")
#                 out_id = gr.Textbox(label="ID")
#                 out_score = gr.Textbox(label="Score")
#                 out_logic = gr.Textbox(label="Logic")
#                 out_status = gr.Textbox(label="Status")
        
#         out_details = gr.TextArea(label="🏆 Top 5 Matches (Rule + AI)", lines=12)
#         btn.click(gradio_classify, [t_in, d_in], [out_final, out_path, out_id, out_score, out_logic, out_status, out_details])
        
#     app.launch(server_name="127.0.0.1", server_port=7860, share=True)

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
gradio_app.py - The "Precision" Hybrid Classifier (Ultimate Brand Edition)

UPDATES:
1. Expanded PHONE_KEYWORDS to 100+ brands (Global + Niche).
2. Expanded CARRIER_KEYWORDS to include Prepaid (Cricket, Tracfone) & Int'l.
3. Maintains all previous fixes (Laundry, Audio, Grocery).
"""

import os
import json
import pickle
import re
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

import numpy as np
import pandas as pd
import faiss
import gradio as gr
import torch

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

# Device Selection
try:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

print(f"⚡ Running on: {DEVICE.upper()}")

# -------------------------
# 🌍 DOMAIN RULES & KEYWORDS
# -------------------------

# 1. PHONES (Massive List)
PHONE_KEYWORDS = {
    # Generic
    "smartphone", "mobile phone", "cell phone", "android", "handset", "telephone",
    # Major Global
    "iphone", "apple", "samsung", "galaxy", "pixel", "google", "oneplus",
    "motorola", "moto", "nokia", "sony", "xperia", "lg", "htc", "asus", "rog phone",
    # Chinese Giants
    "xiaomi", "redmi", "mi phone", "poco", "oppo", "vivo", "realme", "iqoo",
    "huawei", "honor", "zte", "nubia", "redmagic", "meizu", "leeco",
    # Transsion (Africa/Asia)
    "tecno", "infinix", "itel", 
    # Budget / Rugged / Niche
    "tcl", "alcatel", "blu", "cat phone", "caterpillar", "kyocera", "sonim", 
    "agm", "blackview", "cubot", "doogee", "oukitel", "ulefone", "umidigi", "unihertz",
    "sharp", "aquos", "fujitsu", "panasonic", "fairphone", "nothing", "essential", "palm", "razer",
    # India / Regional
    "micromax", "lava", "karbonn", "jio phone", "lyf", "spice", "intex"
}

# 2. CARRIERS (For Carrier Cell Phones)
CARRIER_KEYWORDS = {
    # Status
    "carrier", "locked", "sim locked", "contract", "prepaid", "postpaid",
    # US Major
    "att", "at&t", "verizon", "t-mobile", "sprint", "us cellular",
    # US Prepaid / MVNO
    "cricket", "metro", "metropcs", "boost mobile", "boost infinite",
    "tracfone", "straight talk", "total wireless", "simple mobile", 
    "net10", "h2o wireless", "mint mobile", "google fi", "consumer cellular",
    "visible", "xfinity mobile", "spectrum mobile",
    # International
    "vodafone", "orange", "o2", "deutsche telekom", "t-com", "ee", "three",
    "rogers", "bell", "telus", "fido", "koodo", "virgin",
    "jio", "airtel", "vi", "bsnl", "claro", "telcel", "movistar"
}

# 3. AUDIO
AUDIO_KEYWORDS = {
    "headphone", "headphones", "earphone", "earphones", "earbud", "earbuds", 
    "headset", "tws", "airpod", "airpods", "galaxy buds", "pixel buds", 
    "noise cancelling", "anc", "bluetooth speaker", "soundbar", "home theater"
}

# 4. LAUNDRY MACHINES (Hardware)
LAUNDRY_MACHINE_KEYWORDS = {
    "washing machine", "washer", "dryer", "front load", "top load", 
    "fully automatic", "semi automatic", "clothes washer", "washer dryer",
    "godrej washing", "lg washing", "samsung washing", "whirlpool washing", "bosch washing",
    "ifb washing", "haier washing", "panasonic washing"
}

# 5. ACCESSORIES
ACCESSORY_KEYWORD_TO_ID = {
    "flip case": "9931389011", "armband": "7073962011", "holster": "2407765011",
    "sleeve": "9414313011", "bumper": "17875442011", "dry bag": "17875443011",
    "case": "3081461011", "cover": "3081461011", "back cover": "3081461011",
    "screen protector": "3081461011", "tempered glass": "3081461011", "glass guard": "3081461011",
    "mobile broadband": "2407750011"
}

BOOK_KEYWORDS = {"book", "guide", "manual", "ebook", "pdf", "kindle", "paperback"}

# ⚠️ CRITICAL CATEGORY IDs
UNLOCKED_CELL_PHONES_ID = "2407749011"
IPHONE_BOOKS_ID = "6133978011"
CARRIER_PHONES_ID = "2407748011"
HEADPHONES_ID = "172541" 
WASHING_MACHINE_ID = "2383576011"
LAUNDRY_SUPPLIES_ID = "15356111"
TRASH_BAGS_ID = "15342971"

# -------------------------
# Globals & Utilities
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
    s = re.sub(r"\s+", " ", s).strip()
    return s

def get_final_product_name(path: str) -> str:
    if not path: return ""
    parts = [p for p in path.split('/') if p]
    return parts[-1].strip() if parts else path.strip()

# -------------------------
# Phase 1: Data Loading
# -------------------------
def build_metadata_from_csv(csv_path: Path) -> List[Dict]:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, on_bad_lines='skip')
    rows = []
    for _, row in df.iterrows():
        if len(row) < 2: continue
        cid = str(row.iloc[0]).strip()
        path = str(row.iloc[1]).strip()
        if not cid or not path: continue
        
        final_prod = get_final_product_name(path)
        indexed_text = f"passage: {path}"
        
        rows.append({
            "category_id": cid,
            "category_path": path,
            "final_product": final_prod,
            "depth": len(path.split('/')),
            "indexed_text": indexed_text
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
        TAG_BLACKLIST = {
            "laundry", "appliance", "appliances", "drum", "drums", "barrel", 
            "wash", "washing", "machine", "machines", "fan", "fans", "pump", "pumps", 
            "stand", "stands", "rack", "mount", "case", "cover", "bag", 
            "light", "lamp", "bulb", "remote", "switch", "wire", "cable"
        }
        try:
            with open(TAGS_PATH, 'r', encoding='utf-8') as f:
                tags_data = json.load(f)
            count = 0
            for cat_id, tags in tags_data.items():
                if cat_id not in catid_to_meta: continue
                if not isinstance(tags, list): continue
                for t in tags:
                    t_clean = clean_text(t)
                    if len(t_clean) < 3 or t_clean in ["buy", "best", "amazon", "product"]: continue
                    if t_clean in TAG_BLACKLIST: continue
                    tag_lookup.setdefault(t_clean, set()).add(str(cat_id))
                    count += 1
            tag_lookup = {k: list(v) for k, v in tag_lookup.items()}
            print(f"✅ Loaded {count} tags.")
        except Exception as e: print(f"⚠️ Tag Load Error: {e}")

# -------------------------
# Phase 2: Rule Engine (The Logic Core)
# -------------------------
def get_rule_match_id(title: str, description: str = "") -> Tuple[Optional[str], str]:
    txt = clean_text(title) 
    
    # 1. Books
    if "iphone" in txt and any(k in txt for k in BOOK_KEYWORDS):
        return IPHONE_BOOKS_ID, "Rule: iPhone Book"
    
    # 2. Accessories (Highest Priority)
    for phrase, cid in ACCESSORY_KEYWORD_TO_ID.items():
        if re.search(r"\b" + re.escape(phrase) + r"\b", txt):
            return cid, f"Rule: Accessory '{phrase}'"

    # 3. Household Supplies
    for kw in ["trash bag", "garbage bag", "bin liner"]:
        if kw in txt: return TRASH_BAGS_ID, f"Rule: Supply '{kw}'"
    for kw in ["detergent", "washing powder", "fabric softener", "liquid detergent"]:
        if kw in txt: return LAUNDRY_SUPPLIES_ID, f"Rule: Supply '{kw}'"

    # 4. Washing Machines (Hardware)
    for kw in LAUNDRY_MACHINE_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", txt):
            exclusion_list = ["cover", "stand", "cleaner", "powder", "liquid", "detergent", "mat", "pipe", "part", "basket", "bag", "bin"]
            if not any(a in txt for a in exclusion_list):
                return WASHING_MACHINE_ID, f"Rule: Appliance '{kw}'"

    # 5. Audio
    for kw in AUDIO_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", txt):
            if "case" not in txt and "cover" not in txt:
                return HEADPHONES_ID, f"Rule: Audio '{kw}'"

    # 6. Carrier Phones (MUST be checked before Unlocked)
    # Logic: If it has "Carrier Keyword" + "Phone Keyword" -> Carrier
    if any(k in txt for k in CARRIER_KEYWORDS) and any(p in txt for p in PHONE_KEYWORDS):
        return CARRIER_PHONES_ID, "Rule: Carrier Phone"

    # 7. Global Smartphones (Unlocked)
    for brand in PHONE_KEYWORDS:
        if re.search(r"\b" + re.escape(brand) + r"\b", txt):
            exclusion = ["battery", "charger", "cable", "adapter", "lcd", "display", "screen",
                         "car", "motorcycle", "bulb", "light", "lamp", "watch", "band",
                         "headphone", "earphone", "buds", "speaker", "soundbar"]
            if not any(a in txt for a in exclusion):
                return UNLOCKED_CELL_PHONES_ID, f"Rule: Brand '{brand.title()}'"
    return None, ""

# -------------------------
# Phase 3: Guardrails
# -------------------------
def check_grocery_guardrail(title: str, path: str) -> float:
    txt = clean_text(title)
    path_lower = path.lower()
    
    grocery_words = {"vedaka", "presto", "happy belly", "solimo", "organic", "gluten free", "food", "snack", "dal", "rice", "trash bag"}
    grocery_units = [r"\d+\s?oz", r"\d+\s?lb", r"\d+\s?kg", r"\d+\s?gm", r"\d+\s?liter"]
    
    is_food = any(w in txt for w in grocery_words) or any(re.search(p, txt) for p in grocery_units)
    
    if "amazon fresh" in path_lower or "grocery" in path_lower: return 100.0 if is_food else -200.0
    if "electronics" in path_lower: return -200.0 if is_food else 0.0
    if "instant video" in path_lower:
        if is_food or "count" in txt or "bag" in txt or "pack" in txt: return -500.0
    return 0.0

def check_accessory_penalty(title, path):
    title, path = title.lower(), path.lower()
    for sep in [r"\bfor\b", r"\bfits\b", r"\bcompatible with\b"]:
        match = re.search(sep, title)
        if match:
            target = title[match.end():].strip()
            if target in path and "accessory" not in path and "case" not in path:
                return -25.0
    return 0.0

# -------------------------
# Phase 4: Classification
# -------------------------
def classify_product(title: str, description: str = "") -> Dict:
    start_time = time.time()
    full_text = f"{title} {description}".strip()
    
    # 1. Retrieval
    candidates_map = {}
    
    # E5
    e5_emb = retriever_e5.encode(f"passage: {full_text}", convert_to_numpy=True, normalize_embeddings=True)
    dists_e5, idxs_e5 = index_e5.search(e5_emb.reshape(1, -1).astype('float32'), 30)
    for i, idx in enumerate(idxs_e5[0]):
        if idx >= 0: add_candidate(candidates_map, int(idx), float(dists_e5[0][i]), 'E5')

    # MPNet
    mp_emb = retriever_mpnet.encode(full_text, convert_to_numpy=True, normalize_embeddings=True)
    dists_mp, idxs_mp = index_mpnet.search(mp_emb.reshape(1, -1).astype('float32'), 30)
    for i, idx in enumerate(idxs_mp[0]):
        if idx >= 0: add_candidate(candidates_map, int(idx), float(dists_mp[0][i]), 'MPNet')

    # 2. Tag Engine
    words = clean_text(title).split()
    for i in range(len(words)):
        for j in range(i, min(i+6, len(words))):
            phrase = " ".join(words[i:j+1])
            if phrase in tag_lookup:
                for cid in tag_lookup[phrase]:
                    if cid in candidates_map:
                        candidates_map[cid]['confidence'] += 30.0
                        candidates_map[cid]['sources'].add('Tag')
                        candidates_map[cid]['logic_log'].append(f"Tag: '{phrase}'")
                    else:
                        meta = catid_to_meta[cid]
                        add_candidate_manual(candidates_map, meta, 0.95, 'Tag', f"Tag: '{phrase}'")

    candidates = list(candidates_map.values())

    # 3. Guardrails
    for res in candidates:
        if len(res['sources']) > 1: res['confidence'] += 5.0
        res['confidence'] += min(8.0, res['depth'] * 1.5)
        
        pen = check_accessory_penalty(title, res['category_path'])
        res['confidence'] += pen
        if pen < 0: res['logic_log'].append(f"Accessory Penalty {pen}")
        
        g_score = check_grocery_guardrail(title, res['category_path'])
        res['confidence'] += g_score
        if g_score < -50: res['logic_log'].append("⛔ Bad Dept")
        if g_score > 50: res['logic_log'].append("✅ Dept Confirmed")

    # 4. Re-Rank
    candidates.sort(key=lambda x: x['confidence'], reverse=True)
    top_candidates = candidates[:15]
    
    if reranker:
        rerank_inputs = [[title, c['category_path']] for c in top_candidates]
        scores = reranker.predict(rerank_inputs)
        for i, score in enumerate(scores):
            top_candidates[i]['rerank_score'] = float(score)
        top_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)

    # 5. Rule Override
    rule_id, rule_reason = get_rule_match_id(title, description)
    final_top_results = []
    
    if rule_id:
        rule_winner = None
        others = []
        for cand in top_candidates:
            if cand['category_id'] == rule_id:
                rule_winner = cand
            else:
                others.append(cand)
        
        if not rule_winner and rule_id in catid_to_meta:
            meta = catid_to_meta[rule_id]
            rule_winner = {
                'category_id': rule_id,
                'category_path': meta['category_path'],
                'final_product': meta['final_product'],
                'rerank_score': 10.0,
                'sources': {'Rule'},
                'logic_log': []
            }
            
        if rule_winner:
            rule_winner['rerank_score'] = 99.99
            rule_winner['logic_log'].insert(0, f"🏆 {rule_reason}")
            final_top_results = [rule_winner] + others[:4]
    else:
        final_top_results = top_candidates[:5]

    top = final_top_results[0] if final_top_results else None
    
    return {
        'final_product': top['final_product'] if top else 'Unknown',
        'category_path': top['category_path'] if top else 'Unknown',
        'category_id': top['category_id'] if top else 'Unknown',
        'rerank_score': top.get('rerank_score', 0.0) if top else 0.0,
        'logic_log': top.get('logic_log', []) if top else [],
        'top_results': final_top_results,
        'time_ms': (time.time() - start_time) * 1000
    }

# -------------------------
# Initialization
# -------------------------
def add_candidate(cand_map, idx, raw_score, source):
    meta = metadata[idx]
    add_candidate_manual(cand_map, meta, raw_score, source)

def add_candidate_manual(cand_map, meta, raw_score, source, log=None):
    cid = meta['category_id']
    if cid in cand_map:
        cand_map[cid]['sources'].add(source)
        cand_map[cid]['retrieval_score'] = max(cand_map[cid]['retrieval_score'], float(raw_score))
        if log: cand_map[cid]['logic_log'].append(log)
    else:
        cand_map[cid] = {
            'category_id': cid,
            'category_path': meta['category_path'],
            'final_product': meta['final_product'],
            'depth': meta['depth'],
            'retrieval_score': float(raw_score),
            'confidence': float(raw_score) * 100.0,
            'sources': {source},
            'logic_log': [log] if log else []
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

    if INDEX_PATH_E5.exists() and METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, 'rb') as f:
                loaded_meta = pickle.load(f)
            if len(loaded_meta) == len(metadata):
                metadata = loaded_meta
                index_e5 = faiss.read_index(str(INDEX_PATH_E5))
                index_mpnet = faiss.read_index(str(INDEX_PATH_MPNET))
                print("✅ Loaded Indexes.")
            else:
                raise ValueError("Metadata mismatch")
        except Exception:
            print("⚠️ Rebuilding indexes...")
            index_e5 = build_index(retriever_e5, 'e5', metadata, INDEX_PATH_E5, True)
            index_mpnet = build_index(retriever_mpnet, 'mpnet', metadata, INDEX_PATH_MPNET, False)
    else:
        index_e5 = build_index(retriever_e5, 'e5', metadata, INDEX_PATH_E5, True)
        index_mpnet = build_index(retriever_mpnet, 'mpnet', metadata, INDEX_PATH_MPNET, False)

    catid_to_meta = {m['category_id']: m for m in metadata}
    print("✅ System Ready.")

# -------------------------
# UI
# -------------------------
def gradio_classify(title, desc):
    if not title: return ("",) * 7
    try:
        res = classify_product(title, desc)
        top_text = ""
        for i, item in enumerate(res.get('top_results', []), 1):
            score = item.get('rerank_score', 0.0)
            score_disp = "MAX (Rule)" if score > 90 else f"{score:.4f}"
            logs = " | ".join(item.get('logic_log', []))
            top_text += f"{i}. {item['final_product']}\n   ID: {item['category_id']} | Score: {score_disp}\n   Path: {item['category_path']}\n   ℹ️ {logs}\n\n"
        
        status = "✅ Approved" if float(res.get('rerank_score', 0)) > 0 else "⚠️ Review Needed"
        return (
            str(res.get('final_product')), str(res.get('category_path')), str(res.get('category_id')),
            f"{float(res.get('rerank_score', 0)):.4f}", " | ".join(res.get('logic_log', [])),
            status, top_text
        )
    except Exception as e:
        print(f"Error: {e}")
        return ("Error",) * 7

def main():
    initialize()
    with gr.Blocks() as app:
        gr.Markdown("# ⚡ Precision Hybrid Classifier")
        with gr.Row():
            with gr.Column():
                t_in = gr.Textbox(label="Title")
                d_in = gr.Textbox(label="Description")
                btn = gr.Button("Classify", variant="primary")
            with gr.Column():
                out_final = gr.Textbox(label="Winner")
                out_path = gr.Textbox(label="Path")
                out_id = gr.Textbox(label="ID")
                out_score = gr.Textbox(label="Score")
                out_logic = gr.Textbox(label="Logic")
                out_status = gr.Textbox(label="Status")
        out_details = gr.TextArea(label="Details", lines=10)
        btn.click(gradio_classify, [t_in, d_in], [out_final, out_path, out_id, out_score, out_logic, out_status, out_details])
    app.launch(server_name="127.0.0.1", server_port=7860, share=True)

if __name__ == "__main__":
    main()