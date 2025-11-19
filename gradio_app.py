#!/usr/bin/env python3
"""
Gradio App for Product Category Classification
Model: intfloat/e5-base-v2 (must match training)
Requires: pip install gradio sentence-transformers faiss-cpu numpy pickle5
"""

import gradio as gr
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import re
from pathlib import Path
import time

# ====================================================================
# CONFIG
# ====================================================================
CACHE_DIR = Path("cache")
MODEL_NAME = "intfloat/e5-base-v2"
FAISS_INDEX_PATH = CACHE_DIR / "main_index.faiss"
METADATA_PATH = CACHE_DIR / "metadata.pkl"
SYN_PATH = CACHE_DIR / "cross_store_synonyms.pkl"

encoder = None
faiss_index = None
metadata = []
cross_store_synonyms = {}

# ====================================================================
# UTILITIES
# ====================================================================
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_cross_store_synonyms():
    synonyms = {
        'washing machine': {'laundry machine', 'washer', 'clothes washer', 'washing appliance'},
        'laundry machine': {'washing machine', 'washer', 'clothes washer'},
        'dryer': {'drying machine', 'clothes dryer', 'tumble dryer'},
        'refrigerator': {'fridge', 'cooler', 'ice box', 'cooling appliance'},
        'dishwasher': {'dish washer', 'dish cleaning machine'},
        'microwave': {'microwave oven', 'micro wave'},
        'vacuum': {'vacuum cleaner', 'hoover', 'vac'},
        'tv': {'television', 'telly', 'smart tv', 'display'},
        'laptop': {'notebook', 'portable computer', 'laptop computer'},
        'mobile': {'phone', 'cell phone', 'smartphone', 'cellphone'},
        'tablet': {'ipad', 'tab', 'tablet computer'},
        'headphones': {'headset', 'earphones', 'earbuds', 'ear buds'},
        'speaker': {'audio speaker', 'sound system', 'speakers'},
        'sofa': {'couch', 'settee', 'divan'},
        'wardrobe': {'closet', 'armoire', 'cupboard'},
        'drawer': {'chest of drawers', 'dresser'},
        'pants': {'trousers', 'slacks', 'bottoms'},
        'sweater': {'jumper', 'pullover', 'sweatshirt'},
        'sneakers': {'trainers', 'tennis shoes', 'running shoes'},
        'jacket': {'coat', 'blazer', 'outerwear'},
        'cooker': {'stove', 'range', 'cooking range'},
        'blender': {'mixer', 'food processor', 'liquidizer'},
        'kettle': {'electric kettle', 'water boiler'},
        'stroller': {'pram', 'pushchair', 'buggy', 'baby carriage'},
        'diaper': {'nappy', 'nappies'},
        'pacifier': {'dummy', 'soother'},
        'wrench': {'spanner', 'adjustable wrench'},
        'flashlight': {'torch', 'flash light'},
        'screwdriver': {'screw driver'},
        'tap': {'faucet', 'water tap'},
        'bin': {'trash can', 'garbage can', 'waste bin'},
        'curtain': {'drape', 'window covering'},
        'guillotine': {'paper cutter', 'paper trimmer', 'blade cutter'},
        'trimmer': {'cutter', 'cutting tool', 'edge cutter'},
        'stapler': {'stapling machine', 'staple gun'},
        'magazine': {'periodical', 'journal', 'publication'},
        'comic': {'comic book', 'graphic novel', 'manga'},
        'ebook': {'e-book', 'digital book', 'electronic book'},
        'kids': {'children', 'child', 'childrens', 'youth', 'junior'},
        'women': {'womens', 'ladies', 'female', 'lady'},
        'men': {'mens', 'male', 'gentleman'},
        'baby': {'infant', 'newborn', 'toddler'},
    }

    expanded = {}
    for term, syns in synonyms.items():
        expanded[term] = set(syns)
        for syn in syns:
            if syn not in expanded:
                expanded[syn] = set()
            expanded[syn].add(term)
            expanded[syn].update(syns - {syn})
    return expanded

def extract_cross_store_terms(text: str):
    cleaned = clean_text(text)
    words = cleaned.split()
    all_terms = set()
    all_terms.add(cleaned)
    for word in words:
        if len(word) > 2:
            all_terms.add(word)
            if word in cross_store_synonyms:
                all_terms.update(cross_store_synonyms[word])
    for i in range(len(words) - 1):
        phrase = f"{words[i]} {words[i+1]}"
        all_terms.add(phrase)
        if phrase in cross_store_synonyms:
            all_terms.update(cross_store_synonyms[phrase])
    for i in range(len(words) - 2):
        phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
        all_terms.add(phrase)
    return list(all_terms)

def build_enhanced_query(title, description="", max_synonyms=10):
    title_clean = clean_text(title)
    description_clean = clean_text(description)
    synonyms_list = extract_cross_store_terms(f"{title_clean} {description_clean}")
    enhanced_query = ' '.join([title_clean]*3 + synonyms_list[:max_synonyms])
    return enhanced_query, synonyms_list[:20]

def encode_query(text: str):
    emb = encoder.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    return emb.astype('float32')

# ====================================================================
# CLASSIFICATION
# ====================================================================
def classify_product(title, description="", top_k=5):
    start_time = time.time()
    query_text, matched_terms = build_enhanced_query(title, description)
    query_embedding = encode_query(query_text)
    distances, indices = faiss_index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx >= len(metadata):
            continue
        meta = metadata[idx]
        similarity = 1 - distances[0][i]
        confidence_pct = float(similarity) * 100
        final_product = meta.get('levels', [])[-1] if meta.get('levels') else meta['category_path'].split('/')[-1]
        results.append({
            'rank': i+1,
            'category_id': str(meta['category_id']),
            'category_path': meta['category_path'],
            'final_product': final_product,
            'confidence': round(confidence_pct, 2),
            'depth': meta.get('depth', 0)
        })

    if not results:
        return {
            'error': 'No results found',
            'product': title
        }

    best = results[0]
    conf_pct = best['confidence']
    if conf_pct >= 90:
        conf_level = "EXCELLENT"
    elif conf_pct >= 85:
        conf_level = "VERY HIGH"
    elif conf_pct >= 80:
        conf_level = "HIGH"
    elif conf_pct >= 75:
        conf_level = "GOOD"
    elif conf_pct >= 70:
        conf_level = "MEDIUM"
    else:
        conf_level = "LOW"

    processing_time = (time.time() - start_time) * 1000

    return {
        'product': title,
        'category_id': best['category_id'],
        'category_path': best['category_path'],
        'final_product': best['final_product'],
        'confidence': f"{conf_level} ({conf_pct:.2f}%)",
        'confidence_percent': conf_pct,
        'depth': best['depth'],
        'matched_terms': matched_terms,
        'top_5_results': results,
        'processing_time_ms': round(processing_time, 2)
    }

# ====================================================================
# LOAD MODEL & INDEX
# ====================================================================
def load_model():
    global encoder, faiss_index, metadata, cross_store_synonyms
    print("Loading sentence-transformer model...")
    encoder = SentenceTransformer(MODEL_NAME)
    print("Model loaded.")

    print("Loading FAISS index...")
    faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
    print(f"FAISS index loaded: {faiss_index.ntotal} vectors.")

    print("Loading metadata...")
    with open(METADATA_PATH, 'rb') as f:
        metadata = pickle.load(f)
    print(f"Metadata loaded: {len(metadata)} categories.")

    print("Loading cross-store synonyms...")
    if SYN_PATH.exists():
        with open(SYN_PATH, 'rb') as f:
            cross_store_synonyms = pickle.load(f)
        print(f"Loaded {len(cross_store_synonyms)} synonyms from file.")
    else:
        cross_store_synonyms = build_cross_store_synonyms()
        print(f"Built {len(cross_store_synonyms)} default synonyms.")

# ====================================================================
# GRADIO FUNCTION
# ====================================================================
def classify_gradio(title, description=""):
    result = classify_product(title, description)
    top_match = str(result.get('final_product', ''))
    category_path = str(result.get('category_path', ''))
    confidence = str(result.get('confidence', ''))
    matched_terms = ', '.join(result.get('matched_terms', [])) if result.get('matched_terms') else ''
    top5_html = ""
    for item in result.get('top_5_results', []):
        top5_html += f"{item['rank']}. {item['final_product']} (ID: {item['category_id']}, Confidence: {item['confidence']}%)\n"
    return top_match, category_path, confidence, matched_terms, top5_html

# ====================================================================
# MAIN GRADIO APP
# ====================================================================
def main():
    load_model()
    iface = gr.Interface(
        fn=classify_gradio,
        inputs=[
            gr.Textbox(label="Product Title"),
            gr.Textbox(label="Description")
        ],
        outputs=[
            gr.Textbox(label="Predicted Product"),
            gr.Textbox(label="Category Path"),
            gr.Textbox(label="Confidence"),
            gr.Textbox(label="Matched Terms"),
            gr.Textbox(label="Top 5 Alternatives")
        ],
        title="🎯 Product Category Classifier",
        description="Classify products with full cross-store synonyms and embeddings"
    )
    # Launch with a public shareable link
    iface.launch(share=True)

if __name__ == "__main__":
    main()