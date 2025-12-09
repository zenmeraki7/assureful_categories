#!/usr/bin/env python3
"""
main.py – FastAPI Backend for Precision Hybrid Classifier
Works on Render (no Gradio required)
"""

import os
import json
import pickle
import re
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import faiss
import torch

from sentence_transformers import SentenceTransformer, CrossEncoder
from fastapi import FastAPI
from pydantic import BaseModel

# -------------------------
# CONFIG
# -------------------------
CACHE_DIR = Path("cache")
DATA_DIR = Path("data")
CACHE_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

MODEL_NAME_E5 = "intfloat/e5-base-v2"
MODEL_NAME_MPNET = "sentence-transformers/all-mpnet-base-v2"
MODEL_NAME_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CSV_PATH = DATA_DIR / "categories.csv"
TAGS_PATH = DATA_DIR / "tags.json"
INDEX_PATH_E5 = CACHE_DIR / "index_e5.faiss"
INDEX_PATH_MPNET = CACHE_DIR / "index_mpnet.faiss"
METADATA_PATH = CACHE_DIR / "metadata.pkl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# GLOBALS
# -------------------------
retriever_e5 = None
retriever_mpnet = None
reranker = None
index_e5 = None
index_mpnet = None
metadata: List[Dict] = []
catid_to_meta: Dict[str, Dict] = {}
tag_lookup: Dict[str, List[str]] = {}

# -------------------------
# UTIL FUNCTIONS
# -------------------------
def clean_text(text: str) -> str:
    if not text: return ""
    s = str(text).lower().strip()
    s = re.sub(r"[^\w\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def get_final_product_name(path: str) -> str:
    parts = [p for p in path.split('/') if p]
    return parts[-1] if parts else path

# -------------------------
# LOAD DATA / METADATA
# -------------------------
def build_metadata_from_csv(csv_path: Path):
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    rows = []
    for _, row in df.iterrows():
        cid = str(row.iloc[0]).strip()
        path = str(row.iloc[1]).strip()
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

    metadata = build_metadata_from_csv(CSV_PATH)
    catid_to_meta = {m['category_id']: m for m in metadata}

    # TAG ENGINE
    if TAGS_PATH.exists():
        with open(TAGS_PATH, "r", encoding="utf-8") as f:
            tags_data = json.load(f)

        for cid, tags in tags_data.items():
            if cid not in catid_to_meta: continue
            for t in tags:
                t = clean_text(t)
                if len(t) > 2:
                    tag_lookup.setdefault(t, []).append(cid)


# -------------------------
# RULE ENGINE
# (Same rules as your Gradio version)
# -------------------------

PHONE_KEYWORDS = { "iphone", "samsung", "pixel", "motorola", "oneplus", "vivo", "oppo", "realme",
                   "xiaomi", "mi", "redmi", "huawei", "nokia", "nothing" }

CARRIER_KEYWORDS = {"att", "verizon", "t-mobile", "cricket", "metro", "airtel", "jio"}

BOOK_KEYWORDS = {"book", "guide", "manual", "ebook"}

ACCESSORY_KEYWORD_TO_ID = {
    "back cover": "3081461011",
    "screen protector": "3081461011",
}

UNLOCKED_CELL_PHONES_ID = "2407749011"
IPHONE_BOOKS_ID = "6133978011"
CARRIER_PHONES_ID = "2407748011"

def get_rule_match_id(title: str):
    txt = clean_text(title)

    # iPhone Books
    if "iphone" in txt and any(b in txt for b in BOOK_KEYWORDS):
        return IPHONE_BOOKS_ID, "Rule: iPhone Book"

    # Accessories
    for k, cid in ACCESSORY_KEYWORD_TO_ID.items():
        if k in txt:
            return cid, f"Rule: Accessory {k}"

    # Carrier Phones
    if any(c in txt for c in CARRIER_KEYWORDS) and any(p in txt for p in PHONE_KEYWORDS):
        return CARRIER_PHONES_ID, "Rule: Carrier Phone"

    # Unlocked Phones
    if any(p in txt for p in PHONE_KEYWORDS):
        return UNLOCKED_CELL_PHONES_ID, "Rule: Unlocked Phone"

    return None, ""

# -------------------------
# CLASSIFICATION
# -------------------------
def classify(title: str, description: str = ""):
    full = f"{title} {description}".strip()

    # Encode
    e5_emb = retriever_e5.encode(f"passage: {full}", convert_to_numpy=True, normalize_embeddings=True)
    mp_emb = retriever_mpnet.encode(full, convert_to_numpy=True, normalize_embeddings=True)

    # Retrieve top-30
    d1, i1 = index_e5.search(e5_emb.reshape(1, -1), 30)
    d2, i2 = index_mpnet.search(mp_emb.reshape(1, -1), 30)

    candidates = {}

    def add(idx, score, src):
        m = metadata[idx]
        cid = m['category_id']
        if cid not in candidates:
            candidates[cid] = {
                "category_id": cid,
                "category_path": m['category_path'],
                "final_product": m['final_product'],
                "score": float(score),
                "source": {src},
            }
        else:
            candidates[cid]["score"] = max(candidates[cid]["score"], float(score))
            candidates[cid]["source"].add(src)

    for i, idx in enumerate(i1[0]):
        add(int(idx), float(d1[0][i]), "e5")

    for i, idx in enumerate(i2[0]):
        add(int(idx), float(d2[0][i]), "mpnet")

    candidates = list(candidates.values())

    # RERANK
    rr_inputs = [[title, c['category_path']] for c in candidates[:15]]
    scores = reranker.predict(rr_inputs)

    for i, s in enumerate(scores):
        candidates[i]["rerank"] = float(s)

    candidates.sort(key=lambda x: x.get("rerank", 0), reverse=True)

    # RULE OVERRIDE
    rule_id, rule_reason = get_rule_match_id(title)
    if rule_id:
        for c in candidates:
            if c["category_id"] == rule_id:
                c["rerank"] = 999
                c["rule_applied"] = rule_reason

    candidates.sort(key=lambda x: x["rerank"], reverse=True)

    return candidates[:5]


# -------------------------
# INITIALIZE MODELS
# -------------------------
def initialize():
    global retriever_e5, retriever_mpnet, reranker, index_e5, index_mpnet, metadata

    retriever_e5 = SentenceTransformer(MODEL_NAME_E5, device=DEVICE)
    retriever_mpnet = SentenceTransformer(MODEL_NAME_MPNET, device=DEVICE)
    reranker = CrossEncoder(MODEL_NAME_RERANKER, device=DEVICE)

    load_data()

    if INDEX_PATH_E5.exists():
        index_e5 = faiss.read_index(str(INDEX_PATH_E5))
        index_mpnet = faiss.read_index(str(INDEX_PATH_MPNET))
    else:
        texts_e5 = [m["indexed_text"] for m in metadata]
        emb_e5 = retriever_e5.encode(texts_e5, convert_to_numpy=True, normalize_embeddings=True)
        index_e5 = faiss.IndexFlatIP(emb_e5.shape[1])
        index_e5.add(emb_e5)
        faiss.write_index(index_e5, str(INDEX_PATH_E5))

        texts_mp = [m["category_path"] for m in metadata]
        emb_mp = retriever_mpnet.encode(texts_mp, convert_to_numpy=True, normalize_embeddings=True)
        index_mpnet = faiss.IndexFlatIP(emb_mp.shape[1])
        index_mpnet.add(emb_mp)
        faiss.write_index(index_mpnet, str(INDEX_PATH_MPNET))


# -------------------------
# FASTAPI SERVER
# -------------------------
app = FastAPI(title="Precision Hybrid Classifier")

class Item(BaseModel):
    title: str
    description: str = ""

@app.post("/classify")
def api_classify(item: Item):
    res = classify(item.title, item.description)
    return {"results": res}


@app.get("/")
def root():
    return {"status": "ok", "message": "Hybrid Classifier API running"}

# Run init
initialize()
