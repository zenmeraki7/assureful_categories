#!/usr/bin/env python3
"""
train.py
Build normalized embeddings + FAISS index for category catalog,
build parent embeddings, save synonyms from tags.json and optionally
train a LightGBM classifier and a simple confidence calibrator.

Assumptions / Files:
- categories CSV: category_only_path.csv (Category_ID,Category_path,Final_Category)
- optional: data/tags.json  (map category_id -> list of phrases)
- optional: validation.csv (columns: product_title,category_id) used for calibrator / classifier

Outputs to ./cache:
- main_index.faiss
- metadata.pkl
- parent_embeddings.pkl
- cross_store_synonyms.pkl
- model_info.json
- calibrator.pkl (if validation exists)
- classifier.pkl (if --train-classifier used)
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

# sentence-transformers + faiss
from sentence_transformers import SentenceTransformer
import faiss

# sklearn for calibrator and simple preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# optional LightGBM (install if you plan to train classifier)
try:
    import importlib
    lgb = importlib.import_module("lightgbm")
    LGB_AVAILABLE = True
except Exception:
    lgb = None
    LGB_AVAILABLE = False

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True, parents=True)

DEFAULT_BATCH_SIZE_CPU = 256
DEFAULT_BATCH_SIZE_GPU = 16


def normalize_path_sep(path: str) -> str:
    if not isinstance(path, str):
        return ""
    s = path.strip()
    s = s.replace("/", " > ")
    s = " > ".join([p.strip() for p in s.split(">") if p.strip()])
    return s


def path_to_levels(path: str) -> List[str]:
    n = normalize_path_sep(path)
    return [p.strip() for p in n.split(" > ") if p.strip()]


def safe_pickle_save(obj, p: Path):
    with open(p, "wb") as f:
        pickle.dump(obj, f)


def build_encoder(model_name: str, use_cuda: bool):
    device = "cuda" if use_cuda else "cpu"
    print(f"Loading encoder: {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)
    if use_cuda:
        try:
            import torch
            model = model.half()
            print("Using FP16 on GPU to conserve VRAM.")
        except Exception:
            pass
    return model


def encode_texts(model: SentenceTransformer, texts: List[str], use_cuda: bool) -> np.ndarray:
    batch_size = DEFAULT_BATCH_SIZE_GPU if use_cuda else DEFAULT_BATCH_SIZE_CPU
    print(f"Encoding {len(texts):,} texts in batches of {batch_size} ...")
    all_emb = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        emb = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        all_emb.append(emb.astype("float32"))
    embeddings = np.vstack(all_emb)
    print("Final embeddings shape:", embeddings.shape)
    return embeddings


def build_faiss_index(np_emb: np.ndarray, use_gpu: bool = False):
    d = np_emb.shape[1]
    print(f"Building IndexFlatIP (d={d}) on {'GPU' if use_gpu else 'CPU'}")
    index = faiss.IndexFlatIP(d)
    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print("Converted FAISS index to GPU")
        except Exception as e:
            print("GPU conversion failed; using CPU index:", e)
    index.add(np_emb)
    print("Index ntotal:", index.ntotal)
    return index


def make_parent_embeddings(metadata: List[Dict], embeddings: np.ndarray) -> Dict[str, np.ndarray]:
    """
    For each possible parent path (every prefix), average embeddings of its children.
    This helps hierarchical boosting during inference.
    """
    parent_map = {}
    count_map = {}
    for i, meta in enumerate(metadata):
        levels = meta.get("levels", [])
        for depth in range(1, len(levels)):
            parent = " > ".join(levels[:depth])
            if not parent:
                continue
            parent_map.setdefault(parent, np.zeros(embeddings.shape[1], dtype="float32"))
            count_map.setdefault(parent, 0)
            parent_map[parent] += embeddings[i]
            count_map[parent] += 1

    # average + normalize
    from numpy.linalg import norm
    final = {}
    for k, vec in parent_map.items():
        cnt = count_map.get(k, 1)
        avg = vec / float(cnt)
        nrm = np.linalg.norm(avg) + 1e-12
        final[k] = (avg / nrm).astype("float32")
    return final


def load_tags_json(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # ensure keys are strings
        return {str(k): [str(x) for x in v] for k, v in data.items()}
    except Exception as e:
        print("Failed to load tags.json:", e)
        return {}


def train_calibrator(encoder, metadata, faiss_index, val_path: Path, model_name: str, use_cuda: bool):
    """
    Build a simple calibrator mapping raw cosine similarity of (product -> true category emb)
    to a probability. Uses sklearn LogisticRegression on one feature (raw_score).
    Expects validation.csv with columns product_title,category_id
    """
    print("Training calibrator using:", val_path)
    df = pd.read_csv(val_path, dtype=str, keep_default_na=False)
    if "product_title" not in df.columns or "category_id" not in df.columns:
        print("validation.csv must have 'product_title' and 'category_id' columns. Skipping calibrator.")
        return None

    examples = []
    labels = []
    # Build a mapping category_id -> embedding (from metadata)
    id_to_idx = {m["category_id"]: i for i, m in enumerate(metadata)}

    # prepare product embeddings in batches
    titles = df["product_title"].astype(str).tolist()
    prod_embs = encode_texts(encoder, [f"query: {t}" for t in titles], use_cuda=use_cuda)

    for i, row in df.iterrows():
        cid = str(row["category_id"]).strip()
        if cid not in id_to_idx:
            # not in catalog, skip sample
            continue
        cat_idx = id_to_idx[cid]
        cat_emb = metadata[cat_idx].get("_embedding")  # we will attach embeddings later temporarily
        if cat_emb is None:
            continue
        q_emb = prod_embs[i].reshape(1, -1).astype("float32")
        raw = float(np.dot(q_emb, cat_emb.reshape(-1, 1))[0][0])  # cosine because normalized
        # positive
        examples.append([raw])
        labels.append(1)

        # generate few negatives by sampling other categories
        # sample up to 2 random negatives
        negs = 2
        for _ in range(negs):
            import random
            rand_idx = random.randrange(len(metadata))
            if rand_idx == cat_idx:
                continue
            neg_emb = metadata[rand_idx].get("_embedding")
            if neg_emb is None:
                continue
            raw_neg = float(np.dot(q_emb, neg_emb.reshape(-1, 1))[0][0])
            examples.append([raw_neg])
            labels.append(0)

    if not examples:
        print("No examples for calibrator (maybe category ids mismatch). Skipping.")
        return None

    X = np.array(examples, dtype="float32")
    y = np.array(labels, dtype="int8")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=200)
    clf.fit(Xs, y)
    print("Calibrator trained (logistic regression on raw cosine).")
    return {"calibrator": clf, "scaler": scaler}


def attach_embeddings_to_metadata(metadata: List[Dict], embeddings: np.ndarray):
    for i, m in enumerate(metadata):
        m["_embedding"] = embeddings[i]


def detach_embeddings_from_metadata(metadata: List[Dict]):
    for m in metadata:
        if "_embedding" in m:
            del m["_embedding"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="categories CSV (Category_ID,Category_path,Final_Category)")
    parser.add_argument("--model", default="intfloat/e5-base-v2", help="embedding model")
    parser.add_argument("--gpu", action="store_true", help="use GPU for encoding if available (careful with 4GB)")
    parser.add_argument("--clean-cache", action="store_true", help="delete other cache files after build")
    parser.add_argument("--train-classifier", action="store_true", help="train LightGBM classifier on validation.csv (optional)")
    parser.add_argument("--validation", default="data/validation.csv", help="validation CSV used for calibrator / classifier")
    parser.add_argument("--tags", default="data/tags.json", help="tags.json path (optional)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit("CSV not found: " + str(csv_path))

    print("Reading CSV:", csv_path)
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if df.shape[1] < 2:
        raise SystemExit("CSV must have at least 2 columns: Category_ID, Category_path")

    # columns
    cols = list(df.columns)
    cid_col, path_col = cols[0], cols[1]
    print("Using columns:", cid_col, path_col)

    metadata = []
    texts_for_encoding = []
    for idx, row in df.iterrows():
        cid = str(row[cid_col]).strip()
        raw_path = str(row[path_col]).strip()
        norm_path = normalize_path_sep(raw_path)
        levels = path_to_levels(norm_path)
        final = levels[-1] if levels else norm_path or cid
        # include both path and final in canonical text to encode
        text = f"category: {norm_path}. leaf: {final}."
        metadata.append({
            "category_id": cid,
            "category_path": norm_path,
            "final": final,
            "levels": levels,
            "depth": len(levels)
        })
        texts_for_encoding.append(text)

    print(f"Prepared {len(metadata):,} metadata entries")

    # encoder
    use_cuda = args.gpu
    encoder = build_encoder(args.model, use_cuda=use_cuda)

    # encode categories
    cat_embeddings = encode_texts(encoder, texts_for_encoding, use_cuda=use_cuda)

    # Attach embeddings temporarily for calibrator builder
    attach_embeddings_to_metadata(metadata, cat_embeddings)

    # parent embeddings
    parent_emb = make_parent_embeddings(metadata, cat_embeddings)
    print(f"Built {len(parent_emb):,} parent embeddings")

    # Build CPU FAISS index (IP on normalized vectors -> cosine)
    index = build_faiss_index(cat_embeddings, use_gpu=False)

    # save index (FAISS CPU index)
    faiss_path = CACHE_DIR / "main_index.faiss"
    faiss.write_index(index, str(faiss_path))
    print("Saved FAISS index:", faiss_path)

    # save metadata (we will strip embeddings before saving to reduce pickle size)
    detach_embeddings_from_metadata(metadata)
    meta_path = CACHE_DIR / "metadata.pkl"
    safe_pickle_save(metadata, meta_path)
    print("Saved metadata:", meta_path)

    # save parent embeddings
    parent_path = CACHE_DIR / "parent_embeddings.pkl"
    safe_pickle_save(parent_emb, parent_path)
    print("Saved parent embeddings:", parent_path)

    # save model_info
    info = {
        "model_name": args.model,
        "num_categories": len(metadata),
        "embedding_dim": cat_embeddings.shape[1]
    }
    with open(CACHE_DIR / "model_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print("Saved model_info.json")

    # store tags.json -> cross_store_synonyms (just preserve structure)
    tags = load_tags_json(Path(args.tags))
    if tags:
        syn_p = CACHE_DIR / "cross_store_synonyms.pkl"
        safe_pickle_save(tags, syn_p)
        print("Saved cross_store_synonyms.pkl from tags.json (size: %d)" % len(tags))

    # calibrator: use validation.csv if exists
    val_path = Path(args.validation)
    calibrator_obj = None
    if val_path.exists():
        # we need embeddings attached again for calibrator training
        attach_embeddings_to_metadata(metadata, cat_embeddings)
        calibrator_obj = train_calibrator(encoder, metadata, index, val_path, args.model, use_cuda=use_cuda)
        detach_embeddings_from_metadata(metadata)
        if calibrator_obj:
            safe_pickle_save(calibrator_obj, CACHE_DIR / "calibrator.pkl")
            print("Saved calibrator.pkl")

    # optional LightGBM classifier
    if args.train_classifier:
        if not LGB_AVAILABLE:
            print("LightGBM not available. Install lightgbm to train classifier.")
        else:
            val_path2 = Path(args.validation)
            if not val_path2.exists():
                print("validation.csv required to train classifier. Skipping classifier training.")
            else:
                # create training set from validation.csv
                dfv = pd.read_csv(val_path2, dtype=str, keep_default_na=False)
                if "product_title" not in dfv.columns or "category_id" not in dfv.columns:
                    print("validation.csv must contain product_title and category_id. Skipping classifier.")
                else:
                    # encode product titles
                    prod_texts = [f"query: {t}" for t in dfv["product_title"].astype(str).tolist()]
                    prod_embs = encode_texts(encoder, prod_texts, use_cuda=use_cuda)
                    # map category ids to numeric labels
                    cat_to_label = {m["category_id"]: i for i, m in enumerate(metadata)}
                    labels = []
                    rows = []
                    for i, row in dfv.iterrows():
                        cid = row["category_id"]
                        if cid not in cat_to_label:
                            continue
                        labels.append(cat_to_label[cid])
                        rows.append(prod_embs[i])
                    if len(rows) < 50:
                        print("Not enough training rows for classifier. Need >=50. Skipping.")
                    else:
                        X = np.vstack(rows)
                        y = np.array(labels, dtype=np.int32)
                        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
                        lgb_train = lgb.Dataset(X_train, label=y_train)
                        lgb_eval = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
                        params = {
                            "objective": "multiclass",
                            "num_class": int(max(y) + 1),
                            "metric": "multi_logloss",
                            "verbosity": -1,
                            "num_threads": 4,
                            "learning_rate": 0.1,
                            "num_leaves": 31
                        }
                        print("Training LightGBM classifier (may take time)...")
                        gbm = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_eval], early_stopping_rounds=30, num_boost_round=500)
                        # save classifier and mapping
                        clf_path = CACHE_DIR / "classifier.pkl"
                        safe_pickle_save({"model": gbm, "cat_to_label": cat_to_label, "label_to_cat": {v: k for k, v in cat_to_label.items()}}, clf_path)
                        print("Saved classifier.pkl")

    # cleanup if asked
    if args.clean_cache:
        keep = {"main_index.faiss", "metadata.pkl", "model_info.json", "parent_embeddings.pkl", "cross_store_synonyms.pkl"}
        if calibrator_obj:
            keep.add("calibrator.pkl")
        # remove everything else in cache
        removed = []
        for p in CACHE_DIR.iterdir():
            if p.name in keep:
                continue
            try:
                p.unlink()
                removed.append(p.name)
            except Exception:
                pass
        if removed:
            print("Removed cache files:", removed)

    print("DONE. Index + data saved to cache/")

if __name__ == "__main__":
    main()
