"""
app/utils/prometheus_utils.py

Defines all custom Prometheus metrics for observability.
These are exposed via the /metrics endpoint in main.py.
"""

from prometheus_client import Counter, Histogram, Summary

# --- Request-Level Metrics ---
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds",
    "Application request latency in seconds.",
    ["endpoint", "stage"],  # e.g., endpoint="/classify", stage="full_request"
)

APP_ERRORS = Counter(
    "app_errors_total",
    "Total number of application errors.",
    ["endpoint", "error_type"],
)

# --- ML & Search Metrics ---
FAISS_SEARCH_LATENCY = Histogram(
    "app_faiss_search_latency_seconds",
    "Latency of FAISS index.search() operations."
)

EMBEDDING_LATENCY = Histogram(
    "app_embedding_latency_seconds",
    "Latency of generating sentence embeddings via ModelEnsemble."
)

# --- Indexing Metrics ---
INDEX_REBUILD_LATENCY = Summary(
    "app_index_rebuild_latency_seconds",
    "Time taken to complete a full index build/load."
)

INDEX_VECTORS_TOTAL = Histogram(
    "app_index_vectors_total",
    "Total number of vectors in the active FAISS index.",
    buckets=[10_000, 25_000, 50_000, 100_000], # Customize to your 34k
)