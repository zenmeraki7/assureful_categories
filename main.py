# """
# main.py

# Main application entry point for the FastAPI server.
# Handles application lifecycle, service creation, dependency injection,
# and API routing.
# """

# import logging
# import sys
# from contextlib import asynccontextmanager
# from fastapi import FastAPI, Depends
# from prometheus_fastapi_instrumentator import Instrumentator

# # --- Core App ---
# from app.utils.logging_utils import setup_logging
# from app.core.runtime import RuntimeConfig
# from app.core.system_setup import create_runtime_config
# from app.core.active_indices import get_active_indices, FaissIndices

# # --- Services ---
# from app.services.vocabulary_service import VocabularyService
# from app.services.text_enhancer_service import TextEnhancerService
# from app.services.embedding_service import EmbeddingService
# from app.ml.cross_encoder_service import CrossEncoderService
# from app.services.faiss_service import FaissIndexService
# from app.services.search.direct_search_service import DirectSearchService
# from app.services.search.progressive_search_service import ProgressiveSearchService
# from app.services.search.reranking_search_service import ReRankingSearchService
# from app.services.search.ensemble_strategy_service import EnsembleStrategyService
# from app.services.classification_service import ClassificationService
# from app.services.feedback_service import FeedbackService

# # --- API Routers ---
# from app.api.v1.endpoints import classification, health, admin

# # --- 1. Global State & Lifespan ---
# lifespan_state = {}

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """
#     Handles application startup and shutdown events.
#     All services are created here.
#     """
#     setup_logging()
#     logger = logging.getLogger(__name__)
#     logger.info("--- 🚀 Magnum Opus: Application Startup ---")
    
#     try:
#         # 1. Create RuntimeConfig (loads categories.json)
#         runtime_cfg = create_runtime_config()
#         lifespan_state["runtime_config"] = runtime_cfg
        
#         # 2. Create VocabularyService (pre-computes all keywords)
#         vocab_service = VocabularyService(runtime_cfg)
#         lifespan_state["vocabulary_service"] = vocab_service
        
#         # 3. Create TextEnhancerService
#         text_enhancer_service = TextEnhancerService(vocab_service)
#         lifespan_state["text_enhancer_service"] = text_enhancer_service
        
#         # 4. Create EmbeddingService and load models
#         embedding_service = EmbeddingService(runtime_cfg)
#         embedding_service.load_models() # Blocking, run at startup
#         lifespan_state["embedding_service"] = embedding_service
        
#         # 5. Load Cross-Encoder Model
#         cross_encoder_service = CrossEncoderService(runtime_cfg)
#         lifespan_state["cross_encoder_service"] = cross_encoder_service
        
#         # 6. Get category embeddings (from cache or generate)
#         embeddings = await embedding_service.get_category_embeddings()
        
#         # 7. Create FaissIndexService and build/load indices
#         faiss_service = FaissIndexService(runtime_cfg, embeddings)
#         await faiss_service.build_or_load_indices() # Async
#         lifespan_state["faiss_service"] = faiss_service
        
#         # 8. Initialize All Strategies
#         direct_strategy = DirectSearchService(runtime_cfg, get_active_indices)
#         progressive_strategy = ProgressiveSearchService(runtime_cfg, get_active_indices)
#         reranking_strategy = ReRankingSearchService(
#             runtime_cfg,
#             cross_encoder_service,
#             get_active_indices
#         )
#         ensemble_strategy = EnsembleStrategyService()
        
#         # 9. Initialize Feedback Service
#         feedback_service = FeedbackService()
#         lifespan_state["feedback_service"] = feedback_service

#         # 10. Initialize Main ClassificationService
#         classification_service = ClassificationService(
#             cfg=runtime_cfg,
#             embedding_service=embedding_service,
#             text_enhancer_service=text_enhancer_service,
#             direct_strategy=direct_strategy,
#             progressive_strategy=progressive_strategy,
#             reranking_strategy=reranking_strategy,
#             ensemble_strategy=ensemble_strategy
#         )
#         lifespan_state["classification_service"] = classification_service
        
#         logger.info("--- ✅ Application Startup Complete ---")
        
#     except Exception as e:
#         logger.critical(f"FATAL: Application startup failed: {e}", exc_info=True)
#         sys.exit(1) # Fail fast

#     yield  # --- Application is now running ---

#     # --- Shutdown ---
#     logger.info("--- 🛬 Application Shutdown ---")
#     lifespan_state.clear()


# # --- 2. Create FastAPI App ---
# app = FastAPI(
#     title="Magnum Opus Product Classifier",
#     description="Enterprise-grade classification API "
#                 "with Retrieve & Re-rank.",
#     version="2.0.0",
#     lifespan=lifespan
# )

# # --- 3. Dependency Injectors ---
# # These functions provide services to our API endpoints

# def get_runtime_config() -> RuntimeConfig:
#     return lifespan_state["runtime_config"]

# def get_classification_service() -> ClassificationService:
#     return lifespan_state["classification_service"]

# def get_embedding_service() -> EmbeddingService:
#     return lifespan_state["embedding_service"]
    
# def get_faiss_service() -> FaissIndexService:
#     return lifespan_state["faiss_service"]

# def get_feedback_service() -> FeedbackService:
#     return lifespan_state["feedback_service"]

# # --- 4. Include API Routers ---
# app.include_router(
#     health.router, 
#     prefix="/api/v1", 
#     tags=["Health"]
# )
# app.include_router(
#     classification.router, 
#     prefix="/api/v1", 
#     tags=["Classification"]
# )
# app.include_router(
#     admin.router, 
#     prefix="/api/v1", 
#     tags=["Admin"]
# )

# # --- 5. Add Prometheus Metrics ---
# # This exposes a /metrics endpoint
# Instrumentator(
#     should_instrument_requests=False,
#     excluded_handlers=["/metrics"]
# ).instrument(app).expose(app)

# # --- 6. Root Endpoint ---
# @app.get("/", include_in_schema=False)
# async def root():
#     return {
#         "message": "Magnum Opus Product Classifier API. See /docs for details."
#     }
"""
main.py

Main application entry point for the FastAPI server.
Handles application lifecycle, service creation, dependency injection,
and API routing.
"""

import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

# --- Core App ---
from app.utils.logging_utils import setup_logging
from app.core.runtime import RuntimeConfig
from app.core.system_setup import create_runtime_config
from app.core.active_indices import get_active_indices, FaissIndices

# --- Services ---
from app.services.vocabulary_service import VocabularyService
from app.services.text_enhancer_service import TextEnhancerService
from app.services.embedding_service import EmbeddingService
from app.ml.cross_encoder_service import CrossEncoderService
from app.services.faiss_service import FaissIndexService
from app.services.search.direct_search_service import DirectSearchService
from app.services.search.progressive_search_service import ProgressiveSearchService
from app.services.search.reranking_search_service import ReRankingSearchService
from app.services.search.ensemble_strategy_service import EnsembleStrategyService
from app.services.classification_service import ClassificationService
from app.services.feedback_service import FeedbackService

# --- API Routers ---
from app.api.v1.endpoints import classification, health, admin


# ==============================================================
# 1. Global State & Lifespan
# ==============================================================

lifespan_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    All services are created here.
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("--- 🚀 Magnum Opus: Application Startup ---")

    try:
        # 1. Create RuntimeConfig (loads categories.json)
        runtime_cfg = create_runtime_config()
        lifespan_state["runtime_config"] = runtime_cfg

        # 2. Create VocabularyService (pre-computes all keywords)
        vocab_service = VocabularyService(runtime_cfg)
        lifespan_state["vocabulary_service"] = vocab_service

        # 3. Create TextEnhancerService
        text_enhancer_service = TextEnhancerService(vocab_service)
        lifespan_state["text_enhancer_service"] = text_enhancer_service

        # 4. Create EmbeddingService and load models
        embedding_service = EmbeddingService(runtime_cfg)
        embedding_service.load_models()  # Blocking, run at startup
        lifespan_state["embedding_service"] = embedding_service

        # 5. Load Cross-Encoder Model
        cross_encoder_service = CrossEncoderService(runtime_cfg)
        lifespan_state["cross_encoder_service"] = cross_encoder_service

        # 6. Get category embeddings (from cache or generate)
        embeddings = await embedding_service.get_category_embeddings()

        # 7. Create FaissIndexService and build/load indices
        faiss_service = FaissIndexService(runtime_cfg, embeddings)
        await faiss_service.build_or_load_indices()
        lifespan_state["faiss_service"] = faiss_service

        # 8. Initialize All Search Strategies
        direct_strategy = DirectSearchService(runtime_cfg, get_active_indices)
        progressive_strategy = ProgressiveSearchService(runtime_cfg, get_active_indices)
        reranking_strategy = ReRankingSearchService(
            runtime_cfg,
            cross_encoder_service,
            get_active_indices
        )
        ensemble_strategy = EnsembleStrategyService()

        # 9. Initialize Feedback Service
        feedback_service = FeedbackService()
        lifespan_state["feedback_service"] = feedback_service

        # 10. Initialize Main ClassificationService
        classification_service = ClassificationService(
            cfg=runtime_cfg,
            embedding_service=embedding_service,
            text_enhancer_service=text_enhancer_service,
            direct_strategy=direct_strategy,
            progressive_strategy=progressive_strategy,
            reranking_strategy=reranking_strategy,
            ensemble_strategy=ensemble_strategy
        )
        lifespan_state["classification_service"] = classification_service

        logger.info("--- ✅ Application Startup Complete ---")

    except Exception as e:
        logger.critical(f"FATAL: Application startup failed: {e}", exc_info=True)
        sys.exit(1)  # Fail fast

    yield  # --- Application is now running ---

    # --- Shutdown ---
    logger.info("--- 🛬 Application Shutdown ---")
    lifespan_state.clear()


# ==============================================================
# 2. Create FastAPI App
# ==============================================================

app = FastAPI(
    title="Magnum Opus Product Classifier",
    description="Enterprise-grade classification API with Retrieve & Re-rank.",
    version="2.0.0",
    lifespan=lifespan
)


# ==============================================================
# 3. Dependency Injectors
# ==============================================================

def get_runtime_config() -> RuntimeConfig:
    return lifespan_state["runtime_config"]

def get_classification_service() -> ClassificationService:
    return lifespan_state["classification_service"]

def get_embedding_service() -> EmbeddingService:
    return lifespan_state["embedding_service"]

def get_faiss_service() -> FaissIndexService:
    return lifespan_state["faiss_service"]

def get_feedback_service() -> FeedbackService:
    return lifespan_state["feedback_service"]


# ==============================================================
# 4. Include API Routers
# ==============================================================

app.include_router(
    health.router,
    prefix="/api/v1",
    tags=["Health"]
)
app.include_router(
    classification.router,
    prefix="/api/v1",
    tags=["Classification"]
)
app.include_router(
    admin.router,
    prefix="/api/v1",
    tags=["Admin"]
)


# ==============================================================
# 5. Add Prometheus Metrics
# ==============================================================

# ✅ Fixed version – compatible with latest library
instrumentator = Instrumentator(
    excluded_handlers=["/metrics"]
)
instrumentator.instrument(app).expose(app)


# ==============================================================
# 6. Root Endpoint
# ==============================================================

@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Magnum Opus Product Classifier API. See /docs for details."
    }
