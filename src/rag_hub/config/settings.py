"""Central constants for the RAG hub. Import from here, never hardcode."""
import os
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent  # repo root
DATA_DIR = ROOT_DIR / "data"
PDF_DIR = DATA_DIR / "raw" / "financebench" / "pdfs"
PAGES_CACHE_DIR = DATA_DIR / "processed" / "pages"
CHECKPOINTS_DIR = DATA_DIR / "processed" / "checkpoints"
PARENT_STORE_DIR = DATA_DIR / "processed" / "parent_store"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"
RESULTS_DIR = ROOT_DIR / "eval_results"
SMOKE_20_PATH = EVAL_DIR / "smoke_20.jsonl"
SMOKE_50_PATH = EVAL_DIR / "smoke_50.jsonl"
BM25_INDEX_PATH = PROCESSED_DIR / "bm25_index.pkl"

# Qdrant
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME: str = "financebench_v1"
EMBEDDING_DIM: int = 768

# Model names
GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"
GEMINI_LLM_MODEL: str = "gemini-2.5-flash"
BGE_RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
COLBERT_MODEL: str = "colbert-ir/colbertv2.0"
NLI_MODEL: str = "cross-encoder/nli-deberta-v3-base"

# GCP
GCP_PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "")
GCP_LOCATION: str = os.getenv("GCP_LOCATION", "us-central1")

# Thresholds
HALLUCINATION_THRESHOLD: float = 0.25
CRAG_CONFIDENCE_THRESHOLD: float = 0.5
MAX_GENERATION_ITERATIONS: int = 2
