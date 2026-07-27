import os
from pathlib import Path
from dotenv import load_dotenv

import src.utils  # noqa: F401 - Applies stream flush patch

BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env
load_dotenv(BASE_DIR / ".env")

DOCUMENTS_DIR = BASE_DIR / "documents"
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "index"
CHROMA_DIR = INDEX_DIR / "chroma"
BM25_PATH = INDEX_DIR / "bm25.pkl"
MANIFEST_PATH = INDEX_DIR / "manifest.json"

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))

# Retrieval
TOP_K_VECTOR = int(os.getenv("TOP_K_VECTOR", "12"))
TOP_K_BM25 = int(os.getenv("TOP_K_BM25", "12"))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "6"))
HYBRID_VECTOR_WEIGHT = float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.6"))
HYBRID_BM25_WEIGHT = float(os.getenv("HYBRID_BM25_WEIGHT", "0.4"))

# Models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Generation
MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", "5"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))

