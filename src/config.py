"""
Central configuration: paths, model names, and pipeline parameters.
Edit this file to swap models or tune chunking without touching agent code.
"""

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw_docs"
VECTORSTORE_DIR = BASE_DIR / "src" / "vectorstore" / "chroma_db"

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM
LLM_MODEL = "gpt-4o-mini"

# Chunking
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

# Retrieval
TOP_K = 4
