"""Optimized configuration for faster embeddings."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Directories
BASE_DIR = Path(__file__).parent
PDF_DIRECTORY = os.getenv("PDF_DIRECTORY", str(BASE_DIR / "pdfs"))
CHROMA_DB_CLI = os.getenv("CHROMA_DB_CLI", str(BASE_DIR / "chroma_db"))
CHROMA_DB_STREAMLIT = os.getenv("CHROMA_DB_STREAMLIT", str(BASE_DIR / "chroma_db_streamlit"))

# Model Configuration
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:latest")

# OPTIMIZED: Use dedicated embedding model or sentence-transformers
EMBEDDING_TYPE = os.getenv("EMBEDDING_TYPE", "sentence-transformers")  # Options: "ollama", "sentence-transformers"
DEFAULT_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")  # For Ollama
SENTENCE_TRANSFORMER_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")  # Fast & good quality

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# Document Processing - OPTIMIZED for speed
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))  # Smaller chunks = faster (was 1000)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))  # Less overlap = fewer chunks (was 200)
MAX_DOCUMENT_SIZE = int(os.getenv("MAX_DOCUMENT_SIZE", "10000000"))

# Retrieval
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "4"))

# Timeouts
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "10"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "chatbox.log")

# Performance
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))  # Process embeddings in batches
