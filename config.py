"""Configuration management for ChatBox application."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Directories
BASE_DIR = Path(__file__).parent
PDF_DIRECTORY = os.getenv("PDF_DIRECTORY", str(BASE_DIR / "pdfs"))
CHROMA_DB_CLI = os.getenv("CHROMA_DB_CLI", str(BASE_DIR / "chroma_db"))
CHROMA_DB_STREAMLIT = os.getenv("CHROMA_DB_STREAMLIT", str(BASE_DIR / "chroma_db_streamlit"))

# Model Configuration
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "ikhalid/allam:7b")
DEFAULT_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# Document Processing
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
MAX_DOCUMENT_SIZE = int(os.getenv("MAX_DOCUMENT_SIZE", "10000000"))  # 10MB

# Retrieval
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "4"))

# Timeouts
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "10"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "chatbox.log")

# Language Support
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "en")
SUPPORTED_LANGUAGES = ["en", "ar"]
ENABLE_AUTO_DETECT = os.getenv("ENABLE_AUTO_DETECT", "true").lower() == "true"
ARABIC_FONT = "Cairo"  # Font family for Arabic text

# OCR Configuration
# On Windows, you might need to specify the path to tesseract.exe
# Example: r'C:\Program Files\Tesseract-OCR\tesseract.exe'
TESSERACT_PATH = os.getenv("TESSERACT_PATH", None)
