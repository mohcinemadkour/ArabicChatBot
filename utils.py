import logging
import subprocess
from pathlib import Path
from typing import List, Optional

import arabic_reshaper
from bidi.algorithm import get_display
from langdetect import detect, LangDetectException

import config
import pytesseract
import fitz  # PyMuPDF
from PIL import Image
from langchain_core.documents import Document
from translations import TRANSLATIONS


def setup_logging(log_file: str = None, level: str = None) -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        log_file: Path to log file (defaults to config.LOG_FILE)
        level: Logging level (defaults to config.LOG_LEVEL)
    
    Returns:
        Configured logger instance
    """
    log_file = log_file or config.LOG_FILE
    level = level or config.LOG_LEVEL
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    logger = logging.getLogger('chatbox')
    logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def get_available_ollama_models() -> List[str]:
    """
    Get list of available Ollama models installed locally.
    
    Returns:
        List of model names
    """
    logger = logging.getLogger('chatbox')
    
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=config.OLLAMA_TIMEOUT
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            models = []
            
            for line in lines[1:]:  # Skip header line
                if line.strip():
                    model_name = line.split()[0]
                    # Keep full model name including tag
                    models.append(model_name)
            
            unique_models = sorted(list(set(models)))
            logger.info(f"Found {len(unique_models)} Ollama models: {unique_models}")
            return unique_models
        else:
            logger.error(f"Ollama command failed with return code {result.returncode}")
            return []
            
    except subprocess.TimeoutExpired:
        logger.error(f"Ollama command timed out after {config.OLLAMA_TIMEOUT} seconds")
        return []
    except FileNotFoundError:
        logger.error("Ollama not found. Please install Ollama from https://ollama.ai/")
        return []
    except Exception as e:
        logger.error(f"Error getting Ollama models: {e}")
        return []


def validate_document_size(total_chars: int, max_chars: int = None) -> None:
    """
    Validate that document size is within acceptable limits.
    
    Args:
        total_chars: Total character count
        max_chars: Maximum allowed characters (defaults to config.MAX_DOCUMENT_SIZE)
    
    Raises:
        ValueError: If document exceeds size limit
    """
    max_chars = max_chars or config.MAX_DOCUMENT_SIZE
    
    if total_chars > max_chars:
        raise ValueError(
            f"Documents too large ({total_chars:,} characters). "
            f"Maximum allowed: {max_chars:,} characters. "
            f"Consider splitting into smaller documents or increasing MAX_DOCUMENT_SIZE."
        )


def detect_language(text: str) -> str:
    """
    Detect the language of the given text.
    
    Args:
        text: The text to analyze
        
    Returns:
        Language code ('en' or 'ar')
    """
    if not config.ENABLE_AUTO_DETECT:
        return config.DEFAULT_LANGUAGE
        
    try:
        lang = detect(text)
        if lang == 'ar':
            return 'ar'
        return 'en'
    except LangDetectException:
        return config.DEFAULT_LANGUAGE


def normalize_arabic_text(text: str) -> str:
    """
    Normalize Arabic text by removing diacritics and unifying Alef forms.
    
    Args:
        text: Input Arabic text
        
    Returns:
        Normalized text
    """
    if not text:
        return text
        
    # Common Arabic normalization chars
    text = text.replace("أ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("آ", "ا")
    text = text.replace("ة", "h")
    text = text.replace("ى", "ي")
    
    # Remove tashkeel (diacritics) - simplified version
    tashkeel = ["َ", "ً", "ُ", "ٌ", "ِ", "ٍ", "ْ", "ّ"]
    for char in tashkeel:
        text = text.replace(char, "")
        
    return text


def reshape_arabic_text(text: str) -> str:
    """
    Reshape Arabic text and apply bidirectional algorithm for proper display.
    Useful for terminal output and systems that don't support RTL natively.
    
    Args:
        text: Input text
        
    Returns:
        Reshaped and bidi-processed text
    """
    try:
        reshaped_text = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped_text)
        return bidi_text
    except Exception:
        return text


def get_ui_text(key: str, language: str = 'en', **kwargs) -> str:
    """
    Get translated UI text.
    
    Args:
        key: Translation key
        language: Language code ('en' or 'ar')
        **kwargs: Format arguments
        
    Returns:
        Translated string
    """
    lang_dict = TRANSLATIONS.get(language, TRANSLATIONS['en'])
    text = lang_dict.get(key, TRANSLATIONS['en'].get(key, key))
    
    if kwargs:
        try:
            return text.format(**kwargs)
        except KeyError:
            return text
            
    return text


def extract_text_with_ocr(pdf_path: str) -> List[Document]:
    """
    Extract text from a PDF using OCR as a fallback for scanned images.
    Uses PyMuPDF (fitz) to convert pages to images and Tesseract for OCR.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of Document objects
    """
    logger = logging.getLogger('chatbox')
    logger.info(f"Starting OCR extraction for {pdf_path}...")
    
    if config.TESSERACT_PATH:
        pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_PATH
        
    try:
        documents = []
        # Open the PDF with PyMuPDF
        doc = fitz.open(pdf_path)
        logger.info(f"Opened PDF with {len(doc)} pages for OCR")
        
        for i, page in enumerate(doc):
            # Render page to an image (pixmap)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Zoom for better OCR
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Perform OCR on the image
            text = pytesseract.image_to_string(img)
            
            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={"source": pdf_path, "page": i + 1, "method": "ocr"}
                ))
                logger.info(f"Successfully OCR'd page {i + 1}")
            else:
                logger.warning(f"No text extracted from page {i + 1} during OCR")
                
        doc.close()
        return documents
        
    except Exception as e:
        logger.error(f"OCR extraction failed for {pdf_path}: {e}")
        return []
