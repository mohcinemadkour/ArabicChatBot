import os
import logging
from utils import extract_text_with_ocr, setup_logging
import config

# Setup logging to see results
logger = setup_logging()

def verify_ocr():
    # 1. Path to the scanned test PDF
    pdf_path = os.path.join("pdfs", "scanned_test.pdf")
    
    if not os.path.exists(pdf_path):
        print(f"ERROR: Test file {pdf_path} not found.")
        return

    print(f"Testing OCR on: {pdf_path}")
    print(f"Using Tesseract path: {config.TESSERACT_PATH}")
    
    try:
        # 2. Attempt OCR extraction
        docs = extract_text_with_ocr(pdf_path)
        
        # 3. Check results
        if docs:
            print("\n✅ OCR Extraction SUCCESSFUL!")
            print(f"Extracted {len(docs)} pages.")
            for i, doc in enumerate(docs):
                print(f"\n--- Page {doc.metadata['page']} Content ---")
                print(doc.page_content.strip())
                print("------------------------------")
        else:
            print("\n❌ OCR Extraction FAILED: No text returned.")
            
    except Exception as e:
        print(f"\n❌ OCR Extraction ERROR: {str(e)}")

if __name__ == "__main__":
    verify_ocr()
