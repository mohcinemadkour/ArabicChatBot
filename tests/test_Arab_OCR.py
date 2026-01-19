import fitz  # PyMuPDF
from paddleocr import PaddleOCR
from arabic_reshaper import reshape
from bidi.algorithm import get_display
import os

# Initialize PaddleOCR for Arabic
# use_textline_orientation replaces use_angle_cls in newer versions
# Explicitly using device='cpu' and disabling mkldnn to avoid backend errors
ocr = PaddleOCR(use_textline_orientation=True, lang='ar', device='cpu', enable_mkldnn=False)

def process_arabic_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"ERROR: File not found: {pdf_path}")
        return []
        
    doc = fitz.open(pdf_path)
    extracted_data = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        
        # 1. Extract Images
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_filename = f"page{page_index}_img{img_index}.png"
            with open(img_filename, "wb") as f:
                f.write(base_image["image"])
            print(f"Extracted image: {img_filename}")

        # 2. Extract & Fix Arabic Text
        # We use OCR to ensure the layout/reading order is captured
        # Higher resolution for better OCR
        page_img = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        temp_img = f"temp_page_{page_index}.png"
        page_img.save(temp_img)
        
        result = ocr.ocr(temp_img)
        page_text = ""
        
        # Support for PaddleOCR 3.3.2 (PaddleX) dictionary response
        if result and len(result) > 0 and isinstance(result[0], dict):
            texts = result[0].get('rec_texts', [])
            for text in texts:
                # Fix Arabic formatting for the RAG prompt
                fixed_text = get_display(reshape(text))
                page_text += fixed_text + " "
        # Fallback for standard PaddleOCR list-of-lists format
        elif result and result[0]:
            for line in result[0]:
                try:
                    text = line[1][0]
                    fixed_text = get_display(reshape(text))
                    page_text += fixed_text + " "
                except (IndexError, TypeError):
                    continue
        
        extracted_data.append({"page": page_index, "text": page_text})
        print(f"\n[REAL-TIME] --- Page {page_index} Text Preview ---")
        print(page_text[:200] + ("..." if len(page_text) > 200 else ""))
        
        # Cleanup temp image - DISABLED for inspection
        # if os.path.exists(temp_img):
        #     os.remove(temp_img)
        
    return extracted_data

# Example usage
if __name__ == "__main__":
    # Path from user request
    pdf_path = r"pdfs/أحاديث إصلاح القلوب .pdf"
    
    # Fallback to existing Arabic PDF if the above doesn't exist
    if not os.path.exists(pdf_path):
        fallback_path = os.path.join("pdfs", "أحاديث إصلاح القلوب .pdf")
        if os.path.exists(fallback_path):
            print(f"Specified file not found. Falling back to: {fallback_path}")
            pdf_path = fallback_path

    data = process_arabic_pdf(pdf_path)
    
    for item in data:
        print(f"\n--- Page {item['page']} ---")
        print(item['text'])