from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from PIL import Image, ImageDraw, ImageFont
import os
import io

def create_scanned_pdf(output_path):
    # 1. Create an image with text
    img = Image.new('RGB', (600, 400), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    
    # Try to use a font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    d.text((50, 50), "This is a SCANNED PDF test.", fill=(0, 0, 0), font=font)
    d.text((50, 100), "It contains only an image, no selectable text.", fill=(0, 0, 0), font=font)
    d.text((50, 150), "OCR must be used to extract this message.", fill=(0, 0, 0), font=font)
    
    # 2. Save image to a byte buffer
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    # 3. Create PDF and draw the image onto it
    c = canvas.Canvas(output_path, pagesize=letter)
    
    # Draw the image from buffer
    # Note: reportlab can take a file path, we'll save it temporarily
    temp_img_path = "temp_scanned_page.png"
    img.save(temp_img_path)
    c.drawImage(temp_img_path, 100, 400, width=400, height=300)
    c.save()
    
    # Cleanup
    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)
    
    print(f"Created scanned PDF at: {output_path}")

if __name__ == "__main__":
    output_dir = "pdfs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, "scanned_test.pdf")
    create_scanned_pdf(output_path)
