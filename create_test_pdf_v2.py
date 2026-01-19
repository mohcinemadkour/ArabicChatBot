from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os

output_dir = "pdfs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_path = os.path.join(output_dir, "test_document_v2.pdf")

c = canvas.Canvas(output_path, pagesize=letter)
c.drawString(100, 750, "ChatBox Test Document")
c.drawString(100, 730, "This is a valid PDF document with text content.")
c.drawString(100, 710, "It is used to verify that the text extraction logic works correctly.")
c.drawString(100, 690, "If you can read this, the system is working.")
c.save()

print(f"Created {output_path}")

# Verify it exists
if os.path.exists(output_path):
    print(f"VERIFIED: File exists at {output_path}")
    print(f"Size: {os.path.getsize(output_path)} bytes")
else:
    print("ERROR: File was not created!")
