from fpdf import FPDF
import os

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

text = """
ChatBox Project Verification Document

This is a test document to verify the functionality of the RAG Chatbot.
It proves that the system can read and process text-based PDFs correctly.

English Section:
The ChatBox application uses Ollama for local LLM inference and a vector database for retrieval.
It supports multiple languages including English and Arabic.

Arabic Support Note:
(Note: Standard FPDF doesn't support Arabic script natively without fonts, so we are simulating content here in English for the test, but the system is designed to handle Arabic text if the source PDF has it encoded correctly.)

Configuration:
- Model: ALLaM 7B
- Embeddings: nomic-embed-text
- UI: Streamlit with RTL support
"""

pdf.multi_cell(0, 10, text)

output_dir = "pdfs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_path = os.path.join(output_dir, "test_document.pdf")
pdf.output(output_path)
print(f"Created {output_path}")
