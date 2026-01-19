# ChatBox - RAG Chatbot with Ollama

A Retrieval-Augmented Generation (RAG) chatbot that allows you to chat with your PDF documents using local LLMs via Ollama.

## Features

- 📚 **PDF Processing**: Upload and process multiple PDF documents
- 💬 **Interactive Chat**: CLI and Web UI interfaces
- 🧠 **Local LLM**: Uses Ollama for privacy and offline capability
- 💾 **Persistent Storage**: Vector database persists between sessions
- 🔍 **Source Citations**: See which documents informed each answer
- 🎯 **Conversation Memory**: Maintains context across chat sessions
- ⚙️ **Configurable**: Easy configuration via environment variables
- 🌐 **Arabic Support**: Full Arabic language support with RTL UI and bilingual capabilities

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running
- At least one Ollama model pulled (e.g., `ollama pull llama3.1`)

## Installation

1. Clone the repository:
```bash
git clone <repo-url>
cd ChatBox
```

2. Create virtual environment:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file (optional):
```bash
cp .env.example .env
# Edit .env with your preferred settings
```

## Usage

### CLI Interface

```bash
python main.py
```

**Commands:**
- Type your questions naturally
- `clear` - Clear conversation memory
- `lang` - Switch language (English/Arabic)
- `help` - Show help message
- `quit` or `exit` - Exit the application

### Web Interface (Streamlit)

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Adding Documents

**For CLI:**
Place PDF files in the `pdfs/` directory (or the directory specified in `.env`)

**For Web UI:**
Use the file uploader in the sidebar

## Configuration

All configuration can be done via environment variables. Copy `.env.example` to `.env` and customize:

```bash
# Model Configuration
OLLAMA_MODEL=llama3.1:latest
OLLAMA_EMBEDDING_MODEL=llama3.1:latest
LLM_TEMPERATURE=0.7

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_DOCUMENT_SIZE=10000000

# Retrieval
TOP_K_RESULTS=4

# Language Settings
DEFAULT_LANGUAGE=en  # 'en' or 'ar'
ENABLE_AUTO_DETECT=true
ARABIC_FONT=Cairo
```

See `.env.example` for all available options.

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov

# Run specific test file
pytest tests/test_chatbot.py -v
```

### Code Formatting

```bash
# Format code
black .

# Check formatting
black --check .
```

### Type Checking

```bash
mypy .
```

### Linting

```bash
flake8 .
```

## Project Structure

```
ChatBox/
├── app.py                  # Streamlit web interface
├── main.py                 # CLI interface and RAGChatbot class
├── config.py               # Configuration management
├── utils.py                # Utility functions
├── requirements.txt        # Python dependencies
├── .env.example           # Example environment variables
├── pyproject.toml         # Development tool configuration
├── tests/                 # Test files
│   └── test_chatbot.py
├── pdfs/                  # PDF documents (CLI)
├── chroma_db/             # Vector database (CLI)
└── chroma_db_streamlit/   # Vector database (Web UI)
```

## Troubleshooting

### Issue: "No Ollama models found"
**Solution:** Run `ollama pull llama3.1` to download a model

### Issue: "Out of memory"
**Solution:** Reduce `CHUNK_SIZE` in `.env` or process fewer/smaller PDFs

### Issue: "Documents too large"
**Solution:** Increase `MAX_DOCUMENT_SIZE` in `.env` or split PDFs into smaller files

### Issue: "Failed to load PDF"
**Solution:** Ensure PDF is not corrupted and is a valid PDF file

## Features in Detail

### Conversation Memory
The chatbot maintains conversation context, allowing for follow-up questions and natural dialogue flow.

### Source Citations
Every answer includes references to the source documents and page numbers, ensuring transparency and verifiability.

### Error Handling
Comprehensive error handling with detailed logging helps diagnose issues quickly.

### Logging
All operations are logged to `chatbox.log` for debugging and monitoring.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT

## Acknowledgments

- Built with [LangChain](https://langchain.com/)
- Powered by [Ollama](https://ollama.ai/)
- UI with [Streamlit](https://streamlit.io/)
