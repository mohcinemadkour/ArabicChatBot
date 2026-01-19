"""Unit tests for RAG Chatbot."""

import os
import tempfile
from pathlib import Path

import pytest
from langchain_core.documents import Document

# Import modules to test
import config
from main import RAGChatbot
from utils import validate_document_size, get_available_ollama_models


class TestRAGChatbot:
    """Test cases for RAGChatbot class."""
    
    @pytest.fixture
    def chatbot(self):
        """Create a chatbot instance for testing."""
        return RAGChatbot(
            model_name="llama3.1:latest",
            embedding_model="llama3.1:latest",
            persist_directory="./test_chroma_db"
        )
    
    def test_chatbot_initialization(self, chatbot):
        """Test chatbot initializes correctly."""
        assert chatbot.model_name == "llama3.1:latest"
        assert chatbot.embedding_model == "llama3.1:latest"
        assert chatbot.llm is not None
        assert chatbot.embeddings is not None
        assert chatbot.memory is not None
    
    def test_load_pdfs_empty_directory(self, chatbot):
        """Test loading PDFs from empty directory raises error."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(ValueError, match="No PDF files found"):
                chatbot.load_pdfs(tmp_dir)
    
    def test_process_documents(self, chatbot):
        """Test document processing."""
        docs = [Document(page_content="Test content " * 100)]
        chunks = chatbot.process_documents(docs, chunk_size=100, chunk_overlap=20)
        
        assert len(chunks) > 0
        # Chunks should be roughly chunk_size or smaller
        assert all(len(chunk.page_content) <= 150 for chunk in chunks)
    
    def test_process_documents_with_validation(self, chatbot):
        """Test document size validation."""
        # Create a very large document that exceeds the limit
        large_content = "x" * (config.MAX_DOCUMENT_SIZE + 1000)
        docs = [Document(page_content=large_content)]
        
        with pytest.raises(ValueError, match="Documents too large"):
            chatbot.process_documents(docs)
    
    def test_query_without_vectorstore(self, chatbot):
        """Test querying without initialized vectorstore."""
        with pytest.raises(ValueError, match="Vector store not initialized"):
            chatbot.create_qa_chain()
    
    def test_clear_memory(self, chatbot):
        """Test clearing conversation memory."""
        # Add some content to memory
        chatbot.memory.save_context(
            {"question": "Test question"},
            {"answer": "Test answer"}
        )
        
        # Clear memory
        chatbot.clear_memory()
        
        # Memory should be empty
        assert len(chatbot.memory.chat_memory.messages) == 0


class TestUtils:
    """Test cases for utility functions."""
    
    def test_validate_document_size_valid(self):
        """Test validation with valid document size."""
        # Should not raise any exception
        validate_document_size(1000)
    
    def test_validate_document_size_invalid(self):
        """Test validation with invalid document size."""
        with pytest.raises(ValueError, match="Documents too large"):
            validate_document_size(config.MAX_DOCUMENT_SIZE + 1000)
    
    def test_get_available_ollama_models(self):
        """Test getting available Ollama models."""
        models = get_available_ollama_models()
        # Should return a list (may be empty if Ollama not installed)
        assert isinstance(models, list)


class TestConfig:
    """Test cases for configuration."""
    
    def test_config_values(self):
        """Test that config values are set."""
        assert config.DEFAULT_MODEL is not None
        assert config.DEFAULT_EMBEDDING_MODEL is not None
        assert config.CHUNK_SIZE > 0
        assert config.CHUNK_OVERLAP >= 0
        assert config.MAX_DOCUMENT_SIZE > 0
        assert config.TOP_K_RESULTS > 0
    
    def test_config_types(self):
        """Test that config values have correct types."""
        assert isinstance(config.CHUNK_SIZE, int)
        assert isinstance(config.CHUNK_OVERLAP, int)
        assert isinstance(config.MAX_DOCUMENT_SIZE, int)
        assert isinstance(config.TOP_K_RESULTS, int)
        assert isinstance(config.LLM_TEMPERATURE, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
