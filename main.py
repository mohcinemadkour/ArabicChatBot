"""
RAG Chatbot with Ollama - Main Module

This module implements a Retrieval-Augmented Generation chatbot using Ollama LLMs
and PDF documents as knowledge sources.
"""

import glob
import logging
import os
from typing import List

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
import config
from utils import (
    setup_logging, 
    get_available_ollama_models, 
    validate_document_size,
    detect_language, 
    reshape_arabic_text,
    get_ui_text,
    extract_text_with_ocr
)
from translations import PROMPTS, BILINGUAL_PROMPT

# Setup logging
logger = setup_logging()


class RAGChatbot:
    """RAG Chatbot using Ollama LLMs and vector database for document retrieval."""
    
    def __init__(
        self,
        model_name: str = None,
        embedding_model: str = None,
        persist_directory: str = None,
        temperature: float = None,
        language: str = None
    ):
        """
        Initialize the RAG Chatbot.
        
        Args:
            model_name: Name of the Ollama LLM model (defaults to config.DEFAULT_MODEL)
            embedding_model: Name of the Ollama embedding model (defaults to config.DEFAULT_EMBEDDING_MODEL)
            persist_directory: Directory to store vector database (defaults to config.CHROMA_DB_CLI)
            temperature: LLM temperature setting (defaults to config.LLM_TEMPERATURE)
            language: Interface language 'en' or 'ar' (defaults to config.DEFAULT_LANGUAGE)
        """
        self.model_name = model_name or config.DEFAULT_MODEL
        self.embedding_model = embedding_model or config.DEFAULT_EMBEDDING_MODEL
        self.persist_directory = persist_directory or config.CHROMA_DB_CLI
        temperature = temperature or config.LLM_TEMPERATURE
        self.language = language or config.DEFAULT_LANGUAGE
        
        logger.info(f"Initializing RAGChatbot with model: {self.model_name}, language: {self.language}")
        
        try:
            # Initialize Ollama LLM
            self.llm = Ollama(model=self.model_name, temperature=temperature)
            
            # Initialize embeddings
            self.embeddings = OllamaEmbeddings(model=self.embedding_model)
            
            logger.info("LLM and embeddings initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM or embeddings: {e}")
            raise
        
        # Initialize vector store
        self.vectorstore = None
        self.retriever = None
        
        # Initialize memory for conversation
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Set default prompt based on language settings
        # We start with bilingual prompt if auto-detect is on, otherwise language specific
        if config.ENABLE_AUTO_DETECT:
            self.current_prompt_template = BILINGUAL_PROMPT
        else:
            self.current_prompt_template = PROMPTS.get(self.language, PROMPTS['en'])
        
        self.prompt = PromptTemplate(
            template=self.current_prompt_template,
            input_variables=["context", "question"]
        )
    
    def load_pdfs(self, pdf_directory: str) -> List:
        """
        Load all PDF files from a directory with comprehensive error handling.
        
        Args:
            pdf_directory: Path to directory containing PDFs
            
        Returns:
            List of documents
            
        Raises:
            ValueError: If no PDFs found or all PDFs failed to load
        """
        documents = []
        pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))
        
        if not pdf_files:
            logger.error(f"No PDF files found in {pdf_directory}")
            raise ValueError(f"No PDF files found in {pdf_directory}")
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        failed_files = []
        
        for pdf_file in pdf_files:
            logger.info(f"Loading {os.path.basename(pdf_file)}")
            try:
                loader = PyPDFLoader(pdf_file)
                loaded_docs = loader.load()
                
                if not loaded_docs:
                    failed_files.append((pdf_file, "Empty PDF"))
                    logger.warning(f"PDF is empty: {pdf_file}")
                    continue
                    
                documents.extend(loaded_docs)
                logger.info(f"Successfully loaded {len(loaded_docs)} pages from {os.path.basename(pdf_file)}")
                
            except Exception as e:
                failed_files.append((pdf_file, str(e)))
                logger.error(f"Error loading {pdf_file}: {e}")
                continue
        
        if failed_files:
            logger.warning(f"Failed to load {len(failed_files)} files:")
            for file, error in failed_files:
                logger.warning(f"  - {os.path.basename(file)}: {error}")
        
        if not documents:
            logger.error("No documents were successfully loaded")
            raise ValueError("No documents were successfully loaded from PDFs")
        
        # Validate that we actually extracted text
        total_content_length = sum(len(doc.page_content.strip()) for doc in documents)
        
        if total_content_length == 0:
            logger.warning("Loaded PDFs resulted in empty text content. Attempting OCR fallback...")
            
            ocr_documents = []
            for pdf_file in pdf_files:
                try:
                    ocr_docs = extract_text_with_ocr(pdf_file)
                    ocr_documents.extend(ocr_docs)
                except Exception as e:
                    logger.error(f"OCR failed for {pdf_file}: {e}")
            
            if ocr_documents:
                documents = ocr_documents
                total_content_length = sum(len(doc.page_content.strip()) for doc in documents)
                logger.info(f"✓ OCR extraction successful: {len(documents)} pages loaded")
            else:
                logger.error("OCR fallback failed or returned no text.")
                raise ValueError("Uploaded PDFs contain no extractable text. They might be scanned images and OCR failed.")

        logger.info(f"✓ Total pages loaded: {len(documents)} with {total_content_length} characters of text")
        return documents
    
    def process_documents(
        self,
        documents: List,
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> List:
        """
        Split documents into chunks for embedding with size validation.
        
        Args:
            documents: List of loaded documents
            chunk_size: Size of each chunk (defaults to config.CHUNK_SIZE)
            chunk_overlap: Overlap between chunks (defaults to config.CHUNK_OVERLAP)
            
        Returns:
            List of document chunks
            
        Raises:
            ValueError: If documents exceed maximum size limit
        """
        chunk_size = chunk_size or config.CHUNK_SIZE
        chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        
        # Validate total document size
        total_chars = sum(len(doc.page_content) for doc in documents)
        logger.info(f"Total document size: {total_chars:,} characters")
        
        try:
            validate_document_size(total_chars)
        except ValueError as e:
            logger.error(str(e))
            raise
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")
        return chunks
    
    def create_vectorstore(self, chunks: List, persist: bool = True) -> None:
        """
        Create vector store from document chunks.
        
        Args:
            chunks: List of document chunks
            persist: Whether to persist the vector store to disk
            
        Raises:
            Exception: If vector store creation fails
        """
        logger.info("Creating vector store...")
        
        if not chunks:
            logger.warning("No chunks to process. Skipping vector store creation.")
            return

        try:
            # Verify chunks are not empty
            valid_chunks = [c for c in chunks if c.page_content and c.page_content.strip()]
            if not valid_chunks:
                logger.warning("All chunks were empty after validation.")
                return
                
            logger.info(f"Processing {len(valid_chunks)} valid chunks...")
            
            # Create vectorstore using Chroma
            self.vectorstore = Chroma.from_documents(
                documents=valid_chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory if persist else None,
                collection_metadata={"hnsw:space": "cosine"} # Optimize for cosine similarity
            )
            
            if persist:
                logger.info(f"✓ Vector store persisted to {self.persist_directory}")
            
            # Create retriever
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": config.TOP_K_RESULTS}
            )
            logger.info(f"Retriever configured with k={config.TOP_K_RESULTS}")
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            # Fallback: try to initialize empty if it fails (avoids crashing app)
            if not hasattr(self, 'vectorstore'):
                 raise e
            raise
    
    def load_existing_vectorstore(self) -> bool:
        """
        Load existing vector store from disk.
        
        Returns:
            True if vector store was loaded successfully, False otherwise
        """
        if os.path.exists(self.persist_directory):
            logger.info(f"Loading existing vector store from {self.persist_directory}")
            
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                self.retriever = self.vectorstore.as_retriever(
                    search_kwargs={"k": config.TOP_K_RESULTS}
                )
                logger.info("✓ Vector store loaded successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load existing vector store: {e}")
                return False
        
        logger.info("No existing vector store found")
        return False
    
    def create_qa_chain(self) -> None:
        """
        Create the QA chain with retrieval and conversation memory.
        
        Raises:
            ValueError: If vector store not initialized
        """
        if not self.retriever:
            logger.warning("Retriever not initialized. Cannot create QA chain. (Vector store might be empty)")
            return
        
        logger.info("Creating QA chain with conversation memory")
        
        try:
            # Create conversational retrieval chain with memory
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                memory=self.memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": self.prompt},
                verbose=False
            )
            logger.info("✓ QA chain created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create QA chain: {e}")
            raise
    
    def query(self, question: str) -> dict:
        """
        Query the RAG system.
        
        Args:
            question: Question to ask
            
        Returns:
            Dictionary containing answer and source documents
            
        Raises:
            ValueError: If question is empty
            Exception: If query fails
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
            
        # Detect language if enabled
        if config.ENABLE_AUTO_DETECT:
            detected_lang = detect_language(question)
            logger.info(f"Detected language: {detected_lang}")
            
            # Switch prompt if we have a specific one for this language
            # But use BILINGUAL_PROMPT by default for mixed context
            # Or strict switching:
            # new_prompt = PROMPTS.get(detected_lang, PROMPTS['en'])
            # We stick to BILINGUAL_PROMPT for flexibility as per plan
            pass
        
        # Create or update QA chain if needed
        if not hasattr(self, 'qa_chain'):
            self.create_qa_chain()
            
        # Check if chain exists (it might not if vectorstore creation failed/was skipped)
        if not hasattr(self, 'qa_chain'):
             raise ValueError("Chatbot is not ready. Please ensure valid PDF documents are loaded first.")
        
        logger.info(f"Processing query: {question[:100]}...")
        
        try:
            result = self.qa_chain({"question": question})
            logger.info("✓ Query processed successfully")
            
            return {
                "answer": result["answer"],
                "sources": result.get("source_documents", [])
            }
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise
            
    def set_language(self, language: str) -> None:
        """
        Set the interface language.
        
        Args:
            language: Language code ('en' or 'ar')
        """
        if language in config.SUPPORTED_LANGUAGES:
            self.language = language
            self.current_prompt_template = PROMPTS.get(language, PROMPTS['en'])
            
            # Update prompt template
            self.prompt = PromptTemplate(
                template=self.current_prompt_template,
                input_variables=["context", "question"]
            )
            
            # Recreate chain with new prompt
            if hasattr(self, 'qa_chain'):
                self.create_qa_chain()
                
            # Clear memory on language switch to avoid context confusion
            self.clear_memory()
            logger.info(f"Language switched to {language}")
    
    def clear_memory(self) -> None:
        """Clear conversation memory."""
        self.memory.clear()
        logger.info("Conversation memory cleared")
    
    def interactive_chat(self) -> None:
        """Start interactive chat session."""
        
        def print_ui(key, **kwargs):
            text = get_ui_text(key, self.language, **kwargs)
            if self.language == 'ar':
                text = reshape_arabic_text(text)
            print(text)
            
        print("\n" + "=" * 50)
        print_ui('cli_welcome')
        print("=" * 50)
        print_ui('cli_commands')
        print_ui('cli_quit')
        print_ui('cli_clear')
        print_ui('cli_lang')
        print_ui('cli_help')
        print("=" * 50)
        
        while True:
            try:
                user_label = get_ui_text('you', self.language)
                if self.language == 'ar':
                    user_label = reshape_arabic_text(user_label)
                
                question = input(f"\n{user_label}: ").strip()
                
                if question.lower() in ['quit', 'exit']:
                    print_ui('goodbye')
                    logger.info("Chat session ended by user")
                    break
                    
                elif question.lower() == 'clear':
                    self.clear_memory()
                    print_ui('memory_cleared')
                    continue
                    
                elif question.lower() == 'lang':
                    new_lang = 'en' if self.language == 'ar' else 'ar'
                    self.set_language(new_lang)
                    print_ui('switched_lang')
                    continue
                    
                elif question.lower() == 'help':
                    print("\n" + "=" * 50)
                    print_ui('cli_commands')
                    print_ui('cli_quit')
                    print_ui('cli_clear')
                    print_ui('cli_lang')
                    print_ui('cli_help')
                    print("=" * 50)
                    continue
                    
                elif not question:
                    continue
                
                # Get response
                try:
                    response = self.query(question)
                    
                    # Display answer
                    answer = response['answer']
                    if self.language == 'ar':
                        answer = reshape_arabic_text(answer)
                        
                    assistant_label = get_ui_text('assistant', self.language)
                    if self.language == 'ar':
                        assistant_label = reshape_arabic_text(assistant_label)
                        
                    print(f"\n{assistant_label}: {answer}")
                    
                    # Display sources if available
                    if response['sources']:
                        sources_title = get_ui_text('view_sources', self.language)
                        if self.language == 'ar':
                            sources_title = reshape_arabic_text(sources_title)
                        print(f"\n{sources_title}:")
                        
                        for i, doc in enumerate(response['sources'][:2], 1):
                            source = doc.metadata.get('source', 'Unknown')
                            page = doc.metadata.get('page', 'N/A')
                            print(f"  {i}. {os.path.basename(source)} (Page {page})")
                            
                except Exception as e:
                    print(f"❌ Error: {str(e)}")
                    logger.error(f"Error processing question: {e}")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Exiting...")
                logger.info("Chat session interrupted by user")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {str(e)}")
                logger.error(f"Unexpected error in chat loop: {e}")


def main():
    """Main execution function."""
    logger.info("Starting ChatBox application")
    
    # Create directory for PDFs if it doesn't exist
    os.makedirs(config.PDF_DIRECTORY, exist_ok=True)
    logger.info(f"PDF directory: {config.PDF_DIRECTORY}")
    
    try:
        # Initialize chatbot
        chatbot = RAGChatbot(
            model_name=config.DEFAULT_MODEL,
            embedding_model=config.DEFAULT_EMBEDDING_MODEL,
            persist_directory=config.CHROMA_DB_CLI
        )
        
        # Check if vector store exists
        if not chatbot.load_existing_vectorstore():
            logger.info("No existing vector store found. Processing PDFs...")
            
            # Check if PDFs exist
            pdf_files = glob.glob(os.path.join(config.PDF_DIRECTORY, "*.pdf"))
            if not pdf_files:
                print(f"\n⚠️  No PDF files found in '{config.PDF_DIRECTORY}'")
                print("Please add PDF files to the directory and restart the application.")
                logger.warning("No PDF files found, creating empty vector store")
                
                # Create empty vector store
                chatbot.vectorstore = Chroma(
                    embedding_function=chatbot.embeddings,
                    persist_directory=chatbot.persist_directory
                )
                chatbot.retriever = chatbot.vectorstore.as_retriever()
            else:
                # Load and process PDFs
                try:
                    documents = chatbot.load_pdfs(config.PDF_DIRECTORY)
                    chunks = chatbot.process_documents(documents)
                    chatbot.create_vectorstore(chunks, persist=True)
                except Exception as e:
                    logger.error(f"Failed to process PDFs: {e}")
                    print(f"\n❌ Error processing PDFs: {e}")
                    return
        
        # Create QA chain
        chatbot.create_qa_chain()
        
        # Start interactive chat
        chatbot.interactive_chat()
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"\n❌ Application error: {e}")
        print("Please check the log file for more details.")


if __name__ == "__main__":
    main()