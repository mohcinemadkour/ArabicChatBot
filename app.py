"""
Streamlit UI for RAG Chatbot with Ollama

This module provides a web interface for the RAG chatbot using Streamlit.
"""

import logging
import os

import streamlit as st

import config
from main import RAGChatbot
from utils import get_available_ollama_models, setup_logging, get_ui_text, safe_delete_directory
import shutil

# Setup logging
logger = setup_logging()

# Streamlit page configuration
st.set_page_config(
    page_title="RAG Chatbot with Ollama",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for RTL and Fonts
st.markdown("""
<style>
/* Import Arabic font */
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');

/* Default font for Arabic */
[lang="ar"] {
    font-family: 'Cairo', sans-serif !important;
}

/* RTL Support */
.rtl {
    direction: rtl;
    text-align: right;
}

/* Streamlit specific RTL adjustments */
.stMarkdown[dir="rtl"] {
    text-align: right;
}

/* Adjust chat messages for RTL */
.stChatMessage.rtl {
    flex-direction: row-reverse;
}
</style>
""", unsafe_allow_html=True)


# Initialize language
if 'language' not in st.session_state:
    st.session_state.language = config.DEFAULT_LANGUAGE

# Helper for current language
lang = st.session_state.language

st.title(get_ui_text('page_title', lang))
st.markdown(get_ui_text('subtitle', lang))

# Sidebar for configuration
with st.sidebar:
    st.header(get_ui_text('sidebar_config', lang))
    
    # Language Selection
    selected_lang_label = st.radio(
        get_ui_text('language_select', lang),
        options=['English', 'العربية'],
        index=0 if lang == 'en' else 1,
        horizontal=True
    )
    
    # Update language based on selection
    new_lang = 'en' if selected_lang_label == 'English' else 'ar'
    if new_lang != st.session_state.language:
        st.session_state.language = new_lang
        st.rerun()
    
    # Get available Ollama models
    try:
        available_models = get_available_ollama_models()
    except Exception as e:
        logger.error(f"Error getting Ollama models: {e}")
        available_models = []
    
    # Model selection
    if available_models:
        model_name = st.selectbox(
            get_ui_text('select_model', lang),
            available_models,
            index=0,
            help="Choose the LLM model for chat and embeddings"
        )
    else:
        st.error(get_ui_text('model_error', lang) + " 'ollama pull <model>'")
        st.info("💡 Example: `ollama pull llama3.1`")
        model_name = None
    
    # Refresh models button
    if st.button(get_ui_text('refresh_models', lang)):
        st.rerun()
    
    st.divider()
    
    # PDF Directory Configuration
    st.header(get_ui_text('pdf_docs', lang))
    
    # Show current PDF directory
    pdf_dir = config.PDF_DIRECTORY
    st.text(f"{get_ui_text('directory', lang)}: {pdf_dir}")
    
    # Check for PDFs in directory
    import glob
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    
    if pdf_files:
        st.success(get_ui_text('success_process', lang, count=len(pdf_files)).replace('✅ ', '✅'))
        with st.expander(get_ui_text('view_files', lang)):
            for pdf_file in pdf_files:
                st.text(f"• {os.path.basename(pdf_file)}")
    else:
        st.warning(f"{get_ui_text('no_pdfs', lang)} {pdf_dir}")
        st.info(get_ui_text('add_pdfs_hint', lang))
    
    st.divider()
    
    # Chat controls
    st.header(get_ui_text('chat_controls', lang))
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(get_ui_text('clear_chat', lang), use_container_width=True):
            st.session_state.messages = []
            logger.info("Chat history cleared")
            st.rerun()
    
    with col2:
        if st.button(get_ui_text('reset_db', lang), use_container_width=True):
            st.cache_resource.clear()
            st.session_state.chatbot = None
            logger.info("Vector store reset")
            st.success("✓ Reset complete")
            st.rerun()
    
    # Display statistics
    if 'chatbot' in st.session_state and st.session_state.chatbot:
        st.divider()
        st.header(get_ui_text('stats', lang))
        
        st.metric(get_ui_text('messages_count', lang), len(st.session_state.get('messages', [])))
        
        try:
            if hasattr(st.session_state.chatbot, 'vectorstore') and st.session_state.chatbot.vectorstore:
                try:
                    doc_count = st.session_state.chatbot.vectorstore._collection.count()
                    st.metric(get_ui_text('docs_count', lang), doc_count)
                except:
                    st.metric(get_ui_text('docs_count', lang), "N/A")
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")

# Initialize session state (messages)
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Automated Chatbot Initialization and Processing
@st.cache_resource(show_spinner=False)
def initialize_chatbot(model_name, lang_code, _pdf_files):
    """
    Initialize chatbot and process PDFs. 
    Cached to prevent reloading unless arguments change.
    _pdf_files arg is just for cache invalidation (list of filenames).
    """
    try:
        # Reset vector store on every run
        if os.path.exists(config.CHROMA_DB_STREAMLIT):
            if safe_delete_directory(config.CHROMA_DB_STREAMLIT):
                logger.info("Previous vector store cleared for fresh run")
            else:
                logger.warning("Could not clear previous vector store, will attempt to reuse/overwrite")

        logger.info("Initializing chatbot and processing PDFs...")
        chatbot_instance = RAGChatbot(
            model_name=model_name,
            embedding_model=model_name,
            persist_directory=config.CHROMA_DB_STREAMLIT,
            language=lang_code
        )
        
        # Always check/load documents at startup
        documents = chatbot_instance.load_pdfs(config.PDF_DIRECTORY)
        # Check if we need to process (simple check: if vectorstore empty or force)
        # For now, we follow the existing logic: process and persist
        chunks = chatbot_instance.process_documents(documents)
        chatbot_instance.create_vectorstore(chunks, persist=True)
        chatbot_instance.create_qa_chain()
        
        return chatbot_instance
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {e}")
        raise e

# Main logic to load chatbot
if model_name and pdf_files:
    try:
        # Use a localized spinner message
        with st.spinner(get_ui_text('processing', lang)):
            # Pass list of files to trigger cache reload if files change
            file_hashes = [os.path.getmtime(f) for f in pdf_files]
            
            # This will run once and be cached
            chatbot = initialize_chatbot(model_name, lang, str(file_hashes))
            
            # Store in session state for easy access
            st.session_state.chatbot = chatbot
            
            if 'init_done' not in st.session_state:
                st.session_state.init_done = True
                st.toast(get_ui_text('success_process', lang, count=len(pdf_files)), icon="✅")
            
    except Exception as e:
        error_msg = str(e)
        st.error(f"{get_ui_text('error', lang)}: {error_msg}")
        with st.expander(get_ui_text('error', lang)):
            st.exception(e)

# Main chat interface
st.header(get_ui_text('chat_header', lang))

# Display welcome message if no chatbot
if not st.session_state.get('chatbot'):
    if not pdf_files:
         st.warning(f"{get_ui_text('no_pdfs', lang)} {pdf_dir}")
         st.info(get_ui_text('add_pdfs_hint', lang))
    elif not model_name:
         st.info("👆 Please select a model to start.")
    else:
         # Should be loading...
         st.info(get_ui_text('processing', lang))
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Add RTL class if Arabic
            if lang == 'ar':
                st.markdown(f'<div class="rtl">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(message["content"])
            
            # Display sources if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander(get_ui_text('view_sources', lang), expanded=False):
                    for i, source_info in enumerate(message["sources"], 1):
                        st.markdown(f"**{get_ui_text('source', lang)} {i}**")
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.text(f"📄 {source_info['file']}")
                        with col2:
                            st.text(f"{get_ui_text('page', lang)} {source_info['page']}")
                        
                        # Apply RTL to source content if needed or keep original
                        st.markdown(f"```\n{source_info['content']}\n```")
                        
                        if i < len(message["sources"]):
                            st.divider()
    
    # Chat input
    if prompt := st.chat_input(get_ui_text('input_placeholder', lang)):
        # Validate input
        if not prompt.strip():
            st.warning(get_ui_text('empty_question', lang))
        else:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get response
            with st.chat_message("assistant"):
                with st.spinner(get_ui_text('thinking', lang)):
                    try:
                        response = st.session_state.chatbot.query(prompt)
                        answer = response["answer"]
                        
                        # Display answer
                        if lang == 'ar':
                             st.markdown(f'<div class="rtl">{answer}</div>', unsafe_allow_html=True)
                        else:
                             st.markdown(answer)
                        
                        # Prepare source information
                        sources = []
                        if response.get('sources'):
                            with st.expander(get_ui_text('view_sources', lang), expanded=False):
                                for i, doc in enumerate(response['sources'][:3], 1):
                                    source = doc.metadata.get('source', 'Unknown')
                                    page = doc.metadata.get('page', 'N/A')
                                    content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                                    
                                    # Store source info
                                    sources.append({
                                        'file': os.path.basename(source),
                                        'page': page,
                                        'content': content
                                    })
                                    
                                    st.markdown(f"**{get_ui_text('source', lang)} {i}**")
                                    
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        st.text(f"📄 {os.path.basename(source)}")
                                    with col2:
                                        st.text(f"{get_ui_text('page', lang)} {page}")
                                    
                                    st.markdown(f"```\n{content}\n```")
                                    
                                    if i < len(response['sources'][:3]):
                                        st.divider()
                        
                        # Add assistant message with sources
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                        
                        logger.info(f"Successfully answered query: {prompt[:50]}...")
                        
                    except ValueError as e:
                        error_msg = str(e)
                        st.error(f"❌ {error_msg}")
                        logger.error(f"Validation error in query: {error_msg}")
                        
                    except Exception as e:
                        error_msg = str(e)
                        st.error(f"❌ {get_ui_text('error', lang)}: {error_msg}")
                        logger.error(f"Error processing query: {error_msg}", exc_info=True)
                        
                        with st.expander(get_ui_text('error', lang)):
                            st.exception(e)

# Footer
st.divider()
st.markdown(
    f"""
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    {get_ui_text('tip', lang)}
    </div>
    """,
    unsafe_allow_html=True
)