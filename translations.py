"""
Translation resources for ChatBox application.
Contains UI labels, messages, and prompt templates for supported languages.
"""

TRANSLATIONS = {
    'en': {
        # UI Labels
        'window_title': 'RAG Chatbot with Ollama',
        'page_title': '📚 RAG Chatbot with Ollama',
        'subtitle': 'Chat with PDFs from your configured directory using local LLM',
        'sidebar_config': '⚙️ Configuration',
        'select_model': 'Select Ollama Model',
        'refresh_models': '🔄 Refresh Models',
        'pdf_docs': '📄 PDF Documents',
        'directory': '📁 Directory',
        'view_files': '📋 View PDF Files',
        'no_pdfs': '⚠️ No PDFs found in',
        'add_pdfs_hint': "💡 Add PDF files to the directory and click 'Load PDFs'",
        'load_button': '🚀 Load PDFs from Directory',
        'processing': '📥 Processing PDFs from directory...',
        'ocr_processing': '🔍 No text found. Attempting OCR fallback (this may take a while)...',
        'success_process': '✅ Successfully processed {count} PDF(s) from directory!',
        'ocr_success': '✅ OCR extraction successful: {count} pages loaded!',
        'chat_controls': '💬 Chat Controls',
        'clear_chat': '🗑️ Clear Chat',
        'reset_db': '🔄 Reset DB',
        'stats': '📊 Statistics',
        'messages_count': 'Messages',
        'docs_count': 'Documents in DB',
        'chat_header': '💬 Chat',
        'welcome_msg': '👆 Please upload and process PDF files to start chatting!',
        'input_placeholder': 'Ask a question about your PDFs',
        'thinking': '🤔 Thinking...',
        'view_sources': '📚 View Sources',
        'source': 'Source',
        'page': 'Page',
        'tip': '💡 Tip: Use the sidebar to upload PDFs, configure settings, and manage your chat session',
        
        # CLI Messages
        'cli_welcome': 'RAG Chatbot with Ollama',
        'cli_commands': 'Commands:',
        'cli_quit': "'quit' or 'exit' - Exit the chat",
        'cli_clear': "'clear' - Clear conversation memory",
        'cli_lang': "'lang' - Switch language",
        'cli_help': "'help' - Show this help message",
        'you': '🧑 You',
        'goodbye': '👋 Goodbye!',
        'memory_cleared': '✓ Memory cleared!',
        'assistant': '🤖 Assistant',
        'error': '❌ Error',
        
        # Errors
        'model_error': '❌ No Ollama models found. Please install models using',
        'validation_error': '❌ Validation Error',
        'processing_error': '❌ Error processing PDFs',
        'empty_question': '⚠️ Please enter a question',
        
        # Language
        'language_select': '🌐 Language / اللغة',
        'switched_lang': '✓ Switched language to English',
        'indexed_docs': '📚 Indexed Documents',
        'no_indexed_docs': 'No documents currently indexed.'
    },
    'ar': {
        # UI Labels
        'window_title': 'بوت الدردشة مع أولاما',
        'page_title': '📚 بوت الدردشة مع أولاما',
        'subtitle': 'تحدث مع مستندات PDF باستخدام النماذج المحلية',
        'sidebar_config': '⚙️ الإعدادات',
        'select_model': 'اختر نموذج أولاما',
        'refresh_models': '🔄 تحديث النماذج',
        'pdf_docs': '📄 مستندات PDF',
        'directory': '📁 المجلد',
        'view_files': '📋 عرض الملفات',
        'no_pdfs': '⚠️ لم يتم العثور على ملفات PDF في',
        'add_pdfs_hint': "💡 أضف ملفات PDF إلى المجلد ثم اضغط على 'تحميل الملفات'",
        'load_button': '🚀 تحميل ملفات PDF',
        'processing': '📥 جاري معالجة الملفات...',
        'ocr_processing': '🔍 لم يتم العثور على نص. جاري محاولة استخلاص النص (OCR) (قد يستغرق ذلك وقتاً)...',
        'success_process': '✅ تمت معالجة {count} ملف بنجاح!',
        'ocr_success': '✅ نجح استخلاص النص (OCR): تم تحميل {count} صفحة!',
        'chat_controls': '💬 تحكم الدردشة',
        'clear_chat': '🗑️ مسح الدردشة',
        'reset_db': '🔄 إعادة تعيين قاعدة البيانات',
        'stats': '📊 الإحصائيات',
        'messages_count': 'الرسائل',
        'docs_count': 'المستندات',
        'chat_header': '💬 الدردشة',
        'welcome_msg': '👆 الرجاء تحميل ومعالجة الملفات للبدء!',
        'input_placeholder': 'اطرح سؤالاً حول مستنداتك',
        'thinking': '🤔 جاري التفكير...',
        'view_sources': '📚 عرض المصادر',
        'source': 'المصدر',
        'page': 'صفحة',
        'tip': '💡 نصيحة: استخدم القائمة الجانبية لإدارة الملفات والإعدادات',
        
        # CLI Messages
        'cli_welcome': 'بوت الدردشة مع أولاما',
        'cli_commands': 'الأوامر:',
        'cli_quit': "'quit' أو 'exit' - للخروج",
        'cli_clear': "'clear' - لمسح الذاكرة",
        'cli_lang': "'lang' - تغيير اللغة",
        'cli_help': "'help' - لعرض المساعدة",
        'you': '🧑 أنت',
        'goodbye': '👋 مع السلامة!',
        'memory_cleared': '✓ تم مسح الذاكرة!',
        'assistant': '🤖 المساعد',
        'error': '❌ خطأ',
        
        # Errors
        'model_error': '❌ لم يتم العثور على نماذج. الرجاء تثبيت نموذج باستخدام',
        'validation_error': '❌ خطأ في التحقق',
        'processing_error': '❌ خطأ في معالجة الملفات',
        'empty_question': '⚠️ الرجاء إدخال سؤال',
        
        # Language
        'language_select': '🌐 اللغة / Language',
        'switched_lang': '✓ تم تغيير اللغة إلى العربية',
        'indexed_docs': '📚 المستندات المفهرسة',
        'no_indexed_docs': 'لا توجد مستندات مفهرسة حالياً.'
    }
}

PROMPTS = {
    'ar': """استخدم المعلومات التالية للإجابة على السؤال.
يجب أن تكون الإجابة باللغة العربية دائماً.
إذا كنت لا تعرف الإجابة، قل ذلك ببساطة، ولا تحاول اختلاق إجابة.
اجعل الإجابة موجزة في ثلاث جمل كحد أقصى.

السياق: {context}

السؤال: {question}

الإجابة المفيدة:""",

    'en': """Use the following pieces of context to answer the question at the end.
You must respond in English.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer concise.

Context: {context}

Question: {question}

Helpful Answer:"""
}

# Bilingual prompt for auto-detection scenarios
BILINGUAL_PROMPT = """Use the following context to answer the question. 
If the question is in Arabic, respond in Arabic. If in English, respond in English.
If you don't know the answer, just say so. Keep the answer concise (max 3 sentences).

Context: {context}

Question: {question}

Helpful Answer:"""
