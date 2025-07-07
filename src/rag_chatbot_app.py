import streamlit as st
import tempfile
import os
import time
import torch
import asyncio

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import BitsAndBytesConfig
from utils import process_multiple_pdfs
from models import load_llm as _load_llm


MODEL_LIST = ["lmsys/vicuna-7b-v1.5", "vinai/PhoGPT-4B-Chat"]

# Session state initialization
if 'model_name' not in st.session_state:  # Error syntax: if not st.session_state.model_name
        st.session_state.model_name = MODEL_LIST[0] # Set the first model as default
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'pdf_names' not in st.session_state:  # Changed to store multiple PDF names
    st.session_state.pdf_names = []
if 'total_chunks' not in st.session_state:  # Store total chunks from all PDFs
    st.session_state.total_chunks = 0
if 'lang' not in st.session_state:
    st.session_state.lang = "vi"

#Add language feature
LANG_OPTIONS = {
    "vi": "Tiếng Việt",
    "en": "English"
}

translations = {
    "title": {
        "vi": "📄 Trợ lý PDF RAG",
        "en": "📄 PDF RAG Assistant"
    },
    "description": {
        "vi": "*Trò chuyện với Chatbot để trao đổi về nội dung tài liệu PDF của bạn*",
        "en": "*Chat with the chatbot to explore your PDF content*"
    },
    "model_select": {
        "vi": "Chọn model AI",
        "en": "Select AI model"
    },
    "upload_label": {
        "vi": "Chọn file PDF",
        "en": "Upload PDF file"
    },
    "process_button": {
        "vi": "🔄 Xử lý PDF",
        "en": "🔄 Process PDF"
    },
    "pdf_ready": {
        "vi": "📄 Đã tải",
        "en": "📄 Uploaded"
    },
    "pdf_empty": {
        "vi": "📄 Chưa có tài liệu",
        "en": "📄 No PDF uploaded"
    },
    "chat_input_placeholder": {
        "vi": "Nhập câu hỏi của bạn...",
        "en": "Enter your question..."
    },
    "chat_disabled": {
        "vi": "🔄 Vui lòng upload và xử lý file PDF trước khi bắt đầu chat!",
        "en": "🔄 Please upload and process a PDF before chatting!"
    },
    "thinking": {
        "vi": "Đang suy nghĩ...",
        "en": "Thinking..."
    },
    "model_loading": {
        "vi": "⏳ Đang tải AI models, vui lòng đợi...",
        "en": "⏳ Loading AI models, please wait..."
    },
    "model_ready": {
        "vi": "✅ Models đã sẵn sàng!",
        "en": "✅ Models are ready!"
    },
    "clear_chat": {
        "vi": "🗑️ Xóa lịch sử chat",
        "en": "🗑️ Clear chat history"
    },
    "instructions": {
        "vi": "**Cách sử dụng:**\n1. **Chọn model**\n2. **Upload PDF**\n3. **Đặt câu hỏi**\n4. **Nhận trả lời**",
        "en": "**How to use:**\n1. **Select model**\n2. **Upload PDF**\n3. **Ask questions**\n4. **Receive answers**"
    },
    "used_model":
    {
        "vi": "Model AI được sử dụng: ",
        "en": "Model AI being used: "
    },
    "setting":
    {
        "vi": "⚙️ Cài đặt",
        "en": "⚙️ Settings"
    },
    "upload_pdf":
    {
        "vi": "Upload Tài liệu",
        "en": "Upload PDF"
    }
}

# Functions
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")

load_llm = st.cache_resource(_load_llm)

def add_message(role, content):
    """Thêm tin nhắn vào lịch sử chat"""
    st.session_state.chat_history.append({
        "role": role,
        "content": content,
        "timestamp": time.time()
    })

def clear_chat():
    """Xóa lịch sử chat"""
    st.session_state.chat_history = []

def display_chat():
    """Hiển thị lịch sử chat"""
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write("Xin chào! Tôi là AI assistant. Hãy upload file PDF và bắt đầu đặt câu hỏi về nội dung tài liệu nhé! 😊")

# UI
def main():
    # Sidebar: Language selection
    with st.sidebar:
        st.title("🌐 Language / Ngôn ngữ")
        lang_choice = st.selectbox("Chọn / Select", options=list(LANG_OPTIONS.keys()), format_func=lambda x: LANG_OPTIONS[x])
        st.session_state.lang = lang_choice
        t = lambda key: translations[key][st.session_state.lang]

    st.set_page_config(
        page_title="PDF RAG Chatbot", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title(t("title"))
    st.logo("./logo.png", size="large")
    

    # Main content
    st.markdown(t("description"))
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        display_chat()
    
    # Model selection
    model_name = st.selectbox(t("model_select"), MODEL_LIST, index=MODEL_LIST.index(st.session_state.model_name))
    
    # Chat input
    if st.session_state.models_loaded:
        st.info(t("used_model") + st.session_state.model_name)
        if st.session_state.pdf_processed:
            # User input
            user_input = st.chat_input(t("chat_input_placeholder"))
            
            if user_input:
                # Add user message
                add_message("user", user_input)
                
                # Display user message immediately
                with st.chat_message("user"):
                    st.write(user_input)
                
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner(t("thinking")):
                        try:
                            output = st.session_state.rag_chain.invoke(user_input)
                            # Clean up the response
                            if 'Answer:' in output:
                                answer = output.split('Answer:')[1].strip()
                            else:
                                answer = output.strip()
                            
                            # Display response
                            st.write(answer)
                            
                            # Add assistant message to history
                            add_message("assistant", answer)
                            
                        except Exception as e:
                            error_msg = f"❌ {str(e)}"
                            st.error(error_msg)
                            add_message("assistant", error_msg)
        else:
            st.info(t("chat_disabled"))
            st.chat_input("Nhập câu hỏi của bạn...", disabled=True)
    else:
        st.info(t("model_loading"))
        st.chat_input(t("chat_input_placeholder"), disabled=True)
        
    # Sidebar
    with st.sidebar:
        st.title(t("settings"))
        
        # Load models
        if model_name != st.session_state.model_name:
            st.session_state.model_name = model_name
            st.session_state.models_loaded = False  # Force reload if needed
            st.rerun()
            
        if not st.session_state.models_loaded:
            st.warning(t("model_loading"))
            with st.spinner(t("model_loading")):
                st.session_state.embeddings = load_embeddings()
                st.session_state.llm = load_llm(st.session_state.model_name) 
                st.session_state.models_loaded = True
            st.success(t("model_ready"))
            st.rerun()
        else:
            st.success(t("model_ready"))

        st.markdown("---")
        
        # Upload PDF
        st.subheader("📄 Upload tài liệu")
        uploaded_files = st.file_uploader(t("upload_label"), accept_multiple_files=True, type="pdf")
        
        if uploaded_files:
            if st.button(t("process_button"), use_container_width=True):
                with st.spinner(t("process_button")):
                    progress_bar = st.progress(0)
                    def update_progress(value):
                        progress_bar.progress(value)
                    st.session_state.rag_chain, chunk_counts = process_multiple_pdfs(uploaded_files, update_progress)
                    st.session_state.pdf_processed = True
                    st.session_state.pdf_names = [file.name for file in uploaded_files]
                    st.session_state.total_chunks = sum(chunk_counts)
                    # Reset chat history khi upload PDF mới
                    clear_chat()
                    add_message("assistant", f"{t('pdf_ready')}: {', '.join(st.session_state.pdf_names)}\nChunks: {st.session_state.total_chunks}")
                st.rerun()
        
        # PDF status
        if st.session_state.pdf_processed:
            st.success(f"{t('pdf_ready')}: {', '.join(st.session_state.pdf_names)}")
        else:
            st.info(t("pdf_empty"))
            
        st.markdown("---")
        
        # Chat controls
        st.subheader("💬 Chat")
        if st.button(t("clear_chat"), use_container_width=True):
            clear_chat()
            st.rerun()
            
        st.markdown("---")
        
        # Instructions
        st.markdown("---")
        st.subheader("📋 Guide")
        st.markdown(t("instructions"))

if __name__ == "__main__":
    main()
