import tempfile
import os
import asyncio
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st

async def process_pdf(uploaded_file, update_progress):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    
    semantic_splitter = SemanticChunker(
        embeddings=st.session_state.embeddings,
        buffer_size=1,
        breakpoint_threshold_type="percentile", 
        breakpoint_threshold_amount=95,
        min_chunk_size=500,
        add_start_index=True
    )
    
    docs = semantic_splitter.split_documents(documents)
    
    # Mô phỏng tiến trình (có thể điều chỉnh dựa trên số bước thực tế)
    total_steps = 10
    for i in range(total_steps):
        await asyncio.sleep(0.1)  # Mô phỏng công việc
        update_progress((i + 1) / total_steps / len(files))  # Cập nhật tiến trình
    
    vector_db = Chroma.from_documents(documents=docs, embedding=st.session_state.embeddings)
    retriever = vector_db.as_retriever()
    
    os.unlink(tmp_file_path)
    return retriever, len(docs)

async def process_multiple_pdfs(uploaded_files, update_progress):
    tasks = [process_pdf(file, update_progress) for file in uploaded_files]
    results = await asyncio.gather(*tasks)
    
    # Kết hợp các retriever từ nhiều file
    all_retrievers = [result[0] for result in results]
    all_chunk_counts = [result[1] for result in results]
    
    # Tạo vector database chung (hợp nhất tất cả tài liệu)
    combined_docs = []
    for i, file in enumerate(uploaded_files):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()
        combined_docs.extend(docs)
        os.unlink(tmp_file_path)
    
    semantic_splitter = SemanticChunker(
        embeddings=st.session_state.embeddings,
        buffer_size=1,
        breakpoint_threshold_type="percentile", 
        breakpoint_threshold_amount=95,
        min_chunk_size=500,
        add_start_index=True
    )
    
    combined_chunks = semantic_splitter.split_documents(combined_docs)
    combined_vector_db = Chroma.from_documents(documents=combined_chunks, embedding=st.session_state.embeddings)
    combined_retriever = combined_vector_db.as_retriever()
    
    prompt = hub.pull("rlm/rag-prompt")
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": combined_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt 
        | st.session_state.llm
        | StrOutputParser()
    )
    
    return rag_chain, all_chunk_counts

# Hàm cũ process_pdf vẫn giữ nguyên để tương thích nếu cần
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    
    semantic_splitter = SemanticChunker(
        embeddings=st.session_state.embeddings,
        buffer_size=1,
        breakpoint_threshold_type="percentile", 
        breakpoint_threshold_amount=95,
        min_chunk_size=500,
        add_start_index=True
    )
    
    docs = semantic_splitter.split_documents(documents)
    vector_db = Chroma.from_documents(documents=docs, embedding=st.session_state.embeddings)
    retriever = vector_db.as_retriever()
    
    prompt = hub.pull("rlm/rag-prompt")
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt 
        | st.session_state.llm
        | StrOutputParser()
    )
    
    os.unlink(tmp_file_path)
    return rag_chain, len(docs)
