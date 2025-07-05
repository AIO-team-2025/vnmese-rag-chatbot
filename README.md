# Build RAG Chatbot Application using Streamlit
This is a project for creating a RAG chatbot in Vietnamese languague.

![plot](./logo.png)

## Information
- Embedding model:bkai-foundation-models/vietnamese-bi-encoder
- LLM model (by default): vicuna-7b-v1.5

## Original features
- Upload and embed a PDF file.
- Make questions based on the file.

## Added features
- Upload multiple PDF files.
- AI model customizable: The AI model vicuna-7b-v1.5 is loaded by default. User can choose another LLM model. In this case, the webpage will be reloaded.

## How to run

```bash
streamlit run notebooks/rag_chatbot_app.py & npx localtunnel --port 8501