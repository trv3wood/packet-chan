RAG System (Streamlit + Chroma + Ollama)

This repository is a starter scaffold for a Retrieval-Augmented Generation (RAG) system with a layered architecture:

- Frontend: Streamlit chat UI
- File ingestion: convert documents to plain text and split into chunks
- Vector store: ChromaDB (default) with a simple wrapper
- LLM adapter: Local Ollama model (default), OpenAI-compatible adapter placeholder

Designed for local development and course use.

Quick start (PowerShell)

1. Create a virtual environment and install dependencies

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and edit as needed (model name, chroma directory)

3. Run the Streamlit app

```powershell
streamlit run src/app.py
```

Defaults

- Vector DB: Chroma (local)
- Embeddings: sentence-transformers (`all-MiniLM-L6-v2`)
- LLM: Ollama local model (configure `OLLAMA_MODEL` in `.env`)

Switching to OpenAI or FAISS

See `src/llm_adapter.py` and `src/vector_store.py` — both include simple extension points and configuration comments for switching to OpenAI or FAISS.

File ingestion

Basic readers for `.txt`, `.md`, `.pdf`, `.docx` are provided. For more formats, plug in `markitdown` or other converters and adjust `src/ingest.py`.

Development notes

- Windows PowerShell is the assumed shell for run commands in README.
- This scaffold is intentionally minimal. Add authentication, deployment, and thorough testing before production.
