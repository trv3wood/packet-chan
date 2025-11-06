"""Streamlit app: upload files, ingest into Chroma, ask queries."""
import streamlit as st
from dotenv import load_dotenv
import os
import uuid

from src.ingest import save_uploaded_file, extract_text_from_file, chunk_text
from src.vector_store import VectorStore
from src.retriever import build_prompt
from src.llm_adapter import generate

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")


@st.cache_resource
def get_store():
    return VectorStore(persist_directory=CHROMA_DIR)


def ingest_and_index(uploaded_file):
    tmpdir = os.path.join(".", "uploads")
    path = save_uploaded_file(uploaded_file, tmpdir)
    text = extract_text_from_file(path)
    chunks = chunk_text(text)
    docs = []
    for i, c in enumerate(chunks):
        docs.append({
            "id": f"{uuid.uuid4()}",
            "text": c,
            "metadata": {"source": uploaded_file.name, "chunk_index": i},
        })
    store = get_store()
    store.add_documents(docs)
    st.success(f"Indexed {len(docs)} chunks from {uploaded_file.name}")


def main():
    st.title("RAG Chat — Streamlit + Chroma + Ollama")

    st.sidebar.header("Upload")
    uploaded = st.sidebar.file_uploader("Upload a document to index", accept_multiple_files=False)
    if uploaded is not None:
        if st.sidebar.button("Ingest"):
            ingest_and_index(uploaded)

    st.header("Ask a question")
    question = st.text_input("Your question")
    k = st.slider("retrieval k", 1, 10, 4)

    if st.button("Ask") and question:
        store = get_store()
        hits = store.query(question, k=k)
        prompt = build_prompt(question, hits)
        with st.spinner("Generating answer..."):
            try:
                answer = generate(prompt)
                st.markdown("**Answer**")
                st.write(answer)
                st.markdown("**Sources**")
                for h in hits:
                    st.write(h.get("metadata", {}))
            except Exception as e:
                st.error(f"LLM generation error: {e}")


if __name__ == "__main__":
    main()
