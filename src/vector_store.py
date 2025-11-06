"""Simple vector store wrapper supporting Chroma (default).

Uses sentence-transformers to produce embeddings and Chromadb to store them.
"""
from typing import List, Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self, collection_name: str = "rag_collection"):
        self.client = chromadb.Client()
        self.collection = None
        self.collection_name = collection_name
        # local embedding model
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
        except Exception:
            self.collection = self.client.create_collection(name=self.collection_name)

    def add_documents(self, docs: List[Dict[str, Any]]):
        """docs: list of dicts with keys: id (str), text (str), metadata (dict)
        """
        ids = [d["id"] for d in docs]
        texts = [d["text"] for d in docs]
        metadatas = [d.get("metadata", {}) for d in docs]
        embeddings = self.embed_model.encode(texts, show_progress_bar=False).tolist()
        self.collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)

    def query(self, query_text: str, k: int = 4):
        query_emb = self.embed_model.encode([query_text]).tolist()[0]
        results = self.collection.query(query_embeddings=[query_emb], n_results=k, include=['documents', 'metadatas', 'distances'])
        # results is a dict with lists
        hits = []
        for docs, metas, dists in zip(results.get('documents', []), results.get('metadatas', []), results.get('distances', [])):
            # when querying a single query, chroma returns lists inside lists
            for doc, meta, dist in zip(docs, metas, dists):
                hits.append({"document": doc, "metadata": meta, "distance": dist})
        return hits
