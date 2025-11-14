"""Simple vector store wrapper supporting Chroma (default).

Uses sentence-transformers to produce embeddings and Chromadb to store them.
"""
from typing import List, Dict, Any
import os

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "rag_collection"):
        os.makedirs(persist_directory, exist_ok=True)
        self.persist_directory = persist_directory
        self.client = chromadb.Client()
        self.collection = None
        self.collection_name = collection_name
        # local embedding model
        self.embed_model = SentenceTransformer("BAAI/bge-base-zh-v1.5")
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
        except Exception:
            self.collection = self.client.create_collection(name=self.collection_name)

    def add_documents(self, docs: List[Dict[str, Any]], batch_size: int = 32):
        """docs: list of dicts with keys: id (str), text (str), metadata (dict)
        
        Args:
            batch_size: 批量处理大小，较大的批次可以提高 embedding 生成速度
        """
        ids = [d["id"] for d in docs]
        texts = [d["text"] for d in docs]
        metadatas = [d.get("metadata", {}) for d in docs]
        
        # 批量生成 embeddings（sentence-transformers 会自动批处理，但我们可以优化）
        # 使用更大的 batch_size 可以提高 GPU 利用率（如果有 GPU）
        embeddings = self.embed_model.encode(
            texts, 
            show_progress_bar=False,
            batch_size=batch_size,
            convert_to_numpy=True  # 使用 numpy 格式更快
        ).tolist()
        
        self.collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)

    def query(self, query_text: str, k: int = 4, min_similarity: float = None, max_distance: float = None):
        """查询最相似的文档片段
        
        Args:
            query_text: 查询文本
            k: 返回的结果数量
            min_similarity: 最小相似度阈值（0-1），低于此值的片段将被过滤
            max_distance: 最大距离阈值，超过此距离的片段将被过滤（更直观）
                         如果设置了 max_distance，会优先使用它而不是 min_similarity
        """
        # 检索更多候选（如果设置了阈值，可能需要检索更多然后过滤）
        retrieve_k = k * 5 if (min_similarity is not None or max_distance is not None) else k
        
        query_emb = self.embed_model.encode([query_text]).tolist()[0]
        results = self.collection.query(
            query_embeddings=[query_emb], 
            n_results=retrieve_k, 
            include=['documents', 'metadatas', 'distances']
        )
        
        hits = []
        all_distances = []
        
        # 第一遍：收集所有结果
        for docs, metas, dists in zip(results.get('documents', []), results.get('metadatas', []), results.get('distances', [])):
            for doc, meta, dist in zip(docs, metas, dists):
                all_distances.append(dist)
                hits.append({
                    "document": doc,
                    "metadata": meta,
                    "distance": dist,
                })
        
        if not hits:
            return []
        
        # 使用绝对相似度计算，而不是相对归一化
        # ChromaDB 使用 L2 距离（欧氏距离），对于 384 维向量（all-MiniLM-L6-v2）：
        # - 距离 0.0-0.3: 非常相似（相似度 0.9-1.0）
        # - 距离 0.3-0.6: 中等相似（相似度 0.7-0.9）
        # - 距离 0.6-1.0: 一般相似（相似度 0.5-0.7）
        # - 距离 1.0-1.5: 不太相似（相似度 0.3-0.5）
        # - 距离 > 1.5: 不相似（相似度 < 0.3）
        
        # 使用 sigmoid 函数将距离转换为相似度
        # similarity = 1 / (1 + distance^2) 或使用更平滑的转换
        # 这里使用：similarity = exp(-distance) 或 1 / (1 + distance)
        
        filtered_hits = []
        for hit in hits:
            dist = hit['distance']
            
            # 使用 max_distance 阈值（更直观）
            if max_distance is not None and dist > max_distance:
                continue
            
            # 计算绝对相似度（基于实际距离值）
            # 方法1：使用指数衰减 similarity = exp(-distance)
            # 但这样距离为0时相似度为1，距离为1时相似度约为0.37，可能太低
            
            # 方法2：使用 1 / (1 + distance) 
            # 距离0 -> 相似度1.0，距离1 -> 相似度0.5，距离2 -> 相似度0.33
            
            # 方法3：使用更合理的映射，考虑实际距离范围
            # 对于 L2 距离，使用：similarity = 1 / (1 + distance * scale_factor)
            # scale_factor 可以根据经验调整，比如 1.5
            scale_factor = 1.5
            similarity = 1.0 / (1.0 + dist * scale_factor)
            
            # 或者使用分段函数，更符合实际语义相似度
            if dist <= 0.3:
                # 非常相似
                similarity = 0.9 + 0.1 * (1.0 - dist / 0.3)
            elif dist <= 0.6:
                # 中等相似
                similarity = 0.7 + 0.2 * (1.0 - (dist - 0.3) / 0.3)
            elif dist <= 1.0:
                # 一般相似
                similarity = 0.5 + 0.2 * (1.0 - (dist - 0.6) / 0.4)
            elif dist <= 1.5:
                # 不太相似
                similarity = 0.3 + 0.2 * (1.0 - (dist - 1.0) / 0.5)
            else:
                # 不相似
                similarity = max(0.1, 0.3 * (1.0 - (dist - 1.5) / 2.0))
            
            # 使用 min_similarity 阈值
            if min_similarity is not None and similarity < min_similarity:
                continue
            
            hit['similarity'] = similarity
            filtered_hits.append(hit)
        
        # 按相似度排序（从高到低），并限制返回 k 个
        filtered_hits.sort(key=lambda x: x['similarity'], reverse=True)
        return filtered_hits[:k]

    def clear_all(self):
        """清除集合中的所有文档"""
        try:
            # 获取所有文档的 ID
            results = self.collection.get()
            if results and results.get('ids'):
                # 删除所有文档
                self.collection.delete(ids=results['ids'])
                return len(results['ids'])
            return 0
        except Exception as e:
            # 如果集合不存在或为空，返回 0
            return 0
    
    def delete_collection(self):
        """删除整个集合（用于完全重置）"""
        try:
            self.client.delete_collection(name=self.collection_name)
            # 重新创建空集合
            self._ensure_collection()
        except Exception:
            # 如果集合不存在，直接创建
            self._ensure_collection()
