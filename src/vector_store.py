"""Simple vector store wrapper using in-memory storage.

Uses sentence-transformers to produce embeddings and stores them in memory.
"""
import logging
from typing import List, Dict, Any
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "rag_collection"):
        """初始化向量存储"""
        logger.info("开始初始化VectorStore")
        
        # 内存存储替代ChromaDB
        self.persist_directory = persist_directory  # 保留参数以保持接口兼容性
        self.collection_name = collection_name  # 保留参数以保持接口兼容性
        
        # 内存存储
        self.ids = []
        self.documents = []
        self.embeddings = []
        self.metadata = []
        
        # 尝试加载本地嵌入模型，添加错误处理和超时控制
        self.embed_model = None
        
        # 首先检查是否存在本地缓存的模型
        import os
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "sentence_transformers")
        model_path = os.path.join(cache_dir, "BAAI_bge-base-zh-v1.5")
        logger.info(f"检查本地模型缓存路径: {model_path}")
        
        # 快速检查是否存在本地模型（通过检查是否有config.json文件）
        has_local_model = os.path.exists(os.path.join(model_path, "config.json"))
        logger.info(f"本地模型存在: {has_local_model}")
        
        # 设置严格的超时控制
        import sys
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        from urllib3.exceptions import ReadTimeoutError
        
        # 设置环境变量禁用huggingface的自动下载
        os.environ["HF_HUB_OFFLINE"] = "1"
        
        try:
            # 首先尝试使用本地模型（如果存在）
            if has_local_model:
                logger.info("尝试使用本地缓存的模型")
                self.embed_model = SentenceTransformer(
                    model_path,
                    cache_folder=cache_dir,
                    use_auth_token=False
                )
                logger.info("成功加载本地模型")
            else:
                # 配置requests会话，使用更激进的超时设置
                logger.info("配置requests会话和超时控制")
                session = requests.Session()
                # 设置较少的重试次数和较短的超时
                retry = Retry(total=2, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=None)
                adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
                session.mount('http://', adapter)
                session.mount('https://', adapter)
                
                # 覆盖默认的超时设置
                session.timeout = (5, 10)  # 连接超时5秒，读取超时10秒
                
                # 尝试在超时时间内加载模型
                logger.info("尝试快速加载嵌入模型 BAAI/bge-base-zh-v1.5")
                
                # 使用try-except捕获可能的超时异常
                try:
                    # 在独立线程中尝试加载，设置超时
                    import threading
                    
                    def load_model_thread():
                        try:
                            # 尝试加载模型，设置环境变量控制行为
                            os.environ["HF_HUB_OFFLINE"] = "0"  # 临时启用在线模式
                            self.embed_model = SentenceTransformer(
                                "BAAI/bge-base-zh-v1.5",
                                cache_folder=cache_dir,
                                use_auth_token=False,
                                device="cpu"  # 确保使用CPU以避免额外依赖
                            )
                            logger.info("嵌入模型加载成功")
                        except Exception as inner_e:
                            logger.error(f"模型加载线程异常: {inner_e}")
                            raise
                    
                    # 启动加载线程，设置15秒超时
                    model_thread = threading.Thread(target=load_model_thread)
                    model_thread.start()
                    model_thread.join(timeout=15)  # 15秒超时
                    
                    # 如果线程仍在运行，说明超时
                    if model_thread.is_alive():
                        raise TimeoutError("模型加载超时")
                    
                    # 确保模型已加载
                    if self.embed_model is None:
                        raise ValueError("模型加载失败")
                        
                except (TimeoutError, requests.exceptions.Timeout, ReadTimeoutError) as timeout_e:
                    logger.warning(f"模型加载超时: {timeout_e}")
                    raise
        except Exception as e:
            logger.error(f"无法加载嵌入模型: {e}")
            logger.info("立即切换到回退嵌入模型，避免长时间等待")
            # 立即使用回退模型
            self._create_fallback_embed_model()
        
        logger.info("VectorStore初始化完成")
    
    def _create_fallback_embed_model(self):
        """创建一个简单的回退嵌入模型"""
        logger.info("创建回退嵌入模型")
        
        class FallbackEmbeddingModel:
            def encode(self, texts, show_progress_bar=False, batch_size=32, convert_to_numpy=True):
                logger.info(f"回退模型开始处理文本，数量: {len(texts) if isinstance(texts, list) else 1}")
                if isinstance(texts, str):
                    texts = [texts]
                
                # 返回简单的基于字符频率的嵌入
                embeddings = []
                for i, text in enumerate(texts):
                    if i % 10 == 0 or i == len(texts) - 1:  # 每10个文本或最后一个记录一次日志
                        logger.info(f"处理第 {i+1}/{len(texts)} 个文本")
                    
                    # 简单的嵌入生成方法：基于常见字符的频率
                    embedding = np.zeros(768)  # 保持与原始模型相同的维度
                    for j, char in enumerate(text[:768]):  # 只使用前768个字符
                        embedding[j] = ord(char) % 100 / 100.0  # 归一化到[0,1]
                    embeddings.append(embedding)
                
                result = np.array(embeddings) if convert_to_numpy else embeddings
                logger.info(f"回退模型编码完成，生成嵌入数量: {len(embeddings)}")
                return result
        
        self.embed_model = FallbackEmbeddingModel()
        logger.warning("已启用回退嵌入模型。请注意，搜索质量可能会降低。")

    def add_documents(self, docs: List[Dict[str, Any]], batch_size: int = 32):
        """docs: list of dicts with keys: id (str), text (str), metadata (dict)
        
        Args:
            batch_size: 批量处理大小，较大的批次可以提高 embedding 生成速度
        """
        logger.info(f"开始添加文档，文档数量: {len(docs)}")
        
        new_ids = [d["id"] for d in docs]
        new_texts = [d["text"] for d in docs]
        new_metadatas = [d.get("metadata", {}) for d in docs]
        
        logger.info(f"文档ID和文本已准备好，准备生成嵌入")
        
        # 批量生成 embeddings
        try:
            logger.info(f"开始批量生成嵌入，批大小: {batch_size}")
            new_embeddings = self.embed_model.encode(
                new_texts, 
                show_progress_bar=False,
                batch_size=batch_size,
                convert_to_numpy=True
            )
            logger.info(f"嵌入生成成功，嵌入维度: {new_embeddings.shape[1] if hasattr(new_embeddings, 'shape') else len(new_embeddings[0])}")
        except Exception as e:
            logger.error(f"生成嵌入时出错: {e}")
            # 如果是BAAI模型失败，尝试使用回退模型
            if not isinstance(self.embed_model, type(self._create_fallback_embed_model())):
                logger.info("切换到回退嵌入模型")
                self._create_fallback_embed_model()
                new_embeddings = self.embed_model.encode(
                    new_texts, 
                    show_progress_bar=False,
                    batch_size=batch_size,
                    convert_to_numpy=True
                )
            else:
                logger.error("回退模型也失败了，使用空嵌入")
                new_embeddings = [np.zeros(768) for _ in range(len(new_texts))]
        
        # 添加到内存存储
        self.ids.extend(new_ids)
        self.documents.extend(new_texts)
        self.embeddings.extend(new_embeddings)
        self.metadata.extend(new_metadatas)
        
        logger.info(f"文档添加完成，当前存储文档总数: {len(self.documents)}")

    def query(self, query_text: str, k: int = 4, min_similarity: float = None, max_distance: float = None):
        """查询最相似的文档片段
        
        Args:
            query_text: 查询文本
            k: 返回的结果数量
            min_similarity: 最小相似度阈值（0-1），低于此值的片段将被过滤
            max_distance: 最大距离阈值，超过此距离的片段将被过滤（更直观）
                         如果设置了 max_distance，会优先使用它而不是 min_similarity
        """
        logger.info(f"开始查询，查询文本: {query_text[:100]}..." if len(query_text) > 100 else f"开始查询，查询文本: {query_text}")
        logger.info(f"当前存储文档数量: {len(self.documents)}")
        
        if not self.documents:
            logger.info("没有文档可供查询，返回空结果")
            return []
        
        # 生成查询嵌入
        try:
            logger.info("生成查询嵌入")
            query_emb = self.embed_model.encode([query_text])[0]
            logger.info("查询嵌入生成成功")
        except Exception as e:
            logger.error(f"生成查询嵌入时出错: {e}")
            # 尝试使用回退模型
            if not isinstance(self.embed_model, type(self._create_fallback_embed_model())):
                logger.info("切换到回退嵌入模型")
                self._create_fallback_embed_model()
                query_emb = self.embed_model.encode([query_text])[0]
            else:
                logger.error("回退模型也失败了，使用默认查询嵌入")
                query_emb = np.zeros(768)
        
        hits = []
        
        # 计算与每个文档的相似度
        logger.info("开始计算与每个文档的相似度")
        for i, doc_emb in enumerate(self.embeddings):
            if i % 50 == 0 or i == len(self.embeddings) - 1:  # 每50个嵌入或最后一个记录一次日志
                logger.info(f"计算第 {i+1}/{len(self.embeddings)} 个文档的相似度")
            
            try:
                # 计算欧氏距离（与ChromaDB行为一致）
                distance = np.linalg.norm(query_emb - doc_emb)
                
                hit = {
                    "document": self.documents[i],
                    "metadata": self.metadata[i],
                    "distance": distance,
                }
                hits.append(hit)
            except Exception as e:
                logger.error(f"计算第 {i} 个文档的相似度时出错: {e}")
        
        logger.info("相似度计算完成，开始过滤和排序")
        
        # 应用过滤和排序逻辑（与原始代码保持一致）
        filtered_hits = []
        for hit in hits:
            dist = hit['distance']
            
            # 使用 max_distance 阈值（更直观）
            if max_distance is not None and dist > max_distance:
                continue
            
            # 计算绝对相似度（基于实际距离值）
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
        
        logger.info(f"查询完成，返回前 {min(k, len(filtered_hits))} 个结果")
        return filtered_hits[:k]

    def clear_all(self):
        """清除集合中的所有文档"""
        logger.info(f"清空所有存储的数据，当前文档数量: {len(self.ids)}")
        count = len(self.ids)
        self.ids = []
        self.documents = []
        self.embeddings = []
        self.metadata = []
        logger.info("数据清空完成")
        return count
    
    def delete_collection(self):
        """删除整个集合（用于完全重置）"""
        logger.info("删除集合")
        self.clear_all()
        logger.info("集合删除完成")
