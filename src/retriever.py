"""Retriever: fetch top-k contexts and assemble prompt for LLM."""
from typing import List, Dict


def assemble_context(snippets: List[Dict], max_chars: int = 2000, min_similarity: float = 0.3) -> str:
    """Concatenate snippets into a single context string, limited by max_chars.
    
    Args:
        snippets: 检索到的文档片段列表（已按相似度排序）
        max_chars: 最大字符数限制
        min_similarity: 最小相似度阈值，低于此值的片段将被过滤
    """
    parts = []
    cur = 0
    filtered_count = 0
    
    for s in snippets:
        # 检查相似度阈值
        similarity = s.get('similarity', 1.0 - s.get('distance', 0))
        if similarity < min_similarity:
            filtered_count += 1
            continue
        
        text = s.get('document') or s.get('text') or ""
        if not text:
            continue
        
        if cur + len(text) > max_chars:
            remaining = max_chars - cur
            if remaining <= 0:
                break
            # 改进：尝试在句子边界截断
            if remaining < len(text):
                truncate_pos = text[:remaining].rfind('。')
                if truncate_pos == -1:
                    truncate_pos = text[:remaining].rfind('.')
                if truncate_pos == -1:
                    truncate_pos = text[:remaining].rfind('！')
                if truncate_pos == -1:
                    truncate_pos = text[:remaining].rfind('？')
                
                if truncate_pos > remaining * 0.5:
                    parts.append(text[:truncate_pos + 1])
                else:
                    parts.append(text[:remaining])
            else:
                parts.append(text)
            break
        
        parts.append(text)
        cur += len(text)
    
    # 如果过滤了片段，可以在返回的字符串中记录（可选）
    result = "\n\n---\n\n".join(parts)
    if filtered_count > 0:
        # 可以在结果中添加注释（但 LLM 可能会看到，所以不推荐）
        pass
    
    return result


def rerank_results(question: str, hits: List[Dict], top_k: int = None) -> List[Dict]:
    """使用简单的关键词匹配重排序结果
    
    这是一个简单的重排序方法，可以根据问题中的关键词对结果进行重新排序
    """
    if not hits:
        return hits
    
    # 提取问题中的关键词（简单方法：去除停用词）
    question_lower = question.lower()
    question_words = set(question_lower.split())
    
    # 计算每个片段的得分
    scored_hits = []
    for hit in hits:
        text = (hit.get('document') or hit.get('text') or "").lower()
        
        # 计算关键词匹配度
        matching_words = question_words.intersection(set(text.split()))
        keyword_score = len(matching_words) / len(question_words) if question_words else 0
        
        # 结合原始相似度和关键词匹配度
        original_similarity = hit.get('similarity', 0.5)
        combined_score = original_similarity * 0.7 + keyword_score * 0.3
        
        scored_hits.append({
            **hit,
            'rerank_score': combined_score,
            'keyword_matches': len(matching_words)
        })
    
    # 按重排序得分排序
    scored_hits.sort(key=lambda x: x['rerank_score'], reverse=True)
    
    # 返回 top_k 个结果
    if top_k is not None:
        return scored_hits[:top_k]
    return scored_hits


def build_prompt(question: str, contexts: List[Dict], use_reranking: bool = False) -> str:
    """Build an augmented prompt for the LLM using retrieved contexts.
    
    Args:
        question: 用户问题
        contexts: 检索到的上下文片段
        use_reranking: 是否使用重排序
    """
    # 可选：使用重排序
    if use_reranking:
        contexts = rerank_results(question, contexts)
    
    context_str = assemble_context(contexts)
    prompt = f"""根据上下文回答问题。如果上下文没有答案，说"我不知道"。

上下文：
{context_str}

问题：{question}

回答："""
    return prompt
