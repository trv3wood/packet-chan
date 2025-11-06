"""Retriever: fetch top-k contexts and assemble prompt for LLM."""
from typing import List, Dict


def assemble_context(snippets: List[Dict], max_chars: int = 2000) -> str:
    """Concatenate snippets into a single context string, limited by max_chars."""
    parts = []
    cur = 0
    for s in snippets:
        text = s.get('document') or s.get('text') or ""
        if not text:
            continue
        if cur + len(text) > max_chars:
            # take a tail slice if room remains
            remaining = max_chars - cur
            if remaining <= 0:
                break
            parts.append(text[:remaining])
            cur += remaining
            break
        parts.append(text)
        cur += len(text)
    return "\n\n---\n\n".join(parts)


def build_prompt(question: str, contexts: List[Dict]) -> str:
    """Build an augmented prompt for the LLM using retrieved contexts."""
    context_str = assemble_context(contexts)
    prompt = f"Use the following extracted context to answer the question. If the answer is not contained within the context, say you don't know.\n\nContext:\n{context_str}\n\nQuestion: {question}\n\nAnswer:"
    return prompt
