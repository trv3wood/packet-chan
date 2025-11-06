"""Simple ingestion helpers: extract text and chunk it.

Supports: .txt, .md, .pdf, .docx
Extend this module to integrate Markitdown or other converters for PPTX/images.
"""
from typing import List
import os
import tempfile

from PyPDF2 import PdfReader
import docx


def extract_text_from_pdf(path: str) -> str:
    text = []
    reader = PdfReader(path)
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n\n".join(text)


def extract_text_from_docx(path: str) -> str:
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n\n".join(paragraphs)


def extract_text_from_file(path: str) -> str:
    _, ext = os.path.splitext(path.lower())
    if ext in (".txt", ".md"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    if ext == ".docx":
        return extract_text_from_docx(path)
    # TODO: add pptx, images (OCR), other converters (Markitdown)
    raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Naive sliding window chunker by characters.

    Returns list of text chunks.
    """
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == length:
            break
        start = end - overlap
    return chunks


def save_uploaded_file(uploaded_file, dst_dir: str) -> str:
    """Given a Streamlit UploadedFile-like object, write to a temp file and return path."""
    os.makedirs(dst_dir, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="upload_", dir=dst_dir, suffix="_file")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path
