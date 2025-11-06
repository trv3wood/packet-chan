"""Simple ingestion helpers: extract text and chunk it.

Supports: .txt, .md, .pdf, .docx
Extend this module to integrate Markitdown or other converters for PPTX/images.
"""
from typing import List
import os
import tempfile

from PyPDF2 import PdfReader
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter


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


def extract_text_from_file(path: str, original_filename: str = None) -> str:
    """Extract text from file, using original filename extension if provided."""
    # Use original filename extension if available, otherwise use path extension
    if original_filename:
        _, ext = os.path.splitext(original_filename.lower())
    else:
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
    """LangChain RecursiveCharacterTextSplitter for intelligent text chunking.

    Returns list of text chunks.
    """
    if not text:
        return []
    
    # Create text splitter with the same parameters as the original function
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    
    # Split the text into chunks
    chunks = text_splitter.split_text(text)
    print(chunks)
    return chunks


def save_uploaded_file(uploaded_file, dst_dir: str) -> str:
    """Given a Streamlit UploadedFile-like object, write to a temp file and return path."""
    os.makedirs(dst_dir, exist_ok=True)
    
    # Preserve original file extension in the temporary filename
    original_name = uploaded_file.name
    _, original_ext = os.path.splitext(original_name)
    
    # Create temp file with original extension
    fd, path = tempfile.mkstemp(prefix="upload_", dir=dst_dir, suffix=original_ext)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path
