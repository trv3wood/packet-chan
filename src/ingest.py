"""Simple ingestion helpers: extract text and chunk it.

Supports: .txt, .md, .pdf, .docx
Extend this module to integrate Markitdown or other converters for PPTX/images.
"""
from typing import List
import os
import tempfile
import warnings

# 尝试使用 pypdf（PyPDF2 的升级版，对中文支持更好）
try:
    from pypdf import PdfReader
    USE_PYPDF = True
except ImportError:
    try:
        from PyPDF2 import PdfReader
        USE_PYPDF = False
        warnings.warn("建议使用 pypdf 替代 PyPDF2 以获得更好的中文支持。运行: pip install pypdf")
    except ImportError:
        PdfReader = None
        warnings.warn("未安装 PDF 处理库。运行: pip install pypdf")

import docx


def extract_text_from_pdf(path: str) -> str:
    """从 PDF 文件提取文本，支持多种提取方法和编码"""
    # 首先尝试 pypdf
    try:
        return _extract_with_pypdf(path)
    except Exception as e:
        # 如果 pypdf 失败，尝试 pdfplumber
        try:
            import pdfplumber
            return _extract_with_pdfplumber(path)
        except ImportError:
            raise ValueError(
                f"PDF 提取失败: {e}\n"
                f"建议安装 pdfplumber 以获得更好的支持: pip install pdfplumber"
            )
        except Exception as e2:
            raise ValueError(f"PDF 提取失败（pypdf 和 pdfplumber 都失败）: {e}, {e2}")


def _extract_with_pypdf(path: str) -> str:
    """使用 pypdf 提取文本"""
    text = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*encoding.*")
        
        reader = PdfReader(path, strict=False)
        total_pages = len(reader.pages)
    
    for page_num, page in enumerate(reader.pages, 1):
        try:
            page_text = page.extract_text()
            if page_text and len(page_text.strip()) > 0:
                text.append(page_text)
            else:
                text.append(f"[第 {page_num} 页：无法提取文本]")
        except Exception as e:
            text.append(f"[第 {page_num} 页：提取错误 - {str(e)}]")
    
    result = "\n\n".join(text)
    actual_text = result.replace("[第", "").replace("页：无法提取文本]", "").replace("页：提取错误", "").strip()
    
    if not actual_text or len(actual_text) < 10:
        raise ValueError(f"PDF 文本提取失败：提取的文本太短或为空")
    
    return result


def _extract_with_pdfplumber(path: str) -> str:
    """使用 pdfplumber 提取文本（回退方法）"""
    import pdfplumber
    text = []
    
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            try:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
                else:
                    text.append(f"[第 {page_num} 页：无法提取文本]")
            except Exception as e:
                text.append(f"[第 {page_num} 页：提取错误 - {str(e)}]")
    
    result = "\n\n".join(text)
    actual_text = result.replace("[第", "").replace("页：无法提取文本]", "").replace("页：提取错误", "").strip()
    
    if not actual_text or len(actual_text) < 10:
        raise ValueError(f"PDF 文本提取失败：提取的文本太短或为空")
    
    return result


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
    
    # 获取原始文件的扩展名
    original_name = uploaded_file.name
    _, ext = os.path.splitext(original_name)
    
    # 保留原始扩展名，这样 extract_text_from_file 才能正确识别文件类型
    fd, path = tempfile.mkstemp(prefix="upload_", dir=dst_dir, suffix=ext)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path
