"""Simple ingestion helpers: extract text and chunk it.

Supports: .txt, .md, .pdf, .docx
Extend this module to integrate Markitdown or other converters for PPTX/images.
"""
from typing import List
import os
import tempfile
import warnings
import docx

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

# Try to import the LangChain text splitter; provide a lightweight fallback if unavailable.
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    warnings.warn(
        "langchain not available or RecursiveCharacterTextSplitter import failed; "
        "using a simple fallback splitter. For better splitting, install langchain."
    )

    class _SimpleRecursiveCharacterTextSplitter:
        """A very small fallback to mimic RecursiveCharacterTextSplitter.split_text."""

        def __init__(self, chunk_size=1000, chunk_overlap=200, length_function=len, separators=None):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.length_function = length_function
            self.separators = separators or ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

        def split_text(self, text: str) -> List[str]:
            if not text:
                return []

            # Try to split by the first available separator to produce more natural chunks.
            for sep in self.separators:
                if sep and sep in text:
                    parts = text.split(sep)
                    chunks: List[str] = []
                    for part in parts:
                        part = part.strip()
                        if not part:
                            continue
                        # Ensure parts that are too long are further split with overlap.
                        i = 0
                        while i < len(part):
                            end = min(i + self.chunk_size, len(part))
                            chunks.append(part[i:end].strip())
                            if end >= len(part):
                                break
                            i += max(1, self.chunk_size - self.chunk_overlap)
                    return chunks

            # Fallback sliding-window splitting if no separator matched.
            chunks = []
            start = 0
            text_len = len(text)
            while start < text_len:
                end = min(start + self.chunk_size, text_len)
                chunks.append(text[start:end].strip())
                if end >= text_len:
                    break
                start = max(0, end - self.chunk_overlap)
            return chunks

    # expose the fallback under the expected name
    RecursiveCharacterTextSplitter = _SimpleRecursiveCharacterTextSplitter



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
    
    # 获取原始文件的扩展名
    original_name = uploaded_file.name
    _, ext = os.path.splitext(original_name)
    
    # 保留原始扩展名，这样 extract_text_from_file 才能正确识别文件类型
    fd, path = tempfile.mkstemp(prefix="upload_", dir=dst_dir, suffix=ext)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path
