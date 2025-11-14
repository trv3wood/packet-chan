"""LLM adapter supporting OpenAI-compatible API and Ollama (fallback).

优先使用 OpenAI-compatible API（如果配置了），否则使用本地 Ollama。
"""
import os
import subprocess
from typing import Optional, Tuple
import sys

from dotenv import load_dotenv

load_dotenv()

# OpenAI-compatible API 配置
OPENAI_COMPATIBLE_API_KEY = os.getenv("OPENAI_COMPATIBLE_API_KEY")
OPENAI_COMPATIBLE_API_URL = os.getenv("OPENAI_COMPATIBLE_API_URL")
OPENAI_COMPATIBLE_MODEL = os.getenv("OPENAI_COMPATIBLE_MODEL")

# Ollama 配置（作为回退）
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

# 调试模式
DEBUG_MODE = os.getenv("OPENAI_DEBUG", "false").lower() == "true"


def debug_print(*args, **kwargs):
    """调试输出函数"""
    if DEBUG_MODE:
        print("[OPENAI DEBUG]", *args, **kwargs, file=sys.stderr, flush=True)


def get_llm_status() -> dict:
    """获取当前 LLM 配置状态"""
    load_dotenv(override=True)
    
    openai_api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
    openai_configured = bool(openai_api_key)
    
    # 检查 Ollama 是否可用
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    try:
        import urllib.request
        req = urllib.request.Request(f"{ollama_host}/api/tags")
        urllib.request.urlopen(req)
        ollama_available = True
    except:
        ollama_available = False
    
    openai_model = os.getenv("OPENAI_COMPATIBLE_MODEL", "gpt-3.5-turbo")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama2")
    
    if openai_configured:
        current_service = "OpenAI-compatible API"
        current_model = openai_model
        fallback_service = "Ollama" if ollama_available else "不可用"
    else:
        current_service = "Ollama" if ollama_available else "未配置"
        current_model = ollama_model if ollama_available else "无"
        fallback_service = "无"
    
    return {
        "current_service": current_service,
        "current_model": current_model,
        "openai_configured": openai_configured,
        "ollama_available": ollama_available,
        "fallback_service": fallback_service,
        "openai_api_key_set": bool(openai_api_key),
    }


def generate_with_oai_compat(prompt: str, model: Optional[str] = None, timeout: int = 60) -> str:
    """使用 OpenAI-compatible API 生成回答"""
    from openai import OpenAI
    if not OPENAI_COMPATIBLE_API_KEY:
        raise RuntimeError("OpenAI-compatible API 配置不完整。请设置 OPENAI_COMPATIBLE_API_KEY")
    
    # 使用配置的模型或默认模型
    model = model or OPENAI_COMPATIBLE_MODEL
    if not model:
        raise ValueError("no model specified")
    
    debug_print(f"使用 OpenAI-compatible API 调用模型: {model}")
    debug_print(f"API URL: {OPENAI_COMPATIBLE_API_URL}")
    
    try:
        # 创建 OpenAI 客户端
        client = OpenAI(
            api_key=OPENAI_COMPATIBLE_API_KEY,
            base_url=OPENAI_COMPATIBLE_API_URL,
            timeout=timeout
        )
        
        # 调用聊天完成 API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2048
        )
        
        # 提取回复内容
        answer = response.choices[0].message.content
        if not answer:
            raise RuntimeError("OpenAI-compatible API 返回了空回复")
        
        debug_print(f"OpenAI-compatible API 调用成功，回复长度: {len(answer)}")
        return answer.strip()
        
    except Exception as e:
        debug_print(f"OpenAI-compatible API 调用失败: {e}")
        raise RuntimeError(f"OpenAI-compatible API 调用失败: {e}")
    


def generate_with_ollama(prompt: str, model: Optional[str] = None, timeout: Optional[int] = None) -> str:
    """使用本地 Ollama 生成回答"""
    model = model or OLLAMA_MODEL
    host = OLLAMA_HOST or "http://localhost:11434"
    timeout = timeout or OLLAMA_TIMEOUT
    
    try:
        import urllib.request
        import urllib.error
        import json
        
        url = f"{host}/api/generate"
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                result = json.loads(response.read().decode('utf-8'))
                response_text = result.get("response", "")
                if not response_text:
                    raise RuntimeError(f"Ollama returned empty response. Full result: {result}")
                return response_text.strip()
        except urllib.error.URLError as e:
            if "timed out" in str(e).lower() or isinstance(e, urllib.error.HTTPError):
                raise RuntimeError(f"Ollama request timed out after {timeout} seconds.")
            raise RuntimeError(f"Ollama connection error: {e}")
    except Exception as e:
        raise RuntimeError(f"Ollama error: {e}")


def generate(prompt: str, model: Optional[str] = None) -> Tuple[str, str]:
    """统一的生成函数。优先使用 OpenAI-compatible API（如果配置了），否则使用 Ollama。
    
    返回: (答案, 使用的服务名称)
    """
    # 重新加载配置
    load_dotenv(override=True)
    openai_api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
    
    # 优先使用 OpenAI-compatible API
    if openai_api_key:
        try:
            answer = generate_with_oai_compat(prompt, model=model)
            return answer, "OpenAI-compatible API"
        except Exception as e:
            # 如果 OpenAI-compatible API 失败，回退到 Ollama
            error_msg = str(e)
            print(f"⚠️ OpenAI-compatible API 调用失败，回退到 Ollama: {error_msg}", file=sys.stderr, flush=True)
            try:
                answer = generate_with_ollama(prompt, model=model)
                return answer, "Ollama (回退)"
            except Exception as ollama_error:
                # 如果 Ollama 也失败，抛出原始错误
                raise RuntimeError(f"OpenAI-compatible API 失败: {error_msg}，Ollama 也失败: {ollama_error}")
    else:
        # 使用本地 Ollama
        answer = generate_with_ollama(prompt, model=model)
        return answer, "Ollama"
