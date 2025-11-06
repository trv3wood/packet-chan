"""LLM adapter supporting Ollama (local) and placeholder for OpenAI.

Current default: calls the `ollama` CLI via subprocess. This is a small, explicit approach
so we don't require extra Python client dependencies. Adjust the command or implement an HTTP
client if you run Ollama with a REST API proxy.
"""
import os
import subprocess
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")


def generate_with_ollama(prompt: str, model: Optional[str] = None, timeout: int = 60) -> str:
    model = model or OLLAMA_MODEL
    # Uses `ollama run <model> '<prompt>'`
    # The prompt is passed as an argument to the run command
    cmd = ["ollama", "run", model, prompt]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding='utf-8')
        if proc.returncode != 0:
            raise RuntimeError(f"Ollama error: {proc.stderr}")
        return proc.stdout
    except FileNotFoundError:
        raise RuntimeError("`ollama` CLI not found. Install Ollama and ensure `ollama` is on your PATH.")


def generate(prompt: str, model: Optional[str] = None) -> str:
    """Unified generate function. Current default: Ollama.

    Extend this to call OpenAI's API if OPENAI_API_KEY is present.
    """
    # Future: check OPENAI_API_KEY and call OpenAI client if present.
    return generate_with_ollama(prompt, model=model)
