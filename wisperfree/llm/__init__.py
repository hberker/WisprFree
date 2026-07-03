"""LLM cleanup backends (Stage 2). Selected by ``config.llm.backend``."""

from wisperfree.llm.base import LLMBackend
from wisperfree.llm.prompts import build_cleanup_system_prompt
from wisperfree.config import LLMConfig


def create_llm_backend(config: LLMConfig) -> LLMBackend:
    if config.backend == "ollama":
        from wisperfree.llm.ollama_backend import OllamaBackend

        return OllamaBackend(config)
    raise ValueError(f"unknown LLM backend: {config.backend!r}")


__all__ = ["LLMBackend", "create_llm_backend", "build_cleanup_system_prompt"]
