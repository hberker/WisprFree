"""Ollama backend: local LLM inference over the localhost HTTP API.

Note the ``keep_alive`` option — keeping the model resident between
dictations is what makes the <1.5s combined-latency target reachable.
"""

from __future__ import annotations

import httpx

from wisperfree.config import LLMConfig
from wisperfree.llm.base import LLMBackend


class OllamaBackend(LLMBackend):
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = httpx.Client(
            base_url=config.host, timeout=config.timeout_s
        )

    def warm_up(self) -> None:
        try:
            self._client.post(
                "/api/generate",
                json={
                    "model": self.config.model,
                    "prompt": "",
                    "keep_alive": self.config.keep_alive,
                },
            )
        except httpx.HTTPError:
            pass  # Ollama not running yet; first real call will surface it

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.post(
            "/api/chat",
            json={
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "keep_alive": self.config.keep_alive,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()

    def list_models(self) -> list[str]:
        response = self._client.get("/api/tags")
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]

    def close(self) -> None:
        self._client.close()
