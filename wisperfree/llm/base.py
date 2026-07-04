from __future__ import annotations

from abc import ABC, abstractmethod


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        ...

    def warm_up(self) -> None:
        """Optionally pre-load the model to hide first-call latency."""
