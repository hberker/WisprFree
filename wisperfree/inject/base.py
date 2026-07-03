from __future__ import annotations

from abc import ABC, abstractmethod

from wisperfree.config import InjectionConfig


class TextInjector(ABC):
    def __init__(self, config: InjectionConfig):
        self.config = config

    def inject(self, text: str) -> None:
        if not text:
            return
        if self.config.trailing_space and not text.endswith((" ", "\n")):
            text += " "
        self._type_text(text)

    @abstractmethod
    def _type_text(self, text: str) -> None:
        ...
