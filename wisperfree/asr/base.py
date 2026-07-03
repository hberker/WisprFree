from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TranscriptionResult:
    text: str
    language: str | None = None
    duration_s: float = 0.0
    segments: list[str] = field(default_factory=list)


class ASRBackend(ABC):
    """A local speech-to-text engine.

    ``initial_prompt`` carries the personal-dictionary glossary so the
    decoder is biased toward the user's names/jargon.
    """

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        initial_prompt: str = "",
    ) -> TranscriptionResult:
        ...

    def warm_up(self) -> None:
        """Optionally pre-load the model to hide first-call latency."""

    @staticmethod
    def to_float32(audio: np.ndarray) -> np.ndarray:
        if audio.dtype == np.int16:
            return audio.astype(np.float32) / 32768.0
        if audio.dtype in (np.float32, np.float64):
            return audio.astype(np.float32)
        raise TypeError(f"unsupported dtype {audio.dtype}")
