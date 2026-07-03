"""ASR backends (Stage 1). Selected by ``config.asr.backend``."""

from wisperfree.asr.base import ASRBackend, TranscriptionResult
from wisperfree.config import ASRConfig


def create_asr_backend(config: ASRConfig) -> ASRBackend:
    if config.backend == "faster_whisper":
        from wisperfree.asr.faster_whisper_backend import FasterWhisperBackend

        return FasterWhisperBackend(config)
    if config.backend == "whisper_cpp":
        from wisperfree.asr.whisper_cpp_backend import WhisperCppBackend

        return WhisperCppBackend(config)
    raise ValueError(f"unknown ASR backend: {config.backend!r}")


__all__ = ["ASRBackend", "TranscriptionResult", "create_asr_backend"]
