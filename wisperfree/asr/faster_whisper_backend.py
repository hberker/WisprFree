"""faster-whisper (CTranslate2) backend.

Fastest CPU/GPU path on most machines: quantized weights (int8 on CPU,
float16 on CUDA) and greedy decoding keep Stage 1 well under a second
for short utterances on consumer hardware.
"""

from __future__ import annotations

import numpy as np

from wisperfree.asr.base import ASRBackend, TranscriptionResult
from wisperfree.config import ASRConfig


class FasterWhisperBackend(ASRBackend):
    def __init__(self, config: ASRConfig):
        self.config = config
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel  # lazy heavy import

            device = self.config.device
            compute_type = self.config.compute_type
            if device == "auto":
                device = "cuda" if self._cuda_available() else "cpu"
            if compute_type == "auto":
                compute_type = "float16" if device == "cuda" else "int8"
            self._model = WhisperModel(
                self.config.model, device=device, compute_type=compute_type
            )
        return self._model

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import ctranslate2

            return ctranslate2.get_cuda_device_count() > 0
        except Exception:
            return False

    def warm_up(self) -> None:
        model = self._load()
        # One tiny decode compiles kernels / faults pages in
        silence = np.zeros(16000 // 2, dtype=np.float32)
        list(model.transcribe(silence, beam_size=1)[0])

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        initial_prompt: str = "",
    ) -> TranscriptionResult:
        model = self._load()
        pcm = self.to_float32(audio)
        if sample_rate != 16000:
            pcm = _resample(pcm, sample_rate, 16000)
        segments, info = model.transcribe(
            pcm,
            language=self.config.language,
            beam_size=self.config.beam_size,
            initial_prompt=initial_prompt or None,
            vad_filter=False,  # we already ran VAD upstream
            condition_on_previous_text=False,
        )
        texts = [s.text.strip() for s in segments]
        return TranscriptionResult(
            text=" ".join(t for t in texts if t).strip(),
            language=getattr(info, "language", None),
            duration_s=len(pcm) / 16000.0,
            segments=texts,
        )


def _resample(pcm: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return pcm
    n_out = int(len(pcm) * dst / src)
    x_old = np.linspace(0.0, 1.0, num=len(pcm), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, pcm).astype(np.float32)
