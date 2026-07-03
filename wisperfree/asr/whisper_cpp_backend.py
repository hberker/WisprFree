"""whisper.cpp backend.

Shells out to the ``whisper-cli`` binary (Metal-accelerated on Apple
Silicon, CUDA/OpenVINO builds available). Chosen when
``asr.backend: whisper_cpp`` — useful on macOS where whisper.cpp's Metal
path beats CPU CTranslate2.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from wisperfree.asr.base import ASRBackend, TranscriptionResult
from wisperfree.config import ASRConfig


class WhisperCppBackend(ASRBackend):
    def __init__(self, config: ASRConfig):
        self.config = config
        if not config.whisper_cpp_model_path:
            raise ValueError(
                "asr.whisper_cpp_model_path must point to a ggml/gguf model "
                "file when using the whisper_cpp backend"
            )

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        initial_prompt: str = "",
    ) -> TranscriptionResult:
        pcm = self.to_float32(audio)
        with tempfile.TemporaryDirectory(prefix="wisperfree-") as tmp:
            wav_path = Path(tmp) / "chunk.wav"
            out_prefix = Path(tmp) / "out"
            _write_wav(wav_path, pcm, sample_rate)
            cmd = [
                self.config.whisper_cpp_binary,
                "-m", str(self.config.whisper_cpp_model_path),
                "-f", str(wav_path),
                "--output-json",
                "--output-file", str(out_prefix),
                "--no-prints",
                "--beam-size", str(self.config.beam_size),
            ]
            if self.config.language:
                cmd += ["-l", self.config.language]
            if initial_prompt:
                cmd += ["--prompt", initial_prompt]
            subprocess.run(cmd, check=True, capture_output=True, timeout=120)
            data = json.loads((out_prefix.with_suffix(".json")).read_text())
        texts = [
            seg["text"].strip()
            for seg in data.get("transcription", [])
            if seg.get("text", "").strip()
        ]
        return TranscriptionResult(
            text=" ".join(texts).strip(),
            language=data.get("result", {}).get("language"),
            duration_s=len(pcm) / sample_rate,
            segments=texts,
        )


def _write_wav(path: Path, pcm: np.ndarray, sample_rate: int) -> None:
    ints = (np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(ints.tobytes())
