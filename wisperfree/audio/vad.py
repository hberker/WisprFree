"""Voice Activity Detection.

Uses webrtcvad when installed (fast, battle-tested), otherwise falls
back to an adaptive energy gate implemented in pure numpy so the
pipeline works everywhere. VAD serves two purposes:

  * trim leading/trailing silence so Whisper never sees dead air, and
  * segment the stream into chunks at natural pauses to cut latency.
"""

from __future__ import annotations

import numpy as np

try:  # optional dependency
    import webrtcvad  # type: ignore

    _HAS_WEBRTC = True
except ImportError:
    _HAS_WEBRTC = False


class VoiceActivityDetector:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        backend: str = "auto",
        aggressiveness: int = 2,
        energy_threshold_db: float = -38.0,
    ):
        if frame_ms not in (10, 20, 30):
            raise ValueError("frame_ms must be 10, 20 or 30")
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_len = sample_rate * frame_ms // 1000
        self.energy_threshold_db = energy_threshold_db
        self._noise_floor_db: float | None = None

        if backend == "auto":
            backend = "webrtc" if _HAS_WEBRTC else "energy"
        if backend == "webrtc" and not _HAS_WEBRTC:
            raise RuntimeError("webrtcvad is not installed")
        self.backend = backend
        self._webrtc = (
            webrtcvad.Vad(aggressiveness) if backend == "webrtc" else None
        )

    def is_speech(self, frame: np.ndarray) -> bool:
        """Classify one frame of int16 or float32 mono audio."""
        frame = self._normalize(frame)
        if len(frame) != self.frame_len:
            raise ValueError(
                f"expected {self.frame_len} samples, got {len(frame)}"
            )
        if self._webrtc is not None:
            return self._webrtc.is_speech(frame.tobytes(), self.sample_rate)
        return self._energy_is_speech(frame)

    def _energy_is_speech(self, frame: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        db = 20.0 * np.log10(max(rms, 1.0) / 32768.0)
        if self._noise_floor_db is None:
            self._noise_floor_db = -70.0
        # In noisy rooms the ambient floor lifts the threshold above the
        # fixed default so hum is not mistaken for speech.
        threshold = max(self.energy_threshold_db, self._noise_floor_db + 12.0)
        speech = db > threshold
        if not speech:  # track the noise floor from non-speech frames only
            self._noise_floor_db = 0.95 * self._noise_floor_db + 0.05 * db
        return bool(speech)

    @staticmethod
    def _normalize(frame: np.ndarray) -> np.ndarray:
        if frame.dtype == np.int16:
            return frame
        if frame.dtype in (np.float32, np.float64):
            return (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16)
        raise TypeError(f"unsupported dtype {frame.dtype}")

    def trim_silence(self, audio: np.ndarray, pad_frames: int = 4) -> np.ndarray:
        """Strip leading/trailing non-speech from a full utterance."""
        audio = self._normalize(audio)
        n_frames = len(audio) // self.frame_len
        flags = [
            self.is_speech(audio[i * self.frame_len : (i + 1) * self.frame_len])
            for i in range(n_frames)
        ]
        if not any(flags):
            return audio[:0]
        first = max(0, flags.index(True) - pad_frames)
        last = min(n_frames, n_frames - flags[::-1].index(True) + pad_frames)
        return audio[first * self.frame_len : last * self.frame_len]
