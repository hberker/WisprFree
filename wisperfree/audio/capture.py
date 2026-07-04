"""Microphone capture with VAD-driven chunking.

A background stream feeds fixed-size frames through the VAD. While
dictation is active, speech is accumulated; a chunk is emitted when the
speaker pauses for ``end_silence_ms`` (or the chunk hits ``max_chunk_s``),
so transcription can start while the user is still talking. A short
pre-roll buffer keeps word onsets from being clipped.

``sounddevice`` (PortAudio) is an optional dependency; importing this
module without it still works so the rest of the app (API, settings UI,
tests) runs on machines without audio hardware.
"""

from __future__ import annotations

import queue
import threading
from collections import deque
from typing import Callable, Optional

import numpy as np

from wisperfree.audio.vad import VoiceActivityDetector
from wisperfree.config import AudioConfig

try:  # optional dependency
    import sounddevice  # type: ignore

    _HAS_SOUNDDEVICE = True
except (ImportError, OSError):  # OSError: PortAudio missing
    _HAS_SOUNDDEVICE = False


class MicrophoneCapture:
    """Toggleable mic capture emitting speech chunks via callback."""

    def __init__(
        self,
        config: AudioConfig,
        on_chunk: Callable[[np.ndarray], None],
        vad: Optional[VoiceActivityDetector] = None,
    ):
        if not _HAS_SOUNDDEVICE:
            raise RuntimeError(
                "sounddevice is not installed; install wisperfree[audio]"
            )
        self.config = config
        self.on_chunk = on_chunk
        self.vad = vad or VoiceActivityDetector(
            sample_rate=config.sample_rate,
            frame_ms=config.frame_ms,
            backend=config.vad_backend,
            aggressiveness=config.vad_aggressiveness,
        )
        self._frame_len = config.sample_rate * config.frame_ms // 1000
        self._frames: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream: Optional["sounddevice.InputStream"] = None
        self._worker: Optional[threading.Thread] = None
        self._running = threading.Event()

    @property
    def active(self) -> bool:
        return self._running.is_set()

    def start(self) -> None:
        if self.active:
            return
        self._running.set()
        self._stream = sounddevice.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype="int16",
            blocksize=self._frame_len,
            device=self.config.input_device,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._worker = threading.Thread(target=self._segment_loop, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        if not self.active:
            return
        self._running.clear()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        self._frames.put(indata[:, 0].copy())

    def _segment_loop(self) -> None:
        cfg = self.config
        silence_frames_limit = max(1, cfg.end_silence_ms // cfg.frame_ms)
        pre_roll_frames = max(1, cfg.pre_roll_ms // cfg.frame_ms)
        max_frames = int(cfg.max_chunk_s * 1000 / cfg.frame_ms)

        pre_roll: deque[np.ndarray] = deque(maxlen=pre_roll_frames)
        chunk: list[np.ndarray] = []
        silence_run = 0
        in_speech = False

        def flush() -> None:
            nonlocal chunk, silence_run, in_speech
            if chunk:
                audio = np.concatenate(chunk)
                trimmed = self.vad.trim_silence(audio)
                if len(trimmed) > 0:
                    self.on_chunk(trimmed)
            chunk = []
            silence_run = 0
            in_speech = False

        while self._running.is_set() or not self._frames.empty():
            try:
                frame = self._frames.get(timeout=0.2)
            except queue.Empty:
                continue
            if len(frame) != self._frame_len:
                continue
            if self.vad.is_speech(frame):
                if not in_speech:
                    chunk.extend(pre_roll)
                    pre_roll.clear()
                    in_speech = True
                chunk.append(frame)
                silence_run = 0
            elif in_speech:
                chunk.append(frame)
                silence_run += 1
                if silence_run >= silence_frames_limit:
                    flush()
            else:
                pre_roll.append(frame)
            if len(chunk) >= max_frames:
                flush()
        flush()
