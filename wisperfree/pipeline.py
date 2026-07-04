"""The two-stage dictation pipeline.

    hotkey toggle -> mic capture (VAD-chunked)
        -> Stage 1: Whisper transcription (dictionary-seeded initial_prompt)
        -> Stage 2: LLM cleanup via Ollama (tone- and app-aware)
        -> native text injection at the cursor

Stage 2 is skipped when ``llm.enabled`` is false (Phase-1 behaviour) or
when the LLM is unreachable — raw text still lands at the cursor, which
beats losing the dictation.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from wisperfree.asr.base import ASRBackend
from wisperfree.config import AppConfig
from wisperfree.context.detector import ActiveWindow, ContextDetector
from wisperfree.inject.base import TextInjector
from wisperfree.llm.base import LLMBackend
from wisperfree.llm.prompts import build_cleanup_system_prompt
from wisperfree.storage import DictionaryStore, ToneProfileStore

log = logging.getLogger(__name__)


@dataclass
class DictationEvent:
    raw_text: str
    final_text: str
    app_name: str | None
    tone: str
    asr_latency_s: float
    llm_latency_s: float
    llm_used: bool
    timestamp: float = field(default_factory=time.time)


class DictationPipeline:
    """Wires the stages together; audio chunks go in, injected text comes out."""

    def __init__(
        self,
        config: AppConfig,
        asr: ASRBackend,
        llm: Optional[LLMBackend],
        injector: TextInjector,
        dictionary: DictionaryStore,
        tone_profiles: ToneProfileStore,
        context_detector: Optional[ContextDetector] = None,
    ):
        self.config = config
        self.asr = asr
        self.llm = llm
        self.injector = injector
        self.dictionary = dictionary
        self.tone_profiles = tone_profiles
        self.context_detector = context_detector or ContextDetector()
        self.history: list[DictationEvent] = []
        self._history_lock = threading.Lock()
        self._chunks: "queue.Queue[np.ndarray]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._running = threading.Event()

    def recent_history(self, limit: int = 50) -> list[DictationEvent]:
        """Newest-first snapshot, taken under lock so it can't tear against
        the worker thread mutating ``history``."""
        with self._history_lock:
            return list(reversed(self.history[-limit:]))

    # ---- lifecycle --------------------------------------------------------

    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._running.clear()
        if self._worker:
            self._worker.join(timeout=5.0)
            self._worker = None

    def submit_chunk(self, audio: np.ndarray) -> None:
        """Called by MicrophoneCapture for each VAD-segmented speech chunk."""
        self._chunks.put(audio)

    def warm_up(self) -> None:
        try:
            self.asr.warm_up()
        except Exception:
            log.exception("ASR warm-up failed")
        if self.llm and self.config.llm.enabled:
            try:
                self.llm.warm_up()
            except Exception:
                log.exception("LLM warm-up failed")

    # ---- processing -------------------------------------------------------

    def _loop(self) -> None:
        while self._running.is_set() or not self._chunks.empty():
            try:
                audio = self._chunks.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                event = self.process_chunk(audio)
                if event and event.final_text:
                    self.injector.inject(event.final_text)
            except Exception:
                log.exception("failed to process dictation chunk")

    def process_chunk(self, audio: np.ndarray) -> Optional[DictationEvent]:
        window = (
            self.context_detector.detect()
            if self.config.context.enabled
            else ActiveWindow()
        )
        raw, asr_latency = self._transcribe(audio)
        if not raw:
            return None
        final, tone, llm_latency, llm_used = self._clean(raw, window)
        event = DictationEvent(
            raw_text=raw,
            final_text=final,
            app_name=window.label,
            tone=tone,
            asr_latency_s=asr_latency,
            llm_latency_s=llm_latency,
            llm_used=llm_used,
        )
        with self._history_lock:
            self.history.append(event)
            del self.history[:-50]  # keep the last 50 for the UI / corrections
        log.info(
            "dictated %d chars (asr %.2fs, llm %.2fs, app=%s)",
            len(final), asr_latency, llm_latency, window.label,
        )
        return event

    def process_text(self, raw: str, app_name: str | None = None) -> DictationEvent:
        """Stage 2 only — used by the API for tests/replay and the UI preview."""
        window = ActiveWindow(app_name=app_name)
        final, tone, llm_latency, llm_used = self._clean(raw, window)
        return DictationEvent(
            raw_text=raw,
            final_text=final,
            app_name=window.label,
            tone=tone,
            asr_latency_s=0.0,
            llm_latency_s=llm_latency,
            llm_used=llm_used,
        )

    def _transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        prompt = self.dictionary.initial_prompt(
            max_terms=self.config.asr.max_prompt_terms
        )
        start = time.perf_counter()
        result = self.asr.transcribe(
            audio,
            sample_rate=self.config.audio.sample_rate,
            initial_prompt=prompt,
        )
        return result.text.strip(), time.perf_counter() - start

    def _clean(
        self, raw: str, window: ActiveWindow
    ) -> tuple[str, str, float, bool]:
        profile = self.tone_profiles.match(
            window.label, default_tone=self.config.context.default_tone
        )
        if not (self.llm and self.config.llm.enabled):
            return raw, profile.tone, 0.0, False
        system = build_cleanup_system_prompt(
            tone=profile.tone,
            app_name=window.label,
            dictionary_terms=self.dictionary.terms(limit=100),
            extra_instructions=profile.instructions,
        )
        start = time.perf_counter()
        try:
            cleaned = self.llm.generate(system, raw)
        except Exception:
            log.exception("LLM cleanup failed; injecting raw transcript")
            return raw, profile.tone, time.perf_counter() - start, False
        latency = time.perf_counter() - start
        return (cleaned or raw), profile.tone, latency, True
