"""WisperFree daemon: wires config, storage, pipeline, hotkey, and the
local HTTP API together. The Electron tray app talks to this API on
localhost only — the server binds 127.0.0.1 and nothing else.
"""

from __future__ import annotations

import logging
from typing import Optional

from wisperfree.asr import create_asr_backend
from wisperfree.config import AppConfig, load_config, save_config, update_config
from wisperfree.context.detector import ContextDetector
from wisperfree.inject import create_injector
from wisperfree.llm import create_llm_backend
from wisperfree.finetune import FinetuneRunner
from wisperfree.pipeline import DictationPipeline
from wisperfree.storage import (
    CorrectionStore,
    Database,
    DictionaryStore,
    ToneProfileStore,
)

log = logging.getLogger(__name__)


class Daemon:
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.db = Database(self.config.db_path)
        self.dictionary = DictionaryStore(self.db)
        self.corrections = CorrectionStore(self.db)
        self.tone_profiles = ToneProfileStore(self.db)
        self.finetune = FinetuneRunner(self.config, self.db, self.corrections)

        self.asr = create_asr_backend(self.config.asr)
        self.llm = create_llm_backend(self.config.llm)
        self.injector = create_injector(self.config.injection)
        self.pipeline = DictationPipeline(
            config=self.config,
            asr=self.asr,
            llm=self.llm,
            injector=self.injector,
            dictionary=self.dictionary,
            tone_profiles=self.tone_profiles,
            context_detector=ContextDetector(),
        )
        self._capture = None
        self._hotkey = None
        self.dictating = False

    # ---- dictation control -------------------------------------------------

    def toggle_dictation(self) -> bool:
        if self.dictating:
            self.stop_dictation()
        else:
            self.start_dictation()
        return self.dictating

    def start_dictation(self) -> None:
        if self.dictating:
            return
        from wisperfree.audio.capture import MicrophoneCapture

        if self._capture is None:
            self._capture = MicrophoneCapture(
                self.config.audio, on_chunk=self.pipeline.submit_chunk
            )
        self.pipeline.start()
        self._capture.start()
        self.dictating = True
        log.info("dictation ON")

    def stop_dictation(self) -> None:
        if not self.dictating:
            return
        if self._capture:
            self._capture.stop()
        self.dictating = False
        log.info("dictation OFF")

    # ---- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        self.pipeline.start()
        self.pipeline.warm_up()
        try:
            from wisperfree.hotkey import HotkeyListener

            self._hotkey = HotkeyListener(
                self.config.hotkey.toggle,
                on_toggle=self.toggle_dictation,
            )
            self._hotkey.start()
        except Exception:
            log.warning(
                "global hotkey unavailable (pynput missing or no display); "
                "use the tray app / API to toggle dictation"
            )

    def shutdown(self) -> None:
        self.stop_dictation()
        if self._hotkey:
            self._hotkey.stop()
        self.pipeline.stop()
        self.db.close()

    # ---- config -------------------------------------------------------------

    def apply_config_patch(self, patch: dict) -> AppConfig:
        old = self.config
        self.config = update_config(self.config, patch)
        save_config(self.config)
        # Hot-swap backends when their settings changed
        if self.config.asr != old.asr:
            self.asr = create_asr_backend(self.config.asr)
            self.pipeline.asr = self.asr
        if self.config.llm != old.llm:
            self.llm = create_llm_backend(self.config.llm)
            self.pipeline.llm = self.llm
        if self.config.injection != old.injection:
            self.injector = create_injector(self.config.injection)
            self.pipeline.injector = self.injector
        if self.config.hotkey != old.hotkey and self._hotkey:
            self._hotkey.rebind(self.config.hotkey.toggle)
        self.pipeline.config = self.config
        self.finetune.config = self.config
        return self.config
