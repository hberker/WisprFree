from __future__ import annotations

import numpy as np
import pytest

from wisperfree.asr.base import ASRBackend, TranscriptionResult
from wisperfree.config import AppConfig
from wisperfree.inject.base import TextInjector
from wisperfree.llm.base import LLMBackend
from wisperfree.storage import (
    CorrectionStore,
    Database,
    DictionaryStore,
    ToneProfileStore,
)


class FakeASR(ASRBackend):
    def __init__(self, text="um so meet on Tuesday no wait Wednesday"):
        self.text = text
        self.last_initial_prompt = None

    def transcribe(self, audio, sample_rate, initial_prompt=""):
        self.last_initial_prompt = initial_prompt
        return TranscriptionResult(text=self.text)


class FakeLLM(LLMBackend):
    def __init__(self, reply="Meet on Wednesday."):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []
        self.fail = False

    def generate(self, system_prompt, user_prompt):
        if self.fail:
            raise ConnectionError("ollama down")
        self.calls.append((system_prompt, user_prompt))
        return self.reply


class FakeInjector(TextInjector):
    def __init__(self, config):
        super().__init__(config)
        self.typed: list[str] = []

    def _type_text(self, text):
        self.typed.append(text)


class FakeContextDetector:
    def __init__(self, app_name="Slack", window_title=None):
        from wisperfree.context.detector import ActiveWindow

        self.window = ActiveWindow(app_name=app_name, window_title=window_title)

    def detect(self):
        return self.window


@pytest.fixture
def config(tmp_path) -> AppConfig:
    cfg = AppConfig(data_dir=str(tmp_path / "data"))
    return cfg


@pytest.fixture
def db(config) -> Database:
    database = Database(config.db_path)
    yield database
    database.close()


@pytest.fixture
def dictionary(db) -> DictionaryStore:
    return DictionaryStore(db)


@pytest.fixture
def corrections(db) -> CorrectionStore:
    return CorrectionStore(db)


@pytest.fixture
def tone_profiles(db) -> ToneProfileStore:
    return ToneProfileStore(db)


@pytest.fixture
def client(config, tmp_path, monkeypatch):
    """API test client with hardware/network-facing pieces swapped for fakes."""
    from fastapi.testclient import TestClient

    from wisperfree.api import create_app
    from wisperfree.daemon import Daemon

    # Config lives in tmp so apply_config_patch never touches the real home dir
    monkeypatch.setenv("WISPERFREE_CONFIG_DIR", str(tmp_path / "cfg"))
    daemon = Daemon(config)
    daemon.pipeline.asr = FakeASR()
    daemon.pipeline.llm = FakeLLM(reply="Cleaned text.")
    daemon.pipeline.injector = FakeInjector(config.injection)
    daemon.pipeline.context_detector = FakeContextDetector()
    yield TestClient(create_app(daemon))
    daemon.db.close()


def make_pipeline(config, dictionary, tone_profiles, asr=None, llm=None, app_name="Slack"):
    from wisperfree.pipeline import DictationPipeline

    injector = FakeInjector(config.injection)
    pipeline = DictationPipeline(
        config=config,
        asr=asr or FakeASR(),
        llm=llm,
        injector=injector,
        dictionary=dictionary,
        tone_profiles=tone_profiles,
        context_detector=FakeContextDetector(app_name=app_name),
    )
    return pipeline, injector


def speech_audio(seconds=1.0, sample_rate=16000, freq=220.0) -> np.ndarray:
    """Loud periodic signal the energy VAD classifies as speech."""
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return (np.sin(2 * np.pi * freq * t) * 0.5 * 32767).astype(np.int16)


def silence_audio(seconds=1.0, sample_rate=16000) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.normal(0, 20, int(seconds * sample_rate))).astype(np.int16)
