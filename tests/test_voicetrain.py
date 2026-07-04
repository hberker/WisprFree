import base64
import io
import time
import wave

import numpy as np
import pytest

from tests.conftest import speech_audio
from wisperfree.storage.voice_samples import VoiceSampleStore
from wisperfree.voicetrain.prompts import BASE_SENTENCES, generate_prompts
from wisperfree.voicetrain.runner import VoiceTrainRunner
from wisperfree.voicetrain.trainer import (
    TrainOutcome,
    resolve_base_model,
    split_samples,
)
from wisperfree.voicetrain.wer import corpus_wer, word_error_rate


def wav_bytes(seconds=1.0, sample_rate=16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(speech_audio(seconds, sample_rate).tobytes())
    return buffer.getvalue()


@pytest.fixture
def samples(db, tmp_path) -> VoiceSampleStore:
    return VoiceSampleStore(db, tmp_path / "voice_samples")


# ---- WER ------------------------------------------------------------------

def test_wer_identical():
    assert word_error_rate("hello world", "hello world") == 0.0


def test_wer_ignores_case_and_punctuation():
    assert word_error_rate("Hello, world!", "hello world") == 0.0


def test_wer_substitution_and_deletion():
    assert word_error_rate("a b c d", "a x c") == pytest.approx(0.5)


def test_wer_empty_reference():
    assert word_error_rate("", "anything") == 1.0
    assert word_error_rate("", "") == 0.0


def test_corpus_wer_weighted():
    pairs = [("one two three four", "one two three four"), ("a b", "x y")]
    assert corpus_wer(pairs) == pytest.approx(2 / 6)


# ---- prompts ----------------------------------------------------------------

def test_prompts_count_and_content():
    prompts = generate_prompts(n=10, seed=1)
    assert len(prompts) == 10
    assert all(p in BASE_SENTENCES for p in prompts)


def test_prompts_weave_in_dictionary_terms():
    prompts = generate_prompts(["Kubernetes", "Anthropic"], n=12, seed=1)
    joined = " ".join(prompts)
    assert "Kubernetes" in joined and "Anthropic" in joined


def test_prompts_no_duplicates():
    prompts = generate_prompts(n=20, seed=2)
    assert len(set(prompts)) == len(prompts)


def test_prompts_larger_corpus_supports_more_prompts():
    # the expanded corpus comfortably serves a 40-prompt session
    prompts = generate_prompts(n=40, seed=3)
    assert len(prompts) == 40
    assert len(set(prompts)) == 40


def test_llm_prompts_parse_and_dedupe():
    from wisperfree.voicetrain.prompts import generate_llm_prompts

    class FakeLLM:
        def generate(self, system, user):
            return (
                "1. The morning train was late again today.\n"
                "- She counted forty-two blue marbles.\n"
                '"We should ship the release on Friday."\n'
                "ok\n"  # too short, dropped
                "\n"
                "Kubernetes powers the whole deployment pipeline.\n"
            )

    prompts = generate_llm_prompts(FakeLLM(), ["Kubernetes"], n=3)
    assert len(prompts) == 3
    assert prompts[0] == "The morning train was late again today."
    assert not prompts[1].startswith("-")  # bullet stripped
    assert '"' not in prompts[2]  # quotes stripped


def test_llm_prompts_fall_back_when_llm_fails():
    from wisperfree.voicetrain.prompts import generate_llm_prompts

    class BrokenLLM:
        def generate(self, system, user):
            raise ConnectionError("ollama down")

    prompts = generate_llm_prompts(BrokenLLM(), ["Anthropic"], n=10)
    assert len(prompts) == 10  # curated fallback still delivers


def test_llm_prompts_none_backend_uses_corpus():
    from wisperfree.voicetrain.prompts import generate_llm_prompts

    prompts = generate_llm_prompts(None, n=8)
    assert len(prompts) == 8
    assert all(p in BASE_SENTENCES for p in prompts)


def test_llm_prompts_top_up_when_llm_returns_too_few():
    from wisperfree.voicetrain.prompts import generate_llm_prompts

    class ThinLLM:
        def generate(self, system, user):
            return "Only one usable sentence here today."

    prompts = generate_llm_prompts(ThinLLM(), n=6)
    assert len(prompts) == 6  # topped up from the corpus
    assert prompts[0] == "Only one usable sentence here today."


# ---- sample store -------------------------------------------------------------

def test_sample_add_list_remove(samples):
    sample = samples.add("The quick brown fox.", wav_bytes(2.0))
    assert sample.duration_s == pytest.approx(2.0)
    assert sample.sample_rate == 16000
    assert samples.count() == 1
    assert samples.total_minutes() == pytest.approx(2.0 / 60)

    from pathlib import Path

    path = Path(samples.list()[0].wav_path)
    assert path.exists()
    assert samples.remove(sample.id) is True
    assert not path.exists()
    assert samples.count() == 0
    assert samples.remove(999) is False


def test_sample_rejects_garbage(samples):
    with pytest.raises(ValueError, match="not a valid WAV"):
        samples.add("prompt", b"definitely not a wav")


def test_sample_rejects_too_short(samples):
    with pytest.raises(ValueError, match="too short"):
        samples.add("prompt", wav_bytes(0.2))


def test_sample_rejects_empty_prompt(samples):
    with pytest.raises(ValueError, match="prompt_text"):
        samples.add("  ", wav_bytes(1.0))


# ---- trainer helpers ------------------------------------------------------------

def test_resolve_base_model(config):
    assert (
        resolve_base_model(config.voicetrain, "base.en") == "openai/whisper-base.en"
    )
    config.voicetrain.base_model = "openai/whisper-small"
    assert resolve_base_model(config.voicetrain, "base.en") == "openai/whisper-small"


def test_resolve_base_model_rejects_unknown(config):
    with pytest.raises(ValueError, match="cannot derive"):
        resolve_base_model(config.voicetrain, "/some/tuned/model/dir")


def test_split_samples_deterministic(samples):
    for i in range(20):
        samples.add(f"prompt {i}", wav_bytes(1.0))
    train, holdout = split_samples(samples.list(), holdout_every=10)
    assert len(holdout) == 2
    assert len(train) == 18
    # same split every time
    train2, holdout2 = split_samples(samples.list(), holdout_every=10)
    assert [s.id for s in holdout] == [s.id for s in holdout2]


def test_split_samples_tiny_corpus(samples):
    for i in range(2):
        samples.add(f"prompt {i}", wav_bytes(1.0))
    train, holdout = split_samples(samples.list(), holdout_every=10)
    assert len(train) == 2 and holdout == []

    samples.add("prompt 3", wav_bytes(1.0))
    train, holdout = split_samples(samples.list(), holdout_every=10)
    assert len(holdout) == 1  # forced holdout so WER is always reported


def test_split_by_sentence_no_text_leak(samples):
    # Three takes each of the same 5 sentences. A recording-level split
    # could scatter takes of one sentence across train and holdout; the
    # text-level split must keep every take of a holdout sentence out of
    # training.
    for sentence in range(5):
        for _take in range(3):
            samples.add(f"sentence number {sentence}", wav_bytes(1.0))
    train, holdout = split_samples(samples.list(), holdout_every=3)
    train_texts = {_norm(s.prompt_text) for s in train}
    holdout_texts = {_norm(s.prompt_text) for s in holdout}
    assert holdout_texts  # something was held out
    assert train_texts.isdisjoint(holdout_texts)  # no sentence spans both
    # all three takes of each holdout sentence are held out together
    assert len(holdout) == 3 * len(holdout_texts)


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


# ---- runner --------------------------------------------------------------------

def wait_for(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def make_runner(config, db, samples, calls):
    def fake_train(vt_config, asr_model, sample_list, out_dir) -> TrainOutcome:
        calls.append((asr_model, len(sample_list)))
        model_dir = out_dir / "ct2"
        model_dir.mkdir(parents=True, exist_ok=True)
        return TrainOutcome(model_dir=str(model_dir), wer_before=0.20, wer_after=0.08)

    return VoiceTrainRunner(config, db, samples, train_fn=fake_train)


def test_runner_success_records_wer(config, db, samples):
    calls = []
    config.voicetrain.min_minutes = 0.05  # 3 seconds for the test
    runner = make_runner(config, db, samples, calls)
    for i in range(4):
        samples.add(f"prompt {i}", wav_bytes(1.0))
    assert runner.trigger() is True
    assert wait_for(lambda: not runner.running)
    assert calls == [("base.en", 4)]
    run = runner.runs()[0]
    assert run["status"] == "succeeded"
    assert run["wer_before"] == pytest.approx(0.20)
    assert run["wer_after"] == pytest.approx(0.08)
    assert runner.tuned_models() == [run["model_dir"]]


def test_runner_refuses_below_min_minutes(config, db, samples):
    calls = []
    config.voicetrain.min_minutes = 5.0
    runner = make_runner(config, db, samples, calls)
    samples.add("prompt", wav_bytes(2.0))
    runner.trigger()
    assert wait_for(lambda: not runner.running)
    assert calls == []
    run = runner.runs()[0]
    assert run["status"] == "failed"
    assert "minutes" in run["detail"]


def test_runner_rejects_concurrent(config, db, samples):
    import threading

    started = threading.Event()
    release = threading.Event()

    def slow_train(*args):
        started.set()
        release.wait(timeout=5)
        raise RuntimeError("stop")

    config.voicetrain.min_minutes = 0.01
    runner = VoiceTrainRunner(config, db, samples, train_fn=slow_train)
    samples.add("prompt", wav_bytes(1.0))
    assert runner.trigger() is True
    assert started.wait(timeout=5)
    assert runner.trigger() is False  # already running
    release.set()
    assert wait_for(lambda: not runner.running)


# ---- API -----------------------------------------------------------------------

def test_voicetrain_api(client):
    prompts = client.get("/api/voicetrain/prompts?n=5").json()
    assert len(prompts) == 5

    payload = {
        "prompt_text": prompts[0],
        "wav_base64": base64.b64encode(wav_bytes(1.5)).decode(),
    }
    sample = client.post("/api/voicetrain/samples", json=payload).json()
    assert sample["duration_s"] == pytest.approx(1.5)

    listing = client.get("/api/voicetrain/samples").json()
    assert listing["total_minutes"] == pytest.approx(1.5 / 60, abs=0.01)
    assert len(listing["samples"]) == 1

    bad = client.post(
        "/api/voicetrain/samples",
        json={"prompt_text": "x", "wav_base64": "!!!not base64!!!"},
    )
    assert bad.status_code == 422

    assert client.delete(f"/api/voicetrain/samples/{sample['id']}").json() == {"ok": True}
    assert client.delete("/api/voicetrain/samples/999").status_code == 404

    # trigger runs (fails fast on min_minutes, but the endpoint works)
    assert client.post("/api/voicetrain/trigger").status_code == 200
    assert client.get("/api/voicetrain/models").json() == []
