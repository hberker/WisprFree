import numpy as np
import pytest

from tests.conftest import silence_audio, speech_audio
from wisperfree.audio.vad import VoiceActivityDetector


@pytest.fixture
def vad():
    return VoiceActivityDetector(backend="energy")


def test_speech_frames_detected(vad):
    frame = speech_audio(seconds=0.03)[: vad.frame_len]
    assert vad.is_speech(frame) is True


def test_silence_frames_rejected(vad):
    frame = silence_audio(seconds=0.03)[: vad.frame_len]
    assert vad.is_speech(frame) is False


def test_frame_length_enforced(vad):
    with pytest.raises(ValueError):
        vad.is_speech(np.zeros(10, dtype=np.int16))


def test_invalid_frame_ms():
    with pytest.raises(ValueError):
        VoiceActivityDetector(frame_ms=25)


def test_trim_silence_strips_dead_air(vad):
    audio = np.concatenate(
        [silence_audio(1.0), speech_audio(1.0), silence_audio(1.0)]
    )
    trimmed = vad.trim_silence(audio)
    # At least the bulk of the 2s of silence removed, speech kept
    assert 0 < len(trimmed) < len(audio) * 0.6
    assert len(trimmed) >= 16000 * 0.9


def test_trim_all_silence_returns_empty(vad):
    assert len(vad.trim_silence(silence_audio(1.0))) == 0


def test_float32_input_supported(vad):
    frame = (speech_audio(0.03)[: vad.frame_len].astype(np.float32)) / 32768.0
    assert vad.is_speech(frame) is True
