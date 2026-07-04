"""Configuration for WisperFree.

Everything is driven by a single YAML file (default:
``~/.config/wisperfree/config.yaml``). ASR and LLM backends are selected
by name here so they stay swappable without code changes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


def default_config_dir() -> Path:
    override = os.environ.get("WISPERFREE_CONFIG_DIR")
    if override:
        return Path(override)
    return Path.home() / ".config" / "wisperfree"


def default_data_dir() -> Path:
    override = os.environ.get("WISPERFREE_DATA_DIR")
    if override:
        return Path(override)
    return Path.home() / ".local" / "share" / "wisperfree"


class AudioConfig(BaseModel):
    sample_rate: int = 16000
    channels: int = 1
    input_device: Optional[str] = None  # None = system default microphone
    # VAD tuning
    vad_backend: str = "auto"  # "webrtc" | "energy" | "auto"
    vad_aggressiveness: int = 2  # webrtcvad 0-3
    frame_ms: int = 30
    # End a chunk after this much trailing silence
    end_silence_ms: int = 600
    # Keep this much audio before speech onset so words aren't clipped
    pre_roll_ms: int = 240
    max_chunk_s: float = 30.0


class ASRConfig(BaseModel):
    backend: str = "faster_whisper"  # "faster_whisper" | "whisper_cpp"
    model: str = "base.en"  # tiny/base/small/medium/large-v3 (+ .en variants)
    device: str = "auto"  # auto | cpu | cuda
    compute_type: str = "auto"  # auto | int8 | int8_float16 | float16 ...
    language: Optional[str] = None  # None = autodetect
    beam_size: int = 1  # greedy decoding keeps latency low
    # whisper.cpp specifics (used when backend == "whisper_cpp")
    whisper_cpp_binary: str = "whisper-cli"
    whisper_cpp_model_path: Optional[str] = None
    # How many dictionary terms to seed into Whisper's initial_prompt
    max_prompt_terms: int = 40


class LLMConfig(BaseModel):
    backend: str = "ollama"
    model: str = "llama3.2:3b"  # user-selectable; qwen2.5 etc. also work
    host: str = "http://127.0.0.1:11434"
    enabled: bool = True  # False = Phase-1 behaviour (inject raw transcript)
    timeout_s: float = 30.0
    temperature: float = 0.1
    max_tokens: int = 1024
    keep_alive: str = "10m"  # keep the model warm between dictations


class HotkeyConfig(BaseModel):
    # pynput syntax. Fn is not observable cross-platform, so the portable
    # default is Ctrl+Space; users can rebind in settings.
    toggle: str = "<ctrl>+<space>"


class InjectionConfig(BaseModel):
    # "auto" picks the native mechanism for the current OS:
    # AppKit/CGEvent on macOS, SendInput on Windows, xdotool/wtype on Linux.
    method: str = "auto"
    inter_key_delay_ms: int = 0
    trailing_space: bool = True


class ContextConfig(BaseModel):
    enabled: bool = True
    default_tone: str = "neutral"


class FinetuneConfig(BaseModel):
    # Base HF model used for LoRA; should correspond to the Ollama model.
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    output_dir: Optional[str] = None  # default: <data_dir>/finetune
    auto_schedule: str = "off"  # "off" | "weekly" | "every_n"
    every_n_corrections: int = 50
    min_pairs: int = 8  # refuse to train on fewer labelled pairs
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 2
    max_seq_len: int = 512
    # After training, register the adapter with Ollama under this model name
    ollama_output_model: str = "wisperfree-tuned"


class VoiceTrainConfig(BaseModel):
    """Fine-tuning Whisper itself on the user's recorded voice samples."""

    # "auto" derives the HF model id from asr.model (base.en -> openai/whisper-base.en)
    base_model: str = "auto"
    min_minutes: float = 5.0  # refuse to train on less recorded audio
    holdout_every: int = 10  # every Nth sample held out for the WER report
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    epochs: int = 3
    learning_rate: float = 5e-4
    batch_size: int = 2
    quantization: str = "int8"  # CTranslate2 export quantization


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765


class AppConfig(BaseModel):
    audio: AudioConfig = Field(default_factory=AudioConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    hotkey: HotkeyConfig = Field(default_factory=HotkeyConfig)
    injection: InjectionConfig = Field(default_factory=InjectionConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    finetune: FinetuneConfig = Field(default_factory=FinetuneConfig)
    voicetrain: VoiceTrainConfig = Field(default_factory=VoiceTrainConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    data_dir: str = Field(default_factory=lambda: str(default_data_dir()))

    @property
    def db_path(self) -> Path:
        return Path(self.data_dir) / "wisperfree.db"

    def finetune_dir(self) -> Path:
        return Path(self.finetune.output_dir or Path(self.data_dir) / "finetune")

    def voice_samples_dir(self) -> Path:
        return Path(self.data_dir) / "voice_samples"

    def asr_models_dir(self) -> Path:
        return Path(self.data_dir) / "asr_models"


def config_path() -> Path:
    return default_config_dir() / "config.yaml"


def load_config(path: Optional[Path] = None) -> AppConfig:
    path = path or config_path()
    if path.exists():
        raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        return AppConfig.model_validate(raw)
    return AppConfig()


def save_config(config: AppConfig, path: Optional[Path] = None) -> Path:
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config.model_dump(), sort_keys=False))
    return path


def update_config(config: AppConfig, patch: dict[str, Any]) -> AppConfig:
    """Apply a partial (possibly nested) update and return a new config."""
    merged = _deep_merge(config.model_dump(), patch)
    return AppConfig.model_validate(merged)


def _deep_merge(base: dict, patch: dict) -> dict:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out
