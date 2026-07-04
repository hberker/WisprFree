from pathlib import Path

from wisperfree.config import AppConfig, load_config, save_config, update_config


def test_defaults():
    cfg = AppConfig()
    assert cfg.asr.backend == "faster_whisper"
    assert cfg.llm.backend == "ollama"
    assert cfg.llm.enabled is True
    assert cfg.hotkey.toggle == "<ctrl>+<space>"
    assert cfg.audio.sample_rate == 16000
    assert cfg.server.host == "127.0.0.1"  # localhost only, never exposed


def test_save_load_roundtrip(tmp_path: Path):
    cfg = AppConfig(data_dir=str(tmp_path))
    cfg.asr.model = "small"
    cfg.llm.model = "qwen2.5:7b"
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded == cfg


def test_load_missing_returns_defaults(tmp_path: Path):
    assert load_config(tmp_path / "nope.yaml") == AppConfig()


def test_update_config_nested_patch():
    cfg = AppConfig()
    updated = update_config(cfg, {"llm": {"model": "llama3.2:8b"}, "hotkey": {"toggle": "<alt>+d"}})
    assert updated.llm.model == "llama3.2:8b"
    assert updated.llm.backend == "ollama"  # untouched siblings survive
    assert updated.hotkey.toggle == "<alt>+d"
    assert cfg.llm.model == "llama3.2:3b"  # original unchanged


def test_swappable_backends_via_config():
    cfg = update_config(AppConfig(), {"asr": {"backend": "whisper_cpp"}})
    assert cfg.asr.backend == "whisper_cpp"
