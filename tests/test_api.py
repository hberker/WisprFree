import pytest
from fastapi.testclient import TestClient

from tests.conftest import FakeASR, FakeInjector, FakeLLM, FakeContextDetector
from wisperfree.api import create_app
from wisperfree.daemon import Daemon


@pytest.fixture
def client(config, tmp_path, monkeypatch):
    # Config lives in tmp so apply_config_patch never touches the real home dir
    monkeypatch.setenv("WISPERFREE_CONFIG_DIR", str(tmp_path / "cfg"))
    daemon = Daemon(config)
    # Swap hardware/network-facing pieces for fakes
    daemon.pipeline.asr = FakeASR()
    daemon.pipeline.llm = FakeLLM(reply="Cleaned text.")
    daemon.pipeline.injector = FakeInjector(config.injection)
    daemon.pipeline.context_detector = FakeContextDetector()
    yield TestClient(create_app(daemon))
    daemon.db.close()


def test_status(client):
    body = client.get("/api/status").json()
    assert body["dictating"] is False
    assert body["asr_backend"] == "faster_whisper"
    assert body["llm_model"] == "llama3.2:3b"
    assert body["hotkey"] == "<ctrl>+<space>"


def test_cleanup_endpoint(client):
    body = client.post(
        "/api/cleanup", json={"text": "um hello there", "app_name": "Slack"}
    ).json()
    assert body["final_text"] == "Cleaned text."
    assert body["tone"] == "casual"
    assert body["llm_used"] is True


def test_dictionary_crud_and_suggestions(client):
    assert client.get("/api/dictionary").json() == []
    entry = client.post(
        "/api/dictionary", json={"term": "Kubernetes", "hint": "k8s"}
    ).json()
    assert entry["term"] == "Kubernetes"
    assert len(client.get("/api/dictionary").json()) == 1

    # Repeated corrections generate a suggestion, dismissal hides it
    for i in range(3):
        client.post(
            "/api/corrections",
            json={"raw_transcript": f"use amiga {i}", "final_text": f"Use Amical {i}"},
        )
    suggestions = client.get("/api/dictionary/suggestions").json()
    assert suggestions == [{"term": "Amical", "count": 3}]
    client.post("/api/dictionary/suggestions/Amical/dismiss")
    assert client.get("/api/dictionary/suggestions").json() == []

    assert client.delete(f"/api/dictionary/{entry['id']}").json() == {"ok": True}
    assert client.delete("/api/dictionary/9999").status_code == 404


def test_dictionary_validation(client):
    assert client.post("/api/dictionary", json={"term": "  "}).status_code == 422


def test_tone_profiles(client):
    profiles = client.get("/api/tone-profiles").json()
    assert any(p["app_pattern"] == "slack" for p in profiles)
    created = client.post(
        "/api/tone-profiles",
        json={"app_pattern": "obsidian", "tone": "technical"},
    ).json()
    assert created["tone"] == "technical"
    assert (
        client.post(
            "/api/tone-profiles", json={"app_pattern": "x", "tone": "shouty"}
        ).status_code
        == 422
    )
    assert client.delete(f"/api/tone-profiles/{created['id']}").json() == {"ok": True}


def test_corrections_endpoints(client):
    created = client.post(
        "/api/corrections",
        json={
            "raw_transcript": "helo wrld",
            "final_text": "Hello world",
            "app_name": "Mail",
        },
    ).json()
    listed = client.get("/api/corrections").json()
    assert len(listed) == 1
    assert listed[0]["final_text"] == "Hello world"
    assert client.delete(f"/api/corrections/{created['id']}").json() == {"ok": True}


def test_config_get_and_patch(client):
    cfg = client.get("/api/config").json()
    assert cfg["llm"]["model"] == "llama3.2:3b"
    updated = client.put(
        "/api/config", json={"patch": {"asr": {"model": "small"}}}
    ).json()
    assert updated["asr"]["model"] == "small"
    assert client.get("/api/config").json()["asr"]["model"] == "small"


def test_finetune_endpoints(client):
    runs = client.get("/api/finetune/runs").json()
    assert runs == []
    # Not enough pairs -> the run starts then records failure; trigger returns 200
    response = client.post("/api/finetune/trigger")
    assert response.status_code == 200
