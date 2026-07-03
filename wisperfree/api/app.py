"""Local HTTP API consumed by the tray/settings frontend.

Localhost only. Endpoints cover: dictation toggle + status, config,
dictionary CRUD + suggestions, tone profiles, corrections (history +
logging manual edits), Stage-2 text preview, and fine-tuning control.
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from wisperfree import __version__
from wisperfree.daemon import Daemon


class DictEntryIn(BaseModel):
    term: str
    hint: Optional[str] = None


class ToneProfileIn(BaseModel):
    app_pattern: str
    tone: str
    instructions: Optional[str] = None


class CorrectionIn(BaseModel):
    raw_transcript: str
    final_text: str
    cleaned_transcript: Optional[str] = None
    app_name: Optional[str] = None


class CleanupIn(BaseModel):
    text: str
    app_name: Optional[str] = None


class ConfigPatch(BaseModel):
    patch: dict


def create_app(daemon: Daemon) -> FastAPI:
    app = FastAPI(title="WisperFree", version=__version__)

    @app.get("/api/status")
    def status():
        return {
            "version": __version__,
            "dictating": daemon.dictating,
            "asr_backend": daemon.config.asr.backend,
            "asr_model": daemon.config.asr.model,
            "llm_backend": daemon.config.llm.backend,
            "llm_model": daemon.config.llm.model,
            "llm_enabled": daemon.config.llm.enabled,
            "hotkey": daemon.config.hotkey.toggle,
            "training_running": daemon.finetune.running,
        }

    @app.post("/api/dictation/toggle")
    def toggle():
        try:
            return {"dictating": daemon.toggle_dictation()}
        except RuntimeError as exc:  # e.g. sounddevice missing
            raise HTTPException(status_code=503, detail=str(exc))

    @app.get("/api/history")
    def history():
        return [
            {
                "raw_text": e.raw_text,
                "final_text": e.final_text,
                "app_name": e.app_name,
                "tone": e.tone,
                "asr_latency_s": round(e.asr_latency_s, 3),
                "llm_latency_s": round(e.llm_latency_s, 3),
                "llm_used": e.llm_used,
                "timestamp": e.timestamp,
            }
            for e in reversed(daemon.pipeline.history)
        ]

    # ---- Stage-2 preview (also used headless / in tests) -------------------

    @app.post("/api/cleanup")
    def cleanup(body: CleanupIn):
        event = daemon.pipeline.process_text(body.text, app_name=body.app_name)
        return {
            "raw_text": event.raw_text,
            "final_text": event.final_text,
            "tone": event.tone,
            "llm_used": event.llm_used,
            "llm_latency_s": round(event.llm_latency_s, 3),
        }

    # ---- config -------------------------------------------------------------

    @app.get("/api/config")
    def get_config():
        return daemon.config.model_dump()

    @app.put("/api/config")
    def put_config(body: ConfigPatch):
        try:
            return daemon.apply_config_patch(body.patch).model_dump()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    # ---- dictionary ----------------------------------------------------------

    @app.get("/api/dictionary")
    def list_dictionary():
        return [vars(e) for e in daemon.dictionary.list()]

    @app.post("/api/dictionary")
    def add_dictionary(body: DictEntryIn):
        try:
            return vars(daemon.dictionary.add(body.term, body.hint))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.delete("/api/dictionary/{entry_id}")
    def remove_dictionary(entry_id: int):
        if not daemon.dictionary.remove(entry_id):
            raise HTTPException(status_code=404, detail="entry not found")
        return {"ok": True}

    @app.get("/api/dictionary/suggestions")
    def suggestions():
        return [
            vars(s)
            for s in daemon.corrections.suggestions(daemon.dictionary.terms())
        ]

    @app.post("/api/dictionary/suggestions/{term}/dismiss")
    def dismiss_suggestion(term: str):
        daemon.corrections.dismiss_suggestion(term)
        return {"ok": True}

    # ---- tone profiles --------------------------------------------------------

    @app.get("/api/tone-profiles")
    def list_tone_profiles():
        return [vars(p) for p in daemon.tone_profiles.list()]

    @app.post("/api/tone-profiles")
    def upsert_tone_profile(body: ToneProfileIn):
        try:
            return vars(
                daemon.tone_profiles.upsert(
                    body.app_pattern, body.tone, body.instructions
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.delete("/api/tone-profiles/{profile_id}")
    def remove_tone_profile(profile_id: int):
        if not daemon.tone_profiles.remove(profile_id):
            raise HTTPException(status_code=404, detail="profile not found")
        return {"ok": True}

    # ---- corrections -----------------------------------------------------------

    @app.get("/api/corrections")
    def list_corrections(limit: int = 100, offset: int = 0):
        return [vars(c) for c in daemon.corrections.list(limit, offset)]

    @app.post("/api/corrections")
    def add_correction(body: CorrectionIn):
        correction = daemon.corrections.add(
            raw_transcript=body.raw_transcript,
            final_text=body.final_text,
            cleaned_transcript=body.cleaned_transcript,
            app_name=body.app_name,
        )
        daemon.finetune.maybe_auto_trigger()
        return vars(correction)

    @app.delete("/api/corrections/{correction_id}")
    def remove_correction(correction_id: int):
        if not daemon.corrections.remove(correction_id):
            raise HTTPException(status_code=404, detail="correction not found")
        return {"ok": True}

    # ---- fine-tuning -------------------------------------------------------------

    @app.post("/api/finetune/trigger")
    def trigger_finetune():
        started = daemon.finetune.trigger()
        if not started:
            raise HTTPException(status_code=409, detail="training already running")
        return {"started": True}

    @app.get("/api/finetune/runs")
    def finetune_runs():
        return daemon.finetune.runs()

    return app
