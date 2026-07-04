"""WisperFree — fully offline, privacy-first AI voice dictation.

Two-stage pipeline: local Whisper transcription (Stage 1) followed by a
local LLM cleanup/rewrite pass via Ollama (Stage 2). No cloud calls; no
data leaves the device.
"""

__version__ = "0.1.0"
