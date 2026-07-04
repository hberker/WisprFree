"""Voice-training samples: (spoken audio, ground-truth prompt text) pairs.

WAV files live under ``<data_dir>/voice_samples/`` — local only, and
deleting a sample removes its audio file. These recordings are the only
audio WisperFree ever writes to disk, and only when the user records
them on purpose in the Voice Training tab.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from pathlib import Path

from wisperfree.storage.db import Database


@dataclass
class VoiceSample:
    id: int
    prompt_text: str
    wav_path: str
    duration_s: float
    sample_rate: int
    created_at: str


class VoiceSampleStore:
    def __init__(self, db: Database, samples_dir: str | Path):
        self.db = db
        self.samples_dir = Path(samples_dir)

    def add(self, prompt_text: str, wav_bytes: bytes) -> VoiceSample:
        prompt_text = prompt_text.strip()
        if not prompt_text:
            raise ValueError("prompt_text must not be empty")
        duration_s, sample_rate = _validate_wav(wav_bytes)
        if duration_s < 0.5:
            raise ValueError("recording too short (under 0.5s)")

        self.samples_dir.mkdir(parents=True, exist_ok=True)
        cur = self.db.execute(
            "INSERT INTO voice_samples (prompt_text, wav_path, duration_s, sample_rate) "
            "VALUES (?, ?, ?, ?)",
            (prompt_text, "", duration_s, sample_rate),
        )
        sample_id = cur.lastrowid
        wav_path = self.samples_dir / f"sample-{sample_id:05d}.wav"
        wav_path.write_bytes(wav_bytes)
        self.db.execute(
            "UPDATE voice_samples SET wav_path = ? WHERE id = ?",
            (str(wav_path), sample_id),
        )
        row = self.db.query_one(
            "SELECT * FROM voice_samples WHERE id = ?", (sample_id,)
        )
        return VoiceSample(**dict(row))

    def list(self) -> list[VoiceSample]:
        rows = self.db.query("SELECT * FROM voice_samples ORDER BY id")
        return [VoiceSample(**dict(r)) for r in rows]

    def remove(self, sample_id: int) -> bool:
        row = self.db.query_one(
            "SELECT wav_path FROM voice_samples WHERE id = ?", (sample_id,)
        )
        if row is None:
            return False
        self.db.execute("DELETE FROM voice_samples WHERE id = ?", (sample_id,))
        path = Path(row["wav_path"])
        if path.exists():
            path.unlink()
        return True

    def total_minutes(self) -> float:
        row = self.db.query_one("SELECT COALESCE(SUM(duration_s), 0) FROM voice_samples")
        return row[0] / 60.0

    def count(self) -> int:
        return self.db.query_one("SELECT COUNT(*) FROM voice_samples")[0]


def _validate_wav(wav_bytes: bytes) -> tuple[float, int]:
    """Parse WAV header; returns (duration_s, sample_rate) or raises."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            if wav.getsampwidth() != 2 or wav.getnchannels() != 1:
                raise ValueError("expected 16-bit mono PCM WAV")
            if rate < 8000:
                raise ValueError(f"sample rate too low: {rate}")
            return frames / rate, rate
    except wave.Error as exc:
        raise ValueError(f"not a valid WAV file: {exc}") from exc
