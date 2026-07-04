"""SQLite database for dictionary, corrections, and tone profiles.

Everything stays on-device in a single file. sqlite3 connections are
created per-thread-safe usage with ``check_same_thread=False`` and a
lock, since the daemon serves API calls and the pipeline concurrently.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS dictionary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    term TEXT NOT NULL UNIQUE COLLATE NOCASE,
    hint TEXT,                -- optional pronunciation / usage hint
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_transcript TEXT NOT NULL,      -- Stage 1 output
    cleaned_transcript TEXT,           -- Stage 2 output (as injected)
    final_text TEXT NOT NULL,          -- what the user actually wanted
    app_name TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    used_in_training INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS tone_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    app_pattern TEXT NOT NULL UNIQUE COLLATE NOCASE,  -- substring match on app name
    tone TEXT NOT NULL,                -- casual | formal | technical | neutral
    instructions TEXT,                 -- extra user-editable prompt text
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS dismissed_suggestions (
    term TEXT PRIMARY KEY COLLATE NOCASE,
    dismissed_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',  -- running | succeeded | failed
    pairs_used INTEGER NOT NULL DEFAULT 0,
    detail TEXT
);

CREATE TABLE IF NOT EXISTS voice_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_text TEXT NOT NULL,     -- what the user read aloud (ground truth)
    wav_path TEXT NOT NULL,
    duration_s REAL NOT NULL,
    sample_rate INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS asr_training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',  -- running | succeeded | failed
    samples_used INTEGER NOT NULL DEFAULT 0,
    minutes_used REAL NOT NULL DEFAULT 0,
    wer_before REAL,               -- holdout WER of the base model
    wer_after REAL,                -- holdout WER of the tuned model
    model_dir TEXT,                -- CTranslate2 dir loadable by faster-whisper
    detail TEXT
);
"""

DEFAULT_TONE_PROFILES = [
    ("slack", "casual", None),
    ("discord", "casual", None),
    ("messages", "casual", None),  # iMessage
    ("whatsapp", "casual", None),
    ("mail", "formal", None),
    ("outlook", "formal", None),
    ("gmail", "formal", None),
    ("docs", "formal", None),
    ("word", "formal", None),
    ("terminal", "technical", None),
    ("iterm", "technical", None),
    ("code", "technical", None),  # VS Code
]


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        with self._lock, self._conn:
            self._conn.executescript(SCHEMA)
        self._seed_tone_profiles()

    def _seed_tone_profiles(self) -> None:
        with self._lock, self._conn:
            (count,) = self._conn.execute(
                "SELECT COUNT(*) FROM tone_profiles"
            ).fetchone()
            if count == 0:
                self._conn.executemany(
                    "INSERT INTO tone_profiles (app_pattern, tone, instructions) VALUES (?, ?, ?)",
                    DEFAULT_TONE_PROFILES,
                )

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock, self._conn:
            return self._conn.execute(sql, params)

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
