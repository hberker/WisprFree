"""Per-app tone profiles (casual / formal / technical / neutral).

The context detector reports the active app name; the first profile
whose ``app_pattern`` is a case-insensitive substring of that name wins.
"""

from __future__ import annotations

from dataclasses import dataclass

from wisperfree.storage.db import Database

VALID_TONES = {"casual", "formal", "technical", "neutral"}


@dataclass
class ToneProfile:
    id: int
    app_pattern: str
    tone: str
    instructions: str | None
    created_at: str


class ToneProfileStore:
    def __init__(self, db: Database):
        self.db = db

    def upsert(
        self, app_pattern: str, tone: str, instructions: str | None = None
    ) -> ToneProfile:
        app_pattern = app_pattern.strip()
        if not app_pattern:
            raise ValueError("app_pattern must not be empty")
        if tone not in VALID_TONES:
            raise ValueError(f"tone must be one of {sorted(VALID_TONES)}")
        self.db.execute(
            "INSERT INTO tone_profiles (app_pattern, tone, instructions) VALUES (?, ?, ?) "
            "ON CONFLICT(app_pattern) DO UPDATE SET tone = excluded.tone, "
            "instructions = excluded.instructions",
            (app_pattern, tone, instructions),
        )
        row = self.db.query_one(
            "SELECT * FROM tone_profiles WHERE app_pattern = ? COLLATE NOCASE",
            (app_pattern,),
        )
        return ToneProfile(**dict(row))

    def remove(self, profile_id: int) -> bool:
        cur = self.db.execute("DELETE FROM tone_profiles WHERE id = ?", (profile_id,))
        return cur.rowcount > 0

    def list(self) -> list[ToneProfile]:
        rows = self.db.query(
            "SELECT * FROM tone_profiles ORDER BY app_pattern COLLATE NOCASE"
        )
        return [ToneProfile(**dict(r)) for r in rows]

    def match(self, app_name: str | None, default_tone: str = "neutral") -> ToneProfile:
        """Find the profile for an app name (longest matching pattern wins)."""
        if app_name:
            lowered = app_name.lower()
            best: ToneProfile | None = None
            for profile in self.list():
                if profile.app_pattern.lower() in lowered:
                    if best is None or len(profile.app_pattern) > len(best.app_pattern):
                        best = profile
            if best:
                return best
        return ToneProfile(
            id=0,
            app_pattern="*",
            tone=default_tone,
            instructions=None,
            created_at="",
        )
