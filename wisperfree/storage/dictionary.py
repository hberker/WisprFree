"""Personal dictionary: custom words, names, acronyms, jargon.

Entries bias BOTH pipeline stages — they are joined into Whisper's
``initial_prompt`` and listed in the LLM cleanup system prompt.
"""

from __future__ import annotations

from dataclasses import dataclass

from wisperfree.storage.db import Database


@dataclass
class DictionaryEntry:
    id: int
    term: str
    hint: str | None
    created_at: str


class DictionaryStore:
    def __init__(self, db: Database):
        self.db = db

    def add(self, term: str, hint: str | None = None) -> DictionaryEntry:
        term = term.strip()
        if not term:
            raise ValueError("term must not be empty")
        self.db.execute(
            "INSERT INTO dictionary (term, hint) VALUES (?, ?) "
            "ON CONFLICT(term) DO UPDATE SET hint = excluded.hint",
            (term, hint),
        )
        row = self.db.query_one(
            "SELECT * FROM dictionary WHERE term = ? COLLATE NOCASE", (term,)
        )
        return DictionaryEntry(**dict(row))

    def remove(self, term_or_id: str | int) -> bool:
        if isinstance(term_or_id, int):
            cur = self.db.execute("DELETE FROM dictionary WHERE id = ?", (term_or_id,))
        else:
            cur = self.db.execute(
                "DELETE FROM dictionary WHERE term = ? COLLATE NOCASE", (term_or_id,)
            )
        return cur.rowcount > 0

    def list(self) -> list[DictionaryEntry]:
        rows = self.db.query("SELECT * FROM dictionary ORDER BY term COLLATE NOCASE")
        return [DictionaryEntry(**dict(r)) for r in rows]

    def terms(self, limit: int | None = None) -> list[str]:
        sql = "SELECT term FROM dictionary ORDER BY created_at DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [r["term"] for r in self.db.query(sql)]

    def initial_prompt(self, max_terms: int = 40) -> str:
        """Seed text for Whisper's initial_prompt to bias recognition."""
        terms = self.terms(limit=max_terms)
        if not terms:
            return ""
        # Whisper treats the initial prompt as preceding "context"; a
        # vocabulary-style sentence biases decoding toward these tokens.
        return "Glossary: " + ", ".join(terms) + "."
