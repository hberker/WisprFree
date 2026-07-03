"""Correction log: every manual edit (raw -> final) is a labelled pair.

These pairs feed two features:
  1. Auto-suggest dictionary additions when the user repeatedly corrects
     a transcribed word to the same new word.
  2. The LoRA fine-tuning dataset (see wisperfree.finetune).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from wisperfree.storage.db import Database

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]+")


@dataclass
class Correction:
    id: int
    raw_transcript: str
    cleaned_transcript: str | None
    final_text: str
    app_name: str | None
    created_at: str
    used_in_training: bool


@dataclass
class Suggestion:
    term: str
    count: int


class CorrectionStore:
    def __init__(self, db: Database, suggest_threshold: int = 3):
        self.db = db
        self.suggest_threshold = suggest_threshold

    def add(
        self,
        raw_transcript: str,
        final_text: str,
        cleaned_transcript: str | None = None,
        app_name: str | None = None,
    ) -> Correction:
        cur = self.db.execute(
            "INSERT INTO corrections (raw_transcript, cleaned_transcript, final_text, app_name) "
            "VALUES (?, ?, ?, ?)",
            (raw_transcript, cleaned_transcript, final_text, app_name),
        )
        row = self.db.query_one(
            "SELECT * FROM corrections WHERE id = ?", (cur.lastrowid,)
        )
        return self._row_to_correction(row)

    def list(self, limit: int = 200, offset: int = 0) -> list[Correction]:
        rows = self.db.query(
            "SELECT * FROM corrections ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [self._row_to_correction(r) for r in rows]

    def remove(self, correction_id: int) -> bool:
        cur = self.db.execute(
            "DELETE FROM corrections WHERE id = ?", (correction_id,)
        )
        return cur.rowcount > 0

    def count(self, untrained_only: bool = False) -> int:
        sql = "SELECT COUNT(*) FROM corrections"
        if untrained_only:
            sql += " WHERE used_in_training = 0"
        return self.db.query_one(sql)[0]

    def training_pairs(self, include_trained: bool = False) -> list[Correction]:
        sql = "SELECT * FROM corrections"
        if not include_trained:
            sql += " WHERE used_in_training = 0"
        sql += " ORDER BY id"
        return [self._row_to_correction(r) for r in self.db.query(sql)]

    def mark_trained(self, ids: list[int]) -> None:
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        self.db.execute(
            f"UPDATE corrections SET used_in_training = 1 WHERE id IN ({placeholders})",
            tuple(ids),
        )

    # ---- dictionary auto-suggestions -------------------------------------

    def suggestions(self, dictionary_terms: list[str]) -> list[Suggestion]:
        """Words the user keeps typing in that ASR never produced.

        For each correction pair, collect words present in the final text
        but absent from what the pipeline produced. Words introduced at
        least ``suggest_threshold`` times across distinct corrections are
        suggested for the personal dictionary — capitalised words and
        acronyms are the usual candidates, so plain lowercase words are
        only suggested at double the threshold.
        """
        existing = {t.lower() for t in dictionary_terms}
        dismissed = {
            r["term"].lower()
            for r in self.db.query("SELECT term FROM dismissed_suggestions")
        }
        counts: Counter[str] = Counter()
        canonical: dict[str, str] = {}
        for c in self.list(limit=1000):
            produced = c.cleaned_transcript or c.raw_transcript
            produced_words = {w.lower() for w in _WORD_RE.findall(produced)}
            final_words = _WORD_RE.findall(c.final_text)
            introduced = {
                w for w in final_words if w.lower() not in produced_words
            }
            for word in introduced:
                key = word.lower()
                counts[key] += 1
                # Prefer the cased spelling the user typed
                canonical.setdefault(key, word)
                if word[:1].isupper():
                    canonical[key] = word

        out: list[Suggestion] = []
        for key, n in counts.most_common():
            if key in existing or key in dismissed:
                continue
            word = canonical[key]
            threshold = self.suggest_threshold
            if word.islower():
                threshold *= 2
            if n >= threshold:
                out.append(Suggestion(term=word, count=n))
        return out

    def dismiss_suggestion(self, term: str) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO dismissed_suggestions (term) VALUES (?)",
            (term.strip(),),
        )

    @staticmethod
    def _row_to_correction(row) -> Correction:
        d = dict(row)
        d["used_in_training"] = bool(d["used_in_training"])
        return Correction(**d)
