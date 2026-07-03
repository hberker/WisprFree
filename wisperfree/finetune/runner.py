"""Background fine-tuning job runner.

Exposes the manual "Train on my corrections" trigger and the optional
automatic schedule (weekly, or every N new corrections). Runs in a
daemon thread so dictation stays responsive; run status is persisted to
the ``training_runs`` table for the UI's history view.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
from pathlib import Path
from typing import Callable, Optional

from wisperfree.config import AppConfig
from wisperfree.finetune.dataset import build_dataset
from wisperfree.storage import CorrectionStore, Database

log = logging.getLogger(__name__)

TrainFn = Callable[..., Path]  # injectable for tests


class FinetuneRunner:
    def __init__(
        self,
        config: AppConfig,
        db: Database,
        corrections: CorrectionStore,
        train_fn: Optional[TrainFn] = None,
    ):
        self.config = config
        self.db = db
        self.corrections = corrections
        if train_fn is None:
            from wisperfree.finetune.lora import train_lora

            train_fn = train_lora
        self.train_fn = train_fn
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # ---- status -----------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def runs(self, limit: int = 20) -> list[dict]:
        rows = self.db.query(
            "SELECT * FROM training_runs ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [dict(r) for r in rows]

    # ---- triggers ---------------------------------------------------------

    def trigger(self) -> bool:
        """Manual "Train on my corrections". Returns False if already running."""
        with self._lock:
            if self.running:
                return False
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            return True

    def maybe_auto_trigger(self) -> bool:
        """Called after each new correction / periodically by the daemon."""
        mode = self.config.finetune.auto_schedule
        if mode == "off" or self.running:
            return False
        if mode == "every_n":
            if (
                self.corrections.count(untrained_only=True)
                >= self.config.finetune.every_n_corrections
            ):
                return self.trigger()
        elif mode == "weekly":
            last = self.db.query_one(
                "SELECT started_at FROM training_runs "
                "WHERE status = 'succeeded' ORDER BY id DESC LIMIT 1"
            )
            if last is None:
                return self.trigger()
            started = dt.datetime.fromisoformat(last["started_at"])
            if dt.datetime.utcnow() - started >= dt.timedelta(days=7):
                return self.trigger()
        return False

    # ---- the job ----------------------------------------------------------

    def _run(self) -> None:
        pairs = self.corrections.training_pairs()
        run_id = self.db.execute(
            "INSERT INTO training_runs (status, pairs_used) VALUES ('running', ?)",
            (len(pairs),),
        ).lastrowid
        try:
            if len(pairs) < self.config.finetune.min_pairs:
                raise RuntimeError(
                    f"need at least {self.config.finetune.min_pairs} correction "
                    f"pairs to train, have {len(pairs)}"
                )
            out_dir = self.config.finetune_dir() / dt.datetime.utcnow().strftime(
                "%Y%m%d-%H%M%S"
            )
            dataset_path = out_dir / "dataset.jsonl"
            n = build_dataset(pairs, dataset_path)
            if n < self.config.finetune.min_pairs:
                raise RuntimeError(
                    f"only {n} usable pairs after filtering; need "
                    f"{self.config.finetune.min_pairs}"
                )
            adapter_dir = self.train_fn(self.config.finetune, dataset_path, out_dir)
            self.corrections.mark_trained([p.id for p in pairs])
            self._finish(run_id, "succeeded", f"adapter: {adapter_dir}")
            log.info("fine-tune succeeded: %s", adapter_dir)
        except Exception as exc:
            self._finish(run_id, "failed", str(exc))
            log.exception("fine-tune failed")

    def _finish(self, run_id: int, status: str, detail: str) -> None:
        self.db.execute(
            "UPDATE training_runs SET status = ?, detail = ?, "
            "finished_at = datetime('now') WHERE id = ?",
            (status, detail, run_id),
        )
