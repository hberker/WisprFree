"""Background job runner for voice training (same pattern as the LLM
FinetuneRunner): manual trigger, daemon thread, run history persisted to
``asr_training_runs`` including the before/after WER report."""

from __future__ import annotations

import datetime as dt
import logging
import threading
from pathlib import Path
from typing import Callable, Optional

from wisperfree.config import AppConfig
from wisperfree.storage import Database, VoiceSampleStore
from wisperfree.voicetrain.trainer import TrainOutcome

log = logging.getLogger(__name__)

TrainFn = Callable[..., TrainOutcome]  # injectable for tests


class VoiceTrainRunner:
    def __init__(
        self,
        config: AppConfig,
        db: Database,
        samples: VoiceSampleStore,
        train_fn: Optional[TrainFn] = None,
    ):
        self.config = config
        self.db = db
        self.samples = samples
        if train_fn is None:
            from wisperfree.voicetrain.trainer import train_whisper_lora

            train_fn = train_whisper_lora
        self.train_fn = train_fn
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def runs(self, limit: int = 20) -> list[dict]:
        rows = self.db.query(
            "SELECT * FROM asr_training_runs ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [dict(r) for r in rows]

    def tuned_models(self) -> list[str]:
        """CT2 dirs from succeeded runs that still exist on disk."""
        rows = self.db.query(
            "SELECT model_dir FROM asr_training_runs "
            "WHERE status = 'succeeded' AND model_dir IS NOT NULL ORDER BY id DESC"
        )
        return [r["model_dir"] for r in rows if Path(r["model_dir"]).is_dir()]

    def trigger(self) -> bool:
        """Manual "Train on my voice". Returns False if already running."""
        with self._lock:
            if self.running:
                return False
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            return True

    def _run(self) -> None:
        samples = self.samples.list()
        minutes = self.samples.total_minutes()
        run_id = self.db.execute(
            "INSERT INTO asr_training_runs (status, samples_used, minutes_used) "
            "VALUES ('running', ?, ?)",
            (len(samples), round(minutes, 2)),
        ).lastrowid
        try:
            required = self.config.voicetrain.min_minutes
            if minutes < required:
                raise RuntimeError(
                    f"need at least {required:.0f} minutes of recorded audio, "
                    f"have {minutes:.1f}"
                )
            out_dir = self.config.asr_models_dir() / dt.datetime.utcnow().strftime(
                "%Y%m%d-%H%M%S"
            )
            outcome = self.train_fn(
                self.config.voicetrain, self.config.asr.model, samples, out_dir
            )
            self.db.execute(
                "UPDATE asr_training_runs SET status = 'succeeded', "
                "wer_before = ?, wer_after = ?, model_dir = ?, "
                "detail = ?, finished_at = datetime('now') WHERE id = ?",
                (
                    outcome.wer_before,
                    outcome.wer_after,
                    outcome.model_dir,
                    f"set asr.model to {outcome.model_dir} to use it",
                    run_id,
                ),
            )
            log.info(
                "voice training succeeded: WER %.3f -> %.3f, model %s",
                outcome.wer_before, outcome.wer_after, outcome.model_dir,
            )
        except Exception as exc:
            self.db.execute(
                "UPDATE asr_training_runs SET status = 'failed', detail = ?, "
                "finished_at = datetime('now') WHERE id = ?",
                (str(exc), run_id),
            )
            log.exception("voice training failed")
