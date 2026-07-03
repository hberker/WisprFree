import json
import time
from pathlib import Path

from wisperfree.finetune.dataset import build_dataset, pairs_to_messages
from wisperfree.finetune.runner import FinetuneRunner


def wait_for(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def add_pairs(corrections, n):
    for i in range(n):
        corrections.add(f"uh raw sentence {i}", f"Raw sentence {i}.")


def test_dataset_export(corrections, tmp_path):
    add_pairs(corrections, 3)
    corrections.add("same", "same")  # no-op edits are filtered out
    out = tmp_path / "dataset.jsonl"
    n = build_dataset(corrections.training_pairs(), out)
    assert n == 3
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    roles = [m["role"] for m in lines[0]["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert lines[0]["messages"][1]["content"] == "uh raw sentence 0"
    assert lines[0]["messages"][2]["content"] == "Raw sentence 0."


def test_pairs_to_messages_skips_empty():
    from wisperfree.storage.corrections import Correction

    c = Correction(1, "  ", None, "final", None, "", False)
    assert pairs_to_messages([c]) == []


def make_runner(config, db, corrections, train_calls):
    def fake_train(ft_config, dataset_path: Path, out_dir: Path) -> Path:
        train_calls.append(dataset_path)
        adapter = out_dir / "adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        return adapter

    config.finetune.min_pairs = 2
    return FinetuneRunner(config, db, corrections, train_fn=fake_train)


def test_manual_trigger_trains_and_marks(config, db, corrections):
    calls = []
    runner = make_runner(config, db, corrections, calls)
    add_pairs(corrections, 3)
    assert runner.trigger() is True
    assert wait_for(lambda: not runner.running)
    assert len(calls) == 1
    assert corrections.count(untrained_only=True) == 0
    runs = runner.runs()
    assert runs[0]["status"] == "succeeded"
    assert runs[0]["pairs_used"] == 3


def test_trigger_fails_below_min_pairs(config, db, corrections):
    calls = []
    runner = make_runner(config, db, corrections, calls)
    corrections.add("one", "One.")
    runner.trigger()
    assert wait_for(lambda: not runner.running)
    assert calls == []
    assert runner.runs()[0]["status"] == "failed"
    # pairs stay untrained so a later run can use them
    assert corrections.count(untrained_only=True) == 1


def test_auto_trigger_every_n(config, db, corrections):
    calls = []
    runner = make_runner(config, db, corrections, calls)
    config.finetune.auto_schedule = "every_n"
    config.finetune.every_n_corrections = 3
    add_pairs(corrections, 2)
    assert runner.maybe_auto_trigger() is False
    add_pairs(corrections, 1)
    assert runner.maybe_auto_trigger() is True
    assert wait_for(lambda: not runner.running)
    assert len(calls) == 1


def test_auto_trigger_off_by_default(config, db, corrections):
    calls = []
    runner = make_runner(config, db, corrections, calls)
    add_pairs(corrections, 10)
    assert runner.maybe_auto_trigger() is False
    assert calls == []
