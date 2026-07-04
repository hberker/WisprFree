"""Turn logged corrections into a supervised fine-tuning dataset.

Each correction (raw transcript -> user-edited final text) becomes one
chat example teaching the model the user's phrasing, tone, and recurring
fixes. The same system prompt shape used at inference keeps train/serve
consistent.
"""

from __future__ import annotations

import json
from pathlib import Path

from wisperfree.llm.prompts import build_cleanup_system_prompt
from wisperfree.storage.corrections import Correction

TRAIN_SYSTEM_PROMPT = build_cleanup_system_prompt(tone="neutral")


def pairs_to_messages(corrections: list[Correction]) -> list[dict]:
    examples = []
    for c in corrections:
        target = c.final_text.strip()
        source = c.raw_transcript.strip()
        if not source or not target or source == target:
            continue
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": TRAIN_SYSTEM_PROMPT},
                    {"role": "user", "content": source},
                    {"role": "assistant", "content": target},
                ],
                "correction_id": c.id,
            }
        )
    return examples


def build_dataset(corrections: list[Correction], out_path: Path) -> int:
    """Write a JSONL chat dataset; returns the number of examples."""
    examples = pairs_to_messages(corrections)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return len(examples)
