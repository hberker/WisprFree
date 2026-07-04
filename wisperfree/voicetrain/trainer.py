"""LoRA fine-tuning of Whisper on the user's recorded voice samples.

Pipeline (all local, needs ``wisperfree[finetune]`` + a GPU or patience):

  1. split samples into train / holdout (every Nth held out),
  2. LoRA fine-tune the HF Whisper checkpoint on (audio, prompt) pairs,
  3. measure holdout WER with the base vs. the tuned model — the number
     that tells the user whether this was worth it,
  4. merge the adapter and convert to CTranslate2 so faster-whisper can
     serve the result directly (set ``asr.model`` to the output dir).
"""

from __future__ import annotations

import logging
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wisperfree.config import VoiceTrainConfig
from wisperfree.storage.voice_samples import VoiceSample
from wisperfree.voicetrain.wer import corpus_wer

log = logging.getLogger(__name__)

# asr.model name -> HF checkpoint for training
HF_WHISPER = {
    "tiny": "openai/whisper-tiny",
    "tiny.en": "openai/whisper-tiny.en",
    "base": "openai/whisper-base",
    "base.en": "openai/whisper-base.en",
    "small": "openai/whisper-small",
    "small.en": "openai/whisper-small.en",
    "medium": "openai/whisper-medium",
    "large-v3": "openai/whisper-large-v3",
}


@dataclass
class TrainOutcome:
    model_dir: str  # CTranslate2 dir for faster-whisper
    wer_before: float
    wer_after: float


def resolve_base_model(config: VoiceTrainConfig, asr_model: str) -> str:
    if config.base_model != "auto":
        return config.base_model
    if asr_model in HF_WHISPER:
        return HF_WHISPER[asr_model]
    raise ValueError(
        f"cannot derive an HF checkpoint from asr.model={asr_model!r} "
        "(it may already be a tuned model path); set voicetrain.base_model"
    )


def split_samples(
    samples: list[VoiceSample], holdout_every: int
) -> tuple[list[VoiceSample], list[VoiceSample]]:
    """Deterministic train/holdout split, grouped by sentence text.

    The split is by *sentence*, not by recording: every Nth distinct
    prompt text goes entirely to the holdout. This keeps repeated takes
    of the same sentence from straddling train and holdout, which would
    leak the answer and make the WER report look better than it is.
    """
    if len(samples) < 3:
        return samples, []

    # distinct prompt texts, in first-seen order (deterministic)
    order: list[str] = []
    seen: set[str] = set()
    for s in samples:
        key = _norm_text(s.prompt_text)
        if key not in seen:
            seen.add(key)
            order.append(key)

    holdout_keys = {key for i, key in enumerate(order) if (i + 1) % holdout_every == 0}
    if not holdout_keys and order:
        holdout_keys = {order[-1]}  # force a holdout so WER is always reported

    train, holdout = [], []
    for s in samples:
        (holdout if _norm_text(s.prompt_text) in holdout_keys else train).append(s)
    # never let the holdout swallow every sample
    if not train:
        return samples, []
    return train, holdout


def _norm_text(text: str) -> str:
    return " ".join(text.lower().split())


def load_wav_16k(path: str | Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        rate = wav.getframerate()
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    audio = pcm.astype(np.float32) / 32768.0
    if rate != 16000:
        n_out = int(len(audio) * 16000 / rate)
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
    return audio


def train_whisper_lora(
    config: VoiceTrainConfig,
    asr_model: str,
    samples: list[VoiceSample],
    output_dir: Path,
) -> TrainOutcome:
    try:
        import ctranslate2
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import (
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Voice training dependencies missing; install wisperfree[finetune]"
        ) from exc

    base_model = resolve_base_model(config, asr_model)
    train_samples, holdout = split_samples(samples, config.holdout_every)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = WhisperProcessor.from_pretrained(base_model)
    model = WhisperForConditionalGeneration.from_pretrained(base_model)
    model.config.forced_decoder_ids = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def evaluate(eval_model) -> float:
        eval_model.eval()
        pairs = []
        with torch.no_grad():
            for sample in holdout:
                features = processor(
                    load_wav_16k(sample.wav_path),
                    sampling_rate=16000,
                    return_tensors="pt",
                ).input_features.to(device)
                ids = eval_model.generate(features, max_new_tokens=200)
                hypothesis = processor.batch_decode(ids, skip_special_tokens=True)[0]
                pairs.append((sample.prompt_text, hypothesis))
        return corpus_wer(pairs)

    wer_before = evaluate(model) if holdout else -1.0
    log.info("holdout WER before training: %.3f", wer_before)

    lora = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora)

    features_list = []
    for sample in train_samples:
        inputs = processor(
            load_wav_16k(sample.wav_path), sampling_rate=16000, return_tensors="pt"
        )
        labels = processor.tokenizer(
            sample.prompt_text, return_tensors="pt"
        ).input_ids[0]
        features_list.append(
            {"input_features": inputs.input_features[0], "labels": labels}
        )

    def collate(batch):
        input_features = torch.stack([f["input_features"] for f in batch])
        labels = torch.nn.utils.rnn.pad_sequence(
            [f["labels"] for f in batch],
            batch_first=True,
            padding_value=-100,
        )
        return {"input_features": input_features, "labels": labels}

    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        logging_steps=5,
        save_strategy="no",
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=[],
    )
    Seq2SeqTrainer(
        model=model, args=args, train_dataset=features_list, data_collator=collate
    ).train()

    merged = model.merge_and_unload()
    wer_after = evaluate(merged) if holdout else -1.0
    log.info("holdout WER after training: %.3f", wer_after)

    merged_dir = output_dir / "merged"
    merged.save_pretrained(str(merged_dir))
    processor.save_pretrained(str(merged_dir))

    ct2_dir = output_dir / "ct2"
    from ctranslate2.converters import TransformersConverter

    TransformersConverter(str(merged_dir)).convert(
        str(ct2_dir), quantization=config.quantization, force=True
    )
    return TrainOutcome(
        model_dir=str(ct2_dir), wer_before=wer_before, wer_after=wer_after
    )
