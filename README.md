# 🎙 WisperFree

**Fully offline, privacy-first AI voice dictation.** A Wispr-Flow-style two-stage
pipeline running entirely on local models via [whisper](https://github.com/SYSTRAN/faster-whisper)
and [Ollama](https://ollama.com). No cloud calls. No accounts. No data leaves your device.

## How it works

WisperFree is not plain real-time ASR — it produces *what you meant*, not *what you said*:

```
global hotkey (Ctrl+Space)
   └─► mic capture ─► VAD (trim silence, chunk at pauses)
         └─► Stage 1 · Whisper transcription
         │     · faster-whisper (CTranslate2) or whisper.cpp — swappable
         │     · initial_prompt seeded with your personal dictionary
         └─► Stage 2 · Local LLM cleanup via Ollama
         │     · removes filler (um/uh/like), fixes grammar & punctuation
         │     · resolves self-corrections: "meet Tuesday, wait, Wednesday" → "meet Wednesday"
         │     · formats lists/paragraphs, applies spoken commands ("new line")
         │     · matches tone to the active app (casual → Slack, formal → email)
         └─► native text injection at your cursor
               · macOS CGEvent · Windows SendInput · Linux xdotool/wtype
```

If Ollama is unreachable or Stage 2 is disabled, the raw transcript is injected
instead — dictation never silently drops.

## Features

- **Personal dictionary** (SQLite): names, jargon, acronyms bias *both* stages —
  Whisper's `initial_prompt` and the LLM system prompt. The app auto-suggests
  additions when you repeatedly correct the same word.
- **Context awareness**: active app/window detection (AppKit / Win32 / X11 /
  Hyprland / Sway) drives per-app tone profiles (casual / formal / technical),
  fully user-editable.
- **Trainability**: every manual correction (raw → your edit) is logged locally.
  A LoRA/PEFT fine-tuning job trains the cleanup model on your corrections —
  manually ("Train on my corrections") or on a schedule (weekly / every N).
- **Swappable backends via config**: `asr.backend: faster_whisper | whisper_cpp`,
  Whisper model tiny→large-v3 (quantized, CUDA/Metal-accelerated),
  any Ollama model (`llama3.2:3b`, `qwen2.5:7b`, …).
- **Menu-bar app** (Electron): toggle, settings, dictionary editor, tone
  profiles, correction history, fine-tune trigger.

## Install

```bash
# 1. Backend (Python 3.10+)
pip install -e ".[asr,audio,dev]"        # + [macos] on macOS, + [finetune] for LoRA

# 2. Local LLM
ollama pull llama3.2:3b                  # or qwen2.5:7b etc.

# 3. Linux text injection (pick one)
sudo apt install xdotool                 # X11
sudo apt install wtype                   # Wayland

# 4. Run the daemon (global hotkey + localhost API on 127.0.0.1:8765)
wisperfree

# 5. Tray app + settings UI
cd frontend && npm install && npm start
```

macOS needs Accessibility + Microphone permissions (System Settings → Privacy & Security).

## Usage

Press **Ctrl+Space** (configurable), speak, press again. Text appears at your
cursor in whatever app has focus. Latency target is ≤1–1.5 s end-to-end on an
M-series Mac or mid-range GPU with `base`/`small` Whisper + a 3B model kept
warm (`keep_alive`).

## Configuration

`~/.config/wisperfree/config.yaml` (editable in the settings UI):

```yaml
asr:
  backend: faster_whisper   # or whisper_cpp
  model: base.en            # tiny/base/small/medium/large-v3
  device: auto              # auto/cpu/cuda
llm:
  model: llama3.2:3b
  enabled: true             # false = inject raw transcript (Phase-1 mode)
hotkey:
  toggle: <ctrl>+<space>
finetune:
  auto_schedule: off        # off | weekly | every_n
  every_n_corrections: 50
```

## Fine-tuning on your corrections

Corrections you make in the History tab accumulate in local SQLite. Training
exports them as chat-format JSONL, runs a LoRA fine-tune (PEFT) of the base
model, and emits an Ollama `Modelfile`:

```bash
pip install -e ".[finetune]"
# then click "Train on my corrections", or POST /api/finetune/trigger
ollama create wisperfree-tuned -f ~/.local/share/wisperfree/finetune/<run>/Modelfile
# point llm.model at wisperfree-tuned in settings
```

## Project layout

```
wisperfree/
├── audio/        # mic capture + VAD chunking
├── asr/          # Stage 1 backends: faster-whisper, whisper.cpp
├── llm/          # Stage 2 backends: Ollama + cleanup prompt builder
├── inject/       # native text injection per OS
├── context/      # active app/window detection per OS
├── storage/      # SQLite: dictionary, corrections, tone profiles
├── finetune/     # dataset export + LoRA trainer + job runner
├── pipeline.py   # two-stage orchestration
├── daemon.py     # wiring + lifecycle
└── api/          # localhost FastAPI for the tray app
frontend/         # Electron menu-bar app + settings UI
tests/            # 51 unit tests (fakes for ASR/LLM/injection)
```

## Development

```bash
pip install -e ".[dev]"
python -m pytest
```

## Privacy

- All inference is local (whisper + Ollama on localhost).
- The API binds `127.0.0.1` only.
- Audio is processed in memory and never written to disk.
- Dictionary, corrections, and fine-tune artifacts live in
  `~/.local/share/wisperfree/` — delete the folder, everything is gone.
