# WisperFree on iOS — port plan

**Goal**: same two-stage, fully offline dictation on iPhone. **No server, no
LAN hand-off, no cloud** — both model stages run on the phone itself.

## Why the desktop code doesn't port

| Desktop piece | iOS status |
|---|---|
| Python daemon + Electron tray | Not runnable on iOS |
| Global hotkey | Not allowed |
| System-wide text injection (CGEvent/SendInput) | Forbidden by sandboxing |
| Active-app detection | Not exposed to apps |
| Ollama | Not available |

## The iOS-native equivalent

iOS's one sanctioned way to type into any app is a **custom keyboard
extension** — the same mechanism Wispr Flow uses on iPhone. Architecture:

```
Main app (SwiftUI)                         Keyboard extension
┌─────────────────────────────┐            ┌──────────────────────┐
│ mic capture (AVAudioEngine) │            │ mic key → opens/     │
│ Stage 1: WhisperKit (CoreML)│  App Group │   signals main app   │
│ Stage 2: on-device LLM      │ ─────────► │ inserts final text   │
│ dictionary + corrections    │  (shared   │   via UITextDocument │
│   (SQLite/GRDB)             │   SQLite)  │   Proxy              │
└─────────────────────────────┘            └──────────────────────┘
```

- **Stage 1 — ASR**: [WhisperKit](https://github.com/argmaxinc/WhisperKit)
  (Whisper → Core ML, Neural Engine). `base`/`small` give sub-second
  transcription on A16 and newer. Personal-dictionary biasing via the same
  `initial_prompt` mechanism (WhisperKit supports prompt tokens).
- **Stage 2 — cleanup LLM**, two options:
  1. **Apple Foundation Models framework** (iOS 26+): the system's built-in
     ~3B on-device model. Zero download, zero app memory cost, and on-device
     by OS guarantee. Reuse `wisperfree/llm/prompts.py`'s system prompt
     verbatim.
  2. **llama.cpp (Swift package)** with a bundled quantized model
     (Llama 3.2 1B/3B Q4): works back to iOS 16-ish on ≥6 GB-RAM devices,
     model adds ~0.7–2 GB to the app.
- **Text insertion**: the keyboard extension's `textDocumentProxy.insertText`
  — the legitimate injection path on iOS.
- **Trigger**: mic key on the WisperFree keyboard, the Action Button
  (iPhone 15 Pro+), or an App Intent / Shortcut.

## Hard constraints (and how the design absorbs them)

- **Keyboard extensions can't run the models**: extensions get ~60–70 MB of
  memory and restricted microphone access. Hence: recording + both stages run
  in the **main app**; the keyboard reads the result from the shared App
  Group container and inserts it. The keyboard's mic key deep-links/hands off
  to the main app's recording overlay.
- **No per-app tone detection**: iOS never tells a keyboard which app it's
  in beyond coarse `UITextContentType` hints. Tone profiles become a manual
  toggle in the keyboard (casual/formal/technical chips above the key rows).
- **No on-phone LoRA training**: corrections are still logged to the local
  SQLite store (same schema as desktop), but fine-tuning remains a desktop
  feature. Adapters are not synced anywhere by default.

## What carries over from this repo

- The two-stage design, cleanup system prompt, and tone taxonomy
  (`wisperfree/llm/prompts.py`) — port as-is to Swift strings.
- The SQLite schema for dictionary / corrections / tone profiles
  (`wisperfree/storage/db.py`) — same tables via GRDB.
- The VAD chunking parameters (`wisperfree/audio/`) — Apple's
  `SFVoiceAnalytics`/`AVAudioEngine` + the same energy-gate fallback.

## Deploying to your own iPhone

1. Requires a Mac with Xcode (iOS apps cannot be built from Linux/Windows).
2. Free Apple ID: sideload via Xcode, re-sign every 7 days.
   $99/yr developer account: TestFlight installs that last 90 days, 1-year
   signing.
3. Capabilities needed: App Groups (app ↔ keyboard sharing), microphone
   usage description. **No network entitlement at all** — the app can ship
   with no networking code, which is the strongest possible "data never
   leaves the device" guarantee.
