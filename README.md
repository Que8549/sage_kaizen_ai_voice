# Sage Kaizen AI Voice

Voice pipeline subsystem for the Sage Kaizen local-first AI assistant.

Handles all speech-to-text input and text-to-speech output.  
Integrates with the main Sage Kaizen system via ZeroMQ, or runs standalone.

## Models

| Role | Model | Location |
|------|-------|----------|
| STT  | faster-whisper + distil-large-v3 INT8 | `E:\distil-large-v3-ct2\` |
| TTS  | Kokoro-82M (kokoro-onnx 0.5.0) | `E:\kokoro\` |

## Narrator Voice

Primary voice: **am_fenrir** — deep, resonant American male.  
Target aesthetic: measured, warm, authoritative storyteller (Morgan Freeman style).  
Default speed: **0.87×** (deliberate, unhurried).

## Quick Start

```powershell
# 1. Create venv (Python 3.11 required)
py -3.11 -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install espeak-ng (download MSI, add to PATH)
   https://github.com/espeak-ng/espeak-ng/releases

# 4. Verify everything
python scripts/verify_setup.py

# 5. Run standalone pipeline
python scripts/run_pipeline.py
```

## Persona Profiles

| Persona  | Voice      | Speed | Use Case |
|----------|------------|-------|----------|
| narrator | am_fenrir  | 0.87× | Creative, philosophy, astronomy, long-form |
| mentor   | am_fenrir  | 0.92× | Tutor 6–9, research, code |
| teacher  | am_michael | 0.95× | Tutor K–5, step-by-step |
| chat     | am_fenrir  | 1.00× | Conversational |
| quick    | am_onyx    | 1.05× | Device control, status |

## Project Structure

```
src/
  config.py              — All paths and constants (E:\ model paths)
  voice_pipeline.py      — Main orchestrator (standalone + ZMQ modes)
  stt/
    audio_capture.py     — Microphone capture with energy VAD gate
    transcriber.py       — faster-whisper WhisperModel wrapper
  tts/
    expression_engine.py — intent → persona → SpeechParams
    synthesizer.py       — Kokoro-ONNX synthesis + sentence streaming
    player.py            — Async audio queue + sounddevice playback
scripts/
  verify_setup.py        — Pre-flight check
  run_pipeline.py        — Standalone entry point
tests/
  test_stt.py            — STT smoke test
  test_tts.py            — TTS synthesis test
  test_expression.py     — Expression engine unit tests
```

## Python Version

**Python 3.11.x is required.**  
ctranslate2 (used by faster-whisper) has no wheels for Python 3.13+.  
The main Sage Kaizen app runs Python 3.14.3 in a separate process.
