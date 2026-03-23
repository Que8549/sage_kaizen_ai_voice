# AGENTS.md — Sage Kaizen AI Voice

## Task Guide for Claude Code Agent Sessions

---

## Before Any Code Task

1. Read `CLAUDE.md` fully — it contains the invariants, paths, and design decisions
2. Run `python scripts/verify_setup.py` to confirm the environment is healthy
3. Check which file you are modifying against the file map in `CLAUDE.md`

---

## Common Tasks

### Add a new persona to the expression engine

File: `src/tts/expression_engine.py`

1. Add a new `Persona` enum value
2. Add a `SpeechParams` entry in `_PERSONA_PROFILES`
3. Map relevant intent strings in `_INTENT_TO_PERSONA`
4. Add a test case in `tests/test_expression.py`
5. Add the persona metadata dict in `app/ui/tts_panel.py` (if Streamlit is active)

### Change the narrator voice blend

File: `src/tts/expression_engine.py` — `VOICE_NARRATOR` constant

The blend string format is: `"voice_name:weight,voice_name:weight"`
Weights are percentages; the synthesizer normalises them to 1.0.

Available American male voices: `am_adam, am_echo, am_eric, am_fenrir,
am_liam, am_michael, am_onyx, am_puck`

Available British male voices: `bm_daniel, bm_fable, bm_george, bm_lewis`

### Adjust STT sensitivity

File: `src/stt/audio_capture.py`

- `ENERGY_THRESHOLD` — raise to ignore quieter ambient sounds
- `SILENCE_CHUNKS_TO_STOP` — how many silent 30ms chunks before ending utterance
- `MIN_SPEECH_CHUNKS` — minimum chunk count to consider as real speech

### Add a new intent mapping

File: `src/tts/expression_engine.py` — `_INTENT_TO_PERSONA` dict

Keys must match the intent strings used by the Sage Kaizen router.

---

## Constraints Claude Code Must Always Respect

- **Never change Python version** — 3.11.x only for this venv
- **Never add model downloads** at runtime — models live on `E:\`, pre-installed
- **Never hardcode model paths** — always use `src/config.py` → `PATHS` constants
- **Never use shell=True** in subprocess calls
- **Never commit secrets or API keys** — this is a local-only project
- **Never replace pyzmq** with another message bus
- **Always use `Path(...).resolve()`** before passing paths to external libraries
- **Keep audio capture single-threaded** — PyAudio stream owned by AudioCapture only

---

## Testing

```powershell
# Smoke-test TTS (generates audio file, does not use microphone)
python tests/test_tts.py

# Smoke-test STT (transcribes a test WAV from disk)
python tests/test_stt.py

# Unit-test expression engine (pure functions, no hardware)
python tests/test_expression.py

# Full standalone pipeline (needs microphone + speakers)
python scripts/run_pipeline.py
```

---

## Debugging Tips

| Symptom | Likely Cause | Fix |
|---|---|---|
| `FileNotFoundError: E:\...` | Model path wrong | Run `verify_setup.py` |
| `RuntimeError: PortAudio` | sounddevice can't find audio device | Check Windows audio settings |
| `No module named 'faster_whisper'` | Wrong venv active | Activate `.venv` |
| `ONNX Runtime error` | Wrong onnxruntime version | `pip install --upgrade onnxruntime` |
| `espeak-ng not found` | espeak-ng not in PATH | Install MSI, add to PATH |
| Voice sounds wrong | Wrong persona resolved | Add a `print(params.persona)` in `expression_engine.py` |
| ZMQ connection refused | Main app not running | Switch to standalone mode |
