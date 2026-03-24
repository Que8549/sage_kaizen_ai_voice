# CLAUDE.md — Sage Kaizen AI Voice
## Primary context document for Claude Code (VS Code extension)

---

## Project Overview

## WHO (Stakeholders + Operating Context)
### Primary user / operator / administrator / owner
- **Alquin Cook** (project owner), building Sage Kaizen on a high-end Windows rig.

**sage_kaizen_ai_voice** is the isolated voice pipeline subsystem for the
Sage Kaizen local-first AI assistant. It handles all speech-to-text input
and text-to-speech output for the system.

---

## Host System

| Component | Value |
|---|---|
| OS | Windows 11 Professional |
| CPU | AMD Ryzen 9 9950X3D |
| RAM | 192 GB DDR5 |
| GPU 0 | NVIDIA RTX 5090 32 GB (Fast Brain LLM) |
| GPU 1 | NVIDIA RTX 5080 16 GB (Architect Brain LLM) |
| CUDA | 12.8 |
| Python (this venv) | **3.11.x ONLY** (ctranslate2 constraint) |
| Main Sage Kaizen Python | 3.14.3 (separate process, communicates via ZMQ) |

---

## Python Version Constraint

**This project MUST use Python 3.11.x.** `ctranslate2` (required by
`faster-whisper`) has no published wheels for Python 3.13+.

---

## Architecture

```
Microphone
    │  PyAudio (16kHz, 30ms chunks)
    ▼
AudioCapture  (src/stt/audio_capture.py)
    │  energy VAD gate → audio queue
    ▼
Transcriber   (src/stt/transcriber.py)
    │  faster-whisper + distil-large-v3 INT8 CPU + Silero VAD
    ▼
Transcript  (plain text)
    │
    ▼
ExpressionEngine  (src/tts/expression_engine.py)
    │  intent → Persona → SpeechParams (voice, speed, lang)
    ▼
KokoroSynthesizer  (src/tts/synthesizer.py)
    │  misaki[en]          text → IPA phonemes
    │  phoneme_to_ids()    IPA chars → token IDs (Kokoro vocab)
    │  voices/am_fenrir.bin[seq_len] → style vector (1, 256)
    │  onnxruntime.InferenceSession(model_quantized.onnx)
    │  inputs: input_ids, style, speed → output: float32 audio @ 24kHz
    ▼
AudioPlayer   (src/tts/player.py)
    │  asyncio queue + sounddevice
    ▼
Speakers
```

---

## Model Files (Pre-installed on E:\)

### STT — faster-whisper + distil-large-v3

| Item | Value |
|---|---|
| Source | distil-whisper/distil-large-v3 (HuggingFace) — converted locally to CT2 INT8 |
| Raw download | `E:\distil-large-v3-ct2\` (Transformers format — model.safetensors) |
| CT2 model | `E:\distil-large-v3-ct2-int8\` (CTranslate2 format — model.bin, pure INT8) |
| Device | CPU only |
| Quant | INT8 (direct conversion — smaller than loading FP16 and quantizing at runtime) |
| Speed | 6.3× faster than large-v3, within 1% WER |
| Convert cmd | `ct2-transformers-converter --model E:\distil-large-v3-ct2 --output_dir E:\distil-large-v3-ct2-int8 --quantization int8 --copy_files tokenizer.json preprocessor_config.json` |

### TTS — Kokoro-82M ONNX

| Item | Value |
|---|---|
| Source | **onnx-community/Kokoro-82M-v1.0-ONNX** (HuggingFace) |
| ONNX model | `E:\Kokoro-82M-v1.0-ONNX\onnx\model_quantized.onnx` (89 MB pure INT8) |
| Voices dir | `E:\Kokoro-82M-v1.0-ONNX\voices\` |
| Python API | Raw `onnxruntime.InferenceSession` + `misaki[en]` G2P |
| No pip pkg | **kokoro-onnx is NOT used** — raw onnxruntime only |

> **WARNING**: `model_q8f16.onnx` mixes INT8 weights with fp16 activations and **crashes
> CPUExecutionProvider** at the C++ level (no Python exception). Always use
> `model_quantized.onnx` (pure INT8) for CPU inference.

**Exact file layout on E:\ drive:**
```
E:\Kokoro-82M-v1.0-ONNX\
  onnx\
    model_quantized.onnx  ← DEFAULT (pure INT8, 89 MB, CPU-safe)
    model.onnx            ← optional (fp32, 311 MB)
    model_fp16.onnx       ← optional (fp16, 156 MB)
  voices\
    am_onyx.bin        ← narrator/mentor/chat  (REQUIRED)
    am_michael.bin     ← teacher               (REQUIRED)
    am_echo.bin        ← quick/device control  (REQUIRED)
    af_heart.bin       ← (and all other voices from the repo)
    ...
```

**TTS Inference Pipeline (implemented in synthesizer.py):**
```python
# 1. Text → IPA phonemes
from misaki import en, espeak
g2p = en.G2P(trf=False, british=False, fallback=espeak.EspeakFallback(british=False))
phonemes, _ = g2p("There are more stars than grains of sand.")

# 2. IPA phoneme string → token ID list (using _PHONEME_VOCAB in synthesizer.py)
token_ids = phoneme_to_ids(phonemes)   # max 510 IDs

# 3. Load voice style vector (shape: max_tokens × 1 × 256)
voice_array = np.fromfile("voices/am_fenrir.bin", dtype=np.float32).reshape(-1, 1, 256)
style = voice_array[len(token_ids)]    # index by sequence length → (1, 256)

# 4. Run ONNX inference (v1.0: input key is "input_ids", was "tokens" in older pack)
sess = onnxruntime.InferenceSession("onnx/model_quantized.onnx")
audio = sess.run(None, {
    "input_ids": np.array([[0, *token_ids, 0]], dtype=np.int64),
    "style":     style,                        # (1, 256) float32
    "speed":     np.array([0.87], dtype=np.float32),
})[0]   # float32 audio at 24kHz
```

---

## Narrator Voice Identity — "Sage"

Target: deep, resonant, warm African-American male voice.
Aesthetic reference: am_onyx — modeled on OpenAI's Onyx (98 Hz, gravelly,
"rich and sophisticated", modern authority, conversational warmth).

| Persona | Voice | Speed | Use Case |
|---|---|---|---|
| **narrator** | `am_onyx` | 0.87× | Creative, philosophy, astronomy, long-form |
| **mentor** | `am_onyx` | 0.92× | Tutor 6–9, research, code, architecture |
| **teacher** | `am_michael` | 0.95× | Tutor K–5, step-by-step explainers |
| **chat** | `am_onyx` | 1.00× | Conversational |
| **quick** | `am_echo` | 1.05× | Device control, status ACKs |

---

## ZeroMQ Integration

| Port | Direction | Pattern | Purpose |
|---|---|---|---|
| 5790 | STT → Main | PUSH/PULL | Transcripts |
| 5791 | Main → TTS | PUB/SUB | Token stream |
| 5792 | TTS → Main | PUSH/PULL | Barge-in signal |

Main Sage Kaizen app BINDS. This service CONNECTS.
Pi agents use 5800–5810 — no conflict.

---

## Project File Map

```
sage_kaizen_ai_voice/
├── CLAUDE.md                    ← This file
├── AGENTS.md                    ← Task guide for Claude Code
├── README.md
├── requirements.txt             ← pip install -r requirements.txt
├── pyproject.toml
├── config/
│   ├── paths.yaml               ← E:\ model paths
│   └── voice.yaml               ← ZMQ addrs, persona defaults
├── src/
│   ├── config.py                ← PATHS, STT, TTS, ZMQ constants
│   ├── voice_pipeline.py        ← Main orchestrator
│   ├── _zmq_handlers.py         ← ZeroMQ integration
│   ├── stt/
│   │   ├── audio_capture.py     ← PyAudio + energy VAD
│   │   └── transcriber.py       ← WhisperModel wrapper
│   └── tts/
│       ├── expression_engine.py ← intent → persona → SpeechParams
│       ├── synthesizer.py       ← onnxruntime + misaki G2P
│       └── player.py            ← Async audio queue + sounddevice
├── scripts/
│   ├── verify_setup.py          ← Pre-flight check
│   └── run_pipeline.py          ← Standalone entry point
└── tests/
    ├── test_expression.py       ← Unit tests (no hardware)
    ├── test_tts.py              ← TTS synthesis test
    └── test_stt.py              ← STT test
```

---

## Install Steps

```powershell
# 1. Python 3.11 venv ONLY
# py -3.11 -m venv .venv
# .venv\Scripts\activate

# 2. Install all dependencies (one command, no workarounds)
# pip install -r requirements.txt

# 3. Install espeak-ng MSI → add to PATH
#    https://github.com/espeak-ng/espeak-ng/releases

# 4. Verify
python scripts/verify_setup.py

# 5. Test (no hardware needed)
python tests/test_expression.py
python tests/test_tts.py
```

---

## Hard Invariants

1. **Python 3.11.x only** — ctranslate2 constraint
2. **All paths from `src/config.py`** — never hardcode model paths inline
3. **All paths via `Path(...).resolve()`** before use
4. **No runtime model downloads** — all files pre-exist on E:\
5. **No .bat files in launch paths** — Python launches directly
6. **No `kokoro-onnx` pip package** — use raw `onnxruntime` + `misaki[en]`
7. **Main app BINDS ZMQ sockets; this service CONNECTS**
8. **`model_quantized.onnx` is the default** — change via PATHS.TTS_ONNX_MODEL if needed. NEVER use `model_q8f16.onnx` on CPU — it crashes CPUExecutionProvider.

---

## Why onnx-community/Kokoro-82M-ONNX (not kokoro-onnx pip package)

The `kokoro-onnx` pip package ships its own `kokoro-v1.0.onnx` and
`voices-v1.0.bin` files hosted on the maintainer's GitHub releases —
separate from the HuggingFace repos. Since you already downloaded
`onnx-community/Kokoro-82M-ONNX` from HuggingFace, the project now uses
raw `onnxruntime.InferenceSession` directly with `misaki[en]` for G2P.

This approach is equally stable, avoids an extra download, and gives
direct control over the inference pipeline.

---

### Non-goals (unless explicitly requested)
- Large rewrites that break conventions
- “Magic” behavior without logs/tests

## Notes for Claude (behavioral guidance)
- Prefer small, incremental changes that preserve existing style.
- When adding new features, prefer adding a module rather than tangling existing modules.