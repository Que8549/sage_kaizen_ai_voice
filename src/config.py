"""
src/config.py — Sage Kaizen AI Voice — Centralized Configuration
=================================================================
All model paths, ZeroMQ addresses, and tunable constants live here.

INVARIANT: All paths are resolved via Path(...).resolve() before use.
INVARIANT: No model downloads at runtime — all files must pre-exist on E:\\.
INVARIANT: Never hardcode paths inline in other modules — always import from here.

TTS model source: onnx-community/Kokoro-82M-v1.0-ONNX (HuggingFace)
  Downloaded folder structure:
    E:\\Kokoro-82M-v1.0-ONNX\\onnx\\model_quantized.onnx ← DEFAULT (89 MB, pure INT8, CPU-safe)
    E:\\Kokoro-82M-v1.0-ONNX\\onnx\\model.onnx           ← fp32 (326 MB)
    E:\\Kokoro-82M-v1.0-ONNX\\voices\\am_fenrir.bin      ← narrator/mentor/chat voice
    E:\\Kokoro-82M-v1.0-ONNX\\voices\\am_michael.bin     ← teacher voice
    E:\\Kokoro-82M-v1.0-ONNX\\voices\\am_onyx.bin        ← quick/device voice
    (+ 50+ additional voices from the full v1.0 pack)

Quantization options (v1.0):
    model_quantized.onnx  89 MB  Pure INT8                  ← DEFAULT (CPU-safe)
    model_q8f16.onnx      83 MB  INT8+fp16 — CRASHES on CPU (CPUExecutionProvider unsupported)
    model_uint8f16.onnx  109 MB  Mixed precision
    model_fp16.onnx      156 MB  Half precision
    model.onnx           311 MB  Full fp32

v1.0 breaking change: ONNX input key renamed "tokens" → "input_ids".
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class PATHS:
    # ── STT — faster-whisper distil-large-v3 (INT8, CPU) ──────
    # CTranslate2 INT8 format — converted from distil-whisper/distil-large-v3
    # Run: ct2-transformers-converter --model E:\distil-large-v3-ct2
    #      --output_dir E:\distil-large-v3-ct2-int8 --quantization int8
    #      --copy_files tokenizer.json preprocessor_config.json
    STT_MODEL_DIR = Path(r"E:\distil-large-v3-ct2-int8").resolve()

    # ── TTS — Kokoro-82M ONNX (onnx-community/Kokoro-82M-ONNX) ──
    #
    # TTS_ONNX_MODEL: Path to the ONNX model file.
    #   model_quantized.onnx → pure INT8, 89 MB  ← DEFAULT (CPU-safe, recommended)
    #   model.onnx           → full fp32, 311 MB, marginally better quality
    #   WARNING: model_q8f16.onnx crashes CPUExecutionProvider (fp16 ops unsupported on CPU)
    #
    TTS_ONNX_MODEL = Path(r"E:\Kokoro-82M-v1.0-ONNX\onnx\model_quantized.onnx").resolve()

    # TTS_VOICES_DIR: Directory containing individual voice .bin files.
    #   Each file: <voice_name>.bin  e.g. am_fenrir.bin, am_onyx.bin
    #   Shape when loaded: (max_tokens, 1, 256) float32
    TTS_VOICES_DIR = Path(r"E:\Kokoro-82M-v1.0-ONNX\voices").resolve()

    # TTS_TOKENIZER: tokenizer.json bundled with the model.
    #   Contains the authoritative phoneme→token-id vocabulary at ["model"]["vocab"].
    #   Used by synthesizer.py instead of any hardcoded vocab.
    TTS_TOKENIZER = Path(r"E:\Kokoro-82M-v1.0-ONNX\tokenizer.json").resolve()

    # Output directory for test WAV files
    OUTPUT_DIR = (PROJECT_ROOT / "output").resolve()

    @classmethod
    def validate(cls) -> list[str]:
        """
        Check that all required model files and directories exist.
        Returns a list of error strings (empty list = all good).
        """
        errors: list[str] = []
        checks = [
            (cls.STT_MODEL_DIR,  "STT model directory"),
            (cls.TTS_ONNX_MODEL, "TTS ONNX model"),
            (cls.TTS_VOICES_DIR, "TTS voices directory"),
            (cls.TTS_TOKENIZER,  "TTS tokenizer.json (phoneme vocabulary)"),
        ]
        for path, label in checks:
            if not path.exists():
                errors.append(f"Missing {label}: {path}")

        # Check the three voices used by Sage Kaizen personas
        if cls.TTS_VOICES_DIR.exists():
            for voice in ["am_onyx", "am_michael", "am_echo"]:
                vf = cls.TTS_VOICES_DIR / f"{voice}.bin"
                if not vf.exists():
                    errors.append(f"Missing voice file: {vf}")

        return errors


class STT:
    DEVICE       = "cpu"
    COMPUTE_TYPE = "int8"
    LANGUAGE               = "en"
    BEAM_SIZE              = 1
    CONDITION_ON_PREV_TEXT = False
    VAD_FILTER             = True
    SAMPLE_RATE        = 16_000
    CHANNELS           = 1
    CHUNK_FRAMES       = 480
    FORMAT_PAUDIO      = 8
    ENERGY_THRESHOLD   = 300
    MIN_SPEECH_CHUNKS  = 5
    # ctranslate2 thread tuning — Ryzen 9 9950X3D (Zen 5, 16 physical cores)
    # CPU_THREADS: intra-op parallelism (threads per compute op).
    #   16 = all physical cores; hyperthreads (32) hurt INT8 VNNI throughput.
    # INTER_THREADS: parallel decode streams. 1 = single utterance at a time.
    CPU_THREADS   = 16
    INTER_THREADS = 1

    # Endpoint detection tuning
    # SILENCE_CHUNKS_TO_STOP: chunks of silence before utterance is emitted.
    #   25 chunks × 30ms = 750ms — matches natural inter-word pause duration
    #   in English speech.  Prevents splitting wh-questions like "Who was the
    #   third president…" where "Who?" has a brief pause before the predicate.
    #   Production voice assistants (Vapi, LiveKit) use 750–800 ms.
    #   The energy gate MUST be >= min_silence_duration_ms in VAD_PARAMETERS
    #   or Stage 1 will cut the utterance before Stage 2 (Silero) can apply
    #   its own silence padding, producing false sentence splits.
    SILENCE_CHUNKS_TO_STOP = 25

    # Silero VAD parameters passed to WhisperModel.transcribe().
    # Overrides the built-in defaults (min_silence_duration_ms=2000 is too slow
    # for interactive voice assistants).
    # Keep min_silence_duration_ms <= SILENCE_CHUNKS_TO_STOP × 30ms (750ms)
    # so Silero never fires before the energy gate does.
    VAD_PARAMETERS = {
        "min_silence_duration_ms": 700,  # raised from 500ms; matches 750ms gate
        "speech_pad_ms":           400,  # raised from 200ms; preserves breath pauses
    }


class TTS:
    LANG        = "en-us"
    SAMPLE_RATE = 24_000

    # Narrator voice — deep, resonant African-American male aesthetic
    # am_onyx: modeled on OpenAI's Onyx (98 Hz, gravelly, warm, modern authority)
    DEFAULT_VOICE = "am_onyx"
    DEFAULT_SPEED = 0.87

    MIN_CHUNK_WORDS = 4

    # onnxruntime SessionOptions — Ryzen 9 9950X3D (Zen 5, AVX-512 VNNI, 16 cores)
    # model_quantized.onnx is pure INT8; onnxruntime maps these ops to AVX-512 VNNI
    # kernels (VPDPBUSD etc.) when ORT_ENABLE_ALL is set.
    #
    # ORT_INTRA_THREADS: threads parallelising a single op (matrix multiply).
    #   16 = physical cores only. Hyperthreads (32) hurt VNNI INT8 throughput.
    # ORT_INTER_THREADS: threads executing independent ops concurrently.
    #   1 = sequential — correct for single-stream TTS inference.
    ORT_INTRA_THREADS = 16
    ORT_INTER_THREADS = 1


class ZMQ:
    TRANSCRIPT_BUS = "tcp://127.0.0.1:5790"
    TOKEN_BUS      = "tcp://127.0.0.1:5791"
    INTERRUPT_BUS  = "tcp://127.0.0.1:5792"


class LOGGING:
    LOG_DIR  = (PROJECT_ROOT / "logs").resolve()
    STT_LOG  = LOG_DIR / "stt_service.log"
    TTS_LOG  = LOG_DIR / "tts_service.log"
    MAIN_LOG = LOG_DIR / "voice_pipeline.log"
