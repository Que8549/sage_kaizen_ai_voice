"""
src/config.py — Sage Kaizen AI Voice — Centralized Configuration
=================================================================
All model paths, ZeroMQ addresses, and tunable constants live here.

INVARIANT: All paths are resolved via Path(...).resolve() before use.
INVARIANT: No model downloads at runtime — all files must pre-exist on E:\\.
INVARIANT: Never hardcode paths inline in other modules — always import from here.

TTS model source: onnx-community/Kokoro-82M-ONNX (HuggingFace)
  Downloaded folder structure:
    E:\\Kokoro-82M-ONNX\\onnx\\model_q8f16.onnx     ← ONNX model (INT8, 88 MB)
    E:\\Kokoro-82M-ONNX\\onnx\\onnx\\model.onnx         ← ONNX model (fp32, 310 MB)
    E:\\Kokoro-82M-ONNX\\voices\\am_fenrir.bin    ← narrator voice
    E:\\Kokoro-82M-ONNX\\\\voices\\am_michael.bin   ← teacher voice
    E:\\Kokoro-82M-ONNX\\\\voices\\am_onyx.bin      ← quick/device voice
    (+ all other voice .bin files from the voices/ folder)
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class PATHS:
    # ── STT — faster-whisper distil-large-v3 (INT8, CPU) ──────
    STT_MODEL_DIR = Path(r"E:\distil-large-v3-ct2").resolve()

    # ── TTS — Kokoro-82M ONNX (onnx-community/Kokoro-82M-ONNX) ──
    #
    # TTS_ONNX_MODEL: Path to the ONNX model file.
    #   model_q8.onnx   → INT8 quantised, 88 MB, recommended for CPU (fast + small)
    #   model.onnx      → full fp32, 310 MB, marginally higher quality
    #
    TTS_ONNX_MODEL = Path(r"E:\Kokoro-82M-ONNX\onnx\model_q8f16.onnx").resolve()

    # TTS_VOICES_DIR: Directory containing individual voice .bin files.
    #   Each file: <voice_name>.bin  e.g. am_fenrir.bin, am_onyx.bin
    #   Shape when loaded: (max_tokens, 1, 256) float32
    TTS_VOICES_DIR = Path(r"E:\Kokoro-82M-ONNX\voices").resolve()

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
            (cls.TTS_ONNX_MODEL, "TTS ONNX model (model_q8.onnx)"),
            (cls.TTS_VOICES_DIR, "TTS voices directory"),
        ]
        for path, label in checks:
            if not path.exists():
                errors.append(f"Missing {label}: {path}")

        # Check the three voices used by Sage Kaizen personas
        if cls.TTS_VOICES_DIR.exists():
            for voice in ["am_fenrir", "am_michael", "am_onyx"]:
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
    SILENCE_CHUNKS_TO_STOP = 25
    MIN_SPEECH_CHUNKS  = 5


class TTS:
    LANG        = "en-us"
    SAMPLE_RATE = 24_000

    # Narrator voice (Morgan Freeman aesthetic — deep, warm, deliberate)
    DEFAULT_VOICE = "am_fenrir"
    DEFAULT_SPEED = 0.87

    MIN_CHUNK_WORDS = 4


class ZMQ:
    TRANSCRIPT_BUS = "tcp://127.0.0.1:5790"
    TOKEN_BUS      = "tcp://127.0.0.1:5791"
    INTERRUPT_BUS  = "tcp://127.0.0.1:5792"


class LOGGING:
    LOG_DIR  = (PROJECT_ROOT / "logs").resolve()
    STT_LOG  = LOG_DIR / "stt_service.log"
    TTS_LOG  = LOG_DIR / "tts_service.log"
    MAIN_LOG = LOG_DIR / "voice_pipeline.log"
