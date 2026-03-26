"""
src/stt/transcriber.py — faster-whisper Transcriber (distil-large-v3)
======================================================================
Wraps WhisperModel with:
  - Local model loading from E:\\ (no network calls at runtime)
  - Silero VAD filter (built into faster-whisper)
  - Async-compatible transcription via thread executor
  - Lazy initialization (model loads on first use or explicit init)

Model: distil-large-v3
  - 6.3x faster than Whisper large-v3
  - Within 1% WER on long-form audio
  - Designed for faster-whisper's sequential algorithm
  - MUST use condition_on_previous_text=False to avoid hallucination
  - MUST use beam_size=1 (model is optimized for this)

Verified API (faster-whisper 1.2.1, March 2026):
  WhisperModel(model_size_or_path, device, compute_type, cpu_threads, num_workers)
  model.transcribe(audio, beam_size, language, condition_on_previous_text, vad_filter)
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

from sk_logging import get_logger
from src.config import PATHS, STT


_LOG = get_logger("sage_kaizen.voice.stt.transcriber")


class Transcriber:
    """
    Thread-safe faster-whisper transcriber.

    Usage
    -----
    t = Transcriber()
    t.initialize()  # loads model from E:\\ (~3 GB RAM)

    # Synchronous:
    text = t.transcribe(audio_float32)

    # Async (runs in thread executor):
    text = await t.transcribe_async(audio_float32)
    """

    def __init__(self) -> None:
        self._model: Optional[WhisperModel] = None
        self._lock  = threading.Lock()
        self._ready = False

    # ─────────────────────────────────────────────────────────
    # Initialization
    # ─────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """
        Load the distil-large-v3 model from E:\\ drive.
        Blocks until the model is fully loaded (~5–10 seconds on first call).
        Safe to call multiple times (idempotent).
        """
        if self._ready:
            return

        model_dir = PATHS.STT_MODEL_DIR
        if not model_dir.exists():
            raise FileNotFoundError(
                f"STT model directory not found: {model_dir}\n"
                f"Run scripts/verify_setup.py for diagnostics."
            )

        _LOG.info("Loading STT model from %s ...", model_dir)
        _LOG.info("  device=%s  compute_type=%s", STT.DEVICE, STT.COMPUTE_TYPE)

        with self._lock:
            if not self._ready:  # double-checked locking
                self._model = WhisperModel(
                    model_size_or_path=str(model_dir),
                    device=STT.DEVICE,
                    compute_type=STT.COMPUTE_TYPE,
                    cpu_threads=STT.CPU_THREADS,
                    num_workers=STT.INTER_THREADS,
                )
                self._ready = True

        _LOG.info("STT model ready — distil-large-v3 %s CPU", STT.COMPUTE_TYPE)

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ─────────────────────────────────────────────────────────
    # Transcription
    # ─────────────────────────────────────────────────────────

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe a float32 audio array (16kHz, mono, range [-1, 1]).

        Parameters
        ----------
        audio : np.ndarray
            Float32 audio samples at 16kHz. Produced by AudioCapture.

        Returns
        -------
        str
            Transcribed text, stripped and cleaned. Empty string if no speech.
        """
        if not self._ready:
            self.initialize()

        with self._lock:
            segments, info = self._model.transcribe(
                audio,
                beam_size=STT.BEAM_SIZE,
                language=STT.LANGUAGE,
                condition_on_previous_text=STT.CONDITION_ON_PREV_TEXT,
                vad_filter=STT.VAD_FILTER,
                vad_parameters=STT.VAD_PARAMETERS,
            )
            # segments is a generator — consume it inside the lock
            text = " ".join(seg.text for seg in segments).strip()

        if text:
            _LOG.info("Transcribed: %r  (lang=%s)", text[:80], info.language)
        else:
            _LOG.debug("No speech detected in utterance")

        return text

    async def transcribe_async(self, audio: np.ndarray) -> str:
        """
        Async version — runs transcription in a thread executor so the
        event loop stays responsive during inference.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.transcribe, audio)

    # ─────────────────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────────────────

    def transcribe_file(self, audio_path: str | Path) -> str:
        """
        Transcribe a local audio file (WAV, MP3, FLAC, etc.).
        Useful for testing without a microphone.
        """
        if not self._ready:
            self.initialize()

        path = Path(audio_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        with self._lock:
            segments, info = self._model.transcribe(
                str(path),
                beam_size=STT.BEAM_SIZE,
                language=STT.LANGUAGE,
                condition_on_previous_text=STT.CONDITION_ON_PREV_TEXT,
                vad_filter=STT.VAD_FILTER,
            )
            text = " ".join(seg.text for seg in segments).strip()

        _LOG.info("File transcription: %r", text[:80])
        return text
