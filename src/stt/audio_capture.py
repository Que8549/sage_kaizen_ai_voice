"""
src/stt/audio_capture.py — Microphone Capture with Energy VAD Gate
====================================================================
Owns the PyAudio microphone stream and produces complete audio utterances
ready for transcription.

Two-stage VAD:
  Stage 1 (here): audioop.rms() energy gate — ~0.05ms per chunk, CPU-free
  Stage 2: Silero VAD inside faster-whisper's transcribe() — high accuracy

The energy gate prevents the transcriber from wasting cycles on silence.
The Silero VAD then removes any remaining non-speech frames.

Threading model:
  - AudioCapture runs a daemon thread that reads from PyAudio
  - Completed utterances are placed in a queue
  - The main pipeline consumes from that queue
"""

from __future__ import annotations

import queue
import threading
from typing import Callable, Optional

import numpy as np
import pyaudio

from sk_logging import get_logger
from src.config import STT

_LOG = get_logger("sage_kaizen.voice.stt.capture")


class AudioCapture:
    """
    Captures microphone audio and emits complete speech utterances.

    Parameters
    ----------
    on_utterance : Callable[[np.ndarray], None], optional
        Callback invoked with each utterance as a float32 numpy array
        at 16kHz mono. If None, utterances are queued and accessible
        via get_utterance().
    device_index : int, optional
        PyAudio device index. None = system default.
    """

    def __init__(
        self,
        on_utterance: Optional[Callable[[np.ndarray], None]] = None,
        device_index: Optional[int] = None,
    ) -> None:
        self._callback  = on_utterance
        self._device    = device_index
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._running   = False
        self._thread: Optional[threading.Thread] = None
        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream = None

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open microphone and start capture thread."""
        if self._running:
            return
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=STT.CHANNELS,
            rate=STT.SAMPLE_RATE,
            input=True,
            frames_per_buffer=STT.CHUNK_FRAMES,
            input_device_index=self._device,
        )
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="AudioCapture",
            daemon=True,
        )
        self._thread.start()
        _LOG.info(
            "AudioCapture started — device=%s sample_rate=%d chunk=%dms",
            self._device or "default",
            STT.SAMPLE_RATE,
            STT.CHUNK_FRAMES * 1000 // STT.SAMPLE_RATE,
        )

    def stop(self) -> None:
        """Stop capture and release hardware."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._pa:
            self._pa.terminate()
        _LOG.info("AudioCapture stopped")

    def get_utterance(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        """
        Block until an utterance is available or timeout elapses.
        Returns float32 numpy array at 16kHz, or None on timeout.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def list_devices(self) -> list[dict]:
        """Return a list of available input audio devices."""
        pa = pyaudio.PyAudio()
        devices = []
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                devices.append({"index": i, "name": info["name"]})
        pa.terminate()
        return devices

    # ─────────────────────────────────────────────────────────
    # Capture loop (runs in daemon thread)
    # ─────────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        """
        Reads 30ms chunks from PyAudio.
        Uses RMS energy as a fast first-stage VAD gate.
        Accumulates chunks until silence detected, then emits utterance.
        """
        speech_chunks: list[bytes] = []
        silent_count  = 0
        in_speech     = False

        while self._running:
            try:
                raw = self._stream.read(
                    STT.CHUNK_FRAMES,
                    exception_on_overflow=False,
                )
            except OSError as e:
                _LOG.warning("PyAudio read error: %s", e)
                continue

            # Fast energy gate: RMS of int16 samples via numpy
            energy = int(np.sqrt(np.mean(
                np.frombuffer(raw, dtype=np.int16).astype(np.float32) ** 2
            )))

            if energy > STT.ENERGY_THRESHOLD:
                in_speech    = True
                silent_count = 0
                speech_chunks.append(raw)
            else:
                if in_speech:
                    silent_count += 1
                    speech_chunks.append(raw)  # include trailing silence for context

                    if silent_count >= STT.SILENCE_CHUNKS_TO_STOP:
                        # Utterance complete — check minimum length
                        if len(speech_chunks) >= STT.MIN_SPEECH_CHUNKS:
                            utterance = self._chunks_to_float32(speech_chunks)
                            self._emit(utterance)
                        speech_chunks = []
                        silent_count  = 0
                        in_speech     = False

        _LOG.debug("AudioCapture loop exited")

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _chunks_to_float32(chunks: list[bytes]) -> np.ndarray:
        """
        Concatenate int16 PCM bytes → normalised float32 numpy array.
        faster-whisper expects float32 in [-1.0, 1.0] at 16kHz.
        """
        raw = b"".join(chunks)
        samples = np.frombuffer(raw, dtype=np.int16)
        return samples.astype(np.float32) / 32768.0

    def _emit(self, audio: np.ndarray) -> None:
        """Send utterance to callback or internal queue."""
        if self._callback is not None:
            try:
                self._callback(audio)
            except Exception:
                _LOG.exception("on_utterance callback raised")
        else:
            self._queue.put_nowait(audio)
