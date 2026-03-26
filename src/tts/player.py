"""
src/tts/player.py — Audio Player (non-blocking queue + sounddevice)
====================================================================
Plays numpy audio chunks in sequence without blocking the synthesis loop.

Architecture:
  - Internal asyncio.Queue receives (samples, sr) tuples
  - A background task plays each chunk with sd.play() + sd.wait()
  - interrupt() immediately drains the queue and stops current playback
  - Designed for barge-in: dropping mid-sentence is clean and immediate
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

import numpy as np
import sounddevice as sd

from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.voice.tts.player")


class AudioPlayer:
    """
    Non-blocking audio queue for streaming TTS playback.

    Usage
    -----
    player = AudioPlayer()
    await player.start()         # start background play loop

    player.enqueue(samples, sr)  # enqueue a chunk (non-blocking)
    player.interrupt()           # stop playback, drain queue
    player.reset()               # clear interrupt for new session

    await player.stop()          # shut down cleanly
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Optional[tuple[np.ndarray, int]]] = asyncio.Queue()
        self._interrupt   = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._is_playing  = False  # True while sd.play() is active in the executor

    async def start(self) -> None:
        """Start the background playback loop."""
        try:
            dev = sd.query_devices(kind="output")
            _LOG.info(
                "AudioPlayer: output device = %r  native_rate=%d Hz",
                dev.get("name", "?"),
                int(dev.get("default_samplerate", 0)),
            )
        except Exception as exc:
            _LOG.warning("AudioPlayer: could not query output device: %s", exc)
        self._task = asyncio.create_task(self._play_loop(), name="AudioPlayer")
        _LOG.info("AudioPlayer started")

    async def stop(self) -> None:
        """Drain queue and stop the playback loop."""
        self.interrupt()
        if self._task:
            await self._queue.put(None)          # sentinel to exit loop
            await asyncio.wait_for(self._task, timeout=3.0)
        _LOG.info("AudioPlayer stopped")

    def enqueue(self, samples: np.ndarray, sr: int) -> None:
        """Enqueue an audio chunk. Non-blocking."""
        self._queue.put_nowait((samples, sr))

    def interrupt(self) -> None:
        """
        Signal immediate playback stop.
        Drains the queue and sets the interrupt event.
        sd.stop() will cut off any currently playing audio.
        """
        self._interrupt.set()
        # Drain pending chunks
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        # Stop any currently playing audio
        try:
            sd.stop()
        except Exception:
            pass

    def reset(self) -> None:
        """Clear interrupt flag. Call before starting a new turn."""
        self._interrupt.clear()

    async def drain(self, timeout: float = 12.0) -> None:
        """
        Wait until the queue is empty and the current chunk has finished playing.
        Used after speak_text() to ensure audio completes before proceeding.
        """
        deadline = time.monotonic() + timeout
        while not self._queue.empty() or self._is_playing:
            if time.monotonic() > deadline:
                _LOG.warning("AudioPlayer.drain() timed out after %.1fs", timeout)
                break
            await asyncio.sleep(0.05)

    @property
    def is_playing(self) -> bool:
        return self._is_playing or not self._queue.empty()

    # ─────────────────────────────────────────────────────────
    # Internal play loop
    # ─────────────────────────────────────────────────────────

    async def _play_loop(self) -> None:
        loop = asyncio.get_running_loop()

        while True:
            item = await self._queue.get()

            # Sentinel: stop signal
            if item is None:
                break

            if self._interrupt.is_set():
                # Chunk was queued before interrupt — discard
                continue

            samples, sr = item
            duration = len(samples) / sr if sr else 0.0
            _LOG.info(
                "Playing audio: %d samples @ %d Hz (%.2fs)",
                len(samples), sr, duration,
            )
            try:
                self._is_playing = True
                await loop.run_in_executor(None, self._play_blocking, samples, sr)
            except asyncio.CancelledError:
                break
            except Exception:
                _LOG.exception("Audio playback error")
            finally:
                self._is_playing = False

        _LOG.debug("AudioPlayer loop exited")

    @staticmethod
    def _play_blocking(samples: np.ndarray, sr: int) -> None:
        """Play audio synchronously — runs in thread executor."""
        sd.play(samples, sr, blocking=True, latency="low")
