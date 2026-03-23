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
import logging
from typing import Optional

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)


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

    async def start(self) -> None:
        """Start the background playback loop."""
        self._task = asyncio.create_task(self._play_loop(), name="AudioPlayer")
        log.info("AudioPlayer started")

    async def stop(self) -> None:
        """Drain queue and stop the playback loop."""
        self.interrupt()
        if self._task:
            await self._queue.put(None)          # sentinel to exit loop
            await asyncio.wait_for(self._task, timeout=3.0)
        log.info("AudioPlayer stopped")

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

    @property
    def is_playing(self) -> bool:
        return not self._queue.empty()

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
            try:
                await loop.run_in_executor(None, self._play_blocking, samples, sr)
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("Audio playback error")

        log.debug("AudioPlayer loop exited")

    @staticmethod
    def _play_blocking(samples: np.ndarray, sr: int) -> None:
        """Play audio synchronously — runs in thread executor."""
        sd.play(samples, sr, blocking=True)
