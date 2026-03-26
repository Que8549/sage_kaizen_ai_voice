"""
src/_zmq_handlers.py — ZeroMQ Integration Handlers
====================================================
Used by VoicePipeline.run_integrated() to connect to the main
Sage Kaizen Python 3.14.3 app via ZeroMQ.

Two coroutines run concurrently:
  run_stt_pusher    — captures speech, transcribes, pushes to main app
  run_tts_subscriber — receives LLM tokens, synthesizes, plays audio

Message schemas (JSON over ZMQ):

  Transcript (STT → Main, port 5790):
    {"type": "transcript", "text": "...", "session_id": "uuid"}

  Session start (Main → TTS, port 5791):
    {"type": "session_start", "session_id": "uuid",
     "voice": "am_fenrir", "speed": 0.87, "lang": "en-us", "persona": "narrator"}

  Token (Main → TTS, port 5791):
    {"type": "token", "session_id": "uuid", "text": " word"}

  Turn done (Main → TTS, port 5791):
    {"type": "turn_done", "session_id": "uuid"}

  Interrupt (TTS → Main, port 5792):
    {"type": "interrupt", "session_id": "old-uuid", "reason": "new_speech"}

Synthesis pipeline (pipelined / non-blocking):
  _submit_synth() submits each sentence to a thread executor immediately,
  without awaiting. The returned future + session_id are pushed to an
  asyncio.Queue consumed by _collect_synth(), which awaits each future in
  order and enqueues the audio for playback.

  This means token reception, G2P/tokenisation, and ONNX inference can
  overlap: while the executor runs inference for sentence N, the event loop
  receives tokens and begins G2P for sentence N+1.

  Barge-in safety: each future is tagged with the session_id at submit time.
  The collector discards audio whose session_id no longer matches the live
  session, preventing stale audio from a prior turn playing after reset.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from typing import Optional

import numpy as np

from sk_logging import get_logger
from src.stt.audio_capture import AudioCapture
from src.stt.transcriber import Transcriber
from src.tts.player import AudioPlayer
from src.tts.synthesizer import KokoroSynthesizer
from src.config import TTS

_LOG = get_logger("sage_kaizen.voice.zmq")

_SENTENCE_END_RE = re.compile(r"[.!?](\s|$)")


class _SentenceBuffer:
    """Accumulates streaming tokens and flushes complete sentences."""

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, token: str) -> list[str]:
        self._buf += token
        sentences: list[str] = []
        while True:
            m = _SENTENCE_END_RE.search(self._buf)
            if not m:
                break
            end = m.end()
            sentence = self._buf[:end].strip()
            self._buf = self._buf[end:]
            if len(sentence.split()) >= 4:
                sentences.append(sentence)
        return sentences

    def flush(self) -> str:
        remaining = self._buf.strip()
        self._buf = ""
        return remaining


# ─────────────────────────────────────────────────────────
# STT pusher
# ─────────────────────────────────────────────────────────

async def run_stt_pusher(
    capture:     AudioCapture,
    transcriber: Transcriber,
    push_socket,
) -> None:
    """Capture audio, transcribe, push transcripts to main app."""
    loop = asyncio.get_running_loop()
    _LOG.info("STT pusher running")
    while True:
        audio = await loop.run_in_executor(
            None, lambda: capture.get_utterance(timeout=0.5)
        )
        if audio is None:
            continue
        text = await transcriber.transcribe_async(audio)
        if not text:
            continue
        msg = {
            "type":       "transcript",
            "text":       text,
            "session_id": str(uuid.uuid4()),
        }
        await push_socket.send(json.dumps(msg).encode())
        _LOG.info("Transcript sent: %r", text[:60])


# ─────────────────────────────────────────────────────────
# Pipelined TTS synthesis helpers
# ─────────────────────────────────────────────────────────

def _submit_synth(
    text:        str,
    voice:       str,
    speed:       float,
    lang:        str,
    session_id:  str,
    synthesizer: KokoroSynthesizer,
    synth_queue: asyncio.Queue,
    loop:        asyncio.AbstractEventLoop,
) -> None:
    """
    Submit a synthesis task to the thread executor without blocking.

    The future is immediately pushed to synth_queue alongside the session_id
    that was active at submit time. The collector uses that tag to discard
    audio from stale sessions (barge-in protection).
    """
    if not text.strip():
        return
    fut = loop.run_in_executor(
        None, lambda: synthesizer.synth_one(text, voice, speed, lang)
    )
    synth_queue.put_nowait((fut, session_id))
    _LOG.debug("Synth submitted (session=%.8s): %r", session_id, text[:40])


async def _collect_synth(
    queue:           asyncio.Queue,
    player:          AudioPlayer,
    current_sid_ref: list[Optional[str]],
) -> None:
    """
    Await synthesis futures in submission order and enqueue audio.

    Runs as a background asyncio Task for the lifetime of run_tts_subscriber.
    Exits when it receives a None sentinel.

    Audio from a session that is no longer current (barge-in occurred between
    submit and collect) is silently discarded.
    """
    _LOG.debug("SynthCollector started")
    while True:
        item = await queue.get()
        if item is None:
            break
        fut, sid = item
        try:
            samples, sr = await fut
            if sid == current_sid_ref[0] and len(samples) > 0:
                player.enqueue(np.array(samples, dtype=np.float32), int(sr))
        except Exception:
            _LOG.exception("Synthesis collect error (session=%.8s)", sid)
    _LOG.debug("SynthCollector stopped")


# ─────────────────────────────────────────────────────────
# TTS subscriber
# ─────────────────────────────────────────────────────────

async def run_tts_subscriber(
    sub_socket,
    interrupt_push,
    synthesizer: KokoroSynthesizer,
    player:      AudioPlayer,
) -> None:
    """
    Receive LLM token stream, synthesize sentences in a pipelined fashion,
    play audio in order, and handle barge-in.

    Synthesis is submitted to the thread executor immediately upon sentence
    completion, without awaiting.  A background collector coroutine awaits
    each future in order and enqueues the audio for playback.  This allows
    token reception and inference to overlap:

        Main loop:   recv token → sentence complete → submit → recv token → ...
        Collector:             await inference ──────────────── enqueue audio

    Barge-in: player.interrupt() stops current audio and drains the player
    queue.  Futures already submitted but not yet collected are discarded by
    the collector's session_id check after current_sid_ref is updated.
    """
    session_id: Optional[str] = None
    voice   = TTS.DEFAULT_VOICE
    speed   = TTS.DEFAULT_SPEED
    lang    = TTS.LANG
    buf     = _SentenceBuffer()
    loop    = asyncio.get_running_loop()

    # One-element list so the collector can read the live session_id without
    # a closure-capture issue across coroutine boundaries.
    current_sid_ref: list[Optional[str]] = [None]

    # Ordered queue of (Future, session_id) tuples.
    synth_queue: asyncio.Queue = asyncio.Queue()
    collector = asyncio.create_task(
        _collect_synth(synth_queue, player, current_sid_ref),
        name="SynthCollector",
    )

    _LOG.info("TTS subscriber running")

    try:
        while True:
            try:
                raw = await sub_socket.recv()
                msg = json.loads(raw)
            except Exception:
                _LOG.exception("ZMQ receive error")
                continue

            mtype = msg.get("type")

            if mtype == "session_start":
                old_sid = session_id
                new_sid = msg["session_id"]

                if old_sid and old_sid != new_sid:
                    _LOG.info("Barge-in: old=%s new=%s", old_sid[:8], new_sid[:8])
                    player.interrupt()
                    # Drain unstarted synthesis futures from the old session.
                    # Any already-running futures will be discarded by the
                    # collector once current_sid_ref is updated below.
                    while not synth_queue.empty():
                        try:
                            synth_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    await interrupt_push.send(json.dumps({
                        "type":       "interrupt",
                        "session_id": old_sid,
                        "reason":     "new_speech",
                    }).encode())

                player.reset()
                session_id         = new_sid
                current_sid_ref[0] = new_sid   # collector sees the new session
                voice  = msg.get("voice", TTS.DEFAULT_VOICE)
                speed  = float(msg.get("speed", TTS.DEFAULT_SPEED))
                lang   = msg.get("lang", TTS.LANG)
                buf    = _SentenceBuffer()

            elif mtype == "token":
                if msg.get("session_id") != session_id:
                    continue
                for sentence in buf.feed(msg.get("text", "")):
                    _submit_synth(sentence, voice, speed, lang,
                                  session_id, synthesizer, synth_queue, loop)

            elif mtype == "turn_done":
                if msg.get("session_id") != session_id:
                    continue
                remaining = buf.flush()
                if remaining:
                    _submit_synth(remaining, voice, speed, lang,
                                  session_id, synthesizer, synth_queue, loop)

    finally:
        # Stop the collector gracefully.
        await synth_queue.put(None)
        await collector
