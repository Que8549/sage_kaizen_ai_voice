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

TTS gate / command-only barge-in:
  While TTS is playing (player.is_playing) or within _TTS_DECAY_SECS of
  stopping, run_stt_pusher discards all transcripts EXCEPT new-chat commands.
  On new-chat:
    1. player.interrupt()                 — stop audio immediately
    2. capture.mute() + flush_queue()     — clear buffered echo utterances
    3. current_sid_ref[0] = None          — invalidate stale synthesis futures
    4. Forward command to main app        — UI resets immediately
    5. player.reset() + _speak_local()    — play "Starting new chat."
    6. asyncio.sleep(_TTS_DECAY_SECS)     — 750 ms room-decay
    7. capture.unmute()                   — resume normal operation

  Shared state between the two coroutines:
    current_sid_ref : list[Optional[str]]  — single-element list; owned by
        run_tts_subscriber but readable/writable by run_stt_pusher to
        invalidate stale sessions on new-chat.
    synth_queue     : asyncio.Queue        — shared between run_tts_subscriber
        and its _collect_synth task; created externally in run_integrated().
"""

from __future__ import annotations

import asyncio
import json
import re
import time
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

# Mirrors the regex in ui_streamlit_server.py — must stay in sync.
_NEW_CHAT_RE = re.compile(
    r"^\s*"
    r"(?:hey\s+sage[,\s]+|ok(?:ay)?\s+sage[,\s]+|please\s+)?"
    r"(?:"
    r"(?:start\s+(?:a\s+)?)?new\s+chat"
    r"|(?:start\s+(?:a\s+)?)?new\s+conversation"
    r"|start\s+over"
    r"|clear\s+(?:the\s+)?(?:chat|conversation|history)"
    r"|reset\s+(?:the\s+)?(?:chat|conversation)"
    r")"
    r"\s*[.!?]?\s*$",
    re.IGNORECASE,
)

# Room-decay pause (seconds) after TTS ends before re-enabling the microphone.
# Also used as the post-confirmation unmute delay (matches user's 750 ms requirement).
_TTS_DECAY_SECS = 0.75

# Text spoken as confirmation when a new-chat command is executed.
_NEW_CHAT_CONFIRM_TEXT = "Starting new chat."


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
# Local TTS helper (confirmation audio, bypasses ZMQ)
# ─────────────────────────────────────────────────────────

async def _speak_local(
    text:        str,
    synthesizer: KokoroSynthesizer,
    player:      AudioPlayer,
) -> None:
    """Synthesize and play text through the local player, then await completion.

    Used for in-process confirmations (e.g. "Starting new chat.") that must
    play immediately without going through the ZMQ token bus.  The caller
    must call player.reset() before this so the interrupt flag is clear.
    """
    loop = asyncio.get_running_loop()
    try:
        samples, sr = await loop.run_in_executor(
            None,
            lambda: synthesizer.synth_one(
                text, TTS.DEFAULT_VOICE, TTS.DEFAULT_SPEED, TTS.LANG
            ),
        )
        if samples is not None and len(samples) > 0:
            player.enqueue(np.array(samples, dtype=np.float32), int(sr))
            await player.drain(timeout=10.0)
    except Exception:
        _LOG.exception("_speak_local error for %r", text[:40])


# ─────────────────────────────────────────────────────────
# STT pusher
# ─────────────────────────────────────────────────────────

async def run_stt_pusher(
    capture:         AudioCapture,
    transcriber:     Transcriber,
    push_socket,
    player:          AudioPlayer,
    synthesizer:     KokoroSynthesizer,
    current_sid_ref: list[Optional[str]],
) -> None:
    """Capture audio, transcribe, and push transcripts to the main app.

    TTS gate (command-only barge-in)
    ---------------------------------
    While TTS is playing or within _TTS_DECAY_SECS of stopping, ALL
    transcripts are silently discarded EXCEPT new-chat commands.  This
    prevents the Shure MV7+'s cardioid capsule from picking up speaker
    echo and forwarding it to the LLM as user input.

    New-chat command sequence
    -------------------------
    1. player.interrupt()              — stop TTS audio immediately
    2. capture.mute() + flush_queue()  — block new echo utterances; discard
                                         any already queued
    3. current_sid_ref[0] = None       — invalidate pending synthesis futures
                                         so _collect_synth discards them
    4. Forward command to main app     — UI resets immediately
    5. player.reset() + _speak_local() — play "Starting new chat."
    6. asyncio.sleep(_TTS_DECAY_SECS)  — 750 ms room-decay
    7. capture.unmute()                — resume normal operation
    """
    loop = asyncio.get_running_loop()
    _LOG.info("STT pusher running")

    _tts_was_playing: bool  = False
    _last_tts_end:    float = 0.0      # monotonic timestamp of last TTS→idle edge

    while True:
        audio = await loop.run_in_executor(
            None, lambda: capture.get_utterance(timeout=0.5)
        )

        # ── Update TTS-active tracking on every iteration (even timeouts) ──
        _tts_now = player.is_playing
        if _tts_was_playing and not _tts_now:
            _last_tts_end = time.monotonic()
        _tts_was_playing = _tts_now

        if audio is None:
            continue

        text = await transcriber.transcribe_async(audio)
        if not text:
            continue

        # Refresh after the blocking transcription step
        _tts_now = player.is_playing
        if _tts_was_playing and not _tts_now:
            _last_tts_end = time.monotonic()
        _tts_was_playing = _tts_now

        tts_active = _tts_now or (time.monotonic() - _last_tts_end < _TTS_DECAY_SECS)

        # ── New-chat command — handled regardless of TTS state ─────────────
        if _NEW_CHAT_RE.match(text):
            _LOG.info("New-chat command — stopping TTS, playing confirmation")

            # 1. Stop audio immediately
            player.interrupt()

            # 2. Mute first (closes the emit gate), then flush the queue so
            #    any echo utterances already buffered are discarded.
            capture.mute()
            capture.flush_queue()

            # 3. Invalidate current session so _collect_synth discards any
            #    in-flight synthesis futures from the old turn.
            current_sid_ref[0] = None

            # 4. Notify main app — UI resets right away
            await push_socket.send(json.dumps({
                "type":       "transcript",
                "text":       text,
                "session_id": str(uuid.uuid4()),
            }).encode())
            _LOG.info("New-chat command forwarded to main app")

            # 5. Play confirmation (reset clears the interrupt flag first)
            player.reset()
            await _speak_local(_NEW_CHAT_CONFIRM_TEXT, synthesizer, player)

            # 6. Room-decay pause — let the confirmation audio die in the room
            await asyncio.sleep(_TTS_DECAY_SECS)

            # 7. Re-enable microphone
            capture.unmute()
            _LOG.info("Microphone re-enabled after new-chat confirmation")

            # Reset TTS tracking so the next turn starts clean
            _tts_was_playing = False
            _last_tts_end    = 0.0
            continue

        # ── TTS active — discard everything that isn't a new-chat command ──
        if tts_active:
            _LOG.debug("TTS active — discarding transcript: %r", text[:40])
            continue

        # ── Normal operation — forward transcript to main app ─────────────
        await push_socket.send(json.dumps({
            "type":       "transcript",
            "text":       text,
            "session_id": str(uuid.uuid4()),
        }).encode())
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
    synthesizer:     KokoroSynthesizer,
    player:          AudioPlayer,
    current_sid_ref: list[Optional[str]],
    synth_queue:     asyncio.Queue,
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

    current_sid_ref and synth_queue are created externally (in run_integrated)
    and shared with run_stt_pusher so that a new-chat command can invalidate
    the current session and discard stale synthesis futures.
    """
    session_id: Optional[str] = None
    voice   = TTS.DEFAULT_VOICE
    speed   = TTS.DEFAULT_SPEED
    lang    = TTS.LANG
    buf     = _SentenceBuffer()
    loop    = asyncio.get_running_loop()

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
