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


async def run_tts_subscriber(
    sub_socket,
    interrupt_push,
    synthesizer: KokoroSynthesizer,
    player:      AudioPlayer,
) -> None:
    """Receive LLM token stream, synthesize, play audio, handle barge-in."""
    session_id:   Optional[str] = None
    voice   = TTS.DEFAULT_VOICE
    speed   = TTS.DEFAULT_SPEED
    lang    = TTS.LANG
    buf     = _SentenceBuffer()
    loop    = asyncio.get_running_loop()

    _LOG.info("TTS subscriber running")

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
                await interrupt_push.send(json.dumps({
                    "type":       "interrupt",
                    "session_id": old_sid,
                    "reason":     "new_speech",
                }).encode())

            player.reset()
            session_id = new_sid
            voice  = msg.get("voice", TTS.DEFAULT_VOICE)
            speed  = float(msg.get("speed", TTS.DEFAULT_SPEED))
            lang   = msg.get("lang", TTS.LANG)
            buf    = _SentenceBuffer()

        elif mtype == "token":
            if msg.get("session_id") != session_id:
                continue
            for sentence in buf.feed(msg.get("text", "")):
                await _synth_and_enqueue(sentence, voice, speed, lang,
                                         synthesizer, player, loop)

        elif mtype == "turn_done":
            if msg.get("session_id") != session_id:
                continue
            remaining = buf.flush()
            if remaining:
                await _synth_and_enqueue(remaining, voice, speed, lang,
                                         synthesizer, player, loop)


async def _synth_and_enqueue(
    text:        str,
    voice:       str,
    speed:       float,
    lang:        str,
    synthesizer: KokoroSynthesizer,
    player:      AudioPlayer,
    loop:        asyncio.AbstractEventLoop,
) -> None:
    if not text.strip():
        return
    try:
        samples, sr = await loop.run_in_executor(
            None, lambda: synthesizer._synth_one(text, voice, speed, lang)
        )
        player.enqueue(np.array(samples, dtype=np.float32), int(sr))
    except Exception:
        _LOG.exception("Synthesis error: %r", text[:50])
