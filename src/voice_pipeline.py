"""
src/voice_pipeline.py — Sage Kaizen Voice Pipeline Orchestrator
================================================================
Connects STT → Expression Engine → TTS → Audio Player into a
complete voice interaction loop.

Two operation modes:

  STANDALONE MODE (default):
    Microphone → transcriber → expression engine → synthesizer → speakers
    Used for development, testing, and Pi agent operation.

  INTEGRATED MODE (with main Sage Kaizen app):
    STT sends transcripts via ZeroMQ PUSH to main app (port 5790).
    TTS receives LLM token stream via ZeroMQ SUB from main app (port 5791).
    Barge-in signals pushed back via ZeroMQ PUSH (port 5792).
    Main app BINDS all sockets; this process CONNECTS.

Usage
-----
Standalone:
    pipeline = VoicePipeline()
    await pipeline.run_standalone()

Integrated:
    pipeline = VoicePipeline(mode="integrated")
    await pipeline.run_integrated()
"""

from __future__ import annotations

import asyncio
from typing import Callable, Optional

from sk_logging import get_logger
from src.config import ZMQ
from src.stt.audio_capture import AudioCapture
from src.stt.transcriber import Transcriber
from src.tts.expression_engine import resolve_and_preprocess
from src.tts.player import AudioPlayer
from src.tts.synthesizer import KokoroSynthesizer

_LOG = get_logger("sage_kaizen.voice.pipeline")


class VoicePipeline:
    """
    Integrated voice pipeline: STT → Expression Engine → TTS.

    Parameters
    ----------
    mode : str
        "standalone" — direct callback chain, no ZeroMQ.
        "integrated" — ZeroMQ integration with main Sage Kaizen app.
    on_transcript : Callable[[str], None], optional
        Callback invoked with each transcribed utterance (standalone mode).
        If None in standalone mode, utterances are spoken via TTS directly.
    default_intent : str
        Default intent used when no routing context is available.
    default_persona_override : str, optional
        Force a specific persona regardless of intent.
    """

    def __init__(
        self,
        mode:                    str = "standalone",
        on_transcript:           Optional[Callable[[str], None]] = None,
        default_intent:          str = "chat",
        default_persona_override: Optional[str] = None,
    ) -> None:
        self._mode       = mode
        self._on_transcript = on_transcript
        self._default_intent  = default_intent
        self._persona_override = default_persona_override

        # Core components
        self._capture     = AudioCapture()
        self._transcriber = Transcriber()
        self._synthesizer = KokoroSynthesizer()
        self._player      = AudioPlayer()

        self._running = False

    # ─────────────────────────────────────────────────────────
    # Initialization
    # ─────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Load models and start audio player."""
        _LOG.info("Initializing voice pipeline (mode=%s)...", self._mode)
        loop = asyncio.get_running_loop()

        # Load models in parallel
        await asyncio.gather(
            loop.run_in_executor(None, self._transcriber.initialize),
            loop.run_in_executor(None, self._synthesizer.initialize),
        )

        await self._player.start()
        _LOG.info("Voice pipeline initialized")

    # ─────────────────────────────────────────────────────────
    # Standalone mode
    # ─────────────────────────────────────────────────────────

    async def run_standalone(self) -> None:
        """
        Run the full voice loop in standalone mode.
        Microphone → transcribe → speak response.
        Press Ctrl+C to stop.
        """
        try:
            await self.initialize()
            self._running = True

            # Announce readiness
            await self.speak_text(
                "Sage Kaizen voice pipeline is ready. I'm listening.",
                intent="chat",
            )

            self._capture.start()
            _LOG.info("Listening... (Ctrl+C to stop)")

            while self._running:
                # Block until an utterance arrives (30ms poll interval)
                audio = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: self._capture.get_utterance(timeout=0.5)
                )
                if audio is None:
                    continue

                # Transcribe
                text = await self._transcriber.transcribe_async(audio)
                if not text:
                    continue

                _LOG.info("You said: %r", text)

                # Route to callback or direct TTS
                if self._on_transcript:
                    self._on_transcript(text)
                else:
                    # Echo-mode: speak what was heard (useful for testing)
                    await self.speak_text(text, intent=self._default_intent)

        except asyncio.CancelledError:
            _LOG.info("Pipeline cancelled")
        except KeyboardInterrupt:
            _LOG.info("Pipeline stopped by user")
        finally:
            await self.shutdown()

    # ─────────────────────────────────────────────────────────
    # Integrated mode (ZeroMQ)
    # ─────────────────────────────────────────────────────────

    async def run_integrated(self) -> None:
        """
        Run in integrated mode with the main Sage Kaizen app via ZeroMQ.

        STT: transcripts pushed to main app
        TTS: tokens received from main app → synthesized → played
        Barge-in: interrupt signals pushed back to main app

        Requires pyzmq to be installed.
        """
        try:
            import zmq
            import zmq.asyncio
        except ImportError:
            raise RuntimeError(
                "pyzmq is required for integrated mode. "
                "Run: pip install pyzmq"
            )

        transcript_push = None
        token_sub = None
        interrupt_push = None
        ctx = None
        try:
            await self.initialize()
            self._running = True

            ctx = zmq.asyncio.Context.instance()

            # STT output socket — PUSH to main app
            transcript_push = ctx.socket(zmq.PUSH)
            transcript_push.connect(ZMQ.TRANSCRIPT_BUS)
            _LOG.info("STT transcript bus connected: %s", ZMQ.TRANSCRIPT_BUS)

            # TTS input socket — SUB to receive token stream from main app
            token_sub = ctx.socket(zmq.SUB)
            token_sub.connect(ZMQ.TOKEN_BUS)
            token_sub.setsockopt(zmq.SUBSCRIBE, b"")
            _LOG.info("TTS token bus connected: %s", ZMQ.TOKEN_BUS)

            # Barge-in output socket — PUSH interrupt signals to main app
            interrupt_push = ctx.socket(zmq.PUSH)
            interrupt_push.connect(ZMQ.INTERRUPT_BUS)
            _LOG.info("Interrupt bus connected: %s", ZMQ.INTERRUPT_BUS)

            self._capture.start()
            _LOG.info("Integrated voice pipeline running")

            # Import here to avoid circular import issues in standalone mode
            from src._zmq_handlers import run_stt_pusher, run_tts_subscriber

            await asyncio.gather(
                run_stt_pusher(
                    self._capture, self._transcriber, transcript_push
                ),
                run_tts_subscriber(
                    token_sub, interrupt_push, self._synthesizer, self._player
                ),
            )
        except asyncio.CancelledError:
            _LOG.info("Integrated pipeline cancelled")
        finally:
            if transcript_push:
                transcript_push.close()
            if token_sub:
                token_sub.close()
            if interrupt_push:
                interrupt_push.close()
            if ctx:
                ctx.term()
            await self.shutdown()

    # ─────────────────────────────────────────────────────────
    # Public TTS helper
    # ─────────────────────────────────────────────────────────

    async def speak_text(
        self,
        text:             str,
        intent:           str = "chat",
        brain:            Optional[str] = None,
        grade_level:      Optional[int] = None,
    ) -> None:
        """
        Synthesize and speak text through the expression engine.
        Returns after all audio has been enqueued (non-blocking playback).
        """
        processed, params = resolve_and_preprocess(
            text,
            intent=intent,
            brain=brain,
            grade_level=grade_level,
            override_persona=self._persona_override,
        )
        _LOG.debug("Speaking [%s] speed=%.2f: %r",
                  params.persona.value, params.speed, processed[:60])

        async for samples, sr in self._synthesizer.stream_async(
            processed, params.voice, params.speed, params.lang
        ):
            self._player.enqueue(samples, sr)

    # ─────────────────────────────────────────────────────────
    # Shutdown
    # ─────────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """Clean shutdown of all components."""
        self._running = False
        self._capture.stop()
        await self._player.stop()
        _LOG.info("Voice pipeline shut down")
