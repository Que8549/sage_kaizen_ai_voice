"""
scripts/run_pipeline.py — Voice Pipeline Entry Point
=====================================================
Starts the voice pipeline in either standalone or integrated mode.

Usage:
    # Standalone (echo mode — no main app required):
    python scripts/run_pipeline.py

    # Integrated (ZMQ — requires main Sage Kaizen app to be running):
    python scripts/run_pipeline.py --mode integrated

Standalone mode:
  Loads STT + TTS models, announces readiness, listens for speech, and
  echoes heard text back via TTS.  Useful for testing without the main app.

Integrated mode:
  Connects to the main app via ZeroMQ (ports 5790/5791/5792).
  Speaks "Sage Kaizen online." when models are loaded, then forwards
  voice transcripts to the main app and speaks LLM responses aloud.

Logging: all output goes to logs/sage_kaizen_voice.log via sk_logging.
stdout/stderr are intentionally kept clean — no shell redirection needed.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sk_logging import get_logger
from src.voice_pipeline import VoicePipeline

_LOG = get_logger("sage_kaizen.voice.runner")


async def run(persona: str | None, verbose: bool, mode: str) -> None:
    """Load models and run the voice pipeline in the requested mode."""
    if verbose:
        import logging
        get_logger("sage_kaizen.voice.pipeline").setLevel(logging.DEBUG)
        get_logger("sage_kaizen.voice.tts.synth").setLevel(logging.DEBUG)
        get_logger("sage_kaizen.voice.stt.transcriber").setLevel(logging.DEBUG)

    pipeline = VoicePipeline(
        mode=mode,
        default_intent="chat",
        default_persona_override=persona,
    )

    _LOG.info("run_pipeline: mode=%s persona=%s verbose=%s", mode, persona or "auto", verbose)
    print("   Loading STT + TTS models — this takes 30-60 s on first run...")
    sys.stdout.flush()

    try:
        if mode == "integrated":
            await pipeline.run_integrated()
        else:
            await pipeline.run_standalone()
    except Exception as exc:
        _LOG.exception("Pipeline crashed: %s", exc)
        print(f"\n[ERROR] Pipeline crashed: {exc}", file=sys.stderr)
        print("[ERROR] Full traceback written to logs/sage_kaizen_voice.log", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sage Kaizen AI Voice — voice pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["standalone", "integrated"],
        default="standalone",
        help=(
            "Operation mode: "
            "standalone = echo mode, no main app required; "
            "integrated = ZMQ, requires main Sage Kaizen app to be running"
        ),
    )
    parser.add_argument(
        "--persona",
        choices=["narrator", "mentor", "teacher", "chat", "quick"],
        default=None,
        help="Force a specific TTS persona (default: auto from intent)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    args = parser.parse_args()

    print("\nSage Kaizen AI Voice — Starting...")
    print(f"   Mode    : {args.mode}")
    print(f"   Persona : {args.persona or 'auto'}")
    print(f"   Log     : logs/sage_kaizen_voice.log")
    print("   Stop    : Ctrl+C\n")
    sys.stdout.flush()

    try:
        asyncio.run(run(persona=args.persona, verbose=args.verbose, mode=args.mode))
    except KeyboardInterrupt:
        _LOG.info("run_pipeline: stopped by user (Ctrl+C)")
        print("\nPipeline stopped.")


if __name__ == "__main__":
    main()
