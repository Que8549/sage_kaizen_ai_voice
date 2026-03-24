"""
scripts/run_pipeline.py — Standalone Voice Pipeline Entry Point
================================================================
Starts the full voice pipeline in standalone mode for development
and testing. No ZeroMQ, no main Sage Kaizen app required.

Usage:
    python scripts/run_pipeline.py [--persona narrator|mentor|teacher|chat|quick]

The pipeline will:
  1. Load STT and TTS models from E:\\ drive
  2. Announce readiness with a test phrase
  3. Listen continuously for speech
  4. Echo what was heard back through TTS (for testing)
  5. Stop cleanly on Ctrl+C

Logging: all output goes to logs/sage_kaizen.log via sk_logging.
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


async def run(persona: str | None, verbose: bool) -> None:
    """Load models and run the standalone voice loop."""
    if verbose:
        import logging
        get_logger("sage_kaizen.voice.pipeline").setLevel(logging.DEBUG)
        get_logger("sage_kaizen.voice.tts.synth").setLevel(logging.DEBUG)
        get_logger("sage_kaizen.voice.stt.transcriber").setLevel(logging.DEBUG)

    pipeline = VoicePipeline(
        mode="standalone",
        default_intent="chat",
        default_persona_override=persona,
    )

    _LOG.info("run_pipeline: persona=%s verbose=%s", persona or "auto", verbose)
    print("   Loading STT + TTS models — this takes 30-60 s on first run...")
    sys.stdout.flush()

    try:
        await pipeline.run_standalone()
    except Exception as exc:
        _LOG.exception("Pipeline crashed: %s", exc)
        print(f"\n[ERROR] Pipeline crashed: {exc}", file=sys.stderr)
        print("[ERROR] Full traceback written to logs/sage_kaizen.log", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sage Kaizen AI Voice — standalone pipeline"
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
    print(f"   Persona : {args.persona or 'auto'}")
    print(f"   Log     : logs/sage_kaizen.log")
    print("   Stop    : Ctrl+C\n")
    sys.stdout.flush()

    try:
        asyncio.run(run(persona=args.persona, verbose=args.verbose))
    except KeyboardInterrupt:
        _LOG.info("run_pipeline: stopped by user (Ctrl+C)")
        print("\nPipeline stopped.")


if __name__ == "__main__":
    main()
