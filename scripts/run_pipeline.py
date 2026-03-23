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
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import LOGGING
from src.voice_pipeline import VoicePipeline


def setup_logging(verbose: bool = False) -> None:
    LOGGING.LOG_DIR.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(str(LOGGING.MAIN_LOG), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


async def run(persona: str | None = None) -> None:
    pipeline = VoicePipeline(
        mode="standalone",
        default_intent="chat",
        default_persona_override=persona,
    )
    await pipeline.run_standalone()


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
        help="Enable debug logging",
    )
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    print("\n🎙️  Sage Kaizen AI Voice — Starting...")
    print(f"   Persona: {args.persona or 'auto'}")
    print("   Press Ctrl+C to stop\n")

    try:
        asyncio.run(run(persona=args.persona))
    except KeyboardInterrupt:
        print("\n✅ Pipeline stopped.")


if __name__ == "__main__":
    main()
