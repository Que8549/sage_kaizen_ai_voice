"""
scripts/verify_setup.py — Pre-flight Setup Verification
=========================================================
Run this before starting the voice pipeline to confirm:
  - Python version is correct (3.11.x)
  - All required packages are importable
  - All model files exist on E:\\ drive
  - Audio devices are available
  - espeak-ng is installed and in PATH

Usage:
    python scripts/verify_setup.py
"""

import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = "✅" if ok else "❌"
    msg    = f"  {status}  {label}"
    if detail:
        msg += f"\n       {detail}"
    print(msg)
    return ok


def main() -> int:
    print("\n════════════════════════════════════════════")
    print("  Sage Kaizen AI Voice — Setup Verification ")
    print("════════════════════════════════════════════\n")

    failures = 0

    # ── Python version ──────────────────────────────────────
    print("[ Python ]")
    v = sys.version_info
    py_ok = v.major == 3 and v.minor == 11
    if not check(
        f"Python 3.11.x required — found {v.major}.{v.minor}.{v.micro}",
        py_ok,
        detail="" if py_ok else "ctranslate2 has no wheels for 3.12+. Use Python 3.11.",
    ):
        failures += 1
    print()

    # ── Required packages ───────────────────────────────────
    print("[ Packages ]")
    packages = [
        ("faster_whisper",  "faster-whisper >= 1.2.1"),
        ("misaki",          "misaki[en] >= 0.9.0"),
        ("onnxruntime",     "onnxruntime >= 1.20.0"),
        ("sounddevice",     "sounddevice >= 0.5.0"),
        ("soundfile",       "soundfile >= 0.12.1"),
        ("pyaudio",         "PyAudio >= 0.2.14"),
        ("numpy",           "numpy >= 1.26.0"),
        ("zmq",             "pyzmq >= 25.1.0"),
        ("yaml",            "PyYAML >= 6.0.0"),
    ]
    for mod, label in packages:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "unknown")
            if not check(f"{label}  (installed: {ver})", True):
                failures += 1
        except ImportError:
            if not check(label, False, detail=f"pip install {label.split()[0].lower()}"):
                failures += 1
    print()

    # ── Model files ─────────────────────────────────────────
    print("[ Model Files on E:\\ ]")
    from src.config import PATHS
    model_checks = [
        (PATHS.STT_MODEL_DIR,  "STT: distil-large-v3-ct2 directory"),
        (PATHS.TTS_ONNX_MODEL, "TTS: model_q8f16.onnx (Kokoro-82M-ONNX)"),
        (PATHS.TTS_VOICES_DIR, "TTS: voices directory"),
    ]
    for path, label in model_checks:
        exists = path.exists()
        size   = f" ({path.stat().st_size // 1_048_576} MB)" if exists and path.is_file() else ""
        if not check(f"{label}{size}", exists, detail=str(path) if not exists else ""):
            failures += 1
    # Check individual voice files required by Sage personas
    errors = PATHS.validate()
    for err in errors:
        if not check(err, False):
            failures += 1
    print()

    # ── Audio devices ───────────────────────────────────────
    print("[ Audio Devices ]")
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devs  = [d for d in devices if d["max_input_channels"] > 0]
        output_devs = [d for d in devices if d["max_output_channels"] > 0]
        check(f"Input devices:  {len(input_devs)} found", len(input_devs) > 0)
        check(f"Output devices: {len(output_devs)} found", len(output_devs) > 0)
        if input_devs:
            default_in = sd.query_devices(kind="input")
            print(f"         Default input:  {default_in['name']}")
        if output_devs:
            default_out = sd.query_devices(kind="output")
            print(f"         Default output: {default_out['name']}")
    except Exception as e:
        if not check("sounddevice query", False, detail=str(e)):
            failures += 1
    print()

    # ── espeak-ng ───────────────────────────────────────────
    print("[ espeak-ng (phoneme fallback) ]")
    try:
        result = subprocess.run(
            ["espeak-ng", "--version"],
            capture_output=True, text=True, timeout=5
        )
        ok = result.returncode == 0
        detail = result.stdout.strip().split("\n")[0] if ok else (
            "Download from https://github.com/espeak-ng/espeak-ng/releases\n"
            "       Add install directory to Windows PATH"
        )
        if not check("espeak-ng installed and in PATH", ok, detail=detail):
            failures += 1
    except FileNotFoundError:
        if not check("espeak-ng installed and in PATH", False,
                     detail="Install MSI from github.com/espeak-ng/espeak-ng/releases"):
            failures += 1
    print()

    # ── Quick smoke test ─────────────────────────────────────
    print("[ Quick Smoke Tests ]")
    try:
        from src.tts.expression_engine import resolve_and_preprocess
        text, params = resolve_and_preprocess(
            "Sage Kaizen is ready.", intent="chat"
        )
        check(
            f"Expression engine: [{params.persona.value}] speed={params.speed}",
            bool(text and params),
        )
    except Exception as e:
        if not check("Expression engine import", False, detail=str(e)):
            failures += 1

    print()

    # ── Summary ──────────────────────────────────────────────
    print("════════════════════════════════════════════")
    if failures == 0:
        print("  ✅  All checks passed. Voice pipeline is ready.")
        print("  Run:  python scripts/run_pipeline.py")
    else:
        print(f"  ❌  {failures} check(s) failed. Resolve above before running.")
    print("════════════════════════════════════════════\n")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
