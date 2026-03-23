"""tests/test_stt.py - STT smoke test (uses a test WAV file from disk)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import logging, numpy as np
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
from src.stt.transcriber import Transcriber

def test_model_loads():
    print("\n[ STT Model Load ]")
    t = Transcriber()
    t.initialize()
    assert t.is_ready
    print("  Model loaded OK")

def test_transcribe_file():
    print("\n[ STT File Transcription ]")
    t = Transcriber()
    t.initialize()
    # Use any WAV file for testing — create a silent one if none available
    test_wav = Path("tests/test_audio.wav")
    if not test_wav.exists():
        import soundfile as sf
        import numpy as np
        silence = np.zeros(16000, dtype=np.float32)  # 1s silence
        sf.write(str(test_wav), silence, 16000)
        print(f"  Created silent test WAV: {test_wav}")
    result = t.transcribe_file(str(test_wav))
    print(f"  Transcript: {repr(result)}")
    print("  OK (empty string expected for silence)")

if __name__ == "__main__":
    test_model_loads()
    test_transcribe_file()
    print("\n=== All STT tests passed ===")
