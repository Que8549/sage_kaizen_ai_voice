"""tests/test_tts.py - TTS smoke test (no microphone required)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import logging, numpy as np, soundfile as sf
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
from src.config import PATHS
from src.tts.expression_engine import resolve_and_preprocess
from src.tts.synthesizer import KokoroSynthesizer, split_into_chunks

def test_sentence_chunker():
    print("\n[ Sentence Chunker ]")
    text = (
        "There are more stars in the observable universe than grains of sand on Earth. "
        "Each one burns alone. And yet, from here, they are beautiful."
    )
    chunks = split_into_chunks(text, min_words=4)
    for i, c in enumerate(chunks):
        print(f"  Chunk {i+1}: {c[:60]}")
    assert len(chunks) >= 2
    print("  OK")

def test_synthesis_narrator():
    print("\n[ Narrator synthesis (am_fenrir) ]")
    synth = KokoroSynthesizer()
    synth.initialize()
    text = "There are more stars in the universe than grains of sand on every beach on Earth."
    processed, params = resolve_and_preprocess(text, intent="creative")
    print(f"  Persona: {params.persona.value}  voice: {params.voice}  speed: {params.speed}")
    chunks = list(synth.stream(processed, params.voice, params.speed, params.lang))
    assert chunks, "No audio produced"
    combined = np.concatenate([a for a,_ in chunks]).astype(np.float32)
    sr = chunks[0][1]
    PATHS.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sf.write(str(PATHS.OUTPUT_DIR / "test_narrator.wav"), combined, sr)
    print(f"  Saved test_narrator.wav  ({len(combined)/sr:.2f}s)")
    print("  OK")

def test_synthesis_quick():
    print("\n[ Quick synthesis (am_onyx) ]")
    synth = KokoroSynthesizer()
    synth.initialize()
    processed, params = resolve_and_preprocess("Setting LED mode to cosmic.", intent="device_control")
    print(f"  Persona: {params.persona.value}  voice: {params.voice}  speed: {params.speed}")
    chunks = list(synth.stream(processed, params.voice, params.speed, params.lang))
    assert chunks
    combined = np.concatenate([a for a,_ in chunks]).astype(np.float32)
    sr = chunks[0][1]
    PATHS.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sf.write(str(PATHS.OUTPUT_DIR / "test_quick.wav"), combined, sr)
    print(f"  Saved test_quick.wav  ({len(combined)/sr:.2f}s)")
    print("  OK")

def test_all_personas():
    print("\n[ All persona voices ]")
    synth = KokoroSynthesizer()
    synth.initialize()
    for intent in ["creative", "architecture", "tutor", "chat", "device_control"]:
        _, params = resolve_and_preprocess("Hello.", intent=intent, grade_level=6 if intent=="tutor" else None)
        chunks = list(synth.stream("Hello.", params.voice, params.speed, params.lang))
        assert chunks, f"No audio for intent={intent}"
        print(f"  [{params.persona.value:10s}] voice={params.voice:<12} speed={params.speed:.2f}  OK")

if __name__ == "__main__":
    test_sentence_chunker()
    test_synthesis_narrator()
    test_synthesis_quick()
    test_all_personas()
    print("\n=== All TTS tests passed ===")
