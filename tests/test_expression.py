"""tests/test_expression.py - Expression engine unit tests (no hardware)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.tts.expression_engine import (
    resolve, resolve_and_preprocess, Persona, preprocess_text, _PERSONA_PROFILES
)

def test_intent_routing():
    assert resolve("creative").persona       == Persona.NARRATOR
    assert resolve("device_control").persona  == Persona.QUICK
    assert resolve("chat").persona            == Persona.CHAT
    assert resolve("unknown_intent").persona  == Persona.CHAT
    print("  OK intent routing")

def test_brain_override():
    p = resolve("chat", brain="architect_brain")
    assert p.persona == Persona.NARRATOR, f"Expected NARRATOR, got {p.persona}"
    print("  OK brain override")

def test_grade_level():
    assert resolve("tutor", grade_level=3).persona  == Persona.TEACHER
    assert resolve("tutor", grade_level=7).persona  == Persona.MENTOR
    assert resolve("tutor", grade_level=10).persona == Persona.MENTOR
    print("  OK grade level routing")

def test_persona_override():
    p = resolve("device_control", override_persona="narrator")
    assert p.persona == Persona.NARRATOR
    print("  OK persona override")

def test_text_preprocessing():
    from src.tts.expression_engine import _PERSONA_PROFILES, Persona
    params = _PERSONA_PROFILES[Persona.NARRATOR]
    text = "The RAG pipeline uses GPU acceleration — check the UI."
    result = preprocess_text(text, params)
    assert "Retrieval Augmented Generation" in result
    assert "G-P-U" in result
    assert "user interface" in result
    assert result.endswith(".")
    print("  OK text preprocessing")

def test_all_persona_params_valid():
    for persona, params in _PERSONA_PROFILES.items():
        assert params.voice, f"{persona} missing voice"
        assert 0.5 <= params.speed <= 2.0, f"{persona} speed out of range"
        assert params.lang in ("en-us", "en-gb"), f"{persona} invalid lang"
    print("  OK all persona params valid")

def test_narrator_voice_is_am_fenrir():
    p = resolve("creative")
    assert p.voice == "am_fenrir", f"Expected am_fenrir, got {p.voice}"
    print("  OK narrator voice = am_fenrir")

def test_quick_voice_is_am_onyx():
    p = resolve("device_control")
    assert p.voice == "am_onyx", f"Expected am_onyx, got {p.voice}"
    print("  OK quick voice = am_onyx")

if __name__ == "__main__":
    print("\n[ Expression Engine Unit Tests ]")
    test_intent_routing()
    test_brain_override()
    test_grade_level()
    test_persona_override()
    test_text_preprocessing()
    test_all_persona_params_valid()
    test_narrator_voice_is_am_fenrir()
    test_quick_voice_is_am_onyx()
    print("\n=== All expression tests passed ===")
