"""
src/tts/expression_engine.py — Persona-Aware Speech Expression Engine
=======================================================================
Maps router intent + grade level + brain context to concrete Kokoro
voice parameters.

Target narrator character: deep, warm, resonant African-American male voice.
Aesthetic reference: am_onyx — modeled on OpenAI's Onyx (98 Hz, gravelly,
"rich and sophisticated", modern coolness, conversational authority).

Primary voice: am_onyx (American male, deep and resonant)
  - Available in onnx-community/Kokoro-82M-v1.0-ONNX voices/
  - Single stable voice, no blending, 100% API-stable
  - Each persona uses am_onyx with different speed for variety

Design principles:
  - Pure functions only (no I/O, no side effects)
  - Fully deterministic given same inputs
  - All routing logic in one place for easy tuning
  - Parameters feed directly into KokoroSynthesizer.create()
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────────────────────
# Voice constants — all single-voice, API-stable
# am_onyx:   deep, resonant, gravelly African-American male — primary Sage voice
#            Modeled on OpenAI's Onyx (98 Hz, "rich and sophisticated",
#            warm modern authority). Replaces am_fenrir as the narrator.
# am_michael: warm, clear American male — teacher persona for younger learners
# am_echo:   resonant, clear American male — device-control ACKs
# All present in onnx-community/Kokoro-82M-v1.0-ONNX voices/
# ─────────────────────────────────────────────────────────────

VOICE_NARRATOR = "am_onyx"    # deep, warm, gravelly — African-American male aesthetic
VOICE_MENTOR   = "am_onyx"    # same foundation, slightly faster
VOICE_TEACHER  = "am_michael" # warmer, clearer — better for young children
VOICE_CHAT     = "am_onyx"    # natural baseline
VOICE_QUICK    = "am_echo"    # resonant and clear — device control ACKs


class Persona(str, Enum):
    NARRATOR = "narrator"   # creative, philosophy, astronomy, long-form
    MENTOR   = "mentor"     # 6–9 tutor, research, code, architecture
    TEACHER  = "teacher"    # K–5 tutor, step-by-step explainers
    CHAT     = "chat"       # general conversational exchange
    QUICK    = "quick"      # device_control, status, one-liners


@dataclass(frozen=True)
class SpeechParams:
    """Fully resolved parameters for a single TTS request."""
    voice:                 str    # Kokoro voice name (e.g. "am_fenrir")
    speed:                 float  # 0.75–1.25; 1.0 = baseline
    lang:                  str    # "en-us" | "en-gb"
    persona:               Persona
    expand_abbreviations:  bool = True
    pause_after_colon:     bool = True


_PERSONA_PROFILES: dict[Persona, SpeechParams] = {
    Persona.NARRATOR: SpeechParams(
        voice=VOICE_NARRATOR, speed=0.87, lang="en-us",
        persona=Persona.NARRATOR,
        expand_abbreviations=True, pause_after_colon=True,
    ),
    Persona.MENTOR: SpeechParams(
        voice=VOICE_MENTOR, speed=0.92, lang="en-us",
        persona=Persona.MENTOR,
        expand_abbreviations=True, pause_after_colon=True,
    ),
    Persona.TEACHER: SpeechParams(
        voice=VOICE_TEACHER, speed=0.95, lang="en-us",
        persona=Persona.TEACHER,
        expand_abbreviations=True, pause_after_colon=True,
    ),
    Persona.CHAT: SpeechParams(
        voice=VOICE_CHAT, speed=1.00, lang="en-us",
        persona=Persona.CHAT,
        expand_abbreviations=True, pause_after_colon=False,
    ),
    Persona.QUICK: SpeechParams(
        voice=VOICE_QUICK, speed=1.05, lang="en-us",
        persona=Persona.QUICK,
        expand_abbreviations=False, pause_after_colon=False,
    ),
}

_INTENT_TO_PERSONA: dict[str, Persona] = {
    "creative":       Persona.NARRATOR,
    "philosophy":     Persona.NARRATOR,
    "astronomy":      Persona.NARRATOR,
    "research":       Persona.NARRATOR,
    "long_form":      Persona.NARRATOR,
    "storytelling":   Persona.NARRATOR,
    "architecture":   Persona.MENTOR,
    "code":           Persona.MENTOR,
    "tutor_6":        Persona.MENTOR,
    "tutor_9":        Persona.MENTOR,
    "tutor_1":        Persona.TEACHER,
    "tutor_5":        Persona.TEACHER,
    "explain_simple": Persona.TEACHER,
    "chat":           Persona.CHAT,
    "general":        Persona.CHAT,
    "question":       Persona.CHAT,
    "device_control": Persona.QUICK,
    "led_control":    Persona.QUICK,
    "status":         Persona.QUICK,
    "system":         Persona.QUICK,
}

_BRAIN_PERSONA_OVERRIDE: dict[str, Persona] = {
    "architect_brain": Persona.NARRATOR,
}

_ABBREVIATIONS: dict[str, str] = {
    r"\bRAG\b":   "Retrieval Augmented Generation",
    r"\bLLM\b":   "Large Language Model",
    r"\bSTT\b":   "Speech to Text",
    r"\bTTS\b":   "Text to Speech",
    r"\bGPU\b":   "G-P-U",
    r"\bCPU\b":   "C-P-U",
    r"\bAPI\b":   "A-P-I",
    r"\bUI\b":    "user interface",
    r"\bSQL\b":   "S-Q-L",
    r"\bYAML\b":  "YAML",
    r"\bJSON\b":  "Jason",
    r"\bZMQ\b":   "zero-M-Q",
    r"\bRTX\b":   "R-T-X",
    r"\bRSS\b":   "R-S-S",
    r"\bRAM\b":   "RAM",
}


def _expand_abbreviations(text: str) -> str:
    for pattern, replacement in _ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text)
    return text


def _add_natural_pauses(text: str) -> str:
    text = re.sub(r":(?!\n)", ":,", text)
    text = re.sub(r"\s*—\s*", ", ", text)
    return text


def preprocess_text(text: str, params: SpeechParams) -> str:
    text = text.strip()
    if not text:
        return text
    if params.expand_abbreviations:
        text = _expand_abbreviations(text)
    if params.pause_after_colon:
        text = _add_natural_pauses(text)
    if text and text[-1] not in ".!?":
        text += "."
    return text


def resolve(
    intent:           str,
    brain:            Optional[str] = None,
    grade_level:      Optional[int] = None,
    override_persona: Optional[str] = None,
) -> SpeechParams:
    """
    Resolve a Persona and return SpeechParams.

    Priority: override_persona > brain > grade_level > intent > CHAT fallback
    """
    if override_persona:
        try:
            return _PERSONA_PROFILES[Persona(override_persona)]
        except (ValueError, KeyError):
            pass
    if brain and brain in _BRAIN_PERSONA_OVERRIDE:
        return _PERSONA_PROFILES[_BRAIN_PERSONA_OVERRIDE[brain]]
    if intent == "tutor" and grade_level is not None:
        intent = "tutor_5" if grade_level <= 5 else ("tutor_6" if grade_level <= 8 else "tutor_9")
    persona = _INTENT_TO_PERSONA.get(intent, Persona.CHAT)
    return _PERSONA_PROFILES[persona]


def resolve_and_preprocess(
    text:             str,
    intent:           str,
    brain:            Optional[str] = None,
    grade_level:      Optional[int] = None,
    override_persona: Optional[str] = None,
) -> tuple[str, SpeechParams]:
    """Resolve params AND pre-process text in one call."""
    params = resolve(intent, brain, grade_level, override_persona)
    return preprocess_text(text, params), params


if __name__ == "__main__":
    cases = [("creative",None,None),("device_control",None,None),("tutor",None,6),("research","architect_brain",None)]
    sample = "The RAG pipeline uses GPU acceleration. Check the UI status."
    for intent, brain, grade in cases:
        text, params = resolve_and_preprocess(sample, intent, brain, grade)
        print(f"[{params.persona.value:10s}] voice={params.voice:<12} speed={params.speed}")
        print(f"  → {text[:80]}\n")
