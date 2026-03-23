"""
src/tts/synthesizer.py — Kokoro-ONNX TTS Synthesizer
=====================================================
Uses onnx-community/Kokoro-82M-ONNX directly via onnxruntime.InferenceSession.
NO kokoro-onnx pip package. Raw ONNX inference + misaki G2P.

Downloaded model folder structure (E:\\Kokoro-82M-ONNX\\):
  onnx\\model_q8.onnx          ← recommended (88 MB, CPU-optimised INT8)
  onnx\\model.onnx              ← full fp32 (310 MB, slightly better quality)
  voices\\am_fenrir.bin         ← narrator voice (82 MB each)
  voices\\am_michael.bin        ← teacher voice
  voices\\am_onyx.bin           ← quick/device-control voice

Pipeline:
  Text
    │  misaki en.G2P()   (G2P: grapheme → IPA phonemes)
    │  phoneme_to_ids()  (IPA chars → int token IDs via Kokoro vocab)
    ▼
  token_ids: List[int]
    │  np.fromfile(voices/am_fenrir.bin)  →  style vector (1, 256) at len(tokens)
    │  onnxruntime.InferenceSession.run()
    ▼
  audio: np.ndarray float32 at 24 kHz

ONNX model inputs:
  tokens  int64   [1, seq_len+2]   phoneme IDs padded with 0 at start+end
  style   float32 [1, 256]         voice style vector indexed by seq_len
  speed   float32 [1]              playback speed (1.0 = normal)

Verified from onnx-community/Kokoro-82M-ONNX README (March 2026).
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, Optional

import numpy as np
import onnxruntime as ort

from src.config import PATHS, TTS

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Kokoro phoneme → token ID vocabulary
# Source: hexgrad/Kokoro-82M config.json  (stable — baked into ONNX weights)
# These are the IPA character → integer mappings the model was trained with.
# ─────────────────────────────────────────────────────────────
_PHONEME_VOCAB: dict[str, int] = {
    ";": 1, ":": 2, ",": 3, ".": 4, "!": 5, "?": 6, "—": 9,
    "…": 10, "«": 11, "»": 12, """: 13, """: 14, " ": 16,
    "A": 21, "I": 22, "O": 23, "Q": 24, "S": 25, "T": 26,
    "W": 27, "Y": 28, "a": 43, "b": 44, "c": 45, "d": 46,
    "e": 47, "f": 48, "g": 49, "h": 50, "i": 51, "j": 52,
    "k": 53, "l": 54, "m": 55, "n": 56, "o": 57, "p": 58,
    "q": 59, "r": 60, "s": 61, "t": 62, "u": 63, "v": 64,
    "w": 65, "x": 66, "y": 67, "z": 68, "æ": 69, "ç": 70,
    "ð": 71, "ø": 72, "ŋ": 73, "œ": 74, "ɐ": 75, "ɑ": 76,
    "ɒ": 77, "ɔ": 78, "ɕ": 79, "ɖ": 80, "ɗ": 81, "ɘ": 82,
    "ə": 83, "ɚ": 84, "ɛ": 85, "ɜ": 86, "ɝ": 87, "ɞ": 88,
    "ɟ": 89, "ɠ": 90, "ɡ": 91, "ɢ": 92, "ɣ": 93, "ɤ": 94,
    "ɥ": 95, "ɦ": 96, "ɧ": 97, "ɨ": 98, "ɪ": 99, "ɫ": 100,
    "ɬ": 101, "ɭ": 102, "ɮ": 103, "ɯ": 104, "ɰ": 105,
    "ɱ": 106, "ɲ": 107, "ɳ": 108, "ɴ": 109, "ɵ": 110,
    "ɶ": 111, "ɸ": 112, "ɹ": 113, "ɺ": 114, "ɻ": 115,
    "ɼ": 116, "ɽ": 117, "ɾ": 118, "ɿ": 119, "ʀ": 120,
    "ʁ": 121, "ʂ": 122, "ʃ": 123, "ʄ": 124, "ʈ": 125,
    "ʉ": 126, "ʊ": 127, "ʋ": 128, "ʌ": 129, "ʍ": 130,
    "ʎ": 131, "ʏ": 132, "ʐ": 133, "ʑ": 134, "ʒ": 135,
    "ʔ": 136, "ʕ": 137, "ʘ": 138, "ʙ": 139, "ʛ": 140,
    "ʜ": 141, "ʝ": 142, "ʟ": 143, "ʠ": 144, "ʡ": 145,
    "ʢ": 146, "ʦ": 147, "ʧ": 148, "ʨ": 149, "ʩ": 150,
    "ʪ": 151, "ʫ": 152, "ʬ": 153, "ʭ": 154, "ˈ": 155,
    "ˌ": 156, "ː": 157, "ˑ": 158, "˘": 159, "̃": 160,
    "̈": 161, "̊": 162, "̽": 163, "͡": 164, "β": 165,
    "θ": 166, "χ": 167, "ᵻ": 168,
}


def phoneme_to_ids(phonemes: str) -> list[int]:
    """
    Convert an IPA phoneme string to a list of integer token IDs.
    Unknown characters are silently skipped.
    Returns at most 510 IDs (model context limit is 512 with padding tokens).
    """
    ids = [_PHONEME_VOCAB[ch] for ch in phonemes if ch in _PHONEME_VOCAB]
    return ids[:510]


# ─────────────────────────────────────────────────────────────
# G2P (grapheme-to-phoneme) using misaki
# ─────────────────────────────────────────────────────────────

class _G2P:
    """Lazy-loaded misaki G2P engine with espeak-ng fallback."""

    def __init__(self) -> None:
        self._g2p: Optional[Any] = None
        self._lock = threading.Lock()

    def _load(self):
        if self._g2p is not None:
            return
        try:
            from misaki import en, espeak
            fallback = espeak.EspeakFallback(british=False)
            self._g2p = en.G2P(trf=False, british=False, fallback=fallback)
            log.info("misaki G2P loaded with espeak-ng fallback")
        except Exception:
            log.warning("espeak-ng not available; loading misaki without fallback")
            from misaki import en
            self._g2p = en.G2P(trf=False, british=False, fallback=None)

    def __call__(self, text: str) -> str:
        """Return IPA phoneme string for the given text."""
        with self._lock:
            self._load()
            assert self._g2p is not None
            phonemes, _ = self._g2p(text)
            return phonemes


_g2p = _G2P()


# ─────────────────────────────────────────────────────────────
# Sentence splitter
# ─────────────────────────────────────────────────────────────

_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z\"])"
    r"|(?<=\.\.\.\s)"
    r"|(?<=:,)\s"
)


def split_into_chunks(text: str, min_words: int = TTS.MIN_CHUNK_WORDS) -> list[str]:
    """Split text into synthesis-ready chunks, merging short ones."""
    raw = _SENTENCE_SPLIT_RE.split(text.strip())
    merged: list[str] = []
    buf = ""
    for chunk in raw:
        chunk = chunk.strip()
        if not chunk:
            continue
        buf = (buf + " " + chunk).strip() if buf else chunk
        if len(buf.split()) >= min_words:
            merged.append(buf)
            buf = ""
    if buf:
        if merged:
            merged[-1] += " " + buf
        else:
            merged.append(buf)
    return merged or [text.strip()]


# ─────────────────────────────────────────────────────────────
# KokoroSynthesizer — raw onnxruntime + misaki
# ─────────────────────────────────────────────────────────────

class KokoroSynthesizer:
    """
    Thread-safe Kokoro-ONNX synthesizer using raw onnxruntime.

    Loads:
      - ONNX model from PATHS.TTS_ONNX_MODEL
          (e.g. E:\\Kokoro-82M-ONNX\\onnx\\model_q8f16.onnx)
      - Voice .bin files from PATHS.TTS_VOICES_DIR
          (e.g. E:\\Kokoro-82M-ONNX\\voices\\am_fenrir.bin)

    Usage
    -----
    synth = KokoroSynthesizer()
    synth.initialize()

    audio, sr = synth.synthesize("Hello.", voice="am_fenrir", speed=0.87)

    for audio, sr in synth.stream("Long text...", voice="am_fenrir", speed=0.87):
        play(audio)

    async for audio, sr in synth.stream_async("...", voice="am_fenrir"):
        await enqueue(audio)
    """

    def __init__(self) -> None:
        self._session: Optional[ort.InferenceSession] = None
        self._voice_cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._ready = False

    def initialize(self) -> None:
        """Load ONNX model from E:\\ drive. Idempotent, thread-safe."""
        if self._ready:
            return

        model_path = PATHS.TTS_ONNX_MODEL
        voices_dir = PATHS.TTS_VOICES_DIR

        if not model_path.exists():
            raise FileNotFoundError(
                f"TTS ONNX model not found: {model_path}\n"
                "Run scripts/verify_setup.py for diagnostics.\n"
                "Expected location: E:\\Kokoro-82M-ONNX\\onnx\\model_q8f16.onnx"
            )
        if not voices_dir.exists():
            raise FileNotFoundError(
                f"TTS voices directory not found: {voices_dir}\n"
                "Expected location: E:\\Kokoro-82M-ONNX\\voices\\"
            )

        log.info("Loading Kokoro ONNX model from %s ...", model_path)
        with self._lock:
            if not self._ready:
                self._session = ort.InferenceSession(
                    str(model_path),
                    providers=["CPUExecutionProvider"],
                )
                self._ready = True
        log.info("Kokoro ONNX model ready")

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ── Voice loading ──────────────────────────────────────────

    def _load_voice(self, voice: str) -> np.ndarray:
        """
        Load and cache a voice .bin file.

        Voice files have shape (max_tokens, 1, 256).
        We index at seq_len to get the style vector for that sequence length.

        File path: PATHS.TTS_VOICES_DIR / f"{voice}.bin"
        e.g. E:\\Kokoro-82M-ONNX\\voices\\am_fenrir.bin
        """
        if voice in self._voice_cache:
            return self._voice_cache[voice]

        voice_path = PATHS.TTS_VOICES_DIR / f"{voice}.bin"
        if not voice_path.exists():
            raise FileNotFoundError(
                f"Voice file not found: {voice_path}\n"
                f"Available voices: {[p.stem for p in PATHS.TTS_VOICES_DIR.glob('*.bin')]}"
            )
        # Shape: (max_tokens, 1, 256)
        voice_array = np.fromfile(str(voice_path), dtype=np.float32).reshape(-1, 1, 256)
        self._voice_cache[voice] = voice_array
        log.debug("Loaded voice: %s  shape=%s", voice, voice_array.shape)
        return voice_array

    def _get_style(self, voice: str, seq_len: int) -> np.ndarray:
        """Get style vector for voice at given sequence length. Shape: (1, 256)."""
        voice_array = self._load_voice(voice)
        idx = min(seq_len, voice_array.shape[0] - 1)
        return voice_array[idx]  # (1, 256)

    # ── Core synthesis ─────────────────────────────────────────

    def _synth_text(self, text: str, voice: str, speed: float) -> tuple[np.ndarray, int]:
        """
        Synthesize a single chunk of text.
        Returns (float32 audio samples, sample_rate=24000).
        """
        # Step 1: Text → IPA phonemes via misaki
        phonemes = _g2p(text)

        # Step 2: Phonemes → token IDs
        token_ids = phoneme_to_ids(phonemes)
        if not token_ids:
            log.warning("No phoneme IDs for text: %r", text[:50])
            return np.array([], dtype=np.float32), TTS.SAMPLE_RATE

        # Step 3: Prepare ONNX inputs
        #   tokens:  [0, *token_ids, 0]  (padded with 0 at start and end)
        #   style:   voice_array[len(token_ids)]  shape (1, 256)
        #   speed:   float32 array [1]
        padded_tokens = np.array([[0, *token_ids, 0]], dtype=np.int64)
        style = self._get_style(voice, len(token_ids))
        speed_arr = np.array([speed], dtype=np.float32)

        # Step 4: Run ONNX inference
        with self._lock:
            assert self._session is not None
            outputs = self._session.run(
                None,
                {
                    "tokens": padded_tokens,
                    "style":  style,
                    "speed":  speed_arr,
                },
            )
        # Output is audio waveform at 24kHz
        audio = np.array(outputs[0], dtype=np.float32).flatten()
        return audio, TTS.SAMPLE_RATE

    # ── Public API ─────────────────────────────────────────────

    def synthesize(
        self,
        text:  str,
        voice: str   = TTS.DEFAULT_VOICE,
        speed: float = TTS.DEFAULT_SPEED,
        lang:  str   = TTS.LANG,   # lang param kept for API compat, not used here
    ) -> tuple[np.ndarray, int]:
        """Synthesize full text into a single audio array."""
        if not self._ready:
            self.initialize()
        chunks = list(self.stream(text, voice, speed))
        if not chunks:
            return np.array([], dtype=np.float32), TTS.SAMPLE_RATE
        return np.concatenate([a for a, _ in chunks]).astype(np.float32), chunks[0][1]

    def stream(
        self,
        text:  str,
        voice: str   = TTS.DEFAULT_VOICE,
        speed: float = TTS.DEFAULT_SPEED,
        lang:  str   = TTS.LANG,
    ) -> Iterator[tuple[np.ndarray, int]]:
        """
        Synchronous streaming generator.
        Yields (float32 audio samples, sample_rate) per sentence chunk.
        """
        if not self._ready:
            self.initialize()

        sentence_chunks = split_into_chunks(text)
        log.debug("Synthesizing %d chunk(s) | voice=%s | speed=%.2f",
                  len(sentence_chunks), voice, speed)

        for chunk_text in sentence_chunks:
            if not chunk_text.strip():
                continue
            try:
                audio, sr = self._synth_text(chunk_text, voice, speed)
                if len(audio) > 0:
                    yield audio, sr
            except Exception:
                log.exception("Synthesis error on chunk: %r", chunk_text[:50])

    async def stream_async(
        self,
        text:  str,
        voice: str   = TTS.DEFAULT_VOICE,
        speed: float = TTS.DEFAULT_SPEED,
        lang:  str   = TTS.LANG,
    ) -> AsyncIterator[tuple[np.ndarray, int]]:
        """Async streaming generator — synthesis runs in thread executor."""
        if not self._ready:
            self.initialize()
        loop = asyncio.get_running_loop()
        for chunk_text in split_into_chunks(text):
            if not chunk_text.strip():
                continue
            audio, sr = await loop.run_in_executor(
                None,
                lambda t=chunk_text: self._synth_text(t, voice, speed),
            )
            if len(audio) > 0:
                yield audio, sr

    def _synth_one(self, text: str, voice: str, speed: float, lang: str = "en-us"):
        """Compatibility method for ZMQ handler."""
        return self._synth_text(text, voice, speed)
