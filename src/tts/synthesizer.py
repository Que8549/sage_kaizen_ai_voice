"""
src/tts/synthesizer.py — Kokoro-ONNX TTS Synthesizer
=====================================================
Uses onnx-community/Kokoro-82M-v1.0-ONNX directly via onnxruntime.InferenceSession.
NO kokoro-onnx pip package. Raw ONNX inference + misaki G2P.

Downloaded model folder structure (E:\\Kokoro-82M-v1.0-ONNX\\):
  onnx\\model_quantized.onnx  ← recommended (89 MB, pure INT8, CPU-safe)
  onnx\\model.onnx         ← full fp32 (326 MB)
  voices\\am_fenrir.bin   ← narrator voice
  voices\\am_michael.bin  ← teacher voice
  voices\\am_onyx.bin     ← quick/device-control voice
  voices\\...             ← 50+ additional voices in the v1.0 pack

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

ONNX model inputs (v1.0 — note: key renamed from "tokens" to "input_ids"):
  input_ids  int64   [1, seq_len+2]   phoneme IDs padded with 0 at start+end
  style      float32 [1, 256]         voice style vector indexed by seq_len
  speed      float32 [1]              playback speed (1.0 = normal)

Verified from onnx-community/Kokoro-82M-v1.0-ONNX README (March 2026).
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
from typing import Any, AsyncIterator, Iterator, Optional

import numpy as np
import onnxruntime as ort

from sk_logging import get_logger
from src.config import PATHS, TTS

_LOG = get_logger("sage_kaizen.voice.tts.synth")

# ─────────────────────────────────────────────────────────────
# Kokoro phoneme → token ID vocabulary
# Loaded from PATHS.TTS_TOKENIZER (tokenizer.json bundled with the model).
# This is the authoritative source — never hardcode these mappings.
# ID 0 ("$") is the BOS/EOS pad token; it is not a phoneme.
# ─────────────────────────────────────────────────────────────

def _load_phoneme_vocab() -> dict[str, int]:
    tokenizer_path = PATHS.TTS_TOKENIZER
    with tokenizer_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    vocab: dict[str, int] = data["model"]["vocab"]
    # Exclude the pad token "$" (id 0) — it is used explicitly as BOS/EOS padding,
    # not as a phoneme character that misaki would ever emit.
    return {ch: idx for ch, idx in vocab.items() if ch != "$"}

_PHONEME_VOCAB: dict[str, int] = _load_phoneme_vocab()


def phoneme_to_ids(phonemes: str) -> list[int]:
    """
    Convert an IPA phoneme string to a list of integer token IDs.
    Unknown characters are silently skipped.
    Returns at most 510 IDs (model context limit is 512 with BOS/EOS padding tokens).
    """
    ids = [_PHONEME_VOCAB[ch] for ch in phonemes if ch in _PHONEME_VOCAB]
    return ids[:510]


# ─────────────────────────────────────────────────────────────
# Custom pronunciation lexicon
#
# Maps lowercase word → Kokoro IPA string.
# Applied as whole-word, case-insensitive substitutions before G2P.
# Use Kokoro's uppercase diphthong tokens where applicable:
#   A = /eɪ/ (FACE)   I = /aɪ/ (PRICE)   W = /aʊ/ (MOUTH)   Y = /ɔɪ/ (CHOICE)
#
# To verify a word's current G2P output:
#   from src.tts.synthesizer import _g2p; print(repr(_g2p("word")))
# ─────────────────────────────────────────────────────────────
_CUSTOM_LEXICON: dict[str, str] = {
    # Proper name — misaki stresses the wrong syllable (əlkwˈɪn → "ul-KWIN")
    "alquin":  "ˈælkwɪn",   # æl.kwɪn  →  AL-kwin
    # Already correct in misaki (sˈAʤ), but pinned to prevent regression
    "sage":    "sˈAʤ",      # seɪdʒ    →  single syllable, rhymes with "page"
    # misaki adds an unwanted schwa (kˈIzᵊn → "KAI-zun"); drop it
    "kaizen":  "kˈIzn",     # kaɪzn    →  KAI-zn  (two syllables, no schwa)
}

_CUSTOM_LEXICON_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _CUSTOM_LEXICON) + r")\b",
    re.IGNORECASE,
)


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
            _LOG.info("misaki G2P loaded with espeak-ng fallback")
        except Exception:
            _LOG.warning("espeak-ng not available; loading misaki without fallback")
            from misaki import en
            self._g2p = en.G2P(trf=False, british=False, fallback=None)

    def __call__(self, text: str) -> str:
        """Return IPA phoneme string for the given text."""
        with self._lock:
            self._load()
            assert self._g2p is not None

            # Fast path: no custom words in this text.
            parts = _CUSTOM_LEXICON_RE.split(text)
            if len(parts) == 1:
                phonemes, _ = self._g2p(text)
                return phonemes

            # Split around custom words, apply direct IPA, G2P the rest.
            segments: list[str] = []
            for part in parts:
                if not part:
                    continue
                custom = _CUSTOM_LEXICON.get(part.lower())
                if custom is not None:
                    segments.append(custom)
                elif part.isspace():
                    # Preserve inter-word whitespace directly — space is phoneme token 16.
                    segments.append(part)
                else:
                    # Strip surrounding whitespace before G2P (it drops leading spaces),
                    # then re-inject any leading/trailing space as phoneme token directly.
                    lstripped = part.lstrip()
                    leading = part[: len(part) - len(lstripped)]
                    rstripped = lstripped.rstrip()
                    trailing = lstripped[len(rstripped):]
                    if leading:
                        segments.append(leading)
                    if rstripped:
                        phonemes, _ = self._g2p(rstripped)
                        segments.append(phonemes)
                    if trailing:
                        segments.append(trailing)
            return "".join(segments)


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
          (e.g. E:\\Kokoro-82M-v1.0-ONNX\\onnx\\model_quantized.onnx)
      - Voice .bin files from PATHS.TTS_VOICES_DIR
          (e.g. E:\\Kokoro-82M-v1.0-ONNX\\voices\\am_fenrir.bin)

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
                "Expected: E:\\Kokoro-82M-v1.0-ONNX\\onnx\\model_quantized.onnx\n"
                "WARNING: Do NOT use model_q8f16.onnx — it crashes CPUExecutionProvider."
            )
        if not voices_dir.exists():
            raise FileNotFoundError(
                f"TTS voices directory not found: {voices_dir}\n"
                "Expected location: E:\\Kokoro-82M-v1.0-ONNX\\voices\\"
            )

        _LOG.info("Loading Kokoro ONNX model from %s ...", model_path)
        try:
            with self._lock:
                if not self._ready:
                    opts = ort.SessionOptions()
                    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                    opts.intra_op_num_threads = TTS.ORT_INTRA_THREADS
                    opts.inter_op_num_threads = TTS.ORT_INTER_THREADS
                    self._session = ort.InferenceSession(
                        str(model_path),
                        sess_options=opts,
                        providers=["CPUExecutionProvider"],
                    )
                    self._ready = True
        except Exception:
            _LOG.exception("Failed to load Kokoro ONNX model: %s", model_path)
            raise
        _LOG.info("Kokoro ONNX model ready")

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
        e.g. E:\\Kokoro-82M-v1.0-ONNX\\voices\\am_fenrir.bin
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
        _LOG.debug("Loaded voice: %s  shape=%s", voice, voice_array.shape)
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
            _LOG.warning("No phoneme IDs for text: %r", text[:50])
            return np.array([], dtype=np.float32), TTS.SAMPLE_RATE

        # Step 3: Prepare ONNX inputs
        #   tokens:  [0, *token_ids, 0]  (padded with 0 at start and end)
        #   style:   voice_array[len(token_ids)]  shape (1, 256)
        #   speed:   float32 array [1]
        padded_tokens = np.array([[0, *token_ids, 0]], dtype=np.int64)
        style = self._get_style(voice, len(token_ids))
        speed_arr = np.array([speed], dtype=np.float32)

        # Step 4: Run ONNX inference
        # v1.0 model: input key is "input_ids" (was "tokens" in the old ONNX pack)
        with self._lock:
            assert self._session is not None
            outputs = self._session.run(
                None,
                {
                    "input_ids": padded_tokens,
                    "style":     style,
                    "speed":     speed_arr,
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
        _LOG.debug("Synthesizing %d chunk(s) | voice=%s | speed=%.2f",
                  len(sentence_chunks), voice, speed)

        for chunk_text in sentence_chunks:
            if not chunk_text.strip():
                continue
            try:
                audio, sr = self._synth_text(chunk_text, voice, speed)
                if len(audio) > 0:
                    yield audio, sr
            except Exception:
                _LOG.exception("Synthesis error on chunk: %r", chunk_text[:50])

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
