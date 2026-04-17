"""
German G2P (Grapheme-to-Phoneme) for Kokoro TTS.

Requires:  apt-get install espeak-ng
No extra Python packages beyond existing kokoro dependencies.

Usage:
    from kokoro.de_g2p import DEG2P
    g2p = DEG2P()
    phonemes, tokens = g2p("Guten Morgen, wie geht es Ihnen?")
"""

from __future__ import annotations
from typing import Tuple
from loguru import logger
from .de_normalizer import normalize_text_de


class DEG2P:
    """
    German G2P using espeak-ng via misaki.espeak.EspeakG2P.

    Normalises text with normalize_text_de() first, then phonemises with
    EspeakG2P(language='de').  Returns (phonemes_str, tokens) matching the
    misaki G2P interface used by KPipeline.
    """

    def __init__(self) -> None:
        try:
            from misaki import espeak as _espeak
            self._g2p = _espeak.EspeakG2P(language='de')
            logger.info("DEG2P: initialised EspeakG2P(language='de')")
        except ImportError as exc:
            raise ImportError(
                "misaki is required. Install with: pip install misaki[en]"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                "espeak-ng is required for German TTS. "
                "Install with: apt-get install espeak-ng\n"
                f"Original error: {exc}"
            ) from exc

    def __call__(self, text: str) -> Tuple[str, list]:
        """
        Normalise German text and convert to IPA phonemes.

        Args:
            text: Raw German text.

        Returns:
            (phonemes_str, tokens) — matches the misaki G2P return signature.
        """
        normalised = normalize_text_de(text)
        logger.debug(f"DEG2P in : {text[:80]}")
        logger.debug(f"DEG2P norm: {normalised[:80]}")
        result = self._g2p(normalised)
        # EspeakG2P returns (phonemes, tokens) — guard against str-only return
        if isinstance(result, tuple):
            return result
        return result, []
