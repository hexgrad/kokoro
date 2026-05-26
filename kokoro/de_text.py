"""German text normalization helpers for Kokoro espeak G2P (lang_code='d')."""

from __future__ import annotations

import re

# Text abbreviations only (units handled via _DECIMAL_UNIT_RE below).
_ABBREV = {
    "z.b.": "zum Beispiel",
    "zzgl.": "zuzüglich",
    "ca.": "circa",
    "usw.": "und so weiter",
    "bzw.": "beziehungsweise",
    "stck.": "Stück",
    "min.": "Minuten",
    "max.": "maximal",
    "ltr.": "Liter",
}

_UNIT_WORDS = {
    "kwh": "Kilowattstunden",
    "kw": "Kilowatt",
    "mah": "Milliamperestunden",
    "ma": "Milliampere",
    "g": "Gramm",
    "kg": "Kilogramm",
    "mb": "Megabyte",
    "gb": "Gigabyte",
}

_EUR_RE = re.compile(
    r"(?<!\w)(\d{1,3}(?:\.\d{3})*|\d+),(\d{2})\s*(?:€|eur|euro)(?!\w)",
    re.IGNORECASE,
)
_DECIMAL_UNIT_RE = re.compile(
    r"(?<!\w)(\d+),(\d+)\s*("
    + "|".join(re.escape(u) for u in _UNIT_WORDS)
    + r")(?!\w)",
    re.IGNORECASE,
)


def _apply_abbrevs(text: str) -> str:
    out = text
    for src in sorted(_ABBREV, key=len, reverse=True):
        dst = _ABBREV[src]
        pattern = rf"(?<!\w){re.escape(src)}(?!\w)"
        out = re.sub(pattern, dst, out, flags=re.IGNORECASE)
    return out


def normalize_german(text: str) -> str:
    """Normalize German numerals and abbreviations for more stable TTS."""
    if not text:
        return text

    out = text

    def _dec_unit_sub(m: re.Match[str]) -> str:
        whole, frac, unit = m.group(1), m.group(2), m.group(3)
        unit_word = _UNIT_WORDS.get(unit.lower(), unit)
        return f"{whole} Komma {frac} {unit_word}"

    out = _DECIMAL_UNIT_RE.sub(_dec_unit_sub, out)

    def _eur_sub(m: re.Match[str]) -> str:
        whole = m.group(1).replace(".", "")
        cents = m.group(2)
        return f"{whole} Euro {cents}"

    out = _EUR_RE.sub(_eur_sub, out)
    out = _apply_abbrevs(out)
    return out
