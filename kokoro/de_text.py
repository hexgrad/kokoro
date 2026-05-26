"""German text normalization helpers for Kokoro espeak G2P (lang_code='d')."""

from __future__ import annotations

import re

# Common German abbreviations expanded before phonemization.
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
    "mah": "Milliamperestunden",
    "ma": "Milliampere",
    "kwh": "Kilowattstunden",
    "kw": "Kilowatt",
}

_EUR_RE = re.compile(
    r"(\d{1,3}(?:\.\d{3})*|\d+),(\d{2})\s*(?:€|eur|euro)\b",
    re.IGNORECASE,
)
_DECIMAL_UNIT_RE = re.compile(
    r"(\d+),(\d+)\s*(" + "|".join(re.escape(u) for u in ("kwh", "kw", "mah", "ma", "g", "kg", "mb", "gb")) + r")\b",
    re.IGNORECASE,
)


def normalize_german(text: str) -> str:
    """Normalize German numerals and abbreviations for more stable TTS."""
    if not text:
        return text

    out = text
    for src, dst in _ABBREV.items():
        out = re.sub(rf"\b{re.escape(src)}", dst, out, flags=re.IGNORECASE)

    def _eur_sub(m: re.Match[str]) -> str:
        whole = m.group(1).replace(".", "")
        cents = m.group(2)
        return f"{whole} Euro {cents}"

    out = _EUR_RE.sub(_eur_sub, out)

    def _dec_unit_sub(m: re.Match[str]) -> str:
        whole, frac, unit = m.group(1), m.group(2), m.group(3)
        unit_word = {
            "kwh": "Kilowattstunden",
            "kw": "Kilowatt",
            "mah": "Milliamperestunden",
            "ma": "Milliampere",
            "g": "Gramm",
            "kg": "Kilogramm",
            "mb": "Megabyte",
            "gb": "Gigabyte",
        }.get(unit.lower(), unit)
        return f"{whole} Komma {frac} {unit_word}"

    out = _DECIMAL_UNIT_RE.sub(_dec_unit_sub, out)
    return out
