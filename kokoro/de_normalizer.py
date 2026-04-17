"""
German text normalisation for TTS.

Pure Python, zero external dependencies — safe to import anywhere.
Called by DEG2P before handing text to espeak-ng.
"""

from __future__ import annotations
import re

# ---------------------------------------------------------------------------
# Cardinal numbers
# ---------------------------------------------------------------------------

_ONES = [
    "", "ein", "zwei", "drei", "vier", "fünf", "sechs", "sieben",
    "acht", "neun", "zehn", "elf", "zwölf", "dreizehn", "vierzehn",
    "fünfzehn", "sechzehn", "siebzehn", "achtzehn", "neunzehn",
]
_TENS = [
    "", "", "zwanzig", "dreißig", "vierzig", "fünfzig",
    "sechzig", "siebzig", "achtzig", "neunzig",
]


def int_to_de(n: int) -> str:
    """Convert integer → German words.  Handles negatives and up to ~999 billion."""
    if n < 0:
        return "minus " + int_to_de(-n)
    if n == 0:
        return "null"
    if n < 20:
        return _ONES[n]
    if n < 100:
        ones, tens = n % 10, n // 10
        return (_ONES[ones] + "und" + _TENS[tens]) if ones else _TENS[tens]
    if n < 1_000:
        h, r = divmod(n, 100)
        return ("ein" if h == 1 else _ONES[h]) + "hundert" + (int_to_de(r) if r else "")
    if n < 1_000_000:
        t, r = divmod(n, 1_000)
        return ("ein" if t == 1 else int_to_de(t)) + "tausend" + (int_to_de(r) if r else "")
    if n < 1_000_000_000:
        m, r = divmod(n, 1_000_000)
        head = "eine Million" if m == 1 else int_to_de(m) + " Millionen"
        return head + (" " + int_to_de(r) if r else "")
    b, r = divmod(n, 1_000_000_000)
    head = "eine Milliarde" if b == 1 else int_to_de(b) + " Milliarden"
    return head + (" " + int_to_de(r) if r else "")


# ---------------------------------------------------------------------------
# Ordinals
# ---------------------------------------------------------------------------

_ORD_IRREGULAR = {1: "erst", 2: "zweit", 3: "dritt", 7: "siebt", 8: "acht"}


def ordinal_stem_de(n: int) -> str:
    """Return the uninflected ordinal stem (without case/gender ending)."""
    if n in _ORD_IRREGULAR:
        return _ORD_IRREGULAR[n]
    return int_to_de(n) + ("t" if n < 20 else "st")


# ---------------------------------------------------------------------------
# Years
# ---------------------------------------------------------------------------

def year_de(n: int) -> str:
    if 1100 <= n <= 1999:
        c, r = divmod(n, 100)
        return int_to_de(c) + "hundert" + (int_to_de(r) if r else "")
    return int_to_de(n)


# ---------------------------------------------------------------------------
# Abbreviation table  (pattern, replacement)
# ---------------------------------------------------------------------------

_ABBREVS: list[tuple[re.Pattern, str]] = [
    # Titles
    (re.compile(r'\bDr\.(?=\s)'),          "Doktor"),
    (re.compile(r'\bProf\.(?=\s)'),        "Professor"),
    (re.compile(r'\bHrn?\.\s'),            "Herrn "),
    (re.compile(r'\bFr\.(?=\s[A-ZÄÖÜ])'), "Frau"),
    (re.compile(r'\bDipl\.\s*-?\s*Ing\.'), "Diplom-Ingenieur"),
    (re.compile(r'\bMag\.(?=\s)'),         "Magister"),
    # Locations / units
    (re.compile(r'[Ss]tr\.(?=\s)'),         "Straße"),   # handles both "Str." and "Hauptstr."
    (re.compile(r'\bNr\.(?=\s*\d)'),       "Nummer"),
    (re.compile(r'\bTel\.(?=\s)'),         "Telefon"),
    (re.compile(r'\bAbt\.(?=\s)'),         "Abteilung"),
    # Corporate
    (re.compile(r'\bGmbH\b'),              "Gesellschaft mit beschränkter Haftung"),
    (re.compile(r'\bAG\b(?=[\s,.]|$)'),   "Aktiengesellschaft"),
    # Common phrases
    (re.compile(r'\bz\.\s*B\.'),           "zum Beispiel"),
    (re.compile(r'\bd\.\s*h\.'),           "das heißt"),
    (re.compile(r'\bu\.\s*a\.'),           "unter anderem"),
    (re.compile(r'\bbzw\.'),              "beziehungsweise"),
    (re.compile(r'\busw\.'),              "und so weiter"),
    (re.compile(r'\betc\.', re.I),        "et cetera"),
    (re.compile(r'\bca\.'),               "circa"),
    (re.compile(r'\bvgl\.'),              "vergleiche"),
    (re.compile(r'\binkl\.'),             "inklusive"),
    (re.compile(r'\bexkl\.'),             "exklusive"),
    (re.compile(r'\bggf\.'),              "gegebenenfalls"),
    (re.compile(r'\bi\.\s*d\.\s*R\.'),    "in der Regel"),
    (re.compile(r'\bo\.\s*ä\.'),          "oder ähnliches"),
    (re.compile(r'\bu\.\s*U\.'),          "unter Umständen"),
    # Months
    (re.compile(r'\bJan\.(?=\s)'),  "Januar"),
    (re.compile(r'\bFeb\.(?=\s)'),  "Februar"),
    (re.compile(r'\bMär\.(?=\s)'),  "März"),
    (re.compile(r'\bApr\.(?=\s)'),  "April"),
    (re.compile(r'\bJun\.(?=\s)'),  "Juni"),
    (re.compile(r'\bJul\.(?=\s)'),  "Juli"),
    (re.compile(r'\bAug\.(?=\s)'),  "August"),
    (re.compile(r'\bSep\.(?=\s)'),  "September"),
    (re.compile(r'\bOkt\.(?=\s)'),  "Oktober"),
    (re.compile(r'\bNov\.(?=\s)'),  "November"),
    (re.compile(r'\bDez\.(?=\s)'),  "Dezember"),
]

_DE_MONTHS = [
    "", "Januar", "Februar", "März", "April", "Mai", "Juni",
    "Juli", "August", "September", "Oktober", "November", "Dezember",
]

_CURRENCY_SYM = {"€": "Euro", "$": "Dollar", "£": "Pfund", "¥": "Yen"}
_CURRENCY_PAT = re.compile(
    r'([€$£¥])\s*(\d[\d.,]*)'   # symbol-first:  €1.299,99
    r'|(\d[\d.,]*)\s*([€$£¥])'  # amount-first:  1.299,99€
)


def _currency_repl(m: re.Match) -> str:
    sym, num = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
    word = _CURRENCY_SYM.get(sym, sym)
    cleaned = num.replace(".", "").replace(",", ".")
    try:
        val = float(cleaned)
    except ValueError:
        return m.group(0)
    euros = int(val)
    cents = round((val - euros) * 100)
    if cents == 0:
        return int_to_de(euros) + " " + word
    return int_to_de(euros) + " " + word + " und " + int_to_de(cents) + " Cent"


def _time_repl(m: re.Match) -> str:
    h, mi = int(m.group(1)), int(m.group(2))
    return int_to_de(h) + " Uhr" + (" " + int_to_de(mi) if mi else "")


def _date_repl(m: re.Match) -> str:
    d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= d <= 31 and 1 <= mo <= 12):
        return m.group(0)
    return ordinal_stem_de(d) + "e " + _DE_MONTHS[mo] + " " + year_de(y)


def _ordinal_repl(m: re.Match) -> str:
    return ordinal_stem_de(int(m.group(1))) + "e "


def _number_repl(m: re.Match) -> str:
    raw = m.group(0)
    # German: thousands = '.', decimal = ','
    cleaned = raw.replace(".", "").replace(",", ".")
    try:
        if "." in cleaned:
            int_part, frac = cleaned.split(".", 1)
            digits = " ".join(int_to_de(int(d)) for d in frac)
            return int_to_de(int(int_part)) + " Komma " + digits
        return int_to_de(int(cleaned))
    except (ValueError, OverflowError):
        return raw


# Matches German-format numbers: 1.234.567 or 3,14 or plain integers
_GER_NUM = re.compile(r'\b\d{1,3}(?:\.\d{3})*(?:,\d+)?\b|\b\d+,\d+\b|\b\d+\b')
_YEAR_RE = re.compile(r'\b(\d{4})\b')
_TIME_RE = re.compile(r'\b(\d{1,2}):(\d{2})\b')
_DATE_RE = re.compile(r'\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b')
_ORD_RE  = re.compile(r'(?<!\n)(?<!\d)(\d{1,4})\.\s')


def normalize_text_de(text: str) -> str:
    """
    Full German text normalisation pipeline for TTS.

    Order matters:
      quotes → abbreviations → currency → times → dates → ordinals → years → numbers → whitespace
    """
    if not text:
        return text

    # 1. Quotes
    text = (text
        .replace("\u201e", '"').replace("\u201c", '"')  # „ "
        .replace("\u2018", "'").replace("\u2019", "'")  # ' '
        .replace("\u00ab", '"').replace("\u00bb", '"')  # « »
        .replace("\u2039", '"').replace("\u203a", '"')  # ‹ ›
    )

    # 2. Non-breaking / other whitespace
    text = re.sub(r'[^\S \n]', ' ', text)

    # 3. Abbreviations
    for pat, repl in _ABBREVS:
        text = pat.sub(repl, text)

    # 4. Currency  (before numbers — contains digits)
    text = _CURRENCY_PAT.sub(_currency_repl, text)

    # 5. Times  hh:mm
    text = _TIME_RE.sub(_time_repl, text)

    # 6. Full dates  dd.mm.yyyy
    text = _DATE_RE.sub(_date_repl, text)

    # 7. Ordinals mid-sentence  "am 3. Mai"  →  "am dritte Mai"
    text = _ORD_RE.sub(_ordinal_repl, text)

    # 8. Years (standalone 4-digit 1100-2099)
    def _year_or_num(m: re.Match) -> str:
        n = int(m.group(1))
        return year_de(n) if 1100 <= n <= 2099 else int_to_de(n)
    text = _YEAR_RE.sub(_year_or_num, text)

    # 9. All remaining numbers
    text = _GER_NUM.sub(_number_repl, text)

    # 10. Whitespace cleanup
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
