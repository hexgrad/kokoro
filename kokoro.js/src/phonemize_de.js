/**
 * German text normalisation and phonemisation for kokoro-js.
 *
 * normalize_text_de()  — pure sync function, no network/IO
 * phonemize_de()       — async, calls eSpeak-NG via phonemizer package
 */

import { phonemize as espeakng } from "phonemizer";

// ─── helpers ──────────────────────────────────────────────────────────────────

/** Split text on regex, keeping delimiters as separate items. */
function splitKeepDelim(text, regex) {
  const result = [];
  let prev = 0;
  for (const m of text.matchAll(regex)) {
    if (prev < m.index) result.push({ match: false, text: text.slice(prev, m.index) });
    if (m[0].length > 0) result.push({ match: true, text: m[0] });
    prev = m.index + m[0].length;
  }
  if (prev < text.length) result.push({ match: false, text: text.slice(prev) });
  return result;
}

// ─── cardinal numbers ─────────────────────────────────────────────────────────

const _ONES = [
  "", "ein", "zwei", "drei", "vier", "fünf", "sechs", "sieben",
  "acht", "neun", "zehn", "elf", "zwölf", "dreizehn", "vierzehn",
  "fünfzehn", "sechzehn", "siebzehn", "achtzehn", "neunzehn",
];
const _TENS = [
  "", "", "zwanzig", "dreißig", "vierzig", "fünfzig",
  "sechzig", "siebzig", "achtzig", "neunzig",
];

function intToDE(n) {
  if (n < 0)           return "minus " + intToDE(-n);
  if (n === 0)         return "null";
  if (n < 20)          return _ONES[n];
  if (n < 100) {
    const ones = n % 10, tens = Math.floor(n / 10);
    return ones ? _ONES[ones] + "und" + _TENS[tens] : _TENS[tens];
  }
  if (n < 1_000) {
    const h = Math.floor(n / 100), r = n % 100;
    return (h === 1 ? "ein" : _ONES[h]) + "hundert" + (r ? intToDE(r) : "");
  }
  if (n < 1_000_000) {
    const t = Math.floor(n / 1_000), r = n % 1_000;
    return (t === 1 ? "ein" : intToDE(t)) + "tausend" + (r ? intToDE(r) : "");
  }
  if (n < 1_000_000_000) {
    const m = Math.floor(n / 1_000_000), r = n % 1_000_000;
    return (m === 1 ? "eine Million" : intToDE(m) + " Millionen") + (r ? " " + intToDE(r) : "");
  }
  const b = Math.floor(n / 1_000_000_000), r = n % 1_000_000_000;
  return (b === 1 ? "eine Milliarde" : intToDE(b) + " Milliarden") + (r ? " " + intToDE(r) : "");
}

// ─── ordinals ─────────────────────────────────────────────────────────────────

const _ORD_IRREG = { 1: "erst", 2: "zweit", 3: "dritt", 7: "siebt", 8: "acht" };

function ordinalStemDE(n) {
  if (_ORD_IRREG[n]) return _ORD_IRREG[n];
  return intToDE(n) + (n < 20 ? "t" : "st");
}

// ─── years ────────────────────────────────────────────────────────────────────

function yearDE(n) {
  if (n >= 1100 && n <= 1999) {
    const c = Math.floor(n / 100), r = n % 100;
    return intToDE(c) + "hundert" + (r ? intToDE(r) : "");
  }
  return intToDE(n);
}

// ─── month names ──────────────────────────────────────────────────────────────

const DE_MONTHS = [
  "", "Januar", "Februar", "März", "April", "Mai", "Juni",
  "Juli", "August", "September", "Oktober", "November", "Dezember",
];

// ─── currency ─────────────────────────────────────────────────────────────────

const CURRENCY_WORDS = { "€": "Euro", "$": "Dollar", "£": "Pfund", "¥": "Yen" };

function currencyRepl(sym, num) {
  const word = CURRENCY_WORDS[sym] || sym;
  const cleaned = num.replace(/\./g, "").replace(",", ".");
  const val = parseFloat(cleaned);
  if (isNaN(val)) return sym + num;
  const euros = Math.floor(val);
  const cents = Math.round((val - euros) * 100);
  if (cents === 0) return intToDE(euros) + " " + word;
  return intToDE(euros) + " " + word + " und " + intToDE(cents) + " Cent";
}

// ─── main normalisation ───────────────────────────────────────────────────────

/**
 * Normalise German text for TTS phonemisation.
 * Handles: quotes, abbreviations, currency, times, dates, ordinals, years, numbers.
 *
 * @param {string} text
 * @returns {string}
 */
export function normalize_text_de(text) {
  if (!text) return text;

  // 1. Quotes
  text = text
    .replace(/\u201e|\u201c/g, '"')   // „ "
    .replace(/\u2018|\u2019/g, "'")   // ' '
    .replace(/\u00ab|\u00bb/g, '"')   // « »
    .replace(/\u2039|\u203a/g, '"');  // ‹ ›

  // 2. Non-breaking whitespace
  text = text.replace(/[^\S \n]/g, " ");

  // 3. Abbreviations
  text = text
    .replace(/\bDr\.(?=\s)/g,          "Doktor")
    .replace(/\bProf\.(?=\s)/g,        "Professor")
    .replace(/\bHrn?\.\s/g,            "Herrn ")
    .replace(/\bFr\.(?=\s[A-ZÄÖÜ])/g, "Frau")
    .replace(/\bDipl\.\s*-?\s*Ing\./g, "Diplom-Ingenieur")
    .replace(/[Ss]tr\.(?=\s)/g,        "Straße")   // "Str." and "Hauptstr."
    .replace(/\bNr\.(?=\s*\d)/g,       "Nummer")
    .replace(/\bTel\.(?=\s)/g,         "Telefon")
    .replace(/\bAbt\.(?=\s)/g,         "Abteilung")
    .replace(/\bGmbH\b/g,              "Gesellschaft mit beschränkter Haftung")
    .replace(/\bAG\b(?=[\s,.]|$)/g,   "Aktiengesellschaft")
    .replace(/\bz\.\s*B\./gi,          "zum Beispiel")
    .replace(/\bd\.\s*h\./gi,          "das heißt")
    .replace(/\bu\.\s*a\./gi,          "unter anderem")
    .replace(/\bbzw\./gi,             "beziehungsweise")
    .replace(/\busw\./gi,             "und so weiter")
    .replace(/\betc\./gi,             "et cetera")
    .replace(/\bca\./gi,              "circa")
    .replace(/\bvgl\./gi,             "vergleiche")
    .replace(/\binkl\./gi,            "inklusive")
    .replace(/\bexkl\./gi,            "exklusive")
    .replace(/\bggf\./gi,             "gegebenenfalls")
    .replace(/\bi\.\s*d\.\s*R\./gi,  "in der Regel")
    .replace(/\bo\.\s*ä\./gi,         "oder ähnliches")
    .replace(/\bu\.\s*U\./gi,         "unter Umständen")
    .replace(/\bJan\.(?=\s)/g,  "Januar")
    .replace(/\bFeb\.(?=\s)/g,  "Februar")
    .replace(/\bMär\.(?=\s)/g,  "März")
    .replace(/\bApr\.(?=\s)/g,  "April")
    .replace(/\bJun\.(?=\s)/g,  "Juni")
    .replace(/\bJul\.(?=\s)/g,  "Juli")
    .replace(/\bAug\.(?=\s)/g,  "August")
    .replace(/\bSep\.(?=\s)/g,  "September")
    .replace(/\bOkt\.(?=\s)/g,  "Oktober")
    .replace(/\bNov\.(?=\s)/g,  "November")
    .replace(/\bDez\.(?=\s)/g,  "Dezember");

  // 4. Currency — symbol-first (€9,99) and amount-first (9,99€)
  const cSym = "[€$£¥]";
  text = text.replace(new RegExp(`(${cSym})\\s*(\\d[\\d.,]*)`, "g"), (_, s, n) => currencyRepl(s, n));
  text = text.replace(new RegExp(`(\\d[\\d.,]*)\\s*(${cSym})`, "g"), (_, n, s) => currencyRepl(s, n));

  // 5. Times  hh:mm
  text = text.replace(/\b(\d{1,2}):(\d{2})\b/g, (_, h, m) => {
    const hi = parseInt(h, 10), mi = parseInt(m, 10);
    return intToDE(hi) + " Uhr" + (mi ? " " + intToDE(mi) : "");
  });

  // 6. Full dates  dd.mm.yyyy
  text = text.replace(/\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b/g, (_, d, mo, y) => {
    const day = parseInt(d, 10), month = parseInt(mo, 10), year = parseInt(y, 10);
    if (day < 1 || day > 31 || month < 1 || month > 12) return _;
    return ordinalStemDE(day) + "e " + DE_MONTHS[month] + " " + yearDE(year);
  });

  // 7. Ordinals mid-sentence  "am 3. Mai"
  text = text.replace(/(?<!\n)(?<!\d)(\d{1,4})\.\s/g, (_, n) => ordinalStemDE(parseInt(n, 10)) + "e ");

  // 8. Standalone years (1100-2099)
  text = text.replace(/\b(\d{4})\b/g, (_, n) => {
    const ni = parseInt(n, 10);
    return ni >= 1100 && ni <= 2099 ? yearDE(ni) : intToDE(ni);
  });

  // 9. German-format numbers: 1.234.567 or 3,14 or plain integers
  text = text.replace(/\b\d{1,3}(?:\.\d{3})+(?:,\d+)?\b/g, (m) => {
    const cleaned = m.replace(/\./g, "").replace(",", ".");
    const val = parseFloat(cleaned);
    if (isNaN(val)) return m;
    if (Number.isInteger(val)) return intToDE(Math.round(val));
    const [ip, fp] = cleaned.split(".");
    return intToDE(parseInt(ip, 10)) + " Komma " + fp.split("").map(d => intToDE(parseInt(d, 10))).join(" ");
  });
  text = text.replace(/\b(\d+),(\d+)\b/g, (_, a, b) =>
    intToDE(parseInt(a, 10)) + " Komma " + b.split("").map(d => intToDE(parseInt(d, 10))).join(" ")
  );
  text = text.replace(/\b(\d+)\b/g, (_, n) => intToDE(parseInt(n, 10)));

  // 10. Whitespace
  text = text.replace(/[ \t]{2,}/g, " ").replace(/\n{3,}/g, "\n\n").trim();

  return text;
}

// ─── phonemisation ────────────────────────────────────────────────────────────

const PUNCT = ';:,.!?¡¿—..."«»""(){}[]';
const PUNCT_RE = new RegExp(`(\\s*[${PUNCT.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}]+\\s*)+`, "g");

/**
 * Phonemise German text using eSpeak-NG.
 *
 * @param {string} text
 * @param {boolean} [norm=true]  Run normalize_text_de first
 * @returns {Promise<string>}    IPA phoneme string
 */
export async function phonemize_de(text, norm = true) {
  if (norm) text = normalize_text_de(text);

  const sections = splitKeepDelim(text, PUNCT_RE);
  const parts = await Promise.all(
    sections.map(async ({ match, text: chunk }) => {
      if (match) return chunk;
      const res = await espeakng(chunk, "de");
      return res.join(" ");
    })
  );

  return parts
    .join("")
    // Normalise eSpeak script-g → plain g
    .replace(/ɡ/g, "g")
    // Collapse double spaces
    .replace(/\s{2,}/g, " ")
    .trim();
}
