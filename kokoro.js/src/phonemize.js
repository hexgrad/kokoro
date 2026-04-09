import createEphone, { en_us, en_all, roa, jpx, sit } from "ephone";

/**
 * Helper function to split a string on a regex, but keep the delimiters.
 * This is required, because the JavaScript `.split()` method does not keep the delimiters,
 * and wrapping in a capturing group causes issues with existing capturing groups (due to nesting).
 * @param {string} text The text to split.
 * @param {RegExp} regex The regex to split on.
 * @returns {{match: boolean; text: string}[]} The split string.
 */
function split(text, regex) {
  const result = [];
  let prev = 0;
  for (const match of text.matchAll(regex)) {
    const fullMatch = match[0];
    if (prev < match.index) {
      result.push({ match: false, text: text.slice(prev, match.index) });
    }
    if (fullMatch.length > 0) {
      result.push({ match: true, text: fullMatch });
    }
    prev = match.index + fullMatch.length;
  }
  if (prev < text.length) {
    result.push({ match: false, text: text.slice(prev) });
  }
  return result;
}

/**
 * Helper function to split numbers into phonetic equivalents
 * @param {string} match The matched number
 * @returns {string} The phonetic equivalent
 */
function split_num(match) {
  if (match.includes(".")) {
    return match;
  } else if (match.includes(":")) {
    let [h, m] = match.split(":").map(Number);
    if (m === 0) {
      return `${h} o'clock`;
    } else if (m < 10) {
      return `${h} oh ${m}`;
    }
    return `${h} ${m}`;
  }
  let year = parseInt(match.slice(0, 4), 10);
  if (year < 1100 || year % 1000 < 10) {
    return match;
  }
  let left = match.slice(0, 2);
  let right = parseInt(match.slice(2, 4), 10);
  let suffix = match.endsWith("s") ? "s" : "";
  if (year % 1000 >= 100 && year % 1000 <= 999) {
    if (right === 0) {
      return `${left} hundred${suffix}`;
    } else if (right < 10) {
      return `${left} oh ${right}${suffix}`;
    }
  }
  return `${left} ${right}${suffix}`;
}

/**
 * Helper function to format monetary values
 * @param {string} match The matched currency
 * @returns {string} The formatted currency
 */
function flip_money(match) {
  const bill = match[0] === "$" ? "dollar" : "pound";
  if (isNaN(Number(match.slice(1)))) {
    return `${match.slice(1)} ${bill}s`;
  } else if (!match.includes(".")) {
    let suffix = match.slice(1) === "1" ? "" : "s";
    return `${match.slice(1)} ${bill}${suffix}`;
  }
  const [b, c] = match.slice(1).split(".");
  const d = parseInt(c.padEnd(2, "0"), 10);
  let coins = match[0] === "$" ? (d === 1 ? "cent" : "cents") : d === 1 ? "penny" : "pence";
  return `${b} ${bill}${b === "1" ? "" : "s"} and ${d} ${coins}`;
}

/**
 * Helper function to process decimal numbers
 * @param {string} match The matched number
 * @returns {string} The formatted number
 */
function point_num(match) {
  let [a, b] = match.split(".");
  return `${a} point ${b.split("").join(" ")}`;
}

/**
 * Normalize text for phonemization
 * @param {string} text The text to normalize
 * @param {boolean} english Whether to apply English-specific normalization
 * @returns {string} The normalized text
 */
function normalize_text(text, english = true) {
  // Steps 1-3: universal normalization
  text = text
    // 1. Handle quotes and brackets
    .replace(/['']/g, "'")
    .replace(/«/g, "\u201c")
    .replace(/»/g, "\u201d")
    .replace(/[\u201c\u201d]/g, '"')
    .replace(/\(/g, "«")
    .replace(/\)/g, "»")

    // 2. Replace uncommon punctuation marks
    .replace(/、/g, ", ")
    .replace(/。/g, ". ")
    .replace(/！/g, "! ")
    .replace(/，/g, ", ")
    .replace(/：/g, ": ")
    .replace(/；/g, "; ")
    .replace(/？/g, "? ")

    // 3. Whitespace normalization
    .replace(/[^\S \n]/g, " ")
    .replace(/  +/, " ")
    .replace(/(?<=\n) +(?=\n)/g, "")
    .replace(/\n+/g, " ")

    // Strip leading and trailing whitespace
    .trim();

  if (!english) return text;

  // Steps 4-7: English-specific normalization
  return text
    // 4. Abbreviations
    .replace(/\bD[Rr]\.(?= [A-Z])/g, "Doctor")
    .replace(/\b(?:Mr\.|MR\.(?= [A-Z]))/g, "Mister")
    .replace(/\b(?:Ms\.|MS\.(?= [A-Z]))/g, "Miss")
    .replace(/\b(?:Mrs\.|MRS\.(?= [A-Z]))/g, "Mrs")
    .replace(/\betc\.(?! [A-Z])/gi, "etc")

    // 5. Normalize casual words
    .replace(/\b(y)eah?\b/gi, "$1e'a")

    // 5. Handle numbers and currencies
    .replace(/\d*\.\d+|\b\d{4}s?\b|(?<!:)\b(?:[1-9]|1[0-2]):[0-5]\d\b(?!:)/g, split_num)
    .replace(/(?<=\d),(?=\d)/g, "")
    .replace(/[$£]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[$£]\d+\.\d\d?\b/gi, flip_money)
    .replace(/\d*\.\d+/g, point_num)
    .replace(/(?<=\d)-(?=\d)/g, " to ")
    .replace(/(?<=\d)S/g, " S")

    // 6. Handle possessives
    .replace(/(?<=[BCDFGHJ-NP-TV-Z])'?s\b/g, "'S")
    .replace(/(?<=X')S\b/g, "s")

    // 7. Handle hyphenated words/letters
    .replace(/(?:[A-Za-z]\.){2,} [a-z]/g, (m) => m.replace(/\./g, "-"))
    .replace(/(?<=[A-Z])\.(?=[A-Z])/gi, "-");
}

/**
 * Escapes regular expression special characters from a string by replacing them with their escaped counterparts.
 *
 * @param {string} string The string to escape.
 * @returns {string} The escaped string.
 */
function escapeRegExp(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); // $& means the whole matched string
}

const PUNCTUATION = ';:,.!?¡¿—\u2026"\u00ab\u00bb(){}[]';
const PUNCTUATION_PATTERN = new RegExp(`(\\s*[${escapeRegExp(PUNCTUATION)}]+\\s*)+`, "g");

/**
 * Maps voice language prefix to ephone pack and voice name.
 * @type {Record<string, { pack: import("ephone").ephoneLanguagePack, voice: string }>}
 */
const LANG_CONFIG = {
  a: { pack: en_us, voice: "en-US" },
  b: { pack: en_all, voice: "en" },
  e: { pack: roa, voice: "es" },
  f: { pack: roa, voice: "fr" },
  i: { pack: roa, voice: "it" },
  p: { pack: roa, voice: "pt-BR" },
  j: { pack: jpx, voice: "ja" },
  z: { pack: sit, voice: "cmn" },
  // h = Hindi: handled via lazy dynamic import of the 'all' pack
};

/** @type {Map<Function, Promise<import("ephone").ephoneModule>>} */
const _cache = new Map();

/**
 * Get (or lazily initialize) an ephone instance for the given language pack.
 * @param {import("ephone").ephoneLanguagePack} pack
 * @returns {Promise<import("ephone").ephoneModule>}
 */
function getEphone(pack) {
  if (!_cache.has(pack)) {
    _cache.set(pack, createEphone(pack));
  }
  return _cache.get(pack);
}

/**
 * Phonemize text using ephone (eSpeak-NG WASM)
 * @param {string} text The text to phonemize
 * @param {"a"|"b"|"j"|"z"|"e"|"f"|"h"|"i"|"p"} language The language to use
 * @param {boolean} norm Whether to normalize the text
 * @returns {Promise<string>} The phonemized text
 */
export async function phonemize(text, language = "a", norm = true) {
  const isEnglish = language === "a" || language === "b";

  // 1. Normalize text
  if (norm) {
    text = normalize_text(text, isEnglish);
  }

  // 2. Split into chunks, to ensure we preserve punctuation
  const sections = split(text, PUNCTUATION_PATTERN);

  // 3. Get ephone instance for this language
  /** @type {import("ephone").ephoneModule} */
  let ephone;
  if (language === "h") {
    // Hindi requires the large 'all' language pack (~18MB) — loaded on demand
    const { all } = await import("ephone");
    ephone = await getEphone(all);
  } else {
    const config = LANG_CONFIG[language] ?? LANG_CONFIG["a"];
    ephone = await getEphone(config.pack);
  }

  // Set voice (synchronous — safe to call immediately before textToIpa)
  const voiceName = language === "h" ? "hi" : (LANG_CONFIG[language] ?? LANG_CONFIG["a"]).voice;
  ephone.setVoice(voiceName);

  // 4. Convert each non-punctuation section to IPA phonemes
  const ps = sections
    .map(({ match, text: t }) => {
      if (match) return t;
      if (!t.trim()) return t;
      // ephone always appends a trailing "." — strip it
      return ephone.textToIpa(t).replace(/\.$/, "").trim();
    })
    .join("");

  // 5. Universal post-processing
  let processed = ps.replace(/ʲ/g, "j");

  if (isEnglish) {
    processed = processed
      // Convert eSpeak-NG's /r/ to English retroflex /ɹ/
      .replace(/r/g, "ɹ")
      // https://en.wiktionary.org/wiki/kokoro#English
      .replace(/kəkˈoːɹoʊ/g, "kˈoʊkəɹoʊ")
      .replace(/kəkˈɔːɹəʊ/g, "kˈəʊkəɹəʊ")
      .replace(/x/g, "k")
      .replace(/ɬ/g, "l")
      .replace(/(?<=[a-zɹː])(?=hˈʌndɹɪd)/g, " ")
      .replace(/ z(?=[;:,.!?¡¿—…"«»"" ]|$)/g, "z");

    // Additional post-processing for American English
    if (language === "a") {
      processed = processed.replace(/(?<=nˈaɪn)ti(?!ː)/g, "di");
    }
  }

  return processed.trim();
}
