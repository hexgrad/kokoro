import { pinyin } from "pinyin-pro";

const INITIALS = {
  b: "p", c: "ʦʰ", ch: "ꭧʰ", d: "t", f: "f", g: "k", h: "x",
  j: "ʨ", k: "kʰ", l: "l", m: "m", n: "n", p: "pʰ", q: "ʨʰ",
  r: "ɻ", s: "s", sh: "ʂ", t: "tʰ", x: "ɕ", z: "ʦ", zh: "ꭧ",
};

const FINALS = {
  a: "a", ai: "ai̯", an: "an", ang: "aŋ", ao: "au̯",
  e: "ɤ", ei: "ei̯", en: "ən", eng: "əŋ", er: "ɚ",
  i: "i", ia: "ja", ian: "jɛn", iang: "jaŋ", iao: "jau̯",
  ie: "je", in: "in", iou: "jou̯", ing: "iŋ", iong: "jʊŋ",
  o: "wo", ong: "ʊŋ", ou: "ou̯",
  u: "u", ua: "wa", uai: "wai̯", uan: "wan", uang: "waŋ",
  uei: "wei̯", uen: "wən", ueng: "wəŋ", uo: "wo",
  ü: "y", üe: "ɥe", üan: "ɥɛn", ün: "yn",
};

const TONES = { 1: "→", 2: "↗", 3: "↓", 4: "↘", 5: "" };
const INITIAL_PATTERN = /^(zh|ch|sh|[bpmfdtnlgkhjqxrzcsyw])/;

function normalizeSyllable(syllable) {
  const match = syllable.toLowerCase().match(/^([a-züvê]+)([1-5])$/u);
  if (!match) return null;
  let [, body, tone] = match;
  body = body.replaceAll("v", "ü");

  let initial = body.match(INITIAL_PATTERN)?.[0] ?? "";
  let final = body.slice(initial.length);

  if (initial === "y") {
    const yFinals = {
      i: "i", a: "ia", ao: "iao", e: "ie", ou: "iou", an: "ian",
      in: "in", ang: "iang", ing: "ing", ong: "iong", u: "ü",
      ue: "üe", uan: "üan", un: "ün",
    };
    final = yFinals[final] ?? final;
    initial = "";
  } else if (initial === "w") {
    const wFinals = {
      u: "u", a: "ua", o: "uo", ai: "uai", ei: "uei",
      an: "uan", en: "uen", ang: "uang", eng: "ueng",
    };
    final = wFinals[final] ?? final;
    initial = "";
  } else {
    if (final === "iu") final = "iou";
    if (final === "ui") final = "uei";
    if (final === "un") final = ["j", "q", "x"].includes(initial) ? "ün" : "uen";
    if (["j", "q", "x"].includes(initial) && final.startsWith("u")) {
      final = `ü${final.slice(1)}`;
    }
  }

  if (final === "i" && ["zh", "ch", "sh", "r"].includes(initial)) final = "ɻ̩";
  if (final === "i" && ["z", "c", "s"].includes(initial)) final = "ɹ̩";

  const initialIpa = INITIALS[initial] ?? "";
  const finalIpa = FINALS[final] ?? final;
  return `${initialIpa}${finalIpa}${TONES[Number(tone)]}`;
}

function mapPunctuation(text) {
  return text
    .replace(/[、，]/g, ", ")
    .replace(/[。．]/g, ". ")
    .replace(/！/g, "! ")
    .replace(/：/g, ": ")
    .replace(/；/g, "; ")
    .replace(/？/g, "? ")
    .replace(/[«《「【]/g, " “")
    .replace(/[»》」】]/g, "” ")
    .replace(/（/g, " (")
    .replace(/）/g, ") ");
}

/**
 * Convert Mandarin text to Kokoro-compatible IPA.
 * @param {string} text
 * @param {(text: string) => Promise<string>} [englishPhonemize]
 * @returns {Promise<string>}
 */
export async function phonemizeChinese(text, englishPhonemize) {
  const tokens = pinyin(mapPunctuation(text), {
    toneType: "num",
    type: "array",
    nonZh: "consecutive",
  });

  const output = [];
  for (const token of tokens) {
    const ipa = normalizeSyllable(token);
    if (ipa) {
      output.push(ipa);
    } else if (englishPhonemize && /[A-Za-z]/.test(token)) {
      output.push(await englishPhonemize(token.trim()));
    } else {
      output.push(token);
    }
  }
  return output.join("").replace(/\s+/g, " ").trim();
}

export { normalizeSyllable };
