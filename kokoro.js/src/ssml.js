/**
 * Lightweight SSML parser for kokoro-js.
 *
 * Design principles:
 * - Regex-based, never throws. Malformed or unknown tags are passed through as
 *   literal text so synthesis always degrades gracefully.
 * - Input containing bare `<` that is not part of a recognised SSML tag (e.g.
 *   "2 < 3") is left unchanged.
 * - Nested SSML is not supported; inner tags inside a known tag are treated as
 *   plain text content.
 *
 * Supported tags (priority-ordered per the feature spec):
 *   <phoneme alphabet="ipa" ph="...">word</phoneme>
 *   <break time="500ms"/> / <break time="1.5s"/>
 *   <sub alias="...">text</sub>
 *   <say-as interpret-as="characters|number|ordinal">text</say-as>
 *
 * IPA format note:
 *   The `ph` attribute of `<phoneme>` must use **eSpeak IPA notation**, which
 *   differs from standard (broad) IPA in several important ways:
 *
 *   1. Stress marks go before the stressed **vowel**, not the syllable onset.
 *      - ✅ eSpeak: wˈɜːld  (ˈ immediately before the vowel ɜ)
 *      - ❌ Standard: ˈwɜːld (ˈ before the consonant onset w)
 *      Placing ˈ before a consonant causes the model to vocalize it as a
 *      separate phoneme (often heard as an "ah" sound) rather than as stress.
 *
 *   2. The English rhotic is ɹ, not r.
 *      - ✅ eSpeak: ɹɪd  ❌ Standard: rɪd
 *
 *   To find the correct eSpeak IPA for any word, run:
 *      espeak-ng --ipa -q -v en-us "word"
 */

/**
 * @typedef {{ type: 'text';    value: string }} TextSegment
 * @typedef {{ type: 'phoneme'; text: string; ipa: string }} PhonemeSegment
 * @typedef {{ type: 'break';   ms: number }} BreakSegment
 * @typedef {{ type: 'sub';     text: string; alias: string }} SubSegment
 * @typedef {{ type: 'say-as';  text: string; interpretAs: 'characters'|'number'|'ordinal' }} SayAsSegment
 * @typedef {TextSegment|PhonemeSegment|BreakSegment|SubSegment|SayAsSegment} SSMLSegment
 */

/**
 * Returns true if the text contains any `<` character, indicating it may
 * contain SSML tags. Used as a fast guard to skip parsing on plain text.
 * @param {string} text
 * @returns {boolean}
 */
export function hasSSML(text) {
  return text.includes("<");
}

/**
 * Parse an XML-style attribute string into a key→value map.
 * Handles both single- and double-quoted values.
 * @param {string} attrStr
 * @returns {Record<string, string>}
 */
function parseAttrs(attrStr) {
  const attrs = /** @type {Record<string, string>} */ ({});
  const re = /(\w[\w-]*)=(?:"([^"]*)"|'([^']*)')/g;
  let m;
  while ((m = re.exec(attrStr)) !== null) {
    attrs[m[1]] = m[2] !== undefined ? m[2] : m[3];
  }
  return attrs;
}

/**
 * Convert a SSML break `time` attribute value to milliseconds.
 * Supports "500ms" and "1s" / "1.5s" formats; unknown formats return 0.
 * @param {string} time
 * @returns {number}
 */
export function parseBreakMs(time) {
  if (!time) return 0;
  const t = time.trim();
  if (t.endsWith("ms")) return parseFloat(t) || 0;
  if (t.endsWith("s")) return (parseFloat(t) || 0) * 1000;
  return 0;
}

/**
 * Parse an SSML string into an array of typed segments.
 *
 * Unknown or malformed tags are emitted as `{ type: 'text', value: originalText }`.
 * Text between tags is emitted as-is.
 *
 * @param {string} text
 * @returns {SSMLSegment[]}
 */
export function parseSSML(text) {
  const segments = /** @type {SSMLSegment[]} */ ([]);

  // Matches either:
  //   self-closing:  <tagName attrs/>
  //   paired:        <tagName attrs>inner</tagName>
  // Attribute values may contain single or double quoted strings.
  const TAG_RE = /<([\w-]+)((?:\s+(?:[\w-]+=(?:"[^"]*"|'[^']*')|[\w-]+))*)\s*(?:\/>()|>([\s\S]*?)<\/\1>)/g;

  let lastIndex = 0;

  for (const match of text.matchAll(TAG_RE)) {
    const [fullMatch, tagName, attrStr, selfClose, inner] = match;
    const matchStart = /** @type {number} */ (match.index);

    // Flush plain text before this tag.
    if (matchStart > lastIndex) {
      segments.push({ type: "text", value: text.slice(lastIndex, matchStart) });
    }
    lastIndex = matchStart + fullMatch.length;

    const attrs = parseAttrs(attrStr);
    const isSelfClosing = selfClose !== undefined;
    const content = inner ?? "";

    switch (tagName.toLowerCase()) {
      case "phoneme": {
        // Requires alphabet="ipa" and ph="..." — anything else falls through as text.
        // The ph value must use eSpeak IPA notation (see module-level comment).
        if (attrs["alphabet"]?.toLowerCase() === "ipa" && attrs["ph"]) {
          segments.push({ type: "phoneme", text: content, ipa: attrs["ph"] });
        } else {
          if (attrs["alphabet"] && attrs["alphabet"].toLowerCase() !== "ipa") {
            console.warn(`[kokoro-js] <phoneme>: only alphabet="ipa" is supported (got "${attrs["alphabet"]}"); treating as plain text.`);
          }
          segments.push({ type: "text", value: content });
        }
        break;
      }

      case "break": {
        if (!isSelfClosing) {
          // Malformed — treat as plain text (content is empty for self-closing anyway).
          segments.push({ type: "text", value: fullMatch });
          break;
        }
        const ms = parseBreakMs(attrs["time"] ?? "");
        segments.push({ type: "break", ms });
        break;
      }

      case "sub": {
        if (!attrs["alias"]) {
          // Missing alias — fall back to inner content.
          segments.push({ type: "text", value: content });
        } else {
          segments.push({ type: "sub", text: content, alias: attrs["alias"] });
        }
        break;
      }

      case "say-as": {
        const interpretAs = (attrs["interpret-as"] ?? "").toLowerCase();
        if (interpretAs === "characters" || interpretAs === "number" || interpretAs === "ordinal") {
          segments.push({ type: "say-as", text: content, interpretAs: /** @type {'characters'|'number'|'ordinal'} */ (interpretAs) });
        } else {
          // Unknown interpret-as value — pass inner text through.
          segments.push({ type: "text", value: content });
        }
        break;
      }

      default:
        // Unknown tag — pass the entire raw match through as literal text.
        segments.push({ type: "text", value: fullMatch });
        break;
    }
  }

  // Flush any trailing plain text.
  if (lastIndex < text.length) {
    segments.push({ type: "text", value: text.slice(lastIndex) });
  }

  return segments;
}

/**
 * Split text into alternating text and break segments.
 * Only `<break>` tags are extracted; all other SSML tags are left intact in the
 * text segments so that the phonemize layer can process them.
 *
 * @param {string} text
 * @returns {Array<{type:'text',value:string}|BreakSegment>}
 */
export function splitAtBreaks(text) {
  const BREAK_RE = /<break((?:\s+(?:[\w-]+=(?:"[^"]*"|'[^']*')|[\w-]+))*)\s*\/>/g;
  const result = /** @type {Array<{type:'text',value:string}|BreakSegment>} */ ([]);
  let lastIndex = 0;

  for (const match of text.matchAll(BREAK_RE)) {
    const attrStr = match[1];
    const matchStart = /** @type {number} */ (match.index);

    if (matchStart > lastIndex) {
      result.push({ type: "text", value: text.slice(lastIndex, matchStart) });
    }
    lastIndex = matchStart + match[0].length;

    const ms = parseBreakMs(parseAttrs(attrStr)["time"] ?? "");
    result.push({ type: "break", ms });
  }

  if (lastIndex < text.length) {
    result.push({ type: "text", value: text.slice(lastIndex) });
  }

  // Filter out empty text segments that would produce no audio.
  return result.filter((s) => s.type === "break" || s.value.trim().length > 0);
}
