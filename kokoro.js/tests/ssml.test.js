import { describe, test, expect } from "vitest";
import { hasSSML, parseSSML, parseBreakMs, splitAtBreaks } from "../src/ssml.js";
import { phonemize } from "../src/phonemize.js";

// ─── hasSSML ─────────────────────────────────────────────────────────────────

describe("hasSSML", () => {
  test("plain text → false", () => expect(hasSSML("Hello world")).toBe(false));
  test("text with < but no tag → true (conservative)", () => expect(hasSSML("2 < 3")).toBe(true));
  test("valid SSML tag → true", () => expect(hasSSML('<break time="500ms"/>')).toBe(true));
});

// ─── parseBreakMs ─────────────────────────────────────────────────────────────

describe("parseBreakMs", () => {
  test("ms suffix", () => expect(parseBreakMs("500ms")).toBe(500));
  test("s suffix (integer)", () => expect(parseBreakMs("1s")).toBe(1000));
  test("s suffix (decimal)", () => expect(parseBreakMs("1.5s")).toBe(1500));
  test("empty string → 0", () => expect(parseBreakMs("")).toBe(0));
  test("unknown format → 0", () => expect(parseBreakMs("100")).toBe(0));
  test("whitespace trimmed", () => expect(parseBreakMs(" 250ms ")).toBe(250));
});

// ─── parseSSML ────────────────────────────────────────────────────────────────

describe("parseSSML", () => {
  // Plain text
  test("plain text → single text segment", () => {
    expect(parseSSML("Hello world")).toEqual([{ type: "text", value: "Hello world" }]);
  });

  test("empty string → empty array", () => {
    expect(parseSSML("")).toEqual([]);
  });

  // <break>
  test("<break time='500ms'/> → break segment", () => {
    expect(parseSSML('<break time="500ms"/>')).toEqual([{ type: "break", ms: 500 }]);
  });

  test("<break time='1.5s'/> → break segment", () => {
    expect(parseSSML('<break time="1.5s"/>')).toEqual([{ type: "break", ms: 1500 }]);
  });

  test("<break/> with no time → ms=0", () => {
    expect(parseSSML("<break/>")).toEqual([{ type: "break", ms: 0 }]);
  });

  test("text around break split correctly", () => {
    expect(parseSSML('Hello<break time="500ms"/>World')).toEqual([
      { type: "text", value: "Hello" },
      { type: "break", ms: 500 },
      { type: "text", value: "World" },
    ]);
  });

  // <phoneme>
  test("<phoneme alphabet='ipa' ph='...'>word</phoneme> → phoneme segment", () => {
    expect(parseSSML('<phoneme alphabet="ipa" ph="wɜːld">world</phoneme>')).toEqual([
      { type: "phoneme", text: "world", ipa: "wɜːld" },
    ]);
  });

  test("<phoneme> without ph → text segment (fallback)", () => {
    expect(parseSSML("<phoneme alphabet=\"ipa\">world</phoneme>")).toEqual([
      { type: "text", value: "world" },
    ]);
  });

  test("<phoneme> with non-ipa alphabet → text segment (fallback)", () => {
    expect(parseSSML('<phoneme alphabet="x-sampa" ph="wE:ld">world</phoneme>')).toEqual([
      { type: "text", value: "world" },
    ]);
  });

  // <sub>
  test("<sub alias='...'>text</sub> → sub segment", () => {
    expect(parseSSML('<sub alias="World Wide Web Consortium">W3C</sub>')).toEqual([
      { type: "sub", text: "W3C", alias: "World Wide Web Consortium" },
    ]);
  });

  test("<sub> without alias → text segment (fallback)", () => {
    expect(parseSSML("<sub>W3C</sub>")).toEqual([{ type: "text", value: "W3C" }]);
  });

  // <say-as>
  test("<say-as interpret-as='characters'> → say-as segment", () => {
    expect(parseSSML('<say-as interpret-as="characters">SQL</say-as>')).toEqual([
      { type: "say-as", text: "SQL", interpretAs: "characters" },
    ]);
  });

  test("<say-as interpret-as='ordinal'> → say-as segment", () => {
    expect(parseSSML('<say-as interpret-as="ordinal">42</say-as>')).toEqual([
      { type: "say-as", text: "42", interpretAs: "ordinal" },
    ]);
  });

  test("<say-as interpret-as='number'> → say-as segment", () => {
    expect(parseSSML('<say-as interpret-as="number">100</say-as>')).toEqual([
      { type: "say-as", text: "100", interpretAs: "number" },
    ]);
  });

  test("<say-as> with unknown interpret-as → text segment (fallback)", () => {
    expect(parseSSML('<say-as interpret-as="date">2024-01-01</say-as>')).toEqual([
      { type: "text", value: "2024-01-01" },
    ]);
  });

  // Unknown tags
  test("unknown tag → literal text pass-through", () => {
    expect(parseSSML("<emphasis>hello</emphasis>")).toEqual([
      { type: "text", value: "<emphasis>hello</emphasis>" },
    ]);
  });

  // Malformed / edge cases
  test("bare < not part of a tag → literal text", () => {
    expect(parseSSML("2 < 3")).toEqual([{ type: "text", value: "2 < 3" }]);
  });

  // Mixed
  test("mixed text and tags", () => {
    expect(parseSSML('Say <sub alias="Cascading Style Sheets">CSS</sub> and <break time="300ms"/> done.')).toEqual([
      { type: "text", value: "Say " },
      { type: "sub", text: "CSS", alias: "Cascading Style Sheets" },
      { type: "text", value: " and " },
      { type: "break", ms: 300 },
      { type: "text", value: " done." },
    ]);
  });
});

// ─── splitAtBreaks ────────────────────────────────────────────────────────────

describe("splitAtBreaks", () => {
  test("no breaks → single text segment", () => {
    expect(splitAtBreaks("Hello world")).toEqual([{ type: "text", value: "Hello world" }]);
  });

  test("single break → text, break, text", () => {
    expect(splitAtBreaks('Hello<break time="500ms"/>World')).toEqual([
      { type: "text", value: "Hello" },
      { type: "break", ms: 500 },
      { type: "text", value: "World" },
    ]);
  });

  test("multiple breaks", () => {
    expect(splitAtBreaks('A<break time="100ms"/>B<break time="200ms"/>C')).toEqual([
      { type: "text", value: "A" },
      { type: "break", ms: 100 },
      { type: "text", value: "B" },
      { type: "break", ms: 200 },
      { type: "text", value: "C" },
    ]);
  });

  test("non-break SSML tags left intact in text segments", () => {
    const result = splitAtBreaks('Hello <phoneme alphabet="ipa" ph="wɜːld">world</phoneme><break time="500ms"/>!');
    expect(result).toEqual([
      { type: "text", value: 'Hello <phoneme alphabet="ipa" ph="wɜːld">world</phoneme>' },
      { type: "break", ms: 500 },
      { type: "text", value: "!" },
    ]);
  });

  test("whitespace-only text between breaks is filtered out", () => {
    const result = splitAtBreaks('<break time="100ms"/>   <break time="200ms"/>');
    expect(result).toEqual([
      { type: "break", ms: 100 },
      { type: "break", ms: 200 },
    ]);
  });
});

// ─── phonemize() with SSML tags ───────────────────────────────────────────────

describe("phonemize with SSML", () => {
  // Fast path: plain text is unchanged relative to calling phonemize without SSML.
  test("plain text fast-path is unaffected", async () => {
    expect(await phonemize("Hello World")).toEqual("həlˈoʊ wˈɜːld");
  });

  // <phoneme> — IPA injected directly, G2P bypassed.
  test("<phoneme> injects IPA for tagged word", async () => {
    const result = await phonemize('Hello <phoneme alphabet="ipa" ph="wɜːld">world</phoneme>!');
    expect(result).toContain("wɜːld");
    expect(result).toMatch(/həlˈoʊ/);
  });

  test("<phoneme> without ipa alphabet falls back to plain text synthesis", async () => {
    // No throw — "world" is phonemized normally.
    const result = await phonemize('<phoneme alphabet="x-sampa" ph="wE:ld">world</phoneme>');
    expect(result).toMatch(/wˈɜːld/);
  });

  // <sub> — alias replaces display text.
  test("<sub> synthesizes alias instead of display text", async () => {
    const abbrev = await phonemize("W3C");
    const expanded = await phonemize("World Wide Web Consortium");
    const sub = await phonemize('<sub alias="World Wide Web Consortium">W3C</sub>');
    expect(sub).toEqual(expanded);
    expect(sub).not.toEqual(abbrev);
  });

  // <say-as interpret-as="characters">
  test("<say-as characters> reads each letter individually", async () => {
    // "SQL" as characters should produce phonemes for S, Q, L separately.
    // Plain "SQL" would be read as "sequel" by eSpeak.
    const chars = await phonemize('<say-as interpret-as="characters">SQL</say-as>');
    const plain = await phonemize("SQL");
    expect(chars).not.toEqual(plain);
    // Individual letter names should appear: ɛs (S), kjuː (Q), ɛl (L)
    expect(chars).toMatch(/ˈɛs/);
  });

  // <say-as interpret-as="ordinal">
  test("<say-as ordinal> 1 → 'first'", async () => {
    const result = await phonemize('<say-as interpret-as="ordinal">1</say-as>');
    expect(result).toMatch(/fˈɜːst/);
  });

  test("<say-as ordinal> 42 → '42nd'", async () => {
    const result = await phonemize('<say-as interpret-as="ordinal">42</say-as>');
    const fortySecond = await phonemize("42nd");
    expect(result).toEqual(fortySecond);
  });

  test("<say-as ordinal> 11 → '11th' (not '11st')", async () => {
    const result = await phonemize('<say-as interpret-as="ordinal">11</say-as>');
    const eleventhExpected = await phonemize("11th");
    expect(result).toEqual(eleventhExpected);
  });

  // <say-as interpret-as="number">
  test("<say-as number> behaves like plain number", async () => {
    const ssml = await phonemize('<say-as interpret-as="number">1990</say-as>');
    const plain = await phonemize("1990");
    expect(ssml).toEqual(plain);
  });

  // Mixed
  test("mixed tags in one sentence", async () => {
    const result = await phonemize('The <sub alias="World Wide Web Consortium">W3C</sub> and <phoneme alphabet="ipa" ph="ˈɛskjuːˈɛl">SQL</phoneme>.');
    expect(result).toMatch(/wˈɜːld/); // from "World"
    expect(result).toContain("ˈɛskjuːˈɛl"); // injected IPA
  });
});
