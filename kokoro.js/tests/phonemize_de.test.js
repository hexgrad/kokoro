import { describe, test, expect } from "vitest";
import { normalize_text_de } from "../src/phonemize_de.js";

// ─── intToDE (tested via normalize_text_de) ───────────────────────────────────

describe("intToDE via normalize_text_de", () => {
  const cases = [
    ["0",         "null"],
    ["1",         "ein"],
    ["11",        "elf"],
    ["12",        "zwölf"],
    ["20",        "zwanzig"],
    ["21",        "einundzwanzig"],
    ["42",        "zweiundvierzig"],
    ["99",        "neunundneunzig"],
    ["100",       "einhundert"],
    ["1000",      "eintausend"],
    ["1000000",   "eine Million"],
    ["2000000",   "zwei Millionen"],
  ];
  for (const [input, expected] of cases) {
    test(`${input} → contains "${expected}"`, () => {
      expect(normalize_text_de(`Es gibt ${input} Dinge.`)).toContain(expected);
    });
  }
});

// ─── ordinals ─────────────────────────────────────────────────────────────────

describe("ordinals", () => {
  // Ordinals appear when a number precedes a period mid-sentence
  test("1. → erste", () => expect(normalize_text_de("Am 1. Mai")).toContain("erste"));
  test("3. → dritte", () => expect(normalize_text_de("Am 3. Oktober")).toContain("dritte"));
  test("7. → siebte", () => expect(normalize_text_de("Am 7. Juli")).toContain("siebte"));
  test("20. → zwanzigste", () => expect(normalize_text_de("Am 20. August")).toContain("zwanzigste"));
});

// ─── years ────────────────────────────────────────────────────────────────────

describe("years", () => {
  test("1989 → neunzehnhundert...", () => {
    expect(normalize_text_de("Im Jahr 1989.")).toContain("neunzehnhundert");
  });
  test("1900 → neunzehnhundert", () => {
    expect(normalize_text_de("Seit 1900.")).toContain("neunzehnhundert");
  });
  test("2024 → zweitausend...", () => {
    expect(normalize_text_de("Im Jahr 2024.")).toContain("zweitausend");
  });
  test("1985 digits removed", () => {
    expect(normalize_text_de("Im Jahr 1985.")).not.toContain("1985");
  });
});

// ─── quotes ───────────────────────────────────────────────────────────────────

// describe("quotes", () => {
//   test("„ and " normalised", () => {
//     const r = normalize_text_de('Er sagte: „Guten Morgen."');
//     expect(r).not.toContain("\u201e");
//     expect(r).not.toContain("\u201c");
//   });
//   test("« and » normalised", () => {
//     const r = normalize_text_de("Das ist «toll».");
//     expect(r).not.toContain("«");
//     expect(r).not.toContain("»");
//   });
//   test("‹ and › normalised", () => {
//     const r = normalize_text_de("‹test›");
//     expect(r).not.toContain("‹");
//     expect(r).not.toContain("›");
//   });
//   test("curly apostrophe normalised", () => {
//     expect(normalize_text_de("It\u2019s fine")).not.toContain("\u2019");
//   });
// });
// ─── quotes ───────────────────────────────────────────────────────────────────

describe("quotes", () => {
  test('„ and " normalised', () => {
    const r = normalize_text_de('Er sagte: „Guten Morgen."');
    expect(r).not.toContain("\u201e");
    expect(r).not.toContain("\u201c");
  });
  test("« and » normalised", () => {
    const r = normalize_text_de("Das ist «toll».");
    expect(r).not.toContain("«");
    expect(r).not.toContain("»");
  });
  test("‹ and › normalised", () => {
    const r = normalize_text_de("‹test›");
    expect(r).not.toContain("‹");
    expect(r).not.toContain("›");
  });
  test("curly apostrophe normalised", () => {
    expect(normalize_text_de("It\u2019s fine")).not.toContain("\u2019");
  });
});
// ─── abbreviations ────────────────────────────────────────────────────────────

describe("abbreviations", () => {
  test("Dr. → Doktor", () => expect(normalize_text_de("Dr. Müller")).toContain("Doktor"));
  test("Prof. → Professor", () => expect(normalize_text_de("Prof. Schmidt hält")).toContain("Professor"));

  test("Str. standalone → Straße", () => expect(normalize_text_de("Str. des Friedens")).toContain("Straße"));
  test("Hauptstr. suffix → Straße", () => expect(normalize_text_de("In der Hauptstr. links")).toContain("Straße"));
  test("Musterstr. suffix → Straße", () => expect(normalize_text_de("In der Musterstr. links")).toContain("Straße"));

  test("Nr. → Nummer", () => expect(normalize_text_de("Nr. 5 bitte")).toContain("Nummer"));
  test("Tel. → Telefon", () => expect(normalize_text_de("Tel. 0800 links")).toContain("Telefon"));
  test("z.B. → zum Beispiel", () => expect(normalize_text_de("z.B. morgen")).toContain("zum Beispiel"));
  test("d.h. → das heißt", () => expect(normalize_text_de("d.h. später")).toContain("das heißt"));
  test("usw. → und so weiter", () => expect(normalize_text_de("Äpfel usw.")).toContain("und so weiter"));
  test("bzw. → beziehungsweise", () => expect(normalize_text_de("der Hund bzw. die Katze")).toContain("beziehungsweise"));
  test("etc. → et cetera", () => expect(normalize_text_de("und so etc.")).toContain("et cetera"));
  test("ca. → circa", () => expect(normalize_text_de("ca. 10 Minuten")).toContain("circa"));
  test("GmbH → full form", () => expect(normalize_text_de("Muster GmbH")).toContain("Gesellschaft mit beschränkter Haftung"));
  test("AG → Aktiengesellschaft", () => expect(normalize_text_de("Siemens AG,")).toContain("Aktiengesellschaft"));

  // Month abbreviations
  test("Jan. → Januar", () => expect(normalize_text_de("Jan. 2024 war kalt")).toContain("Januar"));
  test("Feb. → Februar", () => expect(normalize_text_de("Feb. 2024 ")).toContain("Februar"));
  test("Okt. → Oktober", () => expect(normalize_text_de("Okt. 2023 ")).toContain("Oktober"));
  test("Dez. → Dezember", () => expect(normalize_text_de("Dez. 2024 ")).toContain("Dezember"));
});

// ─── numbers ──────────────────────────────────────────────────────────────────

describe("numbers", () => {
  test("0 → null", () => expect(normalize_text_de("Er hat 0 Punkte.")).toContain("null"));
  test("5 → fünf", () => expect(normalize_text_de("5 Katzen.")).toContain("fünf"));
  test("42 → zweiundvierzig", () => expect(normalize_text_de("42 Leute.")).toContain("zweiundvierzig"));
  test("42 digit removed", () => expect(normalize_text_de("42 Leute.")).not.toContain("42"));
  test("100 → einhundert", () => expect(normalize_text_de("100 Punkte.")).toContain("einhundert"));

  test("German thousands 1.000 → eintausend", () => {
    const r = normalize_text_de("1.000 Menschen.");
    expect(r).toContain("tausend");
    expect(r).not.toContain("1.000");
  });

  test("German millions 1.234.567", () => {
    const r = normalize_text_de("1.234.567 Einwohner.");
    expect(r).not.toContain("1.234.567");
  });

  test("decimal comma 36,9 → Komma", () => {
    const r = normalize_text_de("36,9 Grad.");
    expect(r).toContain("Komma");
    expect(r).not.toContain("36,9");
  });

  test("decimal comma 3,14", () => {
    expect(normalize_text_de("Pi ist ca. 3,14.")).toContain("Komma");
  });
});

// ─── currency ─────────────────────────────────────────────────────────────────

describe("currency", () => {
  test("€ before number", () => {
    const r = normalize_text_de("kostet €10");
    expect(r).toContain("Euro");
    expect(r).not.toContain("€");
    expect(r).toContain("zehn");
  });

  test("€ after number", () => {
    const r = normalize_text_de("kostet 10€");
    expect(r).toContain("Euro");
    expect(r).not.toContain("€");
  });

  test("€ with cents", () => {
    const r = normalize_text_de("€9,99 bitte");
    expect(r).toContain("Euro");
    expect(r).toContain("Cent");
    expect(r).not.toContain("€");
  });

  test("€29,99 cents split", () => {
    const r = normalize_text_de("für €29,99");
    expect(r).toContain("neunundzwanzig Euro");
    expect(r).toContain("Cent");
  });

  test("$ → Dollar", () => {
    const r = normalize_text_de("$100 Rabatt");
    expect(r).toContain("Dollar");
    expect(r).not.toContain("$");
  });

  test("£ → Pfund", () => {
    const r = normalize_text_de("£50 Pfund");
    expect(r).toContain("Pfund");
  });
});

// ─── times ────────────────────────────────────────────────────────────────────

describe("times", () => {
  test("14:00 → vierzehn Uhr", () => {
    const r = normalize_text_de("Um 14:00 Uhr.");
    expect(r).toContain("vierzehn Uhr");
    expect(r).not.toContain("14:00");
  });

  test("8:30 → acht Uhr dreißig", () => {
    expect(normalize_text_de("Um 8:30 Uhr.")).toContain("acht Uhr dreißig");
  });

  test("0:00 → null Uhr", () => {
    expect(normalize_text_de("Um 0:00 Uhr.")).toContain("null Uhr");
  });

  test("12:00 → zwölf Uhr", () => {
    expect(normalize_text_de("Um 12:00 Uhr.")).toContain("zwölf Uhr");
  });

  test("15:00 → fünfzehn Uhr (no trailing null)", () => {
    const r = normalize_text_de("Um 15:00");
    expect(r).toContain("fünfzehn Uhr");
    expect(r).not.toContain("null");
  });

  test("9:05 → neun Uhr fünf", () => {
    expect(normalize_text_de("Um 9:05 Uhr.")).toContain("neun Uhr fünf");
  });
});

// ─── dates ────────────────────────────────────────────────────────────────────

describe("dates", () => {
  test("24.12.2024 → Dezember", () => {
    const r = normalize_text_de("Am 24.12.2024.");
    expect(r).toContain("Dezember");
    expect(r).not.toContain("24.12.2024");
  });

  test("1.1.2000 → erste Januar", () => {
    const r = normalize_text_de("Am 1.1.2000.");
    expect(r).toContain("erste");
    expect(r).toContain("Januar");
  });

  test("3.10.1990 → dritte Oktober", () => {
    const r = normalize_text_de("Am 3.10.1990.");
    expect(r).toContain("dritt");
    expect(r).toContain("Oktober");
  });

  test("9.11.1989 → year in output", () => {
    const r = normalize_text_de("Am 9.11.1989.");
    expect(r).toContain("neunzehnhundert");
  });
});

// ─── whitespace ───────────────────────────────────────────────────────────────

describe("whitespace", () => {
  test("double spaces collapsed", () => {
    expect(normalize_text_de("Hallo   Welt")).not.toContain("  ");
  });
  test("leading/trailing trimmed", () => {
    const r = normalize_text_de("  Hallo Welt  ");
    expect(r).toBe(r.trim());
  });
  test("non-breaking space normalised", () => {
    expect(normalize_text_de("Hallo\u00a0Welt")).not.toContain("\u00a0");
  });
});

// ─── umlauts & special chars ──────────────────────────────────────────────────

describe("umlauts", () => {
  test("ä ö ü ß Ä Ö Ü preserved", () => {
    const r = normalize_text_de("Äpfel, Österreich, Überraschung, Größe, schöne");
    expect(r).toContain("Äpfel");
    expect(r).toContain("Österreich");
    expect(r).toContain("Überraschung");
    expect(r).toContain("Größe");
    expect(r).toContain("schöne");
  });
});

// ─── edge cases ───────────────────────────────────────────────────────────────

describe("edge cases", () => {
  test("empty string → empty string", () => {
    expect(normalize_text_de("")).toBe("");
  });

  test("plain text passes through", () => {
    const r = normalize_text_de("Guten Morgen, wie geht es Ihnen?");
    expect(r).toContain("Guten Morgen");
    expect(r).toContain("Ihnen");
  });

  test("complex sentence", () => {
    const t = "Dr. Müller kaufte am 3. Mai 2023 um 14:30 Uhr 3 Pakete für €29,99 bei der Muster GmbH.";
    const r = normalize_text_de(t);
    expect(r).toContain("Doktor");
    expect(r).toContain("Mai");
    expect(r).toContain("vierzehn Uhr dreißig");
    expect(r).toContain("Euro");
    expect(r).toContain("Gesellschaft");
    expect(r).not.toContain("€");
    expect(r).not.toContain("Dr.");
    expect(r).not.toContain("14:30");
  });
});

// ─── voices.js ────────────────────────────────────────────────────────────────

describe("German voices in VOICES", () => {
  test("df_anna exists with correct language", async () => {
    const { VOICES } = await import("../src/voices.js");
    expect(VOICES.df_anna).toBeDefined();
    expect(VOICES.df_anna.language).toBe("de");
    expect(VOICES.df_anna.gender).toBe("Female");
  });

  test("df_emma exists", async () => {
    const { VOICES } = await import("../src/voices.js");
    expect(VOICES.df_emma).toBeDefined();
    expect(VOICES.df_emma.language).toBe("de");
  });

  test("dm_bernd exists", async () => {
    const { VOICES } = await import("../src/voices.js");
    expect(VOICES.dm_bernd).toBeDefined();
    expect(VOICES.dm_bernd.language).toBe("de");
    expect(VOICES.dm_bernd.gender).toBe("Male");
  });

  test("dm_hans exists", async () => {
    const { VOICES } = await import("../src/voices.js");
    expect(VOICES.dm_hans).toBeDefined();
  });

  test("existing English voices still present", async () => {
    const { VOICES } = await import("../src/voices.js");
    expect(VOICES.af_heart).toBeDefined();
    expect(VOICES.bf_emma).toBeDefined();
    expect(VOICES.am_michael).toBeDefined();
  });

  test("German voices have d prefix", async () => {
    const { VOICES } = await import("../src/voices.js");
    const deVoices = Object.entries(VOICES).filter(([, v]) => v.language === "de");
    expect(deVoices.length).toBeGreaterThanOrEqual(4);
    for (const [id] of deVoices) {
      expect(id.startsWith("d")).toBe(true);
    }
  });
});

// ─── kokoro.js routing ────────────────────────────────────────────────────────

describe("KokoroTTS._validate_voice", async () => {
  // We can test _validate_voice without a real model by constructing a
  // minimal instance with null model/tokenizer.
  test("German voice returns 'd'", async () => {
    const { KokoroTTS } = await import("../src/kokoro.js");
    const tts = Object.create(KokoroTTS.prototype);
    expect(tts._validate_voice("df_anna")).toBe("d");
    expect(tts._validate_voice("dm_bernd")).toBe("d");
  });

  test("English voices return 'a' or 'b'", async () => {
    const { KokoroTTS } = await import("../src/kokoro.js");
    const tts = Object.create(KokoroTTS.prototype);
    expect(tts._validate_voice("af_heart")).toBe("a");
    expect(tts._validate_voice("bf_emma")).toBe("b");
  });

  test("unknown voice throws", async () => {
    const { KokoroTTS } = await import("../src/kokoro.js");
    const tts = Object.create(KokoroTTS.prototype);
    expect(() => tts._validate_voice("xx_unknown")).toThrow();
  });
});
