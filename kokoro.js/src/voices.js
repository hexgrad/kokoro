// @ts-nocheck
import path from "path";
import fs from "fs/promises";
// ... rest of the file
import path from "path";
import fs from "fs/promises";

export const VOICES = Object.freeze({
  // ── American English ──────────────────────────────────────────────────────
  af_heart:   { name: "Heart",    language: "en-us", gender: "Female", traits: "❤️", targetQuality: "A",  overallGrade: "A"   },
  af_alloy:   { name: "Alloy",    language: "en-us", gender: "Female",               targetQuality: "B",  overallGrade: "C"   },
  af_aoede:   { name: "Aoede",    language: "en-us", gender: "Female",               targetQuality: "B",  overallGrade: "C+"  },
  af_bella:   { name: "Bella",    language: "en-us", gender: "Female", traits: "🔥", targetQuality: "A",  overallGrade: "A-"  },
  af_jessica: { name: "Jessica",  language: "en-us", gender: "Female",               targetQuality: "C",  overallGrade: "D"   },
  af_kore:    { name: "Kore",     language: "en-us", gender: "Female",               targetQuality: "B",  overallGrade: "C+"  },
  af_nicole:  { name: "Nicole",   language: "en-us", gender: "Female", traits: "🎧", targetQuality: "B",  overallGrade: "B-"  },
  af_nova:    { name: "Nova",     language: "en-us", gender: "Female",               targetQuality: "B",  overallGrade: "C"   },
  af_river:   { name: "River",    language: "en-us", gender: "Female",               targetQuality: "C",  overallGrade: "D"   },
  af_sarah:   { name: "Sarah",    language: "en-us", gender: "Female",               targetQuality: "B",  overallGrade: "C+"  },
  af_sky:     { name: "Sky",      language: "en-us", gender: "Female",               targetQuality: "B",  overallGrade: "C-"  },
  am_adam:    { name: "Adam",     language: "en-us", gender: "Male",                 targetQuality: "D",  overallGrade: "F+"  },
  am_echo:    { name: "Echo",     language: "en-us", gender: "Male",                 targetQuality: "C",  overallGrade: "D"   },
  am_eric:    { name: "Eric",     language: "en-us", gender: "Male",                 targetQuality: "C",  overallGrade: "D"   },
  am_fenrir:  { name: "Fenrir",   language: "en-us", gender: "Male",                 targetQuality: "B",  overallGrade: "C+"  },
  am_liam:    { name: "Liam",     language: "en-us", gender: "Male",                 targetQuality: "C",  overallGrade: "D"   },
  am_michael: { name: "Michael",  language: "en-us", gender: "Male",                 targetQuality: "B",  overallGrade: "C+"  },
  am_onyx:    { name: "Onyx",     language: "en-us", gender: "Male",                 targetQuality: "C",  overallGrade: "D"   },
  am_puck:    { name: "Puck",     language: "en-us", gender: "Male",                 targetQuality: "B",  overallGrade: "C+"  },
  am_santa:   { name: "Santa",    language: "en-us", gender: "Male",                 targetQuality: "C",  overallGrade: "D-"  },

  // ── British English ───────────────────────────────────────────────────────
  bf_emma:     { name: "Emma",     language: "en-gb", gender: "Female", traits: "🚺", targetQuality: "B", overallGrade: "B-" },
  bf_isabella: { name: "Isabella", language: "en-gb", gender: "Female",               targetQuality: "B", overallGrade: "C"  },
  bf_alice:    { name: "Alice",    language: "en-gb", gender: "Female", traits: "🚺", targetQuality: "C", overallGrade: "D"  },
  bf_lily:     { name: "Lily",     language: "en-gb", gender: "Female", traits: "🚺", targetQuality: "C", overallGrade: "D"  },
  bm_george:   { name: "George",   language: "en-gb", gender: "Male",                 targetQuality: "B", overallGrade: "C"  },
  bm_lewis:    { name: "Lewis",    language: "en-gb", gender: "Male",                 targetQuality: "C", overallGrade: "D+" },
  bm_daniel:   { name: "Daniel",   language: "en-gb", gender: "Male",   traits: "🚹", targetQuality: "C", overallGrade: "D"  },
  bm_fable:    { name: "Fable",    language: "en-gb", gender: "Male",   traits: "🚹", targetQuality: "B", overallGrade: "C"  },

  // ── German ────────────────────────────────────────────────────────────────
  // Voice .bin files will be released with the German model checkpoint.
  // Naming convention: d{f|m}_{name}
  df_anna:  { name: "Anna",  language: "de", gender: "Female", traits: "🇩🇪", targetQuality: "B", overallGrade: "pending" },
  df_emma:  { name: "Emma",  language: "de", gender: "Female", traits: "🇩🇪", targetQuality: "B", overallGrade: "pending" },
  dm_bernd: { name: "Bernd", language: "de", gender: "Male",   traits: "🇩🇪", targetQuality: "B", overallGrade: "pending" },
  dm_hans:  { name: "Hans",  language: "de", gender: "Male",   traits: "🇩🇪", targetQuality: "B", overallGrade: "pending" },
});

/** Base URL for fetching voice .bin files. */
let voiceDataUrl = "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices";

export function getVoiceDataUrl() { return voiceDataUrl; }

export function setVoiceDataUrl(url) {
  if (typeof url === "string" && url.trim()) {
    voiceDataUrl = url;
  } else {
    throw new Error("Invalid URL");
  }
}

async function getVoiceFile(id) {
  if (fs && Object.hasOwn(fs, "readFile")) {
    const dirname = typeof __dirname !== "undefined" ? __dirname : import.meta.dirname;
    const file = path.resolve(dirname, `../voices/${id}.bin`);
    const { buffer } = await fs.readFile(file);
    return buffer;
  }

  const url = `${voiceDataUrl}/${id}.bin`;
  let cache;
  try {
    cache = await caches.open("kokoro-voices");
    const cached = await cache.match(url);
    if (cached) return cached.arrayBuffer();
  } catch {}

  const res = await fetch(url);
  const buf = await res.arrayBuffer();
  if (cache) {
    try { await cache.put(url, new Response(buf, { headers: res.headers })); } catch {}
  }
  return buf;
}

const VOICE_CACHE = new Map();

export async function getVoiceData(voice) {
  if (VOICE_CACHE.has(voice)) return VOICE_CACHE.get(voice);
  const buf = new Float32Array(await getVoiceFile(voice));
  VOICE_CACHE.set(voice, buf);
  return buf;
}
