import path from "path";
import fs from "fs/promises";

export const VOICES = Object.freeze({
  af_heart: {
    name: "Heart",
    language: "en-us",
    gender: "Female",
    traits: "❤️",
    targetQuality: "A",
    overallGrade: "A",
  },
  af_alloy: {
    name: "Alloy",
    language: "en-us",
    gender: "Female",
    targetQuality: "B",
    overallGrade: "C",
  },
  af_aoede: {
    name: "Aoede",
    language: "en-us",
    gender: "Female",
    targetQuality: "B",
    overallGrade: "C+",
  },
  af_bella: {
    name: "Bella",
    language: "en-us",
    gender: "Female",
    traits: "🔥",
    targetQuality: "A",
    overallGrade: "A-",
  },
  af_jessica: {
    name: "Jessica",
    language: "en-us",
    gender: "Female",
    targetQuality: "C",
    overallGrade: "D",
  },
  af_kore: {
    name: "Kore",
    language: "en-us",
    gender: "Female",
    targetQuality: "B",
    overallGrade: "C+",
  },
  af_nicole: {
    name: "Nicole",
    language: "en-us",
    gender: "Female",
    traits: "🎧",
    targetQuality: "B",
    overallGrade: "B-",
  },
  af_nova: {
    name: "Nova",
    language: "en-us",
    gender: "Female",
    targetQuality: "B",
    overallGrade: "C",
  },
  af_river: {
    name: "River",
    language: "en-us",
    gender: "Female",
    targetQuality: "C",
    overallGrade: "D",
  },
  af_sarah: {
    name: "Sarah",
    language: "en-us",
    gender: "Female",
    targetQuality: "B",
    overallGrade: "C+",
  },
  af_sky: {
    name: "Sky",
    language: "en-us",
    gender: "Female",
    targetQuality: "B",
    overallGrade: "C-",
  },
  am_adam: {
    name: "Adam",
    language: "en-us",
    gender: "Male",
    targetQuality: "D",
    overallGrade: "F+",
  },
  am_echo: {
    name: "Echo",
    language: "en-us",
    gender: "Male",
    targetQuality: "C",
    overallGrade: "D",
  },
  am_eric: {
    name: "Eric",
    language: "en-us",
    gender: "Male",
    targetQuality: "C",
    overallGrade: "D",
  },
  am_fenrir: {
    name: "Fenrir",
    language: "en-us",
    gender: "Male",
    targetQuality: "B",
    overallGrade: "C+",
  },
  am_liam: {
    name: "Liam",
    language: "en-us",
    gender: "Male",
    targetQuality: "C",
    overallGrade: "D",
  },
  am_michael: {
    name: "Michael",
    language: "en-us",
    gender: "Male",
    targetQuality: "B",
    overallGrade: "C+",
  },
  am_onyx: {
    name: "Onyx",
    language: "en-us",
    gender: "Male",
    targetQuality: "C",
    overallGrade: "D",
  },
  am_puck: {
    name: "Puck",
    language: "en-us",
    gender: "Male",
    targetQuality: "B",
    overallGrade: "C+",
  },
  am_santa: {
    name: "Santa",
    language: "en-us",
    gender: "Male",
    targetQuality: "C",
    overallGrade: "D-",
  },
  bf_emma: {
    name: "Emma",
    language: "en-gb",
    gender: "Female",
    traits: "🚺",
    targetQuality: "B",
    overallGrade: "B-",
  },
  bf_isabella: {
    name: "Isabella",
    language: "en-gb",
    gender: "Female",
    targetQuality: "B",
    overallGrade: "C",
  },
  bm_george: {
    name: "George",
    language: "en-gb",
    gender: "Male",
    targetQuality: "B",
    overallGrade: "C",
  },
  bm_lewis: {
    name: "Lewis",
    language: "en-gb",
    gender: "Male",
    targetQuality: "C",
    overallGrade: "D+",
  },
  bf_alice: {
    name: "Alice",
    language: "en-gb",
    gender: "Female",
    traits: "🚺",
    targetQuality: "C",
    overallGrade: "D",
  },
  bf_lily: {
    name: "Lily",
    language: "en-gb",
    gender: "Female",
    traits: "🚺",
    targetQuality: "C",
    overallGrade: "D",
  },
  bm_daniel: {
    name: "Daniel",
    language: "en-gb",
    gender: "Male",
    traits: "🚹",
    targetQuality: "C",
    overallGrade: "D",
  },
  bm_fable: {
    name: "Fable",
    language: "en-gb",
    gender: "Male",
    traits: "🚹",
    targetQuality: "B",
    overallGrade: "C",
  },

  // Japanese voices
  jf_alpha: {
    name: "Alpha",
    language: "ja",
    gender: "Female",
    targetQuality: "B",
    overallGrade: "C+",
  },
  jf_gongitsune: {
    name: "Gongitsune",
    language: "ja",
    gender: "Female",
    targetQuality: "B",
    overallGrade: "C",
  },
  jf_nezumi: {
    name: "Nezumi",
    language: "ja",
    gender: "Female",
    targetQuality: "B",
    overallGrade: "C-",
  },
  jf_tebukuro: {
    name: "Tebukuro",
    language: "ja",
    gender: "Female",
    targetQuality: "B",
    overallGrade: "C",
  },
  jm_kumo: {
    name: "Kumo",
    language: "ja",
    gender: "Male",
    targetQuality: "B",
    overallGrade: "C-",
  },

  // Chinese (Mandarin) voices
  zf_xiaobei: {
    name: "Xiaobei",
    language: "zh",
    gender: "Female",
    targetQuality: "C",
    overallGrade: "D",
  },
  zf_xiaoni: {
    name: "Xiaoni",
    language: "zh",
    gender: "Female",
    targetQuality: "C",
    overallGrade: "D",
  },
  zf_xiaoxiao: {
    name: "Xiaoxiao",
    language: "zh",
    gender: "Female",
    targetQuality: "C",
    overallGrade: "D",
  },
  zf_xiaoyi: {
    name: "Xiaoyi",
    language: "zh",
    gender: "Female",
    targetQuality: "C",
    overallGrade: "D",
  },
  zm_yunjian: {
    name: "Yunjian",
    language: "zh",
    gender: "Male",
    targetQuality: "C",
    overallGrade: "D",
  },
  zm_yunxi: {
    name: "Yunxi",
    language: "zh",
    gender: "Male",
    targetQuality: "C",
    overallGrade: "D",
  },
  zm_yunxia: {
    name: "Yunxia",
    language: "zh",
    gender: "Male",
    targetQuality: "C",
    overallGrade: "D",
  },
  zm_yunyang: {
    name: "Yunyang",
    language: "zh",
    gender: "Male",
    targetQuality: "C",
    overallGrade: "D",
  },

  // Spanish voices
  ef_dora: {
    name: "Dora",
    language: "es",
    gender: "Female",
    targetQuality: "C",
    overallGrade: "D",
  },
  em_alex: {
    name: "Alex",
    language: "es",
    gender: "Male",
    targetQuality: "C",
    overallGrade: "D",
  },
  em_santa: {
    name: "Santa",
    language: "es",
    gender: "Male",
    targetQuality: "C",
    overallGrade: "D",
  },

  // French voice
  ff_siwis: {
    name: "Siwis",
    language: "fr",
    gender: "Female",
    targetQuality: "B",
    overallGrade: "B-",
  },

  // Hindi voices
  hf_alpha: {
    name: "Alpha",
    language: "hi",
    gender: "Female",
    targetQuality: "B",
    overallGrade: "C",
  },
  hf_beta: {
    name: "Beta",
    language: "hi",
    gender: "Female",
    targetQuality: "B",
    overallGrade: "C",
  },
  hm_omega: {
    name: "Omega",
    language: "hi",
    gender: "Male",
    targetQuality: "B",
    overallGrade: "C",
  },
  hm_psi: {
    name: "Psi",
    language: "hi",
    gender: "Male",
    targetQuality: "B",
    overallGrade: "C",
  },

  // Italian voices
  if_sara: {
    name: "Sara",
    language: "it",
    gender: "Female",
    targetQuality: "B",
    overallGrade: "C",
  },
  im_nicola: {
    name: "Nicola",
    language: "it",
    gender: "Male",
    targetQuality: "B",
    overallGrade: "C",
  },

  // Portuguese (Brazilian) voices
  pf_dora: {
    name: "Dora",
    language: "pt-br",
    gender: "Female",
    targetQuality: "C",
    overallGrade: "D",
  },
  pm_alex: {
    name: "Alex",
    language: "pt-br",
    gender: "Male",
    targetQuality: "C",
    overallGrade: "D",
  },
  pm_santa: {
    name: "Santa",
    language: "pt-br",
    gender: "Male",
    targetQuality: "C",
    overallGrade: "D",
  },
});


/**
 * The base URL for fetching voice data files.
 */
let voiceDataUrl = "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices";


/**
 * Retrieves the current voice data URL.
 * 
 * @returns The current voice data URL.
 */
export function getVoiceDataUrl() {
  return voiceDataUrl;
};

/**
 * Sets a new voice data URL.
 * 
 * @param url - The new URL to set for voice data.
 * @throws Will throw an error if the URL is not a valid non-empty string.
 */
export function setVoiceDataUrl(url) {
  if (typeof url === 'string' && url.trim() !== '') {
    voiceDataUrl = url;
  } else {
    throw new Error("Invalid URL");
  }
};

/**
 *
 * @param {keyof typeof VOICES} id
 * @returns {Promise<ArrayBufferLike>}
 */
async function getVoiceFile(id) {
  if (fs && Object.hasOwn(fs, 'readFile')) {
    const dirname = typeof __dirname !== "undefined" ? __dirname : import.meta.dirname;
    const file = path.resolve(dirname, `../voices/${id}.bin`);
    const { buffer } = await fs.readFile(file);
    return buffer;
  }

  const url = `${voiceDataUrl}/${id}.bin`;

  let cache;
  try {
    cache = await caches.open("kokoro-voices");
    const cachedResponse = await cache.match(url);
    if (cachedResponse) {
      return await cachedResponse.arrayBuffer();
    }
  } catch (e) {
    console.warn("Unable to open cache", e);
  }

  // No cache, or cache failed to open. Fetch the file.
  const response = await fetch(url);
  const buffer = await response.arrayBuffer();

  if (cache) {
    try {
      // NOTE: We use `new Response(buffer, ...)` instead of `response.clone()` to handle LFS files
      await cache.put(
        url,
        new Response(buffer, {
          headers: response.headers,
        }),
      );
    } catch (e) {
      console.warn("Unable to cache file", e);
    }
  }

  return buffer;
}

const VOICE_CACHE = new Map();
export async function getVoiceData(voice) {
  if (VOICE_CACHE.has(voice)) {
    return VOICE_CACHE.get(voice);
  }

  const buffer = new Float32Array(await getVoiceFile(voice));
  VOICE_CACHE.set(voice, buffer);
  return buffer;
}
