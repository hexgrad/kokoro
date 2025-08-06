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

  // TODO: Add support for other languages:
  // jf_alpha: {
  //   name: "alpha",
  //   language: "ja",
  //   gender: "Female",
  //   traits: "🚺",
  //   targetQuality: "B",
  //   overallGrade: "C+",
  // },
  // jf_gongitsune: {
  //   name: "gongitsune",
  //   language: "ja",
  //   gender: "Female",
  //   traits: "🚺",
  //   targetQuality: "B",
  //   overallGrade: "C",
  // },
  // jf_nezumi: {
  //   name: "nezumi",
  //   language: "ja",
  //   gender: "Female",
  //   traits: "🚺",
  //   targetQuality: "B",
  //   overallGrade: "C-",
  // },
  // jf_tebukuro: {
  //   name: "tebukuro",
  //   language: "ja",
  //   gender: "Female",
  //   traits: "🚺",
  //   targetQuality: "B",
  //   overallGrade: "C",
  // },
  // jm_kumo: {
  //   name: "kumo",
  //   language: "ja",
  //   gender: "Male",
  //   traits: "🚹",
  //   targetQuality: "B",
  //   overallGrade: "C-",
  // },
  // zf_xiaobei: {
  //   name: "xiaobei",
  //   language: "zh",
  //   gender: "Female",
  //   traits: "🚺",
  //   targetQuality: "C",
  //   overallGrade: "D",
  // },
  // zf_xiaoni: {
  //   name: "xiaoni",
  //   language: "zh",
  //   gender: "Female",
  //   traits: "🚺",
  //   targetQuality: "C",
  //   overallGrade: "D",
  // },
  // zf_xiaoxiao: {
  //   name: "xiaoxiao",
  //   language: "zh",
  //   gender: "Female",
  //   traits: "🚺",
  //   targetQuality: "C",
  //   overallGrade: "D",
  // },
  // zf_xiaoyi: {
  //   name: "xiaoyi",
  //   language: "zh",
  //   gender: "Female",
  //   traits: "🚺",
  //   targetQuality: "C",
  //   overallGrade: "D",
  // },
  // zm_yunjian: {
  //   name: "yunjian",
  //   language: "zh",
  //   gender: "Male",
  //   traits: "🚹",
  //   targetQuality: "C",
  //   overallGrade: "D",
  // },
  // zm_yunxi: {
  //   name: "yunxi",
  //   language: "zh",
  //   gender: "Male",
  //   traits: "🚹",
  //   targetQuality: "C",
  //   overallGrade: "D",
  // },
  // zm_yunxia: {
  //   name: "yunxia",
  //   language: "zh",
  //   gender: "Male",
  //   traits: "🚹",
  //   targetQuality: "C",
  //   overallGrade: "D",
  // },
  // zm_yunyang: {
  //   name: "yunyang",
  //   language: "zh",
  //   gender: "Male",
  //   traits: "🚹",
  //   targetQuality: "C",
  //   overallGrade: "D",
  // },
  // ef_dora: {
  //   name: "dora",
  //   language: "es",
  //   gender: "Female",
  //   traits: "🚺",
  //   targetQuality: "C",
  //   overallGrade: "D",
  // },
  // em_alex: {
  //   name: "alex",
  //   language: "es",
  //   gender: "Male",
  //   traits: "🚹",
  //   targetQuality: "C",
  //   overallGrade: "D",
  // },
  // em_santa: {
  //   name: "santa",
  //   language: "es",
  //   gender: "Male",
  //   traits: "🚹",
  //   targetQuality: "C",
  //   overallGrade: "D",
  // },
  // ff_siwis: {
  //   name: "siwis",
  //   language: "es",
  //   gender: "Female",
  //   traits: "🚺",
  //   targetQuality: "B",
  //   overallGrade: "B-",
  // },
  // hf_alpha: {
  //   name: "alpha",
  //   language: "hi",
  //   gender: "Female",
  //   traits: "🚺",
  //   targetQuality: "B",
  //   overallGrade: "C",
  // },
  // hf_beta: {
  //   name: "beta",
  //   language: "hi",
  //   gender: "Female",
  //   traits: "🚺",
  //   targetQuality: "B",
  //   overallGrade: "C",
  // },
  // hm_omega: {
  //   name: "omega",
  //   language: "hi",
  //   gender: "Male",
  //   traits: "🚹",
  //   targetQuality: "B",
  //   overallGrade: "C",
  // },
  // hm_psi: {
  //   name: "psi",
  //   language: "hi",
  //   gender: "Male",
  //   traits: "🚹",
  //   targetQuality: "B",
  //   overallGrade: "C",
  // },
  // if_sara: {
  //   name: "sara",
  //   language: "it",
  //   gender: "Female",
  //   traits: "🚺",
  //   targetQuality: "B",
  //   overallGrade: "C",
  // },
  // im_nicola: {
  //   name: "nicola",
  //   language: "it",
  //   gender: "Male",
  //   traits: "🚹",
  //   targetQuality: "B",
  //   overallGrade: "C",
  // },
  // pf_dora: {
  //   name: "dora",
  //   language: "pt-br",
  //   gender: "Female",
  //   traits: "🚺",
  //   targetQuality: "C",
  //   overallGrade: "D",
  // },
  // pm_alex: {
  //   name: "alex",
  //   language: "pt-br",
  //   gender: "Male",
  //   traits: "🚹",
  //   targetQuality: "C",
  //   overallGrade: "D",
  // },
  // pm_santa: {
  //   name: "santa",
  //   language: "pt-br",
  //   gender: "Male",
  //   traits: "🚹",
  //   targetQuality: "C",
  //   overallGrade: "D",
  // },
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
