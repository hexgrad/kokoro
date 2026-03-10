import Module from "./espeakng.worker.js";

const workerPromise = new Promise((resolve) => {
  if (Module.calledRun) {
    resolve(new Module.eSpeakNGWorker());
  } else {
    Module.onRuntimeInitialized = () => resolve(new Module.eSpeakNGWorker());
  }
});

const SUPPORTED_LANGUAGES = [
  "en", // English (American & British)
  "fr", // French
  "es", // Spanish
  "ja", // Japanese
  "zh", // Chinese (Mandarin)
  "hi", // Hindi
  "it", // Italian
  "pt", // Portuguese (Brazilian)
  "ko", // Korean
  "de", // German
];

const initCache = workerPromise.then((worker) => {
  const voices = worker
    .list_voices()
    .map(({ name, identifier, languages }) => ({
      name,
      identifier,
      languages: languages.filter(
        /** @param {{name: string; priority: number}} [lang] */
        (lang) => SUPPORTED_LANGUAGES.includes(lang.name.split("-")[0]),
      ),
    }))
    .filter(
      /** @param {{languages: {name: string; priority: number}[]}} [voice] */
      (voice) => voice.languages.length > 0,
    );

  // Generate list of supported language identifiers:
  const identifiers = new Set();
  for (const voice of voices) {
    identifiers.add(voice.identifier);
    for (const lang of voice.languages) {
      identifiers.add(lang.name);
    }
  }

  return { voices, identifiers };
});

/**
 * List the available voices for the specified language.
 * @param {string} [language] The language identifier
 * @returns {Promise<{name: string; identifier: string; languages: {name: string; priority: number}[]}[]>} A list of available voices
 */
export const list_voices = async (language) => {
  const { voices } = await initCache;
  if (!language) return voices;
  const base = language.split("-")[0];
  return voices.filter(
    /** @param {{languages: {name: string; priority: number}[]}} [voice] */
    (voice) => voice.languages.some((lang) => lang.name === base || lang.name.startsWith(base + "-")),
  );
};

/**
 * Multilingual text to phonemes converter
 *
 * @param {string} text The input text
 * @param {string} [language] The language identifier
 * @returns {Promise<string[]>} A phonemized version of the input
 */
export const phonemize = async (text, language = "en-us") => {
  const worker = await workerPromise;

  const { identifiers } = await initCache;
  if (!identifiers.has(language)) {
    throw new Error(`Invalid language identifier: "${language}". Should be one of: ${Array.from(identifiers).toSorted().join(", ")}.`);
  }
  worker.set_voice(language);

  return (
    worker
      .synthesize_ipa(text)
      .ipa?.split("\n")
      .filter(
        /** @param {string} [x]  */
        (x) => x.length > 0,
      ) ?? []
  );
};
