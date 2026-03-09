import { env as hf, StyleTextToSpeech2Model, AutoTokenizer, Tensor, RawAudio } from "@huggingface/transformers";
import { phonemize } from "./phonemize.js";
import { TextSplitterStream } from "./splitter.js";
import { getVoiceData, VOICES } from "./voices.js";
import { hasSSML, splitAtBreaks } from "./ssml.js";

const STYLE_DIM = 256;
const SAMPLE_RATE = 24000;
const XFADE_LEN = Math.round(SAMPLE_RATE * 0.008); // 8 ms linear cross-fade

/**
 * Generate a silent audio segment.
 * @param {number} durationMs Duration in milliseconds
 * @returns {RawAudio}
 */
function generateSilence(durationMs) {
  const numSamples = Math.round((SAMPLE_RATE * durationMs) / 1000);
  return new RawAudio(new Float32Array(numSamples), SAMPLE_RATE);
}

/**
 * Concatenate multiple RawAudio segments with a short linear cross-fade at each boundary
 * to prevent clicks or pops at splice points.
 * @param {RawAudio[]} audios
 * @returns {RawAudio}
 */
function concatAudio(audios) {
  if (audios.length === 0) return new RawAudio(new Float32Array(0), SAMPLE_RATE);
  if (audios.length === 1) return new RawAudio(audios[0].audio.slice(), SAMPLE_RATE);

  // Clamp cross-fade length to half the shortest segment so it never exceeds any segment.
  const xLen = Math.min(XFADE_LEN, ...audios.map((a) => Math.floor(a.audio.length / 2)));
  const totalLen = audios.reduce((s, a) => s + a.audio.length, 0) - xLen * (audios.length - 1);
  const out = new Float32Array(totalLen);

  let pos = 0;
  for (let i = 0; i < audios.length; i++) {
    const seg = audios[i].audio;
    const isFirst = i === 0;
    const isLast = i === audios.length - 1;

    // Blend this segment's fade-in into the fade-out region already written by the previous segment.
    if (!isFirst) {
      for (let j = 0; j < xLen; j++) {
        out[pos + j] += seg[j] * (j / xLen);
      }
      pos += xLen;
    }

    // Copy the flat (non-overlapping) middle portion of this segment.
    const flatStart = isFirst ? 0 : xLen;
    const flatEnd = isLast ? seg.length : seg.length - xLen;
    out.set(seg.subarray(flatStart, flatEnd), pos);
    pos += flatEnd - flatStart;

    // Write a fade-out tail; the next segment's fade-in will be added on top.
    if (!isLast) {
      for (let j = 0; j < xLen; j++) {
        out[pos + j] = seg[seg.length - xLen + j] * (1 - j / xLen);
      }
      // pos is intentionally NOT advanced here — the next iteration's fade-in writes to the same region.
    }
  }

  return new RawAudio(out, SAMPLE_RATE);
}

/**
 * @typedef {Object} GenerateOptions
 * @property {keyof typeof VOICES} [voice="af_heart"] The voice
 * @property {number} [speed=1] The speaking speed
 */

/**
 * @typedef {Object} StreamProperties
 * @property {RegExp} [split_pattern] The pattern to split the input text. If unset, the default sentence splitter will be used.
 * @typedef {GenerateOptions & StreamProperties} StreamGenerateOptions
 */

export class KokoroTTS {
  /**
   * Create a new KokoroTTS instance.
   * @param {import('@huggingface/transformers').StyleTextToSpeech2Model} model The model
   * @param {import('@huggingface/transformers').PreTrainedTokenizer} tokenizer The tokenizer
   */
  constructor(model, tokenizer) {
    this.model = model;
    this.tokenizer = tokenizer;
  }

  /**
   * Load a KokoroTTS model from the Hugging Face Hub.
   * @param {string} model_id The model id
   * @param {Object} options Additional options
   * @param {"fp32"|"fp16"|"q8"|"q4"|"q4f16"} [options.dtype="fp32"] The data type to use.
   * @param {"wasm"|"webgpu"|"cpu"|null} [options.device=null] The device to run the model on.
   * @param {import("@huggingface/transformers").ProgressCallback} [options.progress_callback=null] A callback function that is called with progress information.
   * @returns {Promise<KokoroTTS>} The loaded model
   */
  static async from_pretrained(model_id, { dtype = "fp32", device = null, progress_callback = null } = {}) {
    const model = StyleTextToSpeech2Model.from_pretrained(model_id, { progress_callback, dtype, device });
    const tokenizer = AutoTokenizer.from_pretrained(model_id, { progress_callback });

    const info = await Promise.all([model, tokenizer]);
    return new KokoroTTS(...info);
  }

  get voices() {
    return VOICES;
  }

  list_voices() {
    console.table(VOICES);
  }

  _validate_voice(voice) {
    if (!VOICES.hasOwnProperty(voice)) {
      console.error(`Voice "${voice}" not found. Available voices:`);
      console.table(VOICES);
      throw new Error(`Voice "${voice}" not found. Should be one of: ${Object.keys(VOICES).join(", ")}.`);
    }
    const language = /** @type {"a"|"b"} */ (voice.at(0)); // "a" or "b"
    return language;
  }

  /**
   * Generate audio from text.
   *
   * @param {string} text The input text
   * @param {GenerateOptions} options Additional options
   * @returns {Promise<RawAudio>} The generated audio
   */
  async generate(text, { voice = "af_heart", speed = 1 } = {}) {
    const language = this._validate_voice(voice);

    // If the text contains <break> tags, split into text/silence segments,
    // generate each independently, and concatenate with cross-fades.
    if (hasSSML(text) && text.includes("<break")) {
      const segments = splitAtBreaks(text);
      const audios = await Promise.all(
        segments.map(async (seg) => {
          if (seg.type === "break") return generateSilence(seg.ms);
          const phonemes = await phonemize(seg.value, language);
          const { input_ids } = this.tokenizer(phonemes, { truncation: true });
          return this.generate_from_ids(input_ids, { voice, speed });
        }),
      );
      return concatAudio(audios);
    }

    const phonemes = await phonemize(text, language);
    const { input_ids } = this.tokenizer(phonemes, {
      truncation: true,
    });

    return this.generate_from_ids(input_ids, { voice, speed });
  }

  /**
   * Generate audio from input ids.
   * @param {Tensor} input_ids The input ids
   * @param {GenerateOptions} options Additional options
   * @returns {Promise<RawAudio>} The generated audio
   */
  async generate_from_ids(input_ids, { voice = "af_heart", speed = 1 } = {}) {
    // Select voice style based on number of input tokens
    const num_tokens = Math.min(Math.max(input_ids.dims.at(-1) - 2, 0), 509);

    // Load voice style
    const data = await getVoiceData(voice);
    const offset = num_tokens * STYLE_DIM;
    const voiceData = data.slice(offset, offset + STYLE_DIM);

    // Prepare model inputs
    const inputs = {
      input_ids,
      style: new Tensor("float32", voiceData, [1, STYLE_DIM]),
      speed: new Tensor("float32", [speed], [1]),
    };

    // Generate audio
    const { waveform } = await this.model(inputs);
    return new RawAudio(waveform.data, SAMPLE_RATE);
  }

  /**
   * Generate audio from text in a streaming fashion.
   * @param {string|TextSplitterStream} text The input text
   * @param {StreamGenerateOptions} options Additional options
   * @returns {AsyncGenerator<{text: string, phonemes: string, audio: RawAudio}, void, void>}
   */
  async *stream(text, { voice = "af_heart", speed = 1, split_pattern = null } = {}) {
    const language = this._validate_voice(voice);

    // If the input is a plain string containing <break> tags, extract breaks
    // first and interleave silence segments with the sentence stream.
    if (typeof text === "string" && hasSSML(text) && text.includes("<break")) {
      const topSegments = splitAtBreaks(text);
      for (const seg of topSegments) {
        if (seg.type === "break") {
          yield { text: "", phonemes: "", audio: generateSilence(seg.ms) };
          continue;
        }
        // Process each text segment through the normal sentence-splitting path.
        const splitter = new TextSplitterStream();
        splitter.push(seg.value);
        splitter.close();
        for await (const sentence of splitter) {
          const phonemes = await phonemize(sentence, language);
          const { input_ids } = this.tokenizer(phonemes, { truncation: true });
          const audio = await this.generate_from_ids(input_ids, { voice, speed });
          yield { text: sentence, phonemes, audio };
        }
      }
      return;
    }

    /** @type {TextSplitterStream} */
    let splitter;
    if (text instanceof TextSplitterStream) {
      splitter = text;
    } else if (typeof text === "string") {
      splitter = new TextSplitterStream();
      const chunks = split_pattern
        ? text
          .split(split_pattern)
          .map((chunk) => chunk.trim())
          .filter((chunk) => chunk.length > 0)
        : [text];
      splitter.push(...chunks);
    } else {
      throw new Error("Invalid input type. Expected string or TextSplitterStream.");
    }
    for await (const sentence of splitter) {
      const phonemes = await phonemize(sentence, language);
      const { input_ids } = this.tokenizer(phonemes, {
        truncation: true,
      });

      // TODO: There may be some cases where - even with splitting - the text is too long.
      // In that case, we should split the text into smaller chunks and process them separately.
      // For now, we just truncate these exceptionally long chunks
      const audio = await this.generate_from_ids(input_ids, { voice, speed });
      yield { text: sentence, phonemes, audio };
    }
  }
}

export const env = {
  set cacheDir(value) {
    hf.cacheDir = value
  },
  get cacheDir() {
    return hf.cacheDir
  },
  set wasmPaths(value) {
    hf.backends.onnx.wasm.wasmPaths = value;
  },
  get wasmPaths() {
    return hf.backends.onnx.wasm.wasmPaths;
  },
};

export { TextSplitterStream };
