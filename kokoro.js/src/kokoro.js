// @ts-nocheck
import { env as hf, StyleTextToSpeech2Model, AutoTokenizer, Tensor, RawAudio } from "@huggingface/transformers";
import { phonemize } from "./phonemize.js";
// ... rest of the file
import { env as hf, StyleTextToSpeech2Model, AutoTokenizer, Tensor, RawAudio } from "@huggingface/transformers";
import { phonemize } from "./phonemize.js";
import { phonemize_de } from "./phonemize_de.js";
import { TextSplitterStream } from "./splitter.js";
import { getVoiceData, VOICES } from "./voices.js";

const STYLE_DIM = 256;
const SAMPLE_RATE = 24000;

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

  /**
   * Validate voice and return its language prefix.
   * @param {string} voice
   * @returns {"a"|"b"|"d"}
   */
  _validate_voice(voice) {
    if (!Object.prototype.hasOwnProperty.call(VOICES, voice)) {
      console.error(`Voice "${voice}" not found. Available voices:`);
      console.table(VOICES);
      throw new Error(`Voice "${voice}" not found. Should be one of: ${Object.keys(VOICES).join(", ")}.`);
    }
    return /** @type {"a"|"b"|"d"} */ (voice.at(0));
  }

  /**
   * Phonemise text for the given language prefix.
   * @param {string} text
   * @param {"a"|"b"|"d"} language
   * @returns {Promise<string>}
   */
  async _phonemize(text, language) {
    if (language === "d") return phonemize_de(text);
    return phonemize(text, language);
  }

  /**
   * Generate audio from text.
   * @param {string} text The input text
   * @param {GenerateOptions} options Additional options
   * @returns {Promise<RawAudio>} The generated audio
   */
  async generate(text, { voice = "af_heart", speed = 1 } = {}) {
    const language = this._validate_voice(voice);
    const phonemes = await this._phonemize(text, language);
    const { input_ids } = this.tokenizer(phonemes, { truncation: true });
    return this.generate_from_ids(input_ids, { voice, speed });
  }

  /**
   * Generate audio from input ids.
   * @param {Tensor} input_ids The input ids
   * @param {GenerateOptions} options Additional options
   * @returns {Promise<RawAudio>} The generated audio
   */
  async generate_from_ids(input_ids, { voice = "af_heart", speed = 1 } = {}) {
    const num_tokens = Math.min(Math.max(input_ids.dims.at(-1) - 2, 0), 509);
    const data = await getVoiceData(voice);
    const offset = num_tokens * STYLE_DIM;
    const voiceData = data.slice(offset, offset + STYLE_DIM);
    const inputs = {
      input_ids,
      style: new Tensor("float32", voiceData, [1, STYLE_DIM]),
      speed: new Tensor("float32", [speed], [1]),
    };
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

    /** @type {TextSplitterStream} */
    let splitter;
    if (text instanceof TextSplitterStream) {
      splitter = text;
    } else if (typeof text === "string") {
      splitter = new TextSplitterStream();
      const chunks = split_pattern
        ? text.split(split_pattern).map((c) => c.trim()).filter((c) => c.length > 0)
        : [text];
      splitter.push(...chunks);
    } else {
      throw new Error("Invalid input type. Expected string or TextSplitterStream.");
    }

    for await (const sentence of splitter) {
      const phonemes = await this._phonemize(sentence, language);
      const { input_ids } = this.tokenizer(phonemes, { truncation: true });
      const audio = await this.generate_from_ids(input_ids, { voice, speed });
      yield { text: sentence, phonemes, audio };
    }
  }
}

export const env = {
  set cacheDir(value) { hf.cacheDir = value; },
  get cacheDir() { return hf.cacheDir; },
  set wasmPaths(value) { hf.backends.onnx.wasm.wasmPaths = value; },
  get wasmPaths() { return hf.backends.onnx.wasm.wasmPaths; },
};

export { TextSplitterStream };
