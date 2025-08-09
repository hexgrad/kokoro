import "@sec-ant/readable-stream/polyfill/asyncIterator";
import { KokoroTTS } from "kokoro-js";
import { detectWebGPU } from "./utils.js";

// Device detection
const device = (await detectWebGPU()) ? "webgpu" : "wasm";
self.postMessage({ status: "device", device });

// Load the model
const model_id = "onnx-community/Kokoro-82M-v1.0-ONNX";
const tts = await KokoroTTS.from_pretrained(model_id, {
  dtype: device === "wasm" ? "q8" : "fp32",
  device,
}).catch((e) => {
  self.postMessage({ status: "error", error: e.message });
  throw e;
});
self.postMessage({ status: "ready", voices: tts.voices, device });

// Listen for messages from the main thread
self.addEventListener("message", async (e) => {
  const { text, voice } = e.data;

  // Generate speech
  const audio = await tts.generate(text, { voice });

  // Send the audio file back to the main thread
  const blob = audio.toBlob();
  self.postMessage({ status: "complete", audio: URL.createObjectURL(blob), text });
});
