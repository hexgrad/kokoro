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
  progress_callback: (progress) => {
    self.postMessage({ status: "progress", progress });
  },
}).catch((e) => {
  self.postMessage({ status: "error", error: e.message });
  throw e;
});

self.postMessage({ status: "ready", voices: tts.voices, device });

// Listen for messages from the main thread
self.addEventListener("message", async (e) => {
  const { text, voice, speed = 1 } = e.data;

  let chunkIndex = 0;
  try {
    for await (const chunk of tts.stream(text, { voice, speed })) {
      const blob = chunk.audio.toBlob();
      self.postMessage({
        status: "chunk",
        index: chunkIndex++,
        audio: URL.createObjectURL(blob),
        text: chunk.text,
        phonemes: chunk.phonemes,
      });
    }
    self.postMessage({ status: "complete" });
  } catch (e) {
    self.postMessage({ status: "error", error: e.message });
  }
});
