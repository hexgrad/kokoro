export interface TTSOptions { speed?: number; voice?: string; language?: string; }
export function prepareAudioBuffer(samples: Float32Array): Buffer {
  return Buffer.from(samples.buffer);
}
