import { describe, expect, test } from "vitest";
import { KokoroTTS } from "../src/kokoro.js";

describe("KokoroTTS.stream", () => {
  test("applies split_pattern to SSML text segments around <break> tags", async () => {
    const tokenizer = () => ({ input_ids: {} });
    const tts = new KokoroTTS({}, tokenizer);

    // Avoid model inference in this behavioral test.
    tts.generate_from_ids = async () => ({ audio: new Float32Array(0), sampling_rate: 24000 });

    const emittedTexts = [];
    const input = 'Alpha. | Beta.<break time="10ms"/>Gamma. | Delta.';

    for await (const chunk of tts.stream(input, { split_pattern: /\s*\|\s*/ })) {
      emittedTexts.push(chunk.text);
    }

    expect(emittedTexts).toEqual(["Alpha.Beta.", "", "Gamma.Delta."]);
    expect(emittedTexts.join(" ")).not.toContain("|");
  });
});
