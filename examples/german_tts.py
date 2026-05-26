#!/usr/bin/env python3
"""German TTS demo using lang_code='d' and a community .pt voice file.

Usage:
  python examples/german_tts.py --voice path/to/voice.pt --text "Guten Tag!"
"""

import argparse
import soundfile as sf
from kokoro import KPipeline


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--voice", required=True, help="Path to German voice .pt pack")
    p.add_argument("--text", default="Guten Tag! Dies ist ein kurzer Test.")
    p.add_argument("--out", default="german_out.wav")
    args = p.parse_args()

    pipeline = KPipeline(lang_code="d")
    for i, (_, _, audio) in enumerate(pipeline(args.text, voice=args.voice)):
        if audio is not None:
            sf.write(args.out, audio.numpy(), 24000)
            print(f"Wrote {args.out}")
            return
    raise SystemExit("No audio generated — check espeak-ng and voice .pt path")


if __name__ == "__main__":
    main()
