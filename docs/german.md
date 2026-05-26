# German (community voices)

Official Kokoro-82M weights are English-first. This guide documents **experimental German support** via:

- `lang_code='d'` — German espeak-ng G2P
- Optional `kokoro.de_text.normalize_german()` for numbers, EUR, and abbreviations
- Community-trained `.pt` voice packs (not bundled in `hexgrad/Kokoro-82M`)

## Quick start

```python
from kokoro import KPipeline
import soundfile as sf
import torch

# 1) Download a community German voice .pt (example repos on Hugging Face)
voice_pt = "path/to/german_voice.pt"  # e.g. community Kokoro-German checkpoints

pipeline = KPipeline(lang_code="d", model=True)
pack = pipeline.load_voice(voice_pt)

text = "Guten Tag! Das ist ein Test mit 2,5 kWh und 49,99 EUR."
for i, (_, ps, audio) in enumerate(pipeline(text, voice=voice_pt, speed=1)):
    if audio is not None:
        sf.write(f"german_{i}.wav", audio.numpy(), 24000)
```

## Voices

Community models (quality varies):

- [Tundragoon/Kokoro-German](https://huggingface.co/Tundragoon/Kokoro-German)
- [huggingFresse/Kokoro-82M-ONNX-German-Martin](https://huggingface.co/huggingFresse/Kokoro-82M-ONNX-German-Martin)

Funding toward **official** German training data is tracked in [GitHub issue #290](https://github.com/hexgrad/kokoro/issues/290).

## Requirements

- `espeak-ng` installed on the system
- `pip install kokoro` (this package)
