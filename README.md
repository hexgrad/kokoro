# kokoro

This WIP repository is intended to be an inference library for https://hf.co/hexgrad/Kokoro-82M

It is under construction and likely will not be useful until the [next base model release](https://huggingface.co/hexgrad/Kokoro-82M/discussions/36).

The goal is to be able to `pip install kokoro` and offer some of the design goals and functionalities laid out below.

### Python usage

First things first: install the package.

```bash
$ pip install kokoro
```

Then, generate some speech.

```python
from kokoro import KPipeline
import soundfile as sf
import numpy as np

# 'a' for American English
pipeline = KPipeline(lang_code='a')

text = """There once was a codebase so light,
That made voices sound crystal bright.
Open source and free,
It spoke with such glee,
Making silicon sing in delight!"""

# Generate and combine all audio segments
audio_segments = [audio for _, _, audio in pipeline(text, voice='af_bella')]
full_audio = np.concatenate(audio_segments)

# Save as 24kHz WAV file
sf.write('output.wav', full_audio, 24000)
```

### G2P will be imported from Misaki
[Misaki](https://github.com/hexgrad/misaki) is a G2P engine with language-specific solutions:
```sh
pip install misaki[en] # installs English
pip install misaki[ja] # installs Japanese
```
Users who don't peek under the hood may not care, since `import kokoro` will simply `import misaki` and life goes on. This is likely the proper separation of responsibilities, and not all users will want or need all languages.

### Smarter LF chunking
Kokoro models have a 512 token context window, which usually amounts to about 30 seconds of audio. Finding natural stopping points in your text to chop is key to smooth long-form (LF) generation, which should be much easier with token-level traces in `misaki[en]` (hopefully other languages to follow).

### Cleaner modeling code
The modeling code could benefit from a touch-up and as a side effect, become ONNX exportable and hopefully slightly faster.

### Experimental features (TBD)
Today, voicepacks are essentially `(510, 256)`-shaped tensors, compiled as average styles per utterance length, with 510 possible lengths. Since most style vectors are computed on synthetic data, each style is essentially a "mean of means", which may explain why the voices are somewhat flat-sounding. It also implies that for any given utterance, currently the only features being used to choose how the voice sounds are (1) the user-selected voice name, like `af` and (2) the length of the utterance. Features like the punctuation texture `.?!` or the text sentiment are not yet being used. Potential solutions could be neural or even classical, e.g. using vector DBs. This, among other things, is still an area of research.

### Community contributions welcome
Within a couple weeks of Kokoro's Christmas 2024 release, talented people already [built](https://github.com/thewh1teagle/kokoro-onnx) [great](https://github.com/remsky/Kokoro-FastAPI) [things](https://github.com/lucasjinreal/Kokoros). If you want to build something, go for it! Kokoro is permissive Apache-licensed software. If you also want to add or improve something here (or [misaki](https://github.com/hexgrad/misaki)), hopefully Kokoro can earn your commit, and feel free to open a PR if so.
