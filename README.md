# kokoro_inference

Personal fork of `kokoro` focused on inference-time performance experiments for
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M).

The main change in this fork is that `KModel` prepares the network for
inference by stripping PyTorch `weight_norm` parametrizations after checkpoint
loading. The normalized weights are materialized once with
`leave_parametrized=True`, so inference no longer pays the parametrization
overhead on every forward pass.

This is an inference optimization project, not a training fork. The goal is to
keep model behavior compatible with upstream Kokoro while making the loaded
model cheaper to run.

## What Changed

- `KModel` strips `weight_norm` by default during initialization.
- `KModel.prepare_for_inference()` can be called manually and returns the number
  of parametrized weights removed.
- `for_training=True` keeps the original training-time parametrizations.
- `local_repo_dir` support lets `KModel` and `KPipeline` load a repo-shaped local
  model directory instead of downloading from Hugging Face.

Only `weight_norm` parametrizations are removed. Normalization layers that are
part of the model computation, such as layer norm and instance norm, are left in
place.

## Install

From this checkout:

```bash
pip install -e .
```

Kokoro uses `misaki` for grapheme-to-phoneme conversion. English support is
included through the package dependency. Some languages and fallback paths also
need `espeak-ng` installed on the system.

macOS:

```bash
brew install espeak-ng
```

Debian/Ubuntu:

```bash
sudo apt-get install espeak-ng
```

## Basic Usage

```python
from kokoro import KPipeline
import soundfile as sf

pipeline = KPipeline(lang_code="a")

text = "Kokoro is an open-weight text-to-speech model."
generator = pipeline(text, voice="af_heart")

for index, result in enumerate(generator):
    if result.audio is None:
        continue
    sf.write(f"{index}.wav", result.audio, 24000)
```

The default `KPipeline` path creates a `KModel`, moves it to the selected device,
sets it to eval mode, and uses the stripped-weight inference form.

## Manual Inference Preparation

Use this if you want to inspect or control the stripping step directly:

```python
from kokoro import KModel

model = KModel(for_training=True)
removed = model.prepare_for_inference()
model.eval()

print(f"removed {removed} weight_norm parametrizations")
```

For training or fine-tuning experiments, keep the original parametrized modules:

```python
from kokoro import KModel

model = KModel(for_training=True)
```

## Local Model Files

To avoid Hugging Face downloads, provide a local directory containing
`config.json`, the Kokoro checkpoint, and `voices/`:

```python
from kokoro import KPipeline

pipeline = KPipeline(
    lang_code="a",
    local_repo_dir="./checkpoints/Kokoro-82M",
)
```

The directory is expected to look like a model repo:

```text
checkpoints/Kokoro-82M/
  config.json
  kokoro-v1_0.pth
  voices/
```

## CLI

```bash
python -m kokoro \
  --text "The quick brown fox jumps over the lazy dog." \
  --voice af_heart \
  --output-file out.wav
```

## Optimization Notes

Kokoro inherits StyleTTS-style modules that wrap many convolution layers with
`torch.nn.utils.parametrizations.weight_norm`. That representation is useful
while training because the effective weight is derived from separate magnitude
and direction parameters.

For inference, the effective weight can be computed once after loading the
checkpoint. This fork does that with:

```python
torch.nn.utils.parametrize.remove_parametrizations(
    module,
    "weight",
    leave_parametrized=True,
)
```

Expected impact:

- lower Python and parametrization overhead during repeated forward passes
- simpler module state for export and deployment experiments
- no intentional checkpoint or architecture change

Actual speedup depends on hardware, backend, text length, batching, and whether
the workload is dominated by model execution or text processing. Benchmark on
your deployment target before treating this as a production optimization.

## Development

Run the test suite:

```bash
pytest
```

Useful files:

- `kokoro/model.py`: model loading and inference preparation
- `kokoro/modules.py`: text encoder and predictor modules
- `kokoro/istftnet.py`: decoder modules using `weight_norm`
- `examples/device_examples.py`: simple device timing example
- `examples/export.py`: ONNX export experiment

## Upstream

This repository is a personal inference-focused fork of upstream Kokoro. For the
original project, model cards, voices, and samples, see:

- https://github.com/hexgrad/kokoro
- https://huggingface.co/hexgrad/Kokoro-82M
