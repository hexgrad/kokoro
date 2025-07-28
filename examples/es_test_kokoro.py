from kokoro import KPipeline
import soundfile as sf
import torch
import numpy as np
from scipy.signal import fftconvolve
import os

# Initialize the pipeline for Spanish, auto-select CUDA if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
pipeline = KPipeline(lang_code='e', device=device)

# Read the text to synthesize from a text file in the demo folder
with open('../demo/es.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# List of available Spanish voices
spanish_voices = ['ef_dora', 'em_alex', 'em_santa']

def simple_reverb(audio, sr=24000, decay=0.08, length=0.07, wet=0.033):
    # Revberberation effect using a simple impulse response
    ir_length = int(length * sr)
    ir = np.logspace(0, -decay, ir_length)
    ir = ir / np.max(ir)
    reverbed = fftconvolve(audio, ir, mode='full')[:len(audio)]
    # Mix the original audio with the reverberated (wet/dry)
    out = (1 - wet) * audio[:len(reverbed)] + wet * reverbed
    # Normalize if necessary
    max_val = np.max(np.abs(out))
    if max_val > 0:
        out = out / max_val * 0.95
    return out

for voice in spanish_voices:
    print(f"\nTesting voice: {voice}")
    speed = 0.95
    split_pattern = r'[.!?]+'
    audio_chunks = []
    generator = list(pipeline(
        text,
        voice=voice,
        speed=speed,
        split_pattern=split_pattern,
    ))
    fade_duration = 0.5  # seconds
    fade_samples = int(fade_duration * 24000)
    minimum_pause = 0  # seconds
    initial_silence = 0.38  # seconds
    initial_silence_samples = int(initial_silence * 24000)

    for i, (gs, ps, audio) in enumerate(generator):
        print(f"Sample {i}")
        print("Graphemes:", gs)
        print("Phonemes:", ps)
        audio_np = audio.cpu().numpy() if hasattr(audio, 'cpu') else np.array(audio)
        if len(audio_np) > fade_samples:
            if i == 0:
                # Add silence at the beginning and apply fade-in to the combined block
                silence = np.zeros(initial_silence_samples, dtype=np.float32)
                audio_with_silence = np.concatenate([silence, audio_np])
                fade_in = np.linspace(0, 1, fade_samples)
                audio_with_silence[:fade_samples] *= fade_in
                # Delete the initial silence after fade-in to avoid delaying the phrase
                audio_np = audio_with_silence[initial_silence_samples:]
            else:
                fade_in = np.linspace(0, 1, fade_samples)
                audio_np[:fade_samples] *= fade_in
            if i < len(generator) - 1:
                fade_out = np.linspace(1, 0, fade_samples)
                audio_np[-fade_samples:] *= fade_out
        audio_chunks.append(audio_np)
        if i < len(generator) - 1:
            pause = np.zeros(int(minimum_pause * 24000), dtype=np.float32)
            audio_chunks.append(pause)
    if audio_chunks:
        full_audio = np.concatenate(audio_chunks)
        # Apply subtle reverb
        full_audio = simple_reverb(full_audio, sr=24000)
        export_dir = 'audio_exports'
        os.makedirs(export_dir, exist_ok=True)
        filename = os.path.join(export_dir, f'output_{voice}.wav')
        sf.write(filename, full_audio, 24000)
        print(f"Audio saved to {filename}")
