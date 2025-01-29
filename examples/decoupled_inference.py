"""Example of using decoupled inference pipeline."""
import torch
from misaki import en
import os
import sys
import numpy as np
from scipy.io import wavfile
# Add parent directory to path for relative imports
# Only needed because we're inside the module at examples/
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from kokoro.models import KModel
from kokoro.inference import KInference
import json
import spacy
from pathlib import Path

MODEL_PATH =  Path(__file__).parent / 'custom_files'/ 'kokoro-v1_0.pth'
CONFIG_PATH = Path(__file__).parent /  'custom_files'/ 'config.json'
VOICE_PATH = Path(__file__).parent / 'custom_files' / 'af_alloy.pt'

# assert the user has placed the desired files
for path in [MODEL_PATH, CONFIG_PATH, VOICE_PATH]:
    assert path.exists(), f'{path} does not exist. Add file to the custom_files folder'

def ensure_pip():
    """Ensure pip is installed in the environment."""
    import importlib.util
    if importlib.util.find_spec('pip') is None:
        print("Installing pip...")
        import ensurepip
        ensurepip.bootstrap()

def main():
    # Download spacy model if needed/have issues 
    # Ensure pip is installed for spacy model download
    if not spacy.util.is_package("en_core_web_sm"):
        ensure_pip()
        print("Downloading spacy model...")
        spacy.cli.download("en_core_web_sm")

    # Load config and model
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = KModel(config, MODEL_PATH).to(device)
    model.eval()
    
    # Initialize G2P and inference pipeline
    g2p = en.G2P(trf=False, british=False)
    inference = KInference(vocab=config['vocab'], phonemizer=g2p)
    
    # Load voice
    voice_embeddings = torch.load(VOICE_PATH, weights_only=True).to(device)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'examples', 'custom_files', 'generated')
    os.makedirs(output_dir, exist_ok=True)

    # Test inference
    text = "Hello! This is a test of the decoupled inference pipeline."
    print(f"\nGenerating speech for: {text}")
    
    # Store all chunks for concatenation
    all_chunks = []
    
    for i, (graphemes, phonemes, audio) in enumerate(inference(model, text, voice_embeddings)):
        print(f"\nChunk: {graphemes}")
        print(f"Phonemes: {phonemes}")
        print(f"Audio shape: {audio.shape}")
        
        # Convert to numpy and save individual chunk

        chunk_path = os.path.join(output_dir, f'chunk_{i}.wav')
        wavfile.write(chunk_path, 24000, audio)
        print(f"Saved chunk to: {chunk_path}")
        
        all_chunks.append(audio)
    
    # Concatenate and save complete audio
    complete_audio = np.concatenate(all_chunks)
    complete_path = os.path.join(output_dir, 'complete.wav')
    wavfile.write(complete_path, 24000, complete_audio)
    print(f"\nSaved complete audio to: {complete_path}")

if __name__ == '__main__':
    main()
