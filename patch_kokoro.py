import re

def patch_pipeline():
    file = "kokoro/pipeline.py"
    with open(file, "r") as f:
        content = f.read()

    start_str = """    def load_single_voice(self, voice: str):
        if voice in self.voices:
            return self.voices[voice]
        if voice.endswith('.pt'):
            f = voice
        else:
            f = hf_hub_download(repo_id=self.repo_id, filename=f'voices/{voice}.pt')"""
            
    replace_str = """    def load_single_voice(self, voice: str):
        if voice in self.voices:
            return self.voices[voice]
        if voice.endswith('.pt'):
            f = voice
            if not os.path.exists(f):
                if self.lang_code == 'd':
                    raise RuntimeError(f"German voice tensor '{voice}' not found locally. Please download a community German voice checkpoint (e.g., from Hugging Face) into your voices directory or specify a custom voice path.")
                raise FileNotFoundError(f"Voice tensor '{voice}' not found locally.")
        else:
            try:
                f = hf_hub_download(repo_id=self.repo_id, filename=f'voices/{voice}.pt')
            except Exception as e:
                if self.lang_code == 'd':
                    logger.warning(f"Voice '{voice}' not found in {self.repo_id}. Attempting community fallback...")
                    try:
                        f = hf_hub_download(repo_id='cryptomilk/kokoro-german-kerstin', filename=f'voices/{voice}.pt')
                    except Exception:
                        raise RuntimeError(f"German voice tensor '{voice}' not found locally. Please download a community German voice checkpoint (e.g., from Hugging Face) into your voices directory or specify a custom voice path.")
                else:
                    raise e"""
                    
    content = content.replace(start_str, replace_str)
    
    with open(file, "w") as f:
        f.write(content)

def rewrite_test():
    file = "test_german_inference.py"
    new_test = """import os
import torch
import soundfile as sf
from kokoro import KPipeline

def test_german_inference():
    pipeline = KPipeline(lang_code='d')
    text = "Kokoro German ist ein Text to Speech Modell mit zweiundachtzig Millionen Parametern."

    # Create a temporary mock voice tensor for CI/unit testing (isolated offline execution)
    mock_voice_path = 'mock_german_voice.pt'
    # Kokoro voice style vectors are typically [1, 256] arrays
    mock_tensor = torch.randn(256) # wait, load_single_voice returns pack. wait, style is pack directly or pack[256]?
    # Actually KModel's style is usually just a 1D tensor or tuple. 
    # Let's inspect KPipeline's load_single_voice pack output if we need to mock it exactly.
    # The prompt says: (matching the exact expected shape and dtype of Kokoro voice style vectors: torch.randn(1, 256) or the native Kokoro embedding dimension). Wait, let's just make it a tensor of shape [1, 256].
    pass
"""
    pass

patch_pipeline()
print("Patched pipeline.py!")
