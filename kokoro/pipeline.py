from .models import KModel
from .inference import KInference
from huggingface_hub import hf_hub_download
from misaki import en, espeak
import json
import os
import re
import torch

LANG_CODES = dict(
    a='American English',
    b='British English',
    e='es',
    f='fr-fr',
    h='hi',
    i='it',
    p='pt-br',
)
REPO_ID = 'hexgrad/Kokoro-82M'

class KPipeline:
    def __init__(self, lang_code='a', config_path=None, model_path=None, trf=False, device=None):
        """Initialize the pipeline with automatic model loading and configuration.
        For more control over model loading and inference, see the KInference class.
        """
        assert lang_code in LANG_CODES, (lang_code, LANG_CODES)
        self.lang_code = lang_code
        
        # Download and load config
        if config_path is None:
            print("Downloading config from HuggingFace hub...")
            config_path = hf_hub_download(repo_id=REPO_ID, filename='config.json')
        assert os.path.exists(config_path), f"Config file not found at {config_path}"
        print(f"Loading config from {config_path}")
        with open(config_path, 'r', encoding='utf-8') as r:
            config = json.load(r)
        print("Config loaded successfully")
        
        # Download and load model
        if model_path is None:
            print("Downloading model from HuggingFace hub...")
            model_path = hf_hub_download(repo_id=REPO_ID, filename='kokoro-v1_0.pth')
        assert os.path.exists(model_path), f"Model file not found at {model_path}"
        print(f"Model file found at {model_path}")
        
        # Setup model and device
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.model = KModel(config, model_path).to(self.device).eval()
        self.voices = {}
        
        # Initialize G2P based on language
        if lang_code in 'ab':
            try:
                fallback = espeak.EspeakFallback(british=lang_code=='b')
            except Exception as e:
                print('WARNING: EspeakFallback not enabled. Out-of-dictionary words will be skipped.', e)
                fallback = None
            g2p = en.G2P(trf=trf, british=lang_code=='b', fallback=fallback)
        else:
            language = LANG_CODES[lang_code]
            print(f"WARNING: Using EspeakG2P(language='{language}'). Chunking logic not yet implemented, so long texts may be truncated unless you split them with '\\n'.")
            g2p = espeak.EspeakG2P(language=language)
            
        # Initialize inference pipeline
        self.inference = KInference(vocab=config['vocab'], phonemizer=g2p)

    def load_voice(self, voice):
        """Load a voice model, downloading it if needed."""
        if voice in self.voices:
            return
            
        v = voice.split('/')[-1]
        if not v.startswith(self.lang_code):
            v = LANG_CODES.get(v, voice)
            p = LANG_CODES.get(self.lang_code, self.lang_code)
            print(f'WARNING: Loading {v} voice into {p} pipeline. Phonemes may be mismatched.')
            
        voice_path = voice if voice.endswith('.pt') else hf_hub_download(repo_id=REPO_ID, filename=f'voices/{voice}.pt')
        assert os.path.exists(voice_path)
        self.voices[voice] = torch.load(voice_path, weights_only=True).to(self.device)

    def __call__(self, text, voice='af', speed=1, split_pattern=r'\n+'):
        """Run the full pipeline with automatic voice loading."""
        assert isinstance(text, str) or isinstance(text, list), type(text)
        self.load_voice(voice)
        return self.inference(self.model, text, self.voices[voice], speed, split_pattern)
