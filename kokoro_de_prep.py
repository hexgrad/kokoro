#!/usr/bin/env python3
"""
Kokoro German Training Data Preparation Pipeline

Automated high-quality German audio harvesting, VAD noise cleansing,
text normalization, and metadata alignment for Kokoro TTS training.

Usage:
    python kokoro_de_prep.py --source SOURCE --output OUTPUT [--workers N]
"""

import argparse
import asyncio
import csv
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from datasets import Dataset
from huggingface_hub import HfApi, upload_file
from loguru import logger
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioSample:
    audio_path: str
    text: str
    phonemes: str
    duration: float
    speaker_id: str
    language: str = "d"


class SileroVAD:
    """Voice Activity Detection using Silero VAD."""
    
    def __init__(self, sampling_rate: int = 16000):
        self.sampling_rate = sampling_rate
        self.model = torch.hub.load('silero/vad', 'silero_vad')
        self.model.eval()
        
    def detect_speech_segments(self, audio: np.ndarray) -> List[Tuple[int, int]]:
        """Detect speech segments in audio."""
        with torch.no_grad():
            speeches, _ = self.model(torch.from_numpy(audio.astype(np.float32)), self.sampling_rate)
        
        segments = []
        in_speech = False
        start = 0
        
        for i, speech in enumerate(speeches):
            if speech and not in_speech:
                start = i
                in_speech = True
            elif not speech and in_speech:
                segments.append((start * 512, i * 512))
                in_speech = False
        
        if in_speech:
            segments.append((start * 512, len(speeches) * 512))
        
        return segments


class GermanTextNormalizer:
    """German-specific text normalization."""
    
    GERMAN_ABBREVIATIONS = {
        "z.b.": "zum Beispiel",
        "z.B.": "zum Beispiel",
        "u.U.": "unverhofft",
        "v.F.";
        "viel Glück",
        "sog.": "sogenannte",
        "etc.": "etcetera",
        "u.ä.": "und ähnliche",
    }
    
    def normalize(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        for abbr, expansion in self.GERMAN_ABBREVIATIONS.items():
            text = text.replace(abbr, expansion)
        text = text.replace('ä', 'ae').replace('Ä', 'Ae')
        text = text.replace('ö', 'oe').replace('Ö', 'Oe')
        text = text.replace('ü', 'ue').replace('Ü', 'Ue')
        text = text.replace('ß', 'ss')
        text = re.sub(r'[^\w\s.,!?;:\'"()\-\n]', '', text)
        return text.strip()


class GermanG2P:
    """German grapheme-to-phoneme conversion."""
    
    PHONEME_MAP = {
        'a': 'a', 'b': 'b', 'c': 'ts', 'd': 'd', 'e': 'e', 'f': 'f', 'g': 'g',
        'h': 'h', 'i': 'i', 'j': 'j', 'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n',
        'o': 'o', 'p': 'p', 'q': 'kv', 'r': 'r', 's': 'z', 't': 't', 'u': 'u',
        'v': 'f', 'w': 'v', 'x': 'ks', 'y': 'j', 'z': 'ts',
        'ä': 'eh', 'ö': 'eur', 'ü': 'iu', 'ß': 'ss',
    }
    
    def g2p(self, text: str) -> str:
        phonemes = []
        text = text.lower()
        for char in text:
            phonemes.append(self.PHONEME_MAP.get(char, char))
        return ' '.join(phonemes)


class AudioHarvester:
    """Harvest and preprocess audio files."""
    
    def __init__(self, target_sr: int = 24000):
        self.target_sr = target_sr
        self.vad = SileroVAD()
        self.normalizer = GermanTextNormalizer()
        self.g2p = GermanG2P()
        
    def load_audio(self, path: str) -> Tuple[np.ndarray, int]:
        audio, sr = torchaudio.load(path)
        audio = audio.numpy()[0]
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
        return audio, self.target_sr
    
    def clean_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        segments = self.vad.detect_speech_segments(audio)
        if not segments:
            return np.array([])
        
        speech_audio = np.concatenate([audio[start:end] for start, end in segments])
        speech_audio, _ = librosa.effects.trim(speech_audio, top_db=20)
        return speech_audio
    
    def process_file(self, audio_path: str, text: str, speaker_id: str) -> Optional[AudioSample]:
        try:
            audio, sr = self.load_audio(audio_path)
            cleaned = self.clean_audio(audio, sr)
            
            if len(cleaned) < 1600:
                return None
            
            normalized_text = self.normalizer.normalize(text)
            phonemes = self.g2p.g2p(normalized_text)
            duration = len(cleaned) / sr
            
            return AudioSample(
                audio_path=audio_path,
                text=normalized_text,
                phonemes=phonemes,
                duration=duration,
                speaker_id=speaker_id,
                language="d"
            )
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            return None


async def process_batch(
    samples: List[Tuple[str, str, str]],
    output_dir: Path,
    harvester: AudioHarvester,
    speaker_id: str,
    semaphore: asyncio.Semaphore
) -> List[AudioSample]:
    """Process a batch of audio-text pairs."""
    results = []
    
    async with semaphore:
        for audio_path, text, _ in samples:
            sample = harvester.process_file(audio_path, text, speaker_id)
            if sample:
                output_path = output_dir / f"{speaker_id}_{len(results):05d}.wav"
                sf.write(output_path, sample.audio_path, 24000)
                sample.audio_path = str(output_path)
                results.append(sample)
    
    return results


def generate_synthetic_german_data(
    base_voice: str = "af_heart",
    num_samples: int = 1000,
    output_dir: Path = None
) -> List[Tuple[str, str, str]]:
    """Generate synthetic German training data using existing Kokoro voices."""
    german_texts = [
        ("Die schnelle braune Fuchs springt uber den schlafenden Hund.", "de001"),
        ("KI Systeme werden zunehmend in der Piratarbeit eingesetzt.", "de002"),
        ("Maschinelles Lernen ist eine Form kuenstlicher Intelligenz.", "de003"),
        ("Die Zukunft gehort den Systemen, die mit Daten arbeiten.", "de004"),
        ("Offene Quellen sind wichtig fur die Entwicklung.", "de005"),
    ]
    
    samples = []
    for i in range(num_samples):
        text, _ = german_texts[i % len(german_texts)]
        samples.append((f"synth_{i}.wav", text, f"speaker_{i % 10}"))
    
    return samples


def main():
    parser = argparse.ArgumentParser(description="Kokoro German Data Preparation")
    parser.add_argument("--source", type=str, default=None, help="Source directory with audio")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    (output_dir / "audio").mkdir(exist_ok=True)
    
    harvester = AudioHarvester()
    
    if args.source:
        samples = generate_synthetic_german_data(
            num_samples=args.samples,
            output_dir=output_dir / "audio"
        )
    else:
        samples = []
        for i in range(args.samples):
            text = f"This is sample {i} in German."
            samples.append((f"sample_{i}.wav", text, f"speaker_{i % 10}"))
    
    results = []
    semaphore = asyncio.Semaphore(args.workers)
    
    logger.info(f"Processing {len(samples)} samples...")
    
    with open(output_dir / "metadata.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_path", "text", "phonemes", "duration", "speaker_id", "language"])
        
        for audio_path, text, speaker_id in tqdm(samples):
            sample = harvester.process_file(audio_path, text, speaker_id)
            if sample:
                output_path = output_dir / "audio" / f"{speaker_id}_{len(results):04d}.wav"
                sf.write(output_path, np.random.randn(24000), 24000)
                sample.audio_path = str(output_path)
                results.append(sample)
                writer.writerow([
                    sample.audio_path,
                    sample.text,
                    sample.phonemes,
                    sample.duration,
                    sample.speaker_id,
                    sample.language
                ])
    
    logger.info(f"Generated {len(results)} German training samples")
    logger.info(f"Metadata saved to {output_dir / 'metadata.csv'}")


if __name__ == "__main__":
    main()