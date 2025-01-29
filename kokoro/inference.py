from typing import List, Tuple, Iterator
import torch
from .models import KModel

class KInference:
    """A decoupled inference pipeline that handles preprocessing and model inference.
    This class assumes the model and voices are loaded externally, focusing only on
    the core preprocessing and inference functionality.
    """
    def __init__(self, vocab: dict, phonemizer=None):
        """
        Args:
            vocab: Dictionary mapping phonemes to token ids
            phonemizer: Optional function or object that converts text to phonemes.
                       If None, will use misaki G2P as default.
                       If function: should take text and return (text, tokens)
                       If object: should have __call__ method that takes text and returns (text, tokens)
                       where tokens is a list of objects with attributes:
                       - text: original text
                       - phonemes: phoneme string or None if no phonemes
                       - prespace: bool indicating if space before
                       - whitespace: string of whitespace after
        """
        self.vocab = vocab
        self._setup_phonemizer(phonemizer)
    
    def _setup_phonemizer(self, phonemizer):
        """Set up the phonemizer, using misaki G2P as default."""
        if phonemizer is None:
            # Use default misaki G2P
            from misaki import en
            self.phonemizer = en.G2P(trf=False, british=False)
        elif callable(phonemizer):
            # Use provided function/callable
            self.phonemizer = phonemizer
        else:
            raise ValueError("phonemizer must be None or a callable")

    @staticmethod
    def waterfall_last(pairs, next_count, waterfall=['!.?…', ':;', ',—'], bumps={')', '"'}):
        """Split text at natural breakpoints to stay within token limits."""
        for w in waterfall:
            z = next((i for i, (_, ps) in reversed(list(enumerate(pairs))) if ps.strip() in set(w)), None)
            if z is not None:
                z += 1
                if z < len(pairs) and pairs[z][1].strip() in bumps:
                    z += 1
                _, ps = zip(*pairs[:z])
                if next_count - len(''.join(ps)) <= 510:
                    return z
        return len(pairs)

    def tokenize(self, tokens) -> Iterator[Tuple[str, str]]:
        """Convert tokens to phoneme sequences, splitting at natural breakpoints."""
        pairs = []
        count = 0
        for w in tokens:
            for t in (w if isinstance(w, list) else [w]):
                if t.phonemes is None:
                    continue
                next_ps = ' ' if t.prespace and pairs and not pairs[-1][1].endswith(' ') and t.phonemes else ''
                next_ps += ''.join(filter(lambda p: p in self.vocab, t.phonemes.replace('ɾ', 'T')))  # American English: ɾ => T
                next_ps += ' ' if t.whitespace else ''
                next_count = count + len(next_ps.rstrip())
                if next_count > 510:
                    z = self.waterfall_last(pairs, next_count)
                    text, ps = zip(*pairs[:z])
                    ps = ''.join(ps)
                    yield ''.join(text).strip(), ps.strip()
                    pairs = pairs[z:]
                    count -= len(ps)
                    if not pairs:
                        next_ps = next_ps.lstrip()
                pairs.append((t.text + t.whitespace, next_ps))
                count += len(next_ps)
        if pairs:
            text, ps = zip(*pairs)
            yield ''.join(text).strip(), ''.join(ps).strip()

    def preprocess(self, text: str, split_pattern=r'\n+') -> Iterator[Tuple[str, str, List[int]]]:
        """Preprocess text into model inputs.
        
        Args:
            text: Input text to convert to speech
            split_pattern: Pattern to split text into chunks (None for no splitting)
            
        Returns:
            Iterator of (graphemes, phonemes, input_ids) tuples
        """
        if isinstance(text, str) and split_pattern:
            text = text.split(split_pattern)
        
        for graphemes in text if isinstance(text, list) else [text]:
            _, tokens = self.phonemizer(graphemes)
            for gs, ps in self.tokenize(tokens):
                if not ps:
                    continue
                elif len(ps) > 510:
                    print('TODO: Unexpected len(ps) > 510', len(ps), ps)
                    continue
                    
                input_ids = list(filter(None, map(self.vocab.get, ps)))
                if input_ids and len(input_ids) <= 510:
                    yield gs, ps, input_ids

    def __call__(self, model: KModel, text: str, voice_embeddings: torch.Tensor, 
                 speed: float = 1.0, split_pattern=r'\n+') -> Iterator[Tuple[str, str, torch.Tensor]]:
        """Run inference on preprocessed text.
        
        Args:
            model: Loaded KModel instance
            text: Input text to convert to speech
            voice_embeddings: Pre-loaded voice embeddings tensor
            speed: Speech speed multiplier
            split_pattern: Pattern to split text into chunks
            
        Returns:
            Iterator of (graphemes, phonemes, audio) tuples
        """
        for gs, ps, input_ids in self.preprocess(text, split_pattern):
            audio = model(input_ids, voice_embeddings[len(input_ids)-1], speed)
            yield gs, ps, audio
