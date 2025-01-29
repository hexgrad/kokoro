import pytest

class MockToken:
    def __init__(self, text, phonemes, prespace=False, whitespace=''):
        self.text = text
        self.phonemes = phonemes
        self.prespace = prespace
        self.whitespace = whitespace

@pytest.fixture
def mock_phonemizer():
    def _phonemizer(text):
        # Convert words to simple 'phonemes' for testing
        tokens = []
        for i, word in enumerate(text.split()):
            tokens.append(MockToken(
                text=word,
                phonemes=word.lower(),  # Simple mock 'phoneme' placeholder
                prespace=i > 0,  # First word has no prespace
                whitespace=' '  # Space after each
            ))
        return text, tokens
    return _phonemizer

@pytest.fixture
def basic_vocab():
    """Basic vocabulary mapping letters to indices"""
    return {c: i for i, c in enumerate('abcdefghijklmnopqrstuvwxyz')}

@pytest.fixture
def inference(mock_phonemizer, basic_vocab):
    """Pre-configured KInference instance for testing"""
    from kokoro.inference import KInference
    return KInference(basic_vocab, phonemizer=mock_phonemizer)