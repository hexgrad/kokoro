import pytest
from kokoro.inference import KInference

def test_inference_init(mock_phonemizer, basic_vocab):
    """Test basic initialization"""
    inf = KInference(basic_vocab, phonemizer=mock_phonemizer)
    assert inf.vocab == basic_vocab
    assert callable(inf.phonemizer)

def test_inference_init_validation(basic_vocab):
    """Test initialization validation"""
    with pytest.raises(ValueError):
        KInference(basic_vocab, phonemizer="not_callable")

def test_waterfall_last():
    """Test the waterfall_last static method"""
    # Note: waterfall=['!.?…', ':;', ',—']
    pairs = [
        ('Hello', 'hello '),
        ('world', '!'),  # Just the punctuation mark
        ('More', 'more ')
    ]
    next_count = len(''.join(p[1] for p in pairs))  # Total length of phonemes
    split_idx = KInference.waterfall_last(pairs, next_count)
    assert split_idx == 2  # Should split after "!"

def test_basic_preprocessing(inference):
    """Test basic preprocessing without model dependencies"""
    text = "Hello world"
    results = list(inference.preprocess(text))
    
    assert len(results) == 1
    graphemes, phonemes, input_ids = results[0]
    assert graphemes == "Hello world"
    assert phonemes == "hello world"
    # input_ids should only include letters (no spaces)
    assert len(input_ids) == len(phonemes.replace(' ', ''))
    # Verify all input_ids are valid vocab indices
    assert all(0 <= id < len(inference.vocab) for id in input_ids)

def test_long_text_preprocessing(inference):
    """Test preprocessing handles text splitting correctly"""
    # Create text that should trigger splitting due to length
    long_text = "Hello world! " * 100
    results = list(inference.preprocess(long_text))
    
    # Should be split into multiple chunks
    assert len(results) > 1
    # Each chunk should have valid input_ids
    for graphemes, phonemes, input_ids in results:
        assert len(input_ids) <= 510  # Max token length
        assert all(0 <= id < len(inference.vocab) for id in input_ids)