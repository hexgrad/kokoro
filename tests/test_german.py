"""Regression tests for German (lang_code='d') support.

German is phonemized with espeak-ng, like the other non-English languages
(Spanish, French, Italian, Hindi, Brazilian Portuguese).
"""

from kokoro.__main__ import languages
from kokoro.pipeline import ALIASES, LANG_CODES, KPipeline

GERMAN_TEXT = 'Guten Morgen, wie geht es Ihnen?'


def test_german_is_registered_as_a_language():
    assert LANG_CODES['d'] == 'de'
    assert ALIASES['de'] == 'd'
    assert ALIASES['de-de'] == 'd'


def test_cli_offers_german_language_code():
    assert 'd' in languages


def test_german_pipeline_uses_the_german_espeak_backend():
    pipeline = KPipeline(lang_code='d', model=False)
    assert pipeline.lang_code == 'd'
    assert pipeline.g2p.language == 'de'


def test_german_pipeline_phonemizes_german_text():
    pipeline = KPipeline(lang_code='d', model=False)
    results = list(pipeline(GERMAN_TEXT, split_pattern=None))
    assert len(results) == 1
    graphemes, phonemes, audio = results[0]
    assert graphemes == GERMAN_TEXT
    # espeak-ng returns IPA, so the phonemes are not the input text
    assert phonemes
    assert phonemes != GERMAN_TEXT
    assert any(ord(c) > 127 for c in phonemes)
    assert audio is None


def test_german_alias_resolves_to_the_same_pipeline():
    pipeline = KPipeline(lang_code='de', model=False)
    assert pipeline.lang_code == 'd'
    phonemes = [result.phonemes for result in pipeline(GERMAN_TEXT, split_pattern=None)]
    assert phonemes == [
        result.phonemes
        for result in KPipeline(lang_code='d', model=False)(GERMAN_TEXT, split_pattern=None)
    ]


def test_german_text_is_chunked_like_other_non_english_languages():
    pipeline = KPipeline(lang_code='d', model=False)
    text = 'Erste Zeile. Zweite Zeile.'
    results = list(pipeline(text, split_pattern=r'\n+'))
    assert results
    assert all(result.phonemes for result in results)
