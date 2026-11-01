import sys
import misaki
import pytest
from misaki import espeak
from kokoro import KPipeline

REPO = 'hexgrad/Kokoro-82M'


def make_pipeline(lang_code, **kwargs):
    return KPipeline(lang_code=lang_code, repo_id=REPO, model=False, **kwargs)


def upstream_pipeline(lang_code):
    pipeline = make_pipeline(lang_code)
    pipeline.foreign.split = lambda text: [(text, None)]
    return pipeline


def phonemize(pipeline, text):
    return ' '.join(r.phonemes for r in pipeline(text))


def espeak_ps(language, text):
    return espeak.EspeakG2P(language=language)(text)[0]


@pytest.mark.parametrize('lang_code,text', [
    ('a', 'The β-catenin level rose by Δ, about π percent. Call [Kokoro](/kˈOkəɹO/) now.'),
    ('e', 'Los partidos políticos tradicionales compiten con los populismos.'),
    ('f', 'Le dromadaire resplendissant déambulait tranquillement.'),
    ('h', 'ट्रांसपोर्टरों की हड़ताल लगातार पांचवें दिन जारी'),
])
def test_text_in_native_script_unchanged(lang_code, text):
    assert phonemize(make_pipeline(lang_code), text) == phonemize(upstream_pipeline(lang_code), text)


@pytest.mark.parametrize('lang_code', ['a', 'e', 'f', 'h'])
def test_cyrillic_uses_russian(lang_code):
    ps = phonemize(make_pipeline(lang_code), 'Привет')
    assert espeak_ps('ru', 'Привет') in ps


def test_english_pipeline_reads_greek_words_not_letters():
    ps = phonemize(make_pipeline('a'), 'She said Καλημέρα.')
    assert espeak_ps('el', 'Καλημέρα') in ps


def test_latin_in_hindi_uses_english():
    english = phonemize(make_pipeline('a'), 'Hello')
    assert english in phonemize(make_pipeline('h'), 'उसने Hello कहा')


def test_han_uses_chinese():
    zh = pytest.importorskip('misaki.zh')
    g2p = zh.ZHG2P()
    for lang_code in 'ae':
        ps = phonemize(make_pipeline(lang_code), '他说：“我们走吧，明天见。”')
        for clause in ['他说', '我们走吧', '明天见']:
            assert g2p(clause)[0] in ps


def test_chinese_pipeline_reads_english():
    pytest.importorskip('misaki.zh')
    english = phonemize(make_pipeline('a'), 'meeting')
    assert english in phonemize(make_pipeline('z'), '我有一个meeting。')
    called = []
    pipeline = make_pipeline('z', en_callable=lambda text: called.append(text) or 'mˈiTɪŋ')
    phonemize(pipeline, '我有一个meeting。')
    assert called == ['meeting']


def test_han_without_mandarin_reading_does_not_crash():
    zh = pytest.importorskip('misaki.zh')
    g2p = zh.ZHG2P()
    ps = phonemize(make_pipeline('a'), '我兙你')
    assert g2p('我')[0] in ps and g2p('你')[0] in ps


def test_without_misaki_zh_han_reads_as_before(monkeypatch):
    monkeypatch.setitem(sys.modules, 'misaki.zh', None)
    monkeypatch.delattr(misaki, 'zh', raising=False)
    text = 'labeled with elements—土火水木金—rather than numbers.'
    assert phonemize(make_pipeline('a'), text) == phonemize(upstream_pipeline('a'), text)


@pytest.mark.parametrize('lang_code,text', [
    ('a', 'Go to [东京](/tˈOkjO/) or [Ελλάδα](/ɛlˈɑdə/) now.'),
    ('a', 'αβ T cells and ΔΔCt values.'),
])
def test_overrides_and_greek_symbols_unchanged(lang_code, text):
    assert phonemize(make_pipeline(lang_code), text) == phonemize(upstream_pipeline(lang_code), text)


def test_long_foreign_run_is_chunked_not_truncated():
    pytest.importorskip('misaki.zh')
    results = list(make_pipeline('a')('Intro: ' + '我们明天去北京看朋友然后回家吃饭' * 12 + ' end.'))
    assert len(results) > 2
    assert all(len(r.phonemes) <= 510 for r in results)
    assert results[-1].phonemes.endswith('ˈɛnd.')


def test_spacing_follows_source_text():
    ps = phonemize(make_pipeline('e'), 'Dijo (Привет) y "Привет" y Привет.')
    russian = espeak_ps('ru', 'Привет')
    assert f'({russian})' in ps and f'"{russian}"' in ps and f' {russian}.' in ps


def test_arabic_sounds_missing_from_vocab_use_nearest_phoneme():
    ps = phonemize(make_pipeline('a'), 'He said مرحبا to علي.')
    assert 'ħ' not in ps and 'ʕ' not in ps
    assert espeak_ps('ar', 'مرحبا').replace('ħ', 'h') in ps
