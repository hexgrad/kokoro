import sys
import pytest
from misaki import espeak
import misaki
from kokoro import KPipeline


@pytest.fixture(scope='module')
def pipeline():
    return KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', model=False)


def upstream_pipeline():
    upstream = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', model=False)
    upstream.g2p.fallback = espeak.EspeakFallback(british=False)
    return upstream


def phonemize(pipeline, text):
    return ' '.join(r.phonemes for r in pipeline(text))


def test_han_uses_chinese_g2p(pipeline):
    zh = pytest.importorskip('misaki.zh')
    ps = phonemize(pipeline, 'labeled with elements—土火水木金—rather than numbers.')
    assert zh.ZHG2P()('土火水木金')[0] in ps
    assert ps.startswith(phonemize(pipeline, 'labeled with elements'))
    assert ps.endswith(phonemize(pipeline, 'rather than numbers.'))


def test_chinese_punctuation_keeps_every_clause(pipeline):
    zh = pytest.importorskip('misaki.zh')
    ps = phonemize(pipeline, '他说：“我们走吧，明天见。”然后离开了。')
    g2p = zh.ZHG2P()
    for clause in ['他说', '我们走吧', '明天见', '然后离开了']:
        assert g2p(clause)[0] in ps


def test_other_scripts_use_espeak_language(pipeline):
    ps = phonemize(pipeline, 'He said Привет and Καλημέρα.')
    assert espeak.EspeakG2P(language='ru')('Привет')[0] in ps
    assert espeak.EspeakG2P(language='el')('Καλημέρα')[0] in ps


def test_english_and_overrides_unchanged(pipeline):
    upstream = upstream_pipeline()
    text = 'Call [北京](/bAʤˈɪŋ/) and Xyzzyplugh now.'
    assert phonemize(pipeline, text) == phonemize(upstream, text)


def test_greek_letter_symbols_keep_english_names(pipeline):
    upstream = upstream_pipeline()
    text = 'The β-catenin level rose by Δ, about π percent.'
    assert phonemize(pipeline, text) == phonemize(upstream, text)


def test_han_without_mandarin_reading_does_not_crash(pipeline):
    zh = pytest.importorskip('misaki.zh')
    ps = phonemize(pipeline, '我兙你')
    g2p = zh.ZHG2P()
    assert g2p('我')[0] in ps and g2p('你')[0] in ps


def test_without_misaki_zh_han_reads_as_before(monkeypatch):
    monkeypatch.setitem(sys.modules, 'misaki.zh', None)
    monkeypatch.delattr(misaki, 'zh', raising=False)
    pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', model=False)
    text = 'labeled with elements—土火水木金—rather than numbers.'
    assert phonemize(pipeline, text) == phonemize(upstream_pipeline(), text)
