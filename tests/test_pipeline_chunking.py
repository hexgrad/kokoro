from kokoro import KPipeline


def fake_g2p(text):
    # Three phonemes per character, so 510 phonemes are 170 characters.
    return ''.join(c * 3 for c in text if not c.isspace()), None


def quiet_pipeline():
    # A Japanese pipeline without a model; the fake G2P avoids misaki[ja] and its dictionary.
    pipeline = KPipeline.__new__(KPipeline)
    pipeline.lang_code = 'j'
    pipeline.model = None
    pipeline.voices = {}
    pipeline.g2p = fake_g2p
    return pipeline


def test_short_text_is_one_chunk():
    text = '今日はいい天気ですね。'
    results = list(quiet_pipeline()(text))
    assert [(r.graphemes, r.phonemes) for r in results] == [(text, fake_g2p(text)[0])]


def test_long_text_is_split_without_dropping_phonemes():
    text = '今日はいい天気ですね。' * 40
    results = list(quiet_pipeline()(text))
    assert all(len(r.phonemes) <= 510 for r in results)
    assert ''.join(r.phonemes for r in results) == fake_g2p(text)[0]
    assert ''.join(r.graphemes for r in results) == text
    assert all(r.graphemes.endswith('。') for r in results)


def test_unpunctuated_text_is_cut_after_hiragana():
    text = '今日はいい天気' * 30
    results = list(quiet_pipeline()(text))
    assert len(results) == 2
    assert ''.join(r.phonemes for r in results) == fake_g2p(text)[0]
    assert results[0].graphemes.endswith('いい') and results[1].graphemes.startswith('天気')


def test_decimals_and_domains_are_not_cut():
    text = 'See example.com or 3.14 now. More text here'
    assert KPipeline.find_cut(text, 10) == text.index('now.') + len('now.')


def test_closing_quote_stays_with_its_sentence():
    text = '「行きます。」と彼は言った。そして'
    assert text[:KPipeline.find_cut(text, 7)] == '「行きます。」'
