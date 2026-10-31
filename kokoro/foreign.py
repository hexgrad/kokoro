from dataclasses import replace
from itertools import groupby
from loguru import logger
from misaki import espeak
from misaki.en import PUNCTS
from typing import Optional
import regex

# English G2P sends words missing from its lexicon to an espeak English fallback,
# which reads non-Latin text as letter names ("Chinese letter", "pe er i ve...").
# ScriptFallback instead phonemizes each run of non-Latin letters with the most
# likely language's G2P, keeping the English voice.

ESPEAK_SCRIPTS = {
    'Arabic': 'ar',
    'Cyrillic': 'ru',
    'Devanagari': 'hi',
    'Greek': 'el',
    'Hangul': 'ko',
    'Hebrew': 'he',
}

# A lone Greek letter is usually a symbol (β-catenin, Δ, π), which English espeak
# already names correctly, so Greek runs need at least two letters.
def script_run(script: str) -> str:
    letter = fr'\p{{{script}}}\p{{Mn}}*'
    return fr'(?P<{script}>{letter}(?:{letter}){"+" if script == "Greek" else "*"})'

RUN_REGEX = regex.compile('|'.join(
    [r'(?P<cjk>[\p{Han}\p{Hiragana}\p{Katakana}ー]+)'] + [script_run(s) for s in ESPEAK_SCRIPTS]
))
KANA = regex.compile(r'[\p{Hiragana}\p{Katakana}]')

# Applied before English G2P so CJK sentences split into tokens and chunks at punctuation.
# '、' is left alone so a Japanese sentence stays in one token for the kana check below.
CJK_PUNCTUATION = str.maketrans({
    '。': '. ', '．': '. ', '，': ', ', '！': '! ', '？': '? ', '：': ': ', '；': '; ',
    '「': '“', '」': '”', '『': '“', '』': '”', '（': ' (', '）': ') ', '　': ' ', '・': ' ',
})
# espeak reads Hebrew with vowel points as letter names.
HEBREW_POINTS = regex.compile(r'\p{Mn}')

class ScriptFallback:
    def __init__(self, british: bool, zh_version: Optional[str] = None):
        self.english = espeak.EspeakFallback(british=british)
        self.zh_version = zh_version
        self.g2ps = {}

    def get_g2p(self, language: str):
        # Chinese and Japanese are optional (misaki[zh], misaki[ja]); without them,
        # those runs keep the English fallback's reading.
        if language not in self.g2ps:
            try:
                if language == 'zh':
                    from misaki import zh
                    self.g2ps[language] = zh.ZHG2P(version=self.zh_version)
                elif language == 'ja':
                    from misaki import ja
                    self.g2ps[language] = ja.JAG2P()
                else:
                    self.g2ps[language] = espeak.EspeakG2P(language=language)
            except ImportError:
                logger.warning(f'Reading {language} text needs `pip install misaki[{language}]`; using the English fallback')
                self.g2ps[language] = None
        return self.g2ps[language]

    def phonemize_other(self, token, text: str) -> str:
        if not any(c.isalnum() for c in text):
            return ''.join(c for c in text.replace('、', ',') if c in PUNCTS)
        return self.english(replace(token, text=text))[0] or ''

    def phonemize_chinese(self, token, g2p, run: str) -> str:
        from pypinyin import lazy_pinyin
        # pypinyin has no reading for some rare characters (radicals, Japanese-only kanji), and
        # misaki's Chinese G2P crashes on or skips them, so they get the English reading as before.
        ps = ''
        for readable, chars in groupby(run, lambda c: lazy_pinyin(c)[0] != c):
            chars = ''.join(chars)
            ps += g2p(chars)[0].strip() if readable else self.phonemize_other(token, chars)
        return ps

    def phonemize_run(self, token, m, cjk_language: str) -> str:
        language = cjk_language if m.lastgroup == 'cjk' else ESPEAK_SCRIPTS[m.lastgroup]
        g2p = self.get_g2p(language)
        if g2p is None:
            return self.phonemize_other(token, m.group())
        if language == 'zh':
            return self.phonemize_chinese(token, g2p, m.group())
        run = HEBREW_POINTS.sub('', m.group()) if m.lastgroup == 'Hebrew' else m.group()
        return g2p(run)[0].strip()

    def __call__(self, token):
        if not RUN_REGEX.search(token.text):
            return self.english(token)
        # Han is shared by Chinese and Japanese; kana in the same token (usually a clause) means Japanese.
        cjk_language = 'ja' if KANA.search(token.text) else 'zh'
        ps, last_end = '', 0
        for m in RUN_REGEX.finditer(token.text):
            ps += self.phonemize_other(token, token.text[last_end:m.start()])
            ps += self.phonemize_run(token, m, cjk_language)
            last_end = m.end()
        return ps + self.phonemize_other(token, token.text[last_end:]), 1
