from itertools import groupby
from loguru import logger
from misaki import en, espeak
from misaki.en import LINK_REGEX
from typing import Callable, List, Optional, Tuple
import regex

# Each pipeline's G2P only handles its own language's script. Other scripts come out
# as symbol names ("Chinese letter"), spelled-out letters, or are silently dropped.
# ForeignScripts splits text into runs of other scripts so each run is phonemized by
# the most likely language's G2P, spoken in the pipeline's voice.

SCRIPT_LANGUAGES = {
    'Latin': 'en',
    'Arabic': 'ar',
    'Cyrillic': 'ru',
    'Devanagari': 'hi',
    'Greek': 'el',
    'Hangul': 'ko',
    'Hebrew': 'he',
}
LANGUAGES = {'a': 'en', 'b': 'en', 'e': 'es', 'f': 'fr-fr', 'h': 'hi', 'i': 'it', 'j': 'ja', 'p': 'pt-br', 'z': 'zh'}
NATIVE_SCRIPTS = {
    'en': 'Latin', 'es': 'Latin', 'fr-fr': 'Latin', 'it': 'Latin', 'pt-br': 'Latin',
    'hi': 'Devanagari',
}

def script_run(script: str) -> str:
    letter = fr'\p{{{script}}}\p{{Mn}}*'
    # Short Greek clusters are usually symbols (β-catenin, ΔΔCt, αβ T cells), which the
    # native G2P already names, so Greek words need at least three letters.
    word = f'(?:{letter}){{3,}}' if script == 'Greek' else f'(?:{letter})+'
    return fr'(?P<{script}>{word}(?:[ \'’\-]+{word})*)'

# '、' stays inside CJK runs so a Japanese clause is checked for kana as a whole.
RUN_REGEX = regex.compile('|'.join(
    [r'(?P<cjk>[\p{Han}\p{Hiragana}\p{Katakana}ー][\p{Han}\p{Hiragana}\p{Katakana}ー、]*)'] +
    [script_run(s) for s in SCRIPT_LANGUAGES]
))
KANA = regex.compile(r'[\p{Hiragana}\p{Katakana}]')
# espeak reads Hebrew with vowel points as letter names.
HEBREW_POINTS = regex.compile(r'\p{Mn}')
# Sounds Kokoro has no phoneme for (Arabic ḥ and ʿayn), mapped to the nearest one it has.
NEAREST_PHONEMES = str.maketrans({'ħ': 'h', 'ʕ': 'ʔ'})

# Applied in pipelines that aren't Chinese or Japanese, so CJK sentences split into
# tokens and chunks at punctuation.
CJK_PUNCTUATION = str.maketrans({
    '。': '. ', '．': '. ', '，': ', ', '！': '! ', '？': '? ', '：': ': ', '；': '; ',
    '「': '“', '」': '”', '『': '“', '』': '”', '（': ' (', '）': ') ', '　': ' ', '・': ' ',
})

class ForeignScripts:
    def __init__(
        self,
        lang_code: str,
        zh_version: Optional[str] = None,
        en_callable: Optional[Callable[[str], str]] = None
    ):
        self.language = LANGUAGES.get(lang_code, lang_code)
        self.zh_version = zh_version
        self.en_callable = en_callable
        self.g2ps = {}

    def get_g2p(self, language: str):
        # Chinese and Japanese are optional (misaki[zh], misaki[ja]); without them,
        # those runs stay with the pipeline's own G2P.
        if language not in self.g2ps:
            try:
                if language == 'zh':
                    from misaki import zh
                    self.g2ps[language] = zh.ZHG2P(version=self.zh_version)
                elif language == 'ja':
                    from misaki import ja
                    self.g2ps[language] = ja.JAG2P()
                elif language == 'en' and self.en_callable is not None:
                    self.g2ps[language] = lambda text: (self.en_callable(text), None)
                elif language == 'en':
                    fallback = espeak.EspeakFallback(british=False)
                    self.g2ps[language] = en.G2P(british=False, fallback=fallback, unk='')
                else:
                    self.g2ps[language] = espeak.EspeakG2P(language=language)
            except Exception as e:
                # e.g. misaki[ja] without `python -m unidic download`
                logger.warning(f'Reading {language} text needs `pip install misaki[{language}]` ({e}); using the pipeline G2P')
                self.g2ps[language] = None
        return self.g2ps[language]

    def run_language(self, m) -> str:
        if m.lastgroup == 'cjk':
            return 'ja' if self.language == 'ja' or KANA.search(m.group()) else 'zh'
        if NATIVE_SCRIPTS.get(self.language) == m.lastgroup:
            return self.language
        return SCRIPT_LANGUAGES[m.lastgroup]

    def foreign_spans(self, text: str):
        links = [m.span() for m in LINK_REGEX.finditer(text)]
        for m in RUN_REGEX.finditer(text):
            if any(start <= m.start() < end for start, end in links):
                continue  # [text](/phonemes/) overrides are the English G2P's to handle
            language = self.run_language(m)
            if language == self.language or self.get_g2p(language) is None:
                continue
            if language != 'zh':
                yield m.start(), m.end(), language
                continue
            # pypinyin has no reading for some rare characters (radicals, Japanese-only kanji),
            # and misaki's Chinese G2P crashes on or skips them, so they stay with the pipeline's G2P.
            from pypinyin import lazy_pinyin
            start = m.start()
            for readable, chars in groupby(m.group(), lambda c: c == '、' or lazy_pinyin(c)[0] != c):
                end = start + len(list(chars))
                if readable:
                    yield start, end, language
                start = end

    def split(self, text: str) -> List[Tuple[str, Optional[str]]]:
        '''Splits text into (segment, language) pairs; language is None for the pipeline's own.'''
        segments, last = [], 0
        for start, end, language in self.foreign_spans(text):
            if start > last:
                segments.append((text[last:start], None))
            segments.append((text[start:end], language))
            last = end
        if last < len(text):
            segments.append((text[last:], None))
        return segments

    def phonemize(self, segment: str, language: str) -> str:
        if language == 'he':
            segment = HEBREW_POINTS.sub('', segment)
        return self.get_g2p(language)(segment)[0].strip().translate(NEAREST_PHONEMES)
