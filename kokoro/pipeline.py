from .model import KModel
from huggingface_hub import hf_hub_download
from misaki import en, espeak
from numbers import Number
from typing import Generator, List, Optional, Tuple, Union
import re
import torch

LANG_CODES = dict(
    a='American English',
    b='British English',
    e='es',
    f='fr-fr',
    h='hi',
    i='it',
    p='pt-br',
)

class KPipeline:
    '''
    KPipeline is a language-aware support class with 2 main responsibilities:
    1. Perform language-specific G2P, mapping (and chunking) text -> phonemes
    2. Manage and store voices, lazily downloaded from HF if needed

    You are expected to have one KPipeline per language. If you have multiple
    KPipelines, you should reuse one KModel instance across all of them.

    KPipeline is designed to work with a KModel, but this is not required.
    There are 2 ways to pass an existing model into a pipeline:
    1. On init: us_pipeline = KPipeline(lang_code='a', model=model)
    2. On call: us_pipeline(text, voice, model=model)

    By default, KPipeline will automatically initialize its own KModel. To
    suppress this, construct a "quiet" KPipeline with model=False.

    A "quiet" KPipeline yields (graphemes, phonemes, None) without generating
    any audio. You can use this to phonemize and chunk your text in advance.

    A "loud" KPipeline _with_ a model yields (graphemes, phonemes, audio).
    '''
    def __init__(self, lang_code: str, model: Union[KModel, bool] = True, trf: bool = False):
        assert lang_code in LANG_CODES, (lang_code, LANG_CODES)
        self.lang_code = lang_code
        self.model = None
        if isinstance(model, KModel):
            self.model = model
        elif model:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = KModel().to(device).eval()
        self.voices = {}
        if lang_code in 'ab':
            try:
                fallback = espeak.EspeakFallback(british=lang_code=='b')
            except Exception as e:
                print('⚠️ WARNING: EspeakFallback not enabled. OOD words will be skipped.', e)
                fallback = None
            self.g2p = en.G2P(trf=trf, british=lang_code=='b', fallback=fallback)
        else:
            language = LANG_CODES[lang_code]
            print(f"⚠️ WARNING: Using EspeakG2P(language='{language}'). Chunking logic not yet implemented, so long texts may be truncated unless you split them with '\\n'.")
            self.g2p = espeak.EspeakG2P(language=language)

    def load_voice(self, voice: str) -> torch.FloatTensor:
        if voice in self.voices:
            return self.voices[voice]
        if voice.endswith('.pt'):
            f = voice
        else:
            f = hf_hub_download(repo_id=KModel.REPO_ID, filename=f'voices/{voice}.pt')
            if not voice.startswith(self.lang_code):
                v = LANG_CODES.get(voice, voice)
                p = LANG_CODES.get(self.lang_code, self.lang_code)
                print(f'⚠️ WARNING: Language mismatch, loading {v} voice into {p} pipeline.')
        pack = torch.load(f, weights_only=True)
        self.voices[voice] = pack
        return pack

    @classmethod
    def waterfall_last(
        cls,
        pairs: List[Tuple[str, str]],
        next_count: int,
        waterfall: List[str] = ['!.?…', ':;', ',—'],
        bumps: List[str] = [')', '”']
    ) -> int:
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

    def en_tokenize(
        self,
        tokens: List[Union[en.MutableToken, List[en.MutableToken]]]
    ) -> Generator[Tuple[str, str], None, None]:
        pairs = []
        count = 0
        for w in tokens:
            for t in (w if isinstance(w, list) else [w]):
                if t.phonemes is None:
                    continue
                next_ps = ' ' if t.prespace and pairs and not pairs[-1][1].endswith(' ') and t.phonemes else ''
                next_ps += t.phonemes.replace('ɾ', 'T') # American English: ɾ => T
                next_ps += ' ' if t.whitespace else ''
                next_count = count + len(next_ps.rstrip())
                if next_count > 510:
                    z = KPipeline.waterfall_last(pairs, next_count)
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

    @classmethod
    def infer(
        cls,
        model: Optional[KModel],
        ps: str,
        pack: torch.FloatTensor,
        speed: Number
    ) -> Optional[torch.FloatTensor]:
        return model(ps, pack[len(ps)-1], speed) if model else None

    def __call__(
        self,
        text: Union[str, List[str]],
        voice: str,
        speed: Number = 1,
        split_pattern: Optional[str] = r'\n+',
        model: Optional[KModel] = None
    ) -> Generator[Tuple[str, str, Optional[torch.FloatTensor]], None, None]:
        pack = self.load_voice(voice)
        model = model or self.model
        pack = pack.to(model.device) if model else pack
        if isinstance(text, str):
            text = re.split(split_pattern, text.strip()) if split_pattern else [text]
        for graphemes in text:
            if self.lang_code in 'ab':
                _, tokens = self.g2p(graphemes)
                for gs, ps in self.en_tokenize(tokens):
                    if not ps:
                        continue
                    elif len(ps) > 510:
                        print(f"⚠️ WARNING: Unexpected len(ps) == {len(ps)} > 510 and ps == '{ps}'")
                        ps = ps[:510]
                    yield gs, ps, KPipeline.infer(model, ps, pack, speed)
            else:
                ps = self.g2p(graphemes)
                if not ps:
                    continue
                elif len(ps) > 510:
                    print(f'⚠️ WARNING: Truncating len(ps) == {len(ps)} > 510')
                    ps = ps[:510]
                yield graphemes, ps, KPipeline.infer(model, ps, pack, speed)
