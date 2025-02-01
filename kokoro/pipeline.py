from .model import KModel
from huggingface_hub import hf_hub_download
from misaki import en, espeak
from numbers import Number
from typing import Generator, List, Optional, Tuple, Union
from loguru import logger
import re
import torch

LANG_CODES = dict(
    # pip install misaki[en]
    a='American English',
    b='British English',

    # espeak-ng
    e='es',
    f='fr-fr',
    h='hi',
    i='it',
    p='pt-br',

    # pip install misaki[ja]
    j='Japanese',

    # pip install misaki[zh]
    z='Mandarin Chinese',
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
    def __init__(
        self,
        lang_code: str,
        model: Union[KModel, bool] = True,
        trf: bool = False,
        device: Optional[str] = None
    ):
        """Initialize a KPipeline.
        
        Args:
            lang_code: Language code for G2P processing
            model: KModel instance, True to create new model, False for no model
            trf: Whether to use transformer-based G2P
            device: Override default device selection ('cuda' or 'cpu', or None for auto)
                   If None, will auto-select cuda if available
                   If 'cuda' and not available, will explicitly raise an error
        """
        assert lang_code in LANG_CODES, (lang_code, LANG_CODES)
        self.lang_code = lang_code
        self.model = None
        if isinstance(model, KModel):
            self.model = model
        elif model:
            if device == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            try:
                self.model = KModel().to(device).eval()
            except RuntimeError as e:
                if device == 'cuda':
                    raise RuntimeError(f"""Failed to initialize model on CUDA: {e}. 
                                       Try setting device='cpu' or check CUDA installation.""")
                raise
        self.voices = {}
        if lang_code in 'ab':
            try:
                fallback = espeak.EspeakFallback(british=lang_code=='b')
            except Exception as e:
                logger.warning("EspeakFallback not Enabled: OOD words will be skipped")
                logger.warning({str(e)})
                fallback = None
            self.g2p = en.G2P(trf=trf, british=lang_code=='b', fallback=fallback)
        elif lang_code == 'j':
            try:
                from misaki import ja
                self.g2p = ja.JAG2P()
            except ImportError:
                logger.error("You need to `pip install misaki[ja]` to use lang_code='j'")
                raise
        elif lang_code == 'z':
            try:
                from misaki import zh
                self.g2p = zh.ZHG2P()
            except ImportError:
                logger.error("You need to `pip install misaki[zh]` to use lang_code='z'")
                raise
        else:
            language = LANG_CODES[lang_code]
            logger.warning(f"Using EspeakG2P(language='{language}'). Chunking logic not yet implemented, so long texts may be truncated unless you split them with '\\n'.")
            self.g2p = espeak.EspeakG2P(language=language)

    def load_single_voice(self, voice: str):
        if voice in self.voices:
            return self.voices[voice]
        if voice.endswith('.pt'):
            f = voice
        else:
            f = hf_hub_download(repo_id=KModel.REPO_ID, filename=f'voices/{voice}.pt')
            if not voice.startswith(self.lang_code):
                v = LANG_CODES.get(voice, voice)
                p = LANG_CODES.get(self.lang_code, self.lang_code)
                logger.warning(f'Language mismatch, loading {v} voice into {p} pipeline.')
        pack = torch.load(f, weights_only=True)
        self.voices[voice] = pack
        return pack

    """
    load_voice is a helper function that lazily downloads and loads a voice:
    Single voice can be requested (e.g. 'af_bella') or multiple voices (e.g. 'af_bella,af_jessica').
    If multiple voices are requested, they are averaged.
    Delimiter is optional and defaults to ','.
    """
    def load_voice(self, voice: str, delimiter: str = ",") -> torch.FloatTensor:
        if voice in self.voices:
            return self.voices[voice]
        packs = [self.load_single_voice(v) for v in voice.split(delimiter)]
        if len(packs) == 1:
            return packs[0]
        self.voices[voice] = torch.mean(torch.stack(packs), dim=0)
        return self.voices[voice]

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
                    text_chunk = ''.join(text).strip()
                    ps_chunk = ps.strip()
                    logger.debug(f"Chunking text at {z}: '{text_chunk[:30]}{'...' if len(text_chunk) > 30 else ''}'")
                    yield text_chunk, ps_chunk
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
        logger.debug(f"Loading voice: {voice}")
        pack = self.load_voice(voice)
        model = model or self.model
        pack = pack.to(model.device) if model else pack
        logger.debug(f"Voice loaded on device: {pack.device if hasattr(pack, 'device') else 'N/A'}")
        if isinstance(text, str):
            text = re.split(split_pattern, text.strip()) if split_pattern else [text]
        for graphemes in text:
            # TODO(misaki): Unify G2P interface between English and non-English
            if self.lang_code in 'ab':
                logger.debug(f"Processing English text: {graphemes[:50]}{'...' if len(graphemes) > 50 else ''}")
                _, tokens = self.g2p(graphemes)
                for gs, ps in self.en_tokenize(tokens):
                    if not ps:
                        continue
                    elif len(ps) > 510:
                        logger.warning(f"Unexpected len(ps) == {len(ps)} > 510 and ps == '{ps}'")
                        ps = ps[:510]
                    yield gs, ps, KPipeline.infer(model, ps, pack, speed)
            else:
                ps = self.g2p(graphemes)
                if not ps:
                    continue
                elif len(ps) > 510:
                    logger.warning(f'Truncating len(ps) == {len(ps)} > 510')
                    ps = ps[:510]
                yield graphemes, ps, KPipeline.infer(model, ps, pack, speed)
