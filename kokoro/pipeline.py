from .models import KModel
from huggingface_hub import hf_hub_download
from misaki import en, espeak
import json
import os
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
REPO_ID = 'hexgrad/Kokoro-82M'

class KPipeline:
    def __init__(self, lang_code='a', config_path=None, model_path=None, trf=False, device=None):
        assert lang_code in LANG_CODES, (lang_code, LANG_CODES)
        self.lang_code = lang_code
        if config_path is None:
            config_path = hf_hub_download(repo_id=REPO_ID, filename='config.json')
        assert os.path.exists(config_path)
        with open(config_path, 'r', encoding='utf8') as r:
            config = json.load(r)
        if model_path is None:
            model_path = hf_hub_download(repo_id=REPO_ID, filename='kokoro-v1_0.pth')
        assert os.path.exists(model_path)
        self.vocab = config['vocab']
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.model = KModel(config, model_path).to(self.device).eval()
        self.voices = {}
        if lang_code in 'ab':
            try:
                fallback = espeak.EspeakFallback(british=lang_code=='b')
            except Exception as e:
                print('WARNING: EspeakFallback not enabled. Out-of-dictionary words will be skipped.', e)
                fallback = None
            self.g2p = en.G2P(trf=trf, british=lang_code=='b', fallback=fallback)
        else:
            language = LANG_CODES[lang_code]
            print(f"WARNING: Using EspeakG2P(language='{language}'). Chunking logic not yet implemented, so long texts may be truncated unless you split them with '\\n'.")
            self.g2p = espeak.EspeakG2P(language=language)

    def load_voice(self, voice):
        if voice in self.voices:
            return
        v = voice.split('/')[-1]
        if not v.startswith(self.lang_code):
            v = LANG_CODES.get(v, voice)
            p = LANG_CODES.get(self.lang_code, self.lang_code)
            print(f'WARNING: Loading {v} voice into {p} pipeline. Phonemes may be mismatched.')
        voice_path = voice if voice.endswith('.pt') else hf_hub_download(repo_id=REPO_ID, filename=f'voices/{voice}.pt')
        assert os.path.exists(voice_path)
        self.voices[voice] = torch.load(voice_path, weights_only=True).to(self.device)

    @classmethod
    def waterfall_last(cls, pairs, next_count, waterfall=['!.?…', ':;', ',—'], bumps={')', '”'}):
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

    def en_tokenize(self, tokens):
        pairs = []
        count = 0
        for w in tokens:
            for t in (w if isinstance(w, list) else [w]):
                if t.phonemes is None:
                    continue
                next_ps = ' ' if t.prespace and pairs and not pairs[-1][1].endswith(' ') and t.phonemes else ''
                next_ps += ''.join(filter(lambda p: p in self.vocab, t.phonemes.replace('ɾ', 'T'))) # American English: ɾ => T
                next_ps += ' ' if t.whitespace else ''
                next_count = count + len(next_ps.rstrip())
                if next_count > 510:
                    z = type(self).waterfall_last(pairs, next_count)
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

    def p2ii(self, ps):
        input_ids = list(filter(lambda i: i is not None, map(lambda p: self.vocab.get(p), ps)))
        assert input_ids and len(input_ids) <= 510, input_ids
        return input_ids

    def __call__(self, text, voice, speed=1, split_pattern=r'\n+'):
        assert isinstance(text, str) or isinstance(text, list), type(text)
        self.load_voice(voice)
        if isinstance(text, str):
            text = re.split(split_pattern, text.strip()) if split_pattern else [text]
        for graphemes in text:
            if self.lang_code in 'ab':
                _, tokens = self.g2p(graphemes)
                for gs, ps in self.en_tokenize(tokens):
                    if not ps:
                        continue
                    elif len(ps) > 510:
                        print(f"TODO: Unexpected len(ps) == {len(ps)} > 510 and ps == '{ps}'")
                        continue
                    input_ids = self.p2ii(ps)
                    yield gs, ps, self.model(input_ids, self.voices[voice][len(input_ids)-1], speed)
            else:
                ps = self.g2p(graphemes)
                if not ps:
                    continue
                elif len(ps) > 510:
                    print(f'WARNING: Truncating len(ps) == {len(ps)} > 510')
                    ps = ps[:510]
                input_ids = self.p2ii(ps)
                yield graphemes, ps, self.model(input_ids, self.voices[voice][len(input_ids)-1], speed)
