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
)
REPO_ID = 'hexgrad/Kokoro-82M'

class KPipeline:
    def __init__(self, lang_code='a', config_path=None, model_path=None, trf=False):
        assert lang_code in LANG_CODES, (lang_code, LANG_CODES)
        self.lang_code = lang_code
        if config_path is None:
            config_path = hf_hub_download(repo_id=REPO_ID, filename='config.json')
        assert os.path.exists(config_path)
        with open(config_path, 'r') as r:
            config = json.load(r)
        if model_path is None:
            model_path = hf_hub_download(repo_id=REPO_ID, filename='model.pth')
        assert os.path.exists(model_path)
        try:
            fallback = espeak.EspeakFallback(british=lang_code=='b')
        except Exception as e:
            print('WARNING: EspeakFallback not enabled. Out-of-dictionary words will be skipped.', e)
            fallback = None
        self.g2p = en.G2P(trf=trf, british=lang_code=='b', fallback=fallback)
        self.vocab = config['vocab']
        self.model = KModel(config, model_path)
        self.voices = {}

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
        self.voices[voice] = torch.load(voice_path, weights_only=True)

    def tokenize(self, tokens):
        text = ''
        ps = ''
        for w in tokens:
            for t in (w if isinstance(w, list) else [w]):
                text += t.text + t.whitespace
                if t.phonemes is None:
                    continue
                if t.prespace and ps and not ps[-1].isspace() and t.phonemes:
                    ps += ' '
                ps += t.phonemes + t.whitespace
        text = text.strip()
        ps = ps.strip()
        ps = ps.replace('ɾ', 'T') # Just for American English
        input_ids = list(filter(lambda x: x is not None, map(lambda c: self.vocab.get(c), ps)))
        # assert len(ps) == len(input_ids), (ps, input_ids)
        return text, input_ids

    @classmethod
    def find_best_split(cls, tokens, punctuation=';:,.!?—…"()“”'):
        middle = len(tokens) // 2
        for targets in ['!.?…', ':;', ',—']:
            best = min(
                (i for i, t in enumerate(tokens[:-1]) if not isinstance(t, list) and t.phonemes in set(targets) and (
                    isinstance(tokens[i+1], list) or tokens[i+1].phonemes not in punctuation
                )),
                key=lambda i: abs(i - middle), default=None
            )
            if best is not None:
                break
        return middle if best is None else (best+1)

    def recursive_split(self, tokens):
        if not tokens:
            return []
        text, input_ids = self.tokenize(tokens)
        if len(input_ids) < 511:
            return [(text, input_ids)] if input_ids else []
        best = type(self).find_best_split(tokens)
        if best in (0, len(tokens)):
            print('TODO: Giving up, not splitting this', len(tokens))
            return []
        return [*self.recursive_split(tokens[:best]), *self.recursive_split(tokens[best:])]

    def __call__(self, text, voice='af', speed=1, split_pattern=r'\n+'):
        assert isinstance(text, str) or isinstance(text, list), type(text)
        self.load_voice(voice)
        if isinstance(text, str) and split:
            text = re.split(split_pattern, text.strip())
        for t in text:
            _, tokens = self.g2p(t)
            for segment_text, input_ids in self.recursive_split(tokens):
                if not input_ids:
                    continue
                assert len(input_ids) < 511, input_ids
                yield segment_text, self.model(input_ids, self.voices[voice][len(input_ids)-1], speed)
