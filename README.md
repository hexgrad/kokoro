# kokoro

An inference library for [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M).

- You can [`pip install kokoro`](https://pypi.org/project/kokoro/)
- 82M weights hosted on HF, auto-downloaded with `huggingface_hub.hf_hub_download`
- Lightweight PyTorch modeling code split across [two](https://github.com/hexgrad/kokoro/blob/main/kokoro/istftnet.py) [files](https://github.com/hexgrad/kokoro/blob/main/kokoro/models.py)
- Uses [`misaki`](https://github.com/hexgrad/misaki) for G2P

### Usage
The following can be run in a single cell on [Google Colab](https://colab.research.google.com/).
```py
# 1️⃣ Install kokoro
!pip install -q kokoro soundfile
# 2️⃣ Install espeak, used for out-of-dictionary fallback
!apt-get -qq -y install espeak-ng > /dev/null 2>&1
# You can skip espeak installation, but OOD words will be skipped unless you provide a fallback

# 3️⃣ Initalize a pipeline
from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
# 🇺🇸 'a' => American English
# 🇬🇧 'b' => British English
pipeline = KPipeline(lang_code='a') # make sure lang_code matches voice

# The following text is for demonstration purposes only, unseen during training
text = '''
The sky above the port was the color of television, tuned to a dead channel.
"It's not like I'm using," Case heard someone say, as he shouldered his way through the crowd around the door of the Chat. "It's like my body's developed this massive drug deficiency."
It was a Sprawl voice and a Sprawl joke. The Chatsubo was a bar for professional expatriates; you could drink there for a week and never hear two words in Japanese.

These were to have an enormous impact, not only because they were associated with Constantine, but also because, as in so many other areas, the decisions taken by Constantine (or in his name) were to have great significance for centuries to come. One of the main issues was the shape that Christian churches were to take, since there was not, apparently, a tradition of monumental church buildings when Constantine decided to help the Christian church build a series of truly spectacular structures. The main form that these churches took was that of the basilica, a multipurpose rectangular structure, based ultimately on the earlier Greek stoa, which could be found in most of the great cities of the empire. Christianity, unlike classical polytheism, needed a large interior space for the celebration of its religious services, and the basilica aptly filled that need. We naturally do not know the degree to which the emperor was involved in the design of new churches, but it is tempting to connect this with the secular basilica that Constantine completed in the Roman forum (the so-called Basilica of Maxentius) and the one he probably built in Trier, in connection with his residence in the city at a time when he was still caesar.
'''

# 4️⃣ Generate, display, and save audio files in a loop.
generator = pipeline(
    text, voice='af_bella',
    speed=1, split_pattern=r'\n+'
)
for i, (gs, ps, audio) in enumerate(generator):
    print(i)  # i => index
    print(gs) # gs => graphemes/text
    print(ps) # ps => phonemes
    display(Audio(data=audio, rate=24000, autoplay=i==0))
    sf.write(f'{i}.wav', audio, 24000) # save each audio file
```
