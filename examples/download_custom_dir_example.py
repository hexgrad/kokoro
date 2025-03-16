import os
import soundfile as sf
from loguru import logger
from kokoro import KPipeline
from typing import Generator

lang_code = 'a'
repo_id = 'hexgrad/Kokoro-82M'
cache_dir = os.path.expanduser('~/kokoro_models/cache')
local_dir = os.path.expanduser('~/kokoro_models/local')
voice = 'am_michael'

# This text is for demonstration purposes only, unseen during training
text = '''
The sky above the port was the color of television, tuned to a dead channel.
"It's not like I'm using," Case heard someone say, as he shouldered his way through the crowd around the door of the Chat. "It's like my body's developed this massive drug deficiency."
It was a Sprawl voice and a Sprawl joke. The Chatsubo was a bar for professional expatriates; you could drink there for a week and never hear two words in Japanese.

These were to have an enormous impact, not only because they were associated with Constantine, but also because, as in so many other areas, the decisions taken by Constantine (or in his name) were to have great significance for centuries to come. One of the main issues was the shape that Christian churches were to take, since there was not, apparently, a tradition of monumental church buildings when Constantine decided to help the Christian church build a series of truly spectacular structures. The main form that these churches took was that of the basilica, a multipurpose rectangular structure, based ultimately on the earlier Greek stoa, which could be found in most of the great cities of the empire. Christianity, unlike classical polytheism, needed a large interior space for the celebration of its religious services, and the basilica aptly filled that need. We naturally do not know the degree to which the emperor was involved in the design of new churches, but it is tempting to connect this with the secular basilica that Constantine completed in the Roman forum (the so-called Basilica of Maxentius) and the one he probably built in Trier, in connection with his residence in the city at a time when he was still caesar.

[Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, [Kokoro](/kˈOkəɹO/) can be deployed anywhere from production environments to personal projects.
'''


def save_audio(generator: Generator, save_dir: str) -> None:
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for i, (gs, ps, audio) in enumerate(generator):
        print(i)  # i => index
        print(gs) # gs => graphemes/text
        print(ps) # ps => phonemes

        # display(Audio(data=audio, rate=24000, autoplay=i==0))
        save_fp = os.path.join(save_dir, f"kokoro_{i}.wav")
        sf.write(save_fp, audio, 24000) # save each audio file


def test_download_into_custom_directory():
    global text, lang_code, repo_id, cache_dir, local_dir, voice

    pipeline = KPipeline(
        lang_code='a',
        repo_id=repo_id,
        cache_dir=cache_dir,
        local_dir=local_dir
    )

    # Generate and save audio files
    generator = pipeline(
        text, voice=voice, # <= change voice here
        speed=1.1, split_pattern=r'\n+'
    )

    save_audio(generator, 'download_custom_dir_example')


def test_load_from_local_file() -> None:
    global text, lang_code, repo_id, cache_dir, local_dir, voice

    # use "local_files_only=True" only if: 
    # You are sure that the model is present in the provided cache directory
    pipeline = KPipeline(
        lang_code='a',
        repo_id=repo_id,
        cache_dir=cache_dir,
        local_dir=local_dir,
        local_files_only=True
    )

    # Generate and save audio files
    generator = pipeline(
        text, voice=voice, # <= change voice here
        speed=1.1, split_pattern=r'\n+'
    )

    save_audio(generator, 'load_from_local_file_example')


if __name__ == '__main__':
    logger.info(f'Testing download into custom cache direcotry.\nCustom Cache Directory Set to: {cache_dir}')
    test_download_into_custom_directory()
    logger.info(f'Testing loading from custom cache direcotry.\nCustom Cache Directory Set to: {cache_dir}')
    test_load_from_local_file()
