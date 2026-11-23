import spaces
from kokoro import KModel, KPipeline
import gradio as gr
import os
import random
import torch

CUDA_AVAILABLE = torch.cuda.is_available()
models = {gpu: KModel().to('cuda' if gpu else 'cpu').eval() for gpu in [False] + ([True] if CUDA_AVAILABLE else [])}
pipelines = {lang_code: KPipeline(lang_code=lang_code, model=False) for lang_code in 'ab'}
pipelines['a'].g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
pipelines['b'].g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'

@spaces.GPU(duration=30)
def forward_gpu(ps, ref_s, speed):
    return models[True](ps, ref_s, speed)

def generate_first(text, voice='af_heart', speed=1, use_gpu=CUDA_AVAILABLE):
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE
    for _, ps, _ in pipeline(text, voice, speed):
        ref_s = pack[len(ps)-1]
        try:
            if use_gpu:
                audio = forward_gpu(ps, ref_s, speed)
            else:
                audio = models[False](ps, ref_s, speed)
        except gr.exceptions.Error as e:
            if use_gpu:
                gr.Warning(str(e))
                gr.Info('Retrying with CPU. To avoid this error, change Hardware to CPU.')
                audio = models[False](ps, ref_s, speed)
            else:
                raise gr.Error(e)
        return (24000, audio.numpy()), ps
    return None, ''

# Arena API
def predict(text, voice='af_heart', speed=1):
    return generate_first(text, voice, speed, use_gpu=False)[0]

def tokenize_first(text, voice='af_heart'):
    pipeline = pipelines[voice[0]]
    for _, ps, _ in pipeline(text, voice):
        return ps
    return ''

def generate_all(text, voice='af_heart', speed=1, use_gpu=CUDA_AVAILABLE):
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE
    first = True
    for _, ps, _ in pipeline(text, voice, speed):
        ref_s = pack[len(ps)-1]
        try:
            if use_gpu:
                audio = forward_gpu(ps, ref_s, speed)
            else:
                audio = models[False](ps, ref_s, speed)
        except gr.exceptions.Error as e:
            if use_gpu:
                gr.Warning(str(e))
                gr.Info('Switching to CPU')
                audio = models[False](ps, ref_s, speed)
            else:
                raise gr.Error(e)
        yield 24000, audio.numpy()
        if first:
            first = False
            yield 24000, torch.zeros(1).numpy()

with open('en.txt', 'r') as r:
    random_quotes = [line.strip() for line in r]

def get_random_quote():
    return random.choice(random_quotes)

def get_gatsby():
    with open('gatsby5k.md', 'r') as r:
        return r.read().strip()

def get_frankenstein():
    with open('frankenstein5k.md', 'r') as r:
        return r.read().strip()

CHOICES = {
'🇺🇸 🚺 Heart ❤️': 'af_heart',
'🇺🇸 🚺 Bella 🔥': 'af_bella',
'🇺🇸 🚺 Nicole 🎧': 'af_nicole',
'🇺🇸 🚺 Aoede': 'af_aoede',
'🇺🇸 🚺 Kore': 'af_kore',
'🇺🇸 🚺 Sarah': 'af_sarah',
'🇺🇸 🚺 Nova': 'af_nova',
'🇺🇸 🚺 Sky': 'af_sky',
'🇺🇸 🚺 Alloy': 'af_alloy',
'🇺🇸 🚺 Jessica': 'af_jessica',
'🇺🇸 🚺 River': 'af_river',
'🇺🇸 🚹 Michael': 'am_michael',
'🇺🇸 🚹 Fenrir': 'am_fenrir',
'🇺🇸 🚹 Puck': 'am_puck',
'🇺🇸 🚹 Echo': 'am_echo',
'🇺🇸 🚹 Eric': 'am_eric',
'🇺🇸 🚹 Liam': 'am_liam',
'🇺🇸 🚹 Onyx': 'am_onyx',
'🇺🇸 🚹 Santa': 'am_santa',
'🇺🇸 🚹 Adam': 'am_adam',
'🇬🇧 🚺 Emma': 'bf_emma',
'🇬🇧 🚺 Isabella': 'bf_isabella',
'🇬🇧 🚺 Alice': 'bf_alice',
'🇬🇧 🚺 Lily': 'bf_lily',
'🇬🇧 🚹 George': 'bm_george',
'🇬🇧 🚹 Fable': 'bm_fable',
'🇬🇧 🚹 Lewis': 'bm_lewis',
'🇬🇧 🚹 Daniel': 'bm_daniel',
}
for v in CHOICES.values():
    pipelines[v[0]].load_voice(v)

TOKEN_NOTE = '''
💡 Customize pronunciation with Markdown link syntax and /slashes/ like `[Kokoro](/kˈOkəɹO/)`

💬 To adjust intonation, try punctuation `;:,.!?—…"()“”` or stress `ˈ` and `ˌ`

⬇️ Lower stress `[1 level](-1)` or `[2 levels](-2)`

⬆️ Raise stress 1 level `[or](+2)` 2 levels (only works on less stressed, usually short words)
'''

with gr.Blocks() as generate_tab:
    out_audio = gr.Audio(label='Output Audio', interactive=False, streaming=False, autoplay=True)
    generate_btn = gr.Button('Generate', variant='primary')
    with gr.Accordion('Output Tokens', open=True):
        out_ps = gr.Textbox(interactive=False, show_label=False, info='Tokens used to generate the audio, up to 510 context length.')
        tokenize_btn = gr.Button('Tokenize', variant='secondary')
        gr.Markdown(TOKEN_NOTE)
        predict_btn = gr.Button('Predict', variant='secondary', visible=False)

STREAM_NOTE = ['⚠️ There is an unknown Gradio bug that might yield no audio the first time you click `Stream`.']
STREAM_NOTE = '\n\n'.join(STREAM_NOTE)

with gr.Blocks() as stream_tab:
    out_stream = gr.Audio(label='Output Audio Stream', interactive=False, streaming=True, autoplay=True)
    stream_btn = gr.Button('Stream', variant='primary')
    gr.Markdown(STREAM_NOTE)

with gr.Blocks() as demo:
    gr.Markdown('# [Kokoro 82M TTS](https://github.com/hexgrad/kokoro) — Support German Language via [Opire](https://opire.dev)')
    gr.Markdown('### 🇩🇪 Want German language support? Click [here to add funding/donation on Opire](https://opire.dev) to sponsor the development!')
    with gr.Tabs():
        with gr.TabItem('Generate'):
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(label='Input Text', value=get_random_quote, lines=5)
                    voice_input = gr.Dropdown(label='Voice', choices=list(CHOICES.keys()), value=list(CHOICES.keys())[0])
                    speed_input = gr.Slider(label='Speed', minimum=0.5, maximum=2.0, value=1.0, step=0.1)
                    with gr.Accordion('Advanced Settings', open=False):
                        hardware_input = gr.Radio(label='Hardware', choices=['GPU', 'CPU'] if CUDA_AVAILABLE else ['CPU'], value='GPU' if CUDA_AVAILABLE else 'CPU')
                    with gr.Row():
                        random_quote_btn = gr.Button('🎲 Random Quote')
                        gatsby_btn = gr.Button('📖 Gatsby 5k')
                        frankenstein_btn = gr.Button('🧪 Frankenstein 5k')
                with gr.Column():
                    generate_tab.render()
        with gr.TabItem('Stream'):
            with gr.Row():
                with gr.Column():
                    text_stream = gr.Textbox(label='Input Text', value=get_random_quote, lines=5)
                    voice_stream = gr.Dropdown(label='Voice', choices=list(CHOICES.keys()), value=list(CHOICES.keys())[0])
                    speed_stream = gr.Slider(label='Speed', minimum=0.5, maximum=2.0, value=1.0, step=0.1)
                with gr.Column():
                    stream_tab.render()

    random_quote_btn.click(get_random_quote, outputs=text_input)
    gatsby_btn.click(get_gatsby, outputs=text_input)
    frankenstein_btn.click(get_frankenstein, outputs=text_input)

    generate_btn.click(
        fn=lambda text, voice, speed, hw: generate_first(text, CHOICES[voice], speed, use_gpu=(hw == 'GPU')),
        inputs=[text_input, voice_input, speed_input, hardware_input],
        outputs=[out_audio, out_ps]
    )

    tokenize_btn.click(
        fn=lambda text, voice: tokenize_first(text, CHOICES[voice]),
        inputs=[text_input, voice_input],
        outputs=out_ps
    )

    predict_btn.click(
        fn=lambda text, voice, speed: predict(text, CHOICES[voice], speed),
        inputs=[text_input, voice_input, speed_input],
        outputs=out_audio
    )

    stream_btn.click(
        fn=lambda text, voice, speed: generate_all(text, CHOICES[voice], speed),
        inputs=[text_stream, voice_stream, speed_stream],
        outputs=out_stream
    )

if __name__ == '__main__':
    demo.queue().launch(show_api=False)