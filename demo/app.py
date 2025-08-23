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
    with gr.Row():
        stream_btn = gr.Button('Stream', variant='primary')
        stop_btn = gr.Button('Stop', variant='stop')
    with gr.Accordion('Note', open=True):
        gr.Markdown(STREAM_NOTE)
        gr.DuplicateButton()

API_OPEN = True
with gr.Blocks() as app:
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label='Input Text', info=f"Arbitrarily many characters supported")
            with gr.Row():
                voice = gr.Dropdown(list(CHOICES.items()), value='af_heart', label='Voice', info='Quality and availability vary by language')
                use_gpu = gr.Dropdown(
                    [('CPU 🐌', False), ('ZeroGPU 🚀', True)],
                    value=CUDA_AVAILABLE,
                    label='Hardware',
                    info='GPU is usually faster, but has a usage quota',
                    interactive=CUDA_AVAILABLE
                )
            speed = gr.Slider(minimum=0.5, maximum=2, value=1, step=0.1, label='Speed')
            random_btn = gr.Button('🎲 Random Quote 💬', variant='secondary')
            with gr.Row():
                gatsby_btn = gr.Button('🥂 Gatsby 📕', variant='secondary')
                frankenstein_btn = gr.Button('💀 Frankenstein 📗', variant='secondary')
        with gr.Column():
            gr.TabbedInterface([generate_tab, stream_tab], ['Generate', 'Stream'])
    random_btn.click(fn=get_random_quote, inputs=[], outputs=[text])
    gatsby_btn.click(fn=get_gatsby, inputs=[], outputs=[text])
    frankenstein_btn.click(fn=get_frankenstein, inputs=[], outputs=[text])
    generate_btn.click(fn=generate_first, inputs=[text, voice, speed, use_gpu], outputs=[out_audio, out_ps])
    tokenize_btn.click(fn=tokenize_first, inputs=[text, voice], outputs=[out_ps])
    stream_event = stream_btn.click(fn=generate_all, inputs=[text, voice, speed, use_gpu], outputs=[out_stream])
    stop_btn.click(fn=None, cancels=stream_event)
    predict_btn.click(fn=predict, inputs=[text, voice, speed], outputs=[out_audio])

if __name__ == '__main__':
    app.queue(api_open=API_OPEN).launch(server_name="0.0.0.0", server_port=40001, show_api=API_OPEN)
