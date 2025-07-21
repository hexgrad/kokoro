import argparse
import os
import torch
import onnx
import onnxruntime as ort
import sounddevice as sd

from kokoro import KModel, KPipeline
from kokoro.model import KModelForONNX

def export_onnx(model, output):
    onnx_file = output + "/" + "kokoro.onnx"

    input_ids = torch.randint(1, 100, (48,)).numpy()
    input_ids = torch.LongTensor([[0, *input_ids, 0]])
    style = torch.randn(1, 256)
    speed = torch.randint(1, 10, (1,)).int()

    torch.onnx.export(
        model, 
        args = (input_ids, style, speed), 
        f = onnx_file, 
        export_params = True, 
        verbose = True, 
        input_names = [ 'input_ids', 'style', 'speed' ], 
        output_names = [ 'waveform', 'duration' ],
        opset_version = 17, 
        dynamic_axes = {
            'input_ids': {0: "batch_size", 1: 'input_ids_len' }, 
            'style': {0: "batch_size"}, 
            "speed": {0: "batch_size"}
        }, 
        do_constant_folding = True, 
    )

    print('export kokoro.onnx ok!')

    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)
    print('onnx check ok!')

def load_input_ids(pipeline, text):
    if pipeline.lang_code in 'ab':
        _, tokens = pipeline.g2p(text)
        for gs, ps, tks in pipeline.en_tokenize(tokens):
            if not ps:
                continue
    else:
        ps, _ = pipeline.g2p(text)

    if len(ps) > 510:
        ps = ps[:510]

    input_ids = list(filter(lambda i: i is not None, map(lambda p: pipeline.model.vocab.get(p), ps)))
    print(f"text: {text} -> phonemes: {ps} -> input_ids: {input_ids}")
    input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(pipeline.model.device)
    return ps, input_ids

def load_voice(pipeline, voice, phonemes):
    pack = pipeline.load_voice(voice).to('cpu')
    return pack[len(phonemes) - 1]

def load_sample(model):
    pipeline = KPipeline(lang_code='a', model=model.kmodel, device='cpu')
    text = '''
    In today's fast-paced tech world, building software applications has never been easier — thanks to AI-powered coding assistants.'
    '''
    text = '''
    The sky above the port was the color of television, tuned to a dead channel.
    '''
    voice = 'checkpoints/voices/af_heart.pt'

    pipeline = KPipeline(lang_code='z', model=model.kmodel, device='cpu')
    text = '''
    2月15日晚，猫眼专业版数据显示，截至发稿，《哪吒之魔童闹海》（或称《哪吒2》）今日票房已达7.8亿元，累计票房（含预售）超过114亿元。
    '''
    voice = 'checkpoints/voices/zf_xiaoxiao.pt'

    phonemes, input_ids = load_input_ids(pipeline, text)
    style = load_voice(pipeline, voice, phonemes)
    speed = torch.IntTensor([1])

    return input_ids, style, speed

def inference_onnx(model, output):
    onnx_file = output + "/" + "kokoro.onnx"
    session = ort.InferenceSession(onnx_file)

    input_ids, style, speed = load_sample(model)

    outputs = session.run(None, {
        'input_ids': input_ids.numpy(), 
        'style': style.numpy(), 
        'speed': speed.numpy(), 
    })

    output = torch.from_numpy(outputs[0])
    print(f'output: {output.shape}')
    print(output)

    audio = output.numpy()
    sd.play(audio, 24000)
    sd.wait()

def check_model(model):
    input_ids, style, speed = load_sample(model)
    output, duration = model(input_ids, style, speed)

    print(f'output: {output.shape}')
    print(f'duration: {duration.shape}')
    print(output)

    audio = output.numpy()
    sd.play(audio, 24000)
    sd.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Export kokoro Model to ONNX", add_help=True)
    parser.add_argument("--inference", "-t", help="test kokoro.onnx model", action="store_true")
    parser.add_argument("--check", "-m", help="check kokoro model", action="store_true")
    parser.add_argument(
        "--config_file", "-c", type=str, default="checkpoints/config.json", help="path to config file"
    )
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, default="checkpoints/kokoro-v1_0.pth", help="path to checkpoint file"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="onnx", help="output directory"
    )

    args = parser.parse_args()

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    output_dir = args.output_dir
    
    # make dir
    os.makedirs(output_dir, exist_ok=True)

    kmodel = KModel(config=config_file, model=checkpoint_path, disable_complex=True)
    model = KModelForONNX(kmodel).eval()

    if args.inference:
        inference_onnx(model, output_dir)
    elif args.check:
        check_model(model)
    else:
        export_onnx(model, output_dir)
