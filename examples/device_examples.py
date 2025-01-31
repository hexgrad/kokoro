"""
Quick example to show how device selection can be controlled, and was checked
"""
# import warnings, time
import time
# import warnings
from kokoro import KPipeline
# warnings.filterwarnings('ignore')
# import torch; torch.set_warn_always(False)

def generate_audio(pipeline, text):
    for _, _, audio in pipeline(text, voice='af_bella'):
        samples = audio.shape[0] if audio is not None else 0
        assert samples > 0, "No audio generated"
        return samples

def time_synthesis(device=None):
    try:
        start = time.perf_counter()
        pipeline = KPipeline(lang_code='a', device=device)
        samples = generate_audio(pipeline, "The quick brown fox jumps over the lazy dog.")
        ms = (time.perf_counter() - start) * 1000
        print(f"✓ {device or 'auto':<6} | {ms:>5.1f}ms total | {samples:>6,d} samples")
    except RuntimeError as e:
        print(f"✗ {'cuda' if 'CUDA' in str(e) else device or 'auto':<6} | {'not available' if 'CUDA' in str(e) else str(e)}")

def compare_shared_model():
    try:
        start = time.perf_counter()
        en_us = KPipeline(lang_code='a')
        en_uk = KPipeline(lang_code='a', model=en_us.model)
        
        for pipeline in [en_us, en_uk]:
            generate_audio(pipeline, "Testing model reuse.")
                
        ms = (time.perf_counter() - start) * 1000
        print(f"✓ reuse  | {ms:>5.1f}ms for both models")
    except Exception as e:
        print(f"✗ reuse  | {str(e)}")

if __name__ == '__main__':
    print("\nDevice Selection & Performance:")
    print("----------------------------------------")
    time_synthesis()
    time_synthesis('cuda')
    time_synthesis('cpu') 
    print("----------------------------------------")
    compare_shared_model()