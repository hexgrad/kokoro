"""
Quick example to show how device selection can be controlled, and was checked
"""
import time
from kokoro import KPipeline
from loguru import logger

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
        logger.info(f"✓ {device or 'auto':<6} | {ms:>5.1f}ms total | {samples:>6,d} samples")
    except RuntimeError as e:
        logger.error(f"✗ {'cuda' if 'CUDA' in str(e) else device or 'auto':<6} | {'not available' if 'CUDA' in str(e) else str(e)}")

def compare_shared_model():
    try:
        start = time.perf_counter()
        en_us = KPipeline(lang_code='a')
        en_uk = KPipeline(lang_code='a', model=en_us.model)
        
        for pipeline in [en_us, en_uk]:
            generate_audio(pipeline, "Testing model reuse.")
                
        ms = (time.perf_counter() - start) * 1000
        logger.info(f"✓ reuse  | {ms:>5.1f}ms for both models")
    except Exception as e:
        logger.error(f"✗ reuse  | {str(e)}")

if __name__ == '__main__':
    logger.info("Device Selection & Performance")
    logger.info("-" * 40)
    time_synthesis()
    time_synthesis('cuda')
    time_synthesis('cpu') 
    logger.info("-" * 40)
    compare_shared_model()