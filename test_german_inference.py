import os
import torch
import soundfile as sf
from kokoro import KPipeline

def test_german_inference():
    pipeline = KPipeline(lang_code='d')
    text = "Kokoro German ist ein Text to Speech Modell mit zweiundachtzig Millionen Parametern."

    # Create a temporary mock voice tensor for CI/unit testing (isolated offline execution)
    mock_voice_path = 'voices/mock_german_voice.pt'
    os.makedirs('voices', exist_ok=True)
    
    mock_tensor = torch.randn(510, 1, 256)
    torch.save(mock_tensor, mock_voice_path)

    try:
        # Force the framework to evaluate our newly placed mock voice file off local disk
        generator = pipeline(text, voice=mock_voice_path, speed=1.0)
        for i, (gs, ps, audio) in enumerate(generator):
            assert audio is not None and len(audio) > 0, "Audio generation failed"
            # sf.write(f"german_test_output_{i}.wav", audio, 24000) # skip disk write in CI
        print("CI Test Passed: Generated audio successfully using isolated mock German tensors.")
    finally:
        # Clean up the temporary fixture upon test completion
        if os.path.exists(mock_voice_path):
            os.remove(mock_voice_path)

if __name__ == "__main__":
    test_german_inference()
