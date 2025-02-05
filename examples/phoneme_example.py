from kokoro import KPipeline, KModel
import torch
from scipy.io import wavfile

def save_audio(audio: torch.Tensor, filename: str):
    """Helper function to save audio tensor as WAV file"""
    if audio is not None:
        # Ensure audio is on CPU and in the right format
        audio_cpu = audio.cpu().numpy()
        
        # Save using scipy.io.wavfile
        wavfile.write(
            filename,
            24000,  # Kokoro uses 24kHz sample rate
            audio_cpu
        )
        print(f"Audio saved as '{filename}'")
    else:
        print("No audio was generated")

def main():
    # Initialize pipeline with American English
    pipeline = KPipeline(lang_code='a')
    
    # The phoneme string for:
    # "How are you today? I am doing reasonably well, thank you for asking"
    phonemes = "hˌW ɑɹ ju tədˈA? ˌI ɐm dˈuɪŋ ɹˈizənəbli wˈɛl, θˈæŋk ju fɔɹ ˈæskɪŋ"
    
    try:
        print("\nExample 1: Using generate_from_tokens with raw phonemes")
        results = list(pipeline.generate_from_tokens(
            tokens=phonemes,
            voice="af_bella",
            speed=1.0
        ))
        if results:
            save_audio(results[0].audio, 'phoneme_output_new.wav')
        
        # Example 2: Using generate_from_tokens with pre-processed tokens
        print("\nExample 2: Using generate_from_tokens with pre-processed tokens")
        #  get the tokens through G2P or any other method
        text = "How are you today? I am doing reasonably well, thank you for asking"
        _, tokens = pipeline.g2p(text)
        
        # Then generate from tokens
        for result in pipeline.generate_from_tokens(
            tokens=tokens,
            voice="af_bella",
            speed=1.0
        ):
            # Each result may contain timestamps if available
            if result.tokens:
                for token in result.tokens:
                    if hasattr(token, 'start_ts') and hasattr(token, 'end_ts'):
                        print(f"Token: {token.text} ({token.start_ts:.2f}s - {token.end_ts:.2f}s)")
            save_audio(result.audio, f'token_output_{hash(result.phonemes)}.wav')
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()