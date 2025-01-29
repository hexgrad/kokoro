"""Example of using KPipeline with automatic model and voice loading."""
from kokoro import KPipeline

def main():
    # Initialize pipeline - it will automatically:
    # - Download config and model from HF hub
    # - Initialize G2P (and download spacy model if needed)
    # - Handle voice loading on demand
    print("Creating KPipeline...")
    pipeline = KPipeline(lang_code='a')  # 'a' for American English
    print("Pipeline created successfully")
    
    for graphemes, phonemes, audio in pipeline(text, voice='af_alloy'):
        print(f"\nChunk: {graphemes}")
        print(f"Phonemes: {phonemes}")
        print(f"Audio shape: {audio.shape}")

    # Initialize pipeline - it will automatically:
    # - Download config and model from HF hub
    # - Initialize G2P (and download spacy model if needed)
    # - Handle voice loading on demand
    pipeline = KPipeline(lang_code='a')  # 'a' for American English

    # Generate speech - pipeline will automatically:
    # - Load the voice when needed
    # - Handle text preprocessing
    # - Run inference
    text = "Hello! This is a test of the automatic pipeline."
    print(f"\nGenerating speech for: {text}")
    
    for graphemes, phonemes, audio in pipeline(text, voice='af_alloy'):
        print(f"\nChunk: {graphemes}")
        print(f"Phonemes: {phonemes}")
        print(f"Audio shape: {audio.shape}")

if __name__ == '__main__':
    main()
