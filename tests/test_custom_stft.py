import torch
import numpy as np
import pytest
from kokoro.custom_stft import CustomSTFT
from kokoro.istftnet import TorchSTFT
import torch.nn.functional as F


@pytest.fixture
def sample_audio():
    # Generate a sample audio signal (sine wave)
    sample_rate = 16000
    duration = 1.0  # seconds
    t = torch.linspace(0, duration, int(sample_rate * duration))
    frequency = 440.0  # Hz
    signal = torch.sin(2 * np.pi * frequency * t)
    return signal.unsqueeze(0)  # Add batch dimension


def test_stft_reconstruction(sample_audio):
    # Initialize both STFT implementations
    custom_stft = CustomSTFT(filter_length=800, hop_length=200, win_length=800)
    torch_stft = TorchSTFT(filter_length=800, hop_length=200, win_length=800)

    # Process through both implementations
    custom_output = custom_stft(sample_audio)
    torch_output = torch_stft(sample_audio)

    # Compare outputs
    assert torch.allclose(custom_output, torch_output, rtol=1e-3, atol=1e-3)


def test_magnitude_phase_consistency(sample_audio):
    custom_stft = CustomSTFT(filter_length=800, hop_length=200, win_length=800)
    torch_stft = TorchSTFT(filter_length=800, hop_length=200, win_length=800)

    # Get magnitude and phase from both implementations
    custom_mag, custom_phase = custom_stft.transform(sample_audio)
    torch_mag, torch_phase = torch_stft.transform(sample_audio)

    # Compare magnitudes ignoring the boundary frames
    custom_mag_center = custom_mag[..., 2:-2]
    torch_mag_center = torch_mag[..., 2:-2]
    assert torch.allclose(custom_mag_center, torch_mag_center, rtol=1e-2, atol=1e-2)


def test_batch_processing():
    # Create a batch of signals
    batch_size = 4
    sample_rate = 16000
    duration = 0.1  # shorter duration for faster testing
    t = torch.linspace(0, duration, int(sample_rate * duration))
    frequency = 440.0
    signals = torch.sin(2 * np.pi * frequency * t).unsqueeze(0).repeat(batch_size, 1)

    custom_stft = CustomSTFT(filter_length=800, hop_length=200, win_length=800)

    # Process batch
    output = custom_stft(signals)

    # Check output shape
    assert output.shape[0] == batch_size
    assert len(output.shape) == 3  # (batch, 1, time)


def test_different_window_sizes():
    signal = torch.randn(1, 16000)  # 1 second of random noise

    # Test with different window sizes
    for filter_length in [512, 1024, 2048]:
        custom_stft = CustomSTFT(
            filter_length=filter_length,
            hop_length=filter_length // 4,
            win_length=filter_length,
        )

        # Forward and backward transform
        output = custom_stft(signal)

        # Check that output length is reasonable
        assert output.shape[-1] >= signal.shape[-1]
