import torch
import torch.nn.functional as F
from kokoro.custom_stft import CustomSTFT
from kokoro.istftnet import SineGen
from kokoro.modules import AdaLayerNorm


def audio_rate_sines(gen, f0):
    # The previous algorithm: repeat F0 to audio rate, then interpolate back down to frame rate
    f0 = F.interpolate(f0.transpose(1, 2), scale_factor=gen.upsample_scale).transpose(1, 2)
    rad_values = (f0 * torch.arange(1, gen.dim + 1) / gen.sampling_rate) % 1
    rad_values = F.interpolate(rad_values.transpose(1, 2), scale_factor=1 / gen.upsample_scale, mode="linear").transpose(1, 2)
    phase = torch.cumsum(rad_values, dim=1) * 2 * torch.pi
    phase = F.interpolate(phase.transpose(1, 2) * gen.upsample_scale, scale_factor=gen.upsample_scale, mode="linear").transpose(1, 2)
    return torch.sin(phase)


def test_sine_gen_matches_audio_rate_computation():
    gen = SineGen(24000, upsample_scale=300, harmonic_num=8, voiced_threshold=10)
    torch.manual_seed(0)
    for frames in (1, 2, 7, 150):
        f0 = torch.rand(1, frames, 1) * 400 * (torch.rand(1, frames, 1) > 0.3)
        fn = f0 * torch.arange(1, gen.dim + 1)
        assert torch.equal(gen._f02sine(fn), audio_rate_sines(gen, f0))


def test_sine_gen_output_is_audio_rate():
    gen = SineGen(24000, upsample_scale=300, harmonic_num=8, voiced_threshold=10)
    assert gen(torch.full((1, 5, 1), 200.0)).shape == (1, 5 * 300, 9)


def test_ada_layer_norm_scales_each_channel_by_style():
    norm = AdaLayerNorm(style_dim=4, channels=6)
    x, s = torch.randn(2, 10, 6), torch.randn(2, 4)
    gamma, beta = norm.fc(s).chunk(2, dim=-1)
    expected = (1 + gamma[:, None]) * F.layer_norm(x, (6,)) + beta[:, None]
    assert torch.allclose(norm(x, s), expected)


def test_custom_stft_phase_is_pi_on_negative_real_axis():
    # A negative constant puts the DC bin on the negative real axis, where the imaginary part is -0
    # and atan2 alone would give -pi
    stft = CustomSTFT(filter_length=20, hop_length=5, win_length=20)
    _, phase = stft.transform(torch.full((1, 200), -1.0))
    assert torch.all(phase[:, 0, 2:-2] == torch.pi)
