import pytest
import torch
import torch.nn.functional as F

from kokoro.backends.rocm import ROCmConv1d, enable_rocm


def test_cpu_fallback_keeps_gradients_and_hooks():
    layer = ROCmConv1d(4, 4, 3, padding=1).eval()
    x = torch.randn(2, 4, 31, requires_grad=True)
    calls = []
    layer.register_forward_hook(lambda *args: calls.append(True))
    actual = layer(x)
    expected = F.conv1d(x, layer.weight, layer.bias, padding=1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    assert x.grad is not None and layer.weight.grad is not None
    assert calls == [True]


def test_cpu_enable_rejected_before_mutation():
    class Model(torch.nn.Sequential):
        device = torch.device('cpu')

    model = Model(torch.nn.Conv1d(4, 4, 3)).eval()
    with pytest.raises(ValueError, match='ROCm GPU'):
        enable_rocm(model)
    assert type(model[0]) is torch.nn.Conv1d


@pytest.mark.skipif(not torch.version.hip or not torch.cuda.is_available(), reason='ROCm required')
def test_gpu_affine_snake_preserves_eager_arithmetic():
    from kokoro.backends.activations import affine_snake
    x = torch.randn(1, 128, 8193, device='cuda')
    h = torch.randn(1, 256, device='cuda')
    a = torch.rand(1, 128, 1, device='cuda') + 0.5
    gamma, beta = h[:, :, None].chunk(2, dim=1)
    affine = (1 + gamma) * x + beta
    expected = affine + (1 / a) * torch.sin(a * affine).square()
    torch.testing.assert_close(affine_snake(x, h, a), expected, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not torch.version.hip or not torch.cuda.is_available(), reason='ROCm required')
@pytest.mark.parametrize('length', [97, 8193])
def test_gpu_conv_dispatch(length):
    layer = ROCmConv1d(128, 128, 3, padding=3, dilation=3).cuda().eval()
    x = torch.randn(1, 128, length, device='cuda')
    with torch.no_grad():
        expected = F.conv1d(x, layer.weight, layer.bias, padding=3, dilation=3)
        torch.testing.assert_close(layer(x), expected, rtol=1e-4, atol=1e-5)
