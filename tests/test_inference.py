import pytest
import torch
from torch.nn.utils import weight_norm
from kokoro.inference import prepare_for_inference


def test_parametrized_weightnorm_is_frozen_and_restores_conv_class():
    from torch.nn.utils.parametrizations import weight_norm as parametrized_weight_norm
    layer = parametrized_weight_norm(torch.nn.Conv1d(4, 8, 3)).eval()
    x = torch.randn(1, 4, 21)
    with torch.no_grad():
        expected = layer(x)
        assert prepare_for_inference(layer) is layer
        assert type(layer) is torch.nn.Conv1d
        assert not hasattr(layer, 'parametrizations')
        assert set(layer.state_dict()) == {'weight', 'bias'}
        torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)
        assert prepare_for_inference(layer) is layer


def test_preparation_preserves_output_and_removes_repeated_weightnorm():
    model = torch.nn.Sequential(weight_norm(torch.nn.Conv1d(4, 8, 3))).eval()
    x = torch.randn(1, 4, 21)
    with torch.no_grad():
        expected = model(x)
        assert prepare_for_inference(model) is model
        torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
        assert '0.weight_g' not in model.state_dict()
        assert '0.weight' in model.state_dict()
        assert prepare_for_inference(model) is model


def test_preparation_requires_eval():
    model = weight_norm(torch.nn.Linear(4, 8))
    with pytest.raises(ValueError, match='eval'):
        prepare_for_inference(model)
    assert hasattr(model, 'weight_g')
