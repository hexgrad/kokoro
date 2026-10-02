import warnings

import pytest
import torch

from kokoro.model import KModel


class _Stub(KModel):
    """KModel without the heavy submodules: one weight-normed conv under `conv`."""

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.conv = torch.nn.utils.weight_norm(torch.nn.Conv1d(2, 2, 3))


def _prefixed(sd, rename=None):
    out = {}
    for k, v in sd.items():
        if rename:
            for old, new in rename.items():
                k = k.replace(old, new)
        out["module." + k] = v
    return out


def test_dataparallel_checkpoint_loads_without_warning():
    model = _Stub()
    sd = _prefixed(model.conv.state_dict())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model._load_component("conv", sd)


def test_missing_weight_norm_keys_warn():
    model = _Stub()
    sd = _prefixed(model.conv.state_dict(), {
        "weight_g": "parametrizations.weight.original0",
        "weight_v": "parametrizations.weight.original1",
    })
    with pytest.warns(RuntimeWarning, match="random initialization"):
        model._load_component("conv", sd)
