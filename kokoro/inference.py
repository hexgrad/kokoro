"""Explicit, one-way preparation of a loaded model for inference."""
from torch.nn.utils import remove_weight_norm
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import _WeightNorm


def prepare_for_inference(model):
    """Freeze legacy or parametrized weight normalization after loading weights.

    This changes the state-dict schema from weight_g/weight_v to weight. Load
    checkpoints before calling this function. Training and checkpoint loading
    on a prepared instance are unsupported; construct a fresh model instead.
    Calling this function repeatedly is safe.
    """
    if model.training:
        raise ValueError("Call model.eval() before preparing inference weights")
    for module in model.modules():
        if hasattr(module, 'weight_g') and hasattr(module, 'weight_v'):
            remove_weight_norm(module)
        elif parametrize.is_parametrized(module, 'weight'):
            transforms = module.parametrizations.weight
            # Other transformations must retain their normal behavior.
            if len(transforms) == 1 and isinstance(transforms[0], _WeightNorm):
                parametrize.remove_parametrizations(module, 'weight', leave_parametrized=True)
    return model
