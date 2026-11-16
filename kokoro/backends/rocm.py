"""Experimental, explicitly enabled ROCm fp32 inference backend."""
import torch
from ..istftnet import AdaINResBlock1


def eligible_tensor(x):
    return (x.device.type == 'cuda' and x.dtype == torch.float32
            and x.ndim == 3 and x.shape[0] == 1 and x.is_contiguous())


class ROCmConv1d(torch.nn.Conv1d):
    def forward(self, x):
        if (self.training or torch.is_grad_enabled() or not eligible_tensor(x)
                or x.shape[-1] < 8192 or not self.weight.is_contiguous()):
            return super().forward(x)
        from .conv1d import conv
        if self.in_channels == 256 and self.kernel_size == (11,):
            padded = torch.nn.functional.pad(x, (self.padding[0], self.padding[0]))[0]
            result = torch.zeros((256, x.shape[-1]), device=x.device, dtype=x.dtype)
            for tap in range(11):
                lo = tap * self.dilation[0]
                result.addmm_(self.weight[:, :, tap], padded[:, lo:lo+x.shape[-1]])
            return (result + self.bias[:, None])[None]
        return conv(x, self.weight, self.bias, self.dilation[0], (64, 128, 16, 4))


class ROCmAdaINResBlock1(AdaINResBlock1):
    def forward(self, x, s):
        if (self.training or torch.is_grad_enabled() or not eligible_tensor(x)
                or not s.is_contiguous() or s.dtype != x.dtype or s.device != x.device):
            return super().forward(x, s)
        from .activations import affine_snake
        for c1, c2, n1, n2, a1, a2 in zip(
                self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2):
            xt = affine_snake(n1.norm(x), n1.fc(s), a1)
            xt = c1(xt)
            xt = affine_snake(n2.norm(xt), n2.fc(s), a2)
            x = c2(xt) + x
        return x


def enable_rocm(model):
    if model.training:
        raise ValueError('Call model.eval() before enabling the inference backend')
    if not torch.version.hip or model.device.type != 'cuda':
        raise ValueError('The ROCm backend requires a ROCm GPU model')
    arch = torch.cuda.get_device_properties(model.device).gcnArchName.split(':')[0]
    if arch != 'gfx1201':
        raise ValueError(f'ROCm inference backend has only been validated on gfx1201, got {arch}')
    if any(p.dtype != torch.float32 for p in model.parameters() if p.is_floating_point()):
        raise ValueError('The ROCm inference backend requires fp32 model parameters')
    # Resolve optional dependencies before making any changes.
    from . import activations, conv1d
    model.prepare_for_inference()
    for module in model.modules():
        if type(module) is AdaINResBlock1:
            module.__class__ = ROCmAdaINResBlock1
        if (type(module) is torch.nn.Conv1d and module.groups == 1
                and module.stride == (1,) and module.padding_mode == 'zeros'
                and module.in_channels == module.out_channels
                and module.in_channels in (128, 256) and module.kernel_size in ((3,), (7,), (11,))
                and module.dilation in ((1,), (3,), (5,)) and module.bias is not None
                and module.padding == (((module.kernel_size[0]-1)//2)*module.dilation[0],)):
            module.__class__ = ROCmConv1d
    return model
