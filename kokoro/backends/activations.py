"""Pointwise Snake activation using fp32 device sine and eager operation order."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def affine_snake_kernel(X,H,A,Y,N:tl.constexpr,L:tl.constexpr,C:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    channel=i//L
    normalized=tl.load(X+i,i<N,other=0)
    gamma=tl.load(H+channel,i<N,other=0)
    beta=tl.load(H+C+channel,i<N,other=0)
    a=tl.load(A+channel,i<N,other=1)
    x=(1.0+gamma)*normalized+beta
    s=libdevice.sin(a*x)
    y=x+(1.0/a)*(s*s)
    tl.store(Y+i,y,i<N)

def affine_snake(normalized,h,a):
    assert normalized.shape[0]==1 and normalized.is_contiguous()
    y=torch.empty_like(normalized)
    affine_snake_kernel[(triton.cdiv(normalized.numel(),256),)](
        normalized,h,a,y,normalized.numel(),normalized.shape[-1],normalized.shape[1],256,
        enable_fp_fusion=False)
    return y
