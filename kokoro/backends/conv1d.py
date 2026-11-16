"""Strict-fp32 implicit Conv1d for the optional ROCm inference backend."""
import triton
import triton.language as tl
import torch

@triton.jit
def conv_kernel(X,W,B,Y,L:tl.constexpr,CI:tl.constexpr,CO:tl.constexpr,
                KERNEL:tl.constexpr,DIL:tl.constexpr, BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    m=tl.program_id(0)*BM+tl.arange(0,BM)
    n=tl.program_id(1)*BN+tl.arange(0,BN)
    k=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for block in range(triton.cdiv(CI*KERNEL,BK)):
        kk=block*BK+k
        ci=kk//KERNEL
        tap=kk%KERNEL
        t=n[None,:]+tap[:,None]*DIL-DIL*(KERNEL-1)//2
        a=tl.load(W+m[:,None]*(CI*KERNEL)+(ci*KERNEL+tap)[None,:],
                  (m[:,None]<CO)&(kk[None,:]<CI*KERNEL),other=0)
        b=tl.load(X+ci[:,None]*L+t,(kk[:,None]<CI*KERNEL)&(ci[:,None]<CI)&(t>=0)&(t<L),other=0)
        acc+=tl.dot(a,b,input_precision="ieee")
    bias=tl.load(B+m,m<CO,other=0)
    tl.store(Y+m[:,None]*L+n[None,:],acc+bias[:,None],(m[:,None]<CO)&(n[None,:]<L))

def conv(x,w,b,dilation=1,config=(64,128,16,4)):
    assert x.shape[0]==1 and x.dtype==w.dtype==b.dtype==torch.float32
    ci,length=x.shape[1:];co,_,kernel=w.shape
    bm,bn,bk,warps=config
    y=torch.empty((1,co,length),dtype=x.dtype,device=x.device)
    conv_kernel[(triton.cdiv(co,bm),triton.cdiv(length,bn))](
        x,w,b,y,length,ci,co,kernel,dilation,bm,bn,bk,num_warps=warps)
    return y
