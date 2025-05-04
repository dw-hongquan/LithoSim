'''
Author: Hongquan
Date: Apr 06, 2025
Description: ValueCrossAttn and PosCrossAttn
'''
import sys
sys.path.append('.')

from inspect import isfunction
import torch
from torch import nn, Tensor, einsum, complex64
from einops import rearrange, repeat
from typing import Optional
from src.models.components.complex_net import ComplexDropout, ComplexSoftmax

class ChunkedPosCrossAttn(nn.Module):
    def __init__(self,
                 query_dim: int,
                 context_dim: int = 65,
                 heads: int = 8,
                 dim_head: int = 4,
                 dropout: float = 0.1,
                 chunk_size: int = 8) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.query_dim = query_dim
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.chunk_size = chunk_size

        self.to_q = nn.Conv2d(
            in_channels=query_dim,
            out_channels=inner_dim,
            kernel_size=1,
            bias=False
        )
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, query_dim, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self,
                img_feats: Tensor,
                context: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
        """
        :param img_feats: [B, C, H, W] (Query)
        :param context: [B, N, context_dim]
        :return: [B, C, H, W]
        """
        _, _, H, W = img_feats.shape
        chunk_size = self.chunk_size

        assert H % chunk_size == 0 and W % chunk_size == 0, \
            f"Dimensions must be divisible by {chunk_size}, got H:{H} W:{W}"

        num_blocks_h = H // chunk_size
        num_blocks_w = W // chunk_size

        img_feats = rearrange(img_feats, 
                            'b c (nh bh) (nw bw) -> (b nh nw) c bh bw', 
                            bh=chunk_size, bw=chunk_size).contiguous()

        q = self.to_q(img_feats)
        q = rearrange(q, 'b (h d) bh bw -> (b h) (bh bw) d', 
                      h=self.heads).contiguous()

        k = self.to_k(context)
        k = rearrange(k, 'b n (h d) -> (b h) n d', h=self.heads).contiguous()
        k = repeat(k, 'bh n d -> (bh nh nw) n d', 
                   nh=num_blocks_h, nw=num_blocks_w).contiguous()

        v = self.to_v(context)
        v = rearrange(v, 'b n (h d) -> (b h) n d', h=self.heads).contiguous()
        v = repeat(v, 'bh n d -> (bh nh nw) n d', 
                   nh=num_blocks_h, nw=num_blocks_w).contiguous()

        sim = einsum('b i d, b j d -> b i j', q, k).contiguous() * self.scale

        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = max_neg_value(sim)
            mask = repeat(mask, 
                         'b j -> (b h nh nw) () j', 
                         h=self.heads, 
                         nh=num_blocks_h, 
                         nw=num_blocks_w)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v).contiguous()

        out = rearrange(out, 
                       '(b nh nw h) (bh bw) d -> b (h d) (nh bh) (nw bw)', 
                       h=self.heads,
                       nh=num_blocks_h,
                       nw=num_blocks_w,
                       bh=chunk_size,
                       bw=chunk_size).contiguous()

        return self.to_out(out)

class ChunkedValueCrossAttn(nn.Module):
    def __init__(self,
                 query_dim: int,
                 context_dim: int = 16,
                 heads: int = 8,
                 dim_head: int = 4,
                 dropout: float = 0.1,
                 chunk_size: int = 32) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.chunk_size = chunk_size

        self.to_q = nn.Conv2d(query_dim, inner_dim, kernel_size=1, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, query_dim, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        """
        :param x: [B, C, H, W] (Query)
        :param context: [B, context_dim]
        :return: [B, C, H, W]
        """
        _, _, H, W = x.shape
        cs = self.chunk_size

        assert H % cs == 0 and W % cs == 0, \
            f"Dimensions must be divisible by {cs}, got H:{H} W:{W}"

        x_chunks = rearrange(x, 
            'b c (nh h) (nw w) -> (b nh nw) c h w', 
            h=cs, w=cs).contiguous()  # [B*N, C, cs, cs]
        
        q = self.to_q(x_chunks)  # [B*N, inner_dim, cs, cs]
        q = rearrange(q, 'b c h w -> b (h w) c').contiguous()  # [B*N, cs^2, inner_dim]

        num_blocks = (H//cs) * (W//cs)
        context = context.repeat_interleave(num_blocks, dim=0).contiguous()  # [B*N, context_dim]
        context = context.unsqueeze(1)  # [B*N, 1, context_dim]
        
        k = self.to_k(context)  # [B*N, 1, inner_dim]
        v = self.to_v(context)  # [B*N, 1, inner_dim]
        k = rearrange(k, 'b s c -> b c s').contiguous()  # [B*N, inner_dim, 1]
        v = rearrange(v, 'b s c -> b c s').contiguous()  # [B*N, inner_dim, 1]

        attn = torch.softmax(torch.matmul(q, k) * self.scale, dim=-1)  # [B*N, cs^2, 1]
        out = torch.matmul(attn, v.transpose(1, 2).contiguous())  # [B*N, cs^2, inner_dim]
        out = rearrange(out, 'b (h w) c -> b c h w', h=cs, w=cs).contiguous()  # [B*N, inner_dim, cs, cs]
        out = rearrange(out, 
            '(b nh nw) c h w -> b c (nh h) (nw w)', 
            nh=H//cs, nw=W//cs, h=cs, w=cs).contiguous()  # [B, C, H, W]

        return self.to_out(out)

class ValueCrossAttn(nn.Module):
    def __init__(self,
                 query_dim: int,
                 context_dim: int = 16,
                 heads: int = 8,
                 dim_head: int = 64,
                 dropout: int = 0.) -> None:
        super().__init__()
        inner_dim = dim_head * heads

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(query_dim, inner_dim, kernel_size=1, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, query_dim, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self,
                x: Tensor,
                context: Tensor) -> Tensor:
        """
        :x: [B, C, H, W] (Query)
        :context: [B, context_dim]
        :return: [B, C, H, W]
        """
        x = x.contiguous()
        context = context.contiguous()

        _, _, H, W = x.shape

        q = self.to_q(x).contiguous()  # [B, inner_dim, H, W]

        context = context.unsqueeze(1)
        k = self.to_k(context).contiguous()  # [B, seq_len, inner_dim]
        v = self.to_v(context).contiguous()

        q = rearrange(q, 'b c h w -> b (h w) c').contiguous()  # [B, H*W, inner_dim]
        k = rearrange(k, 'b s c -> b c s').contiguous()        # [B, inner_dim, seq_len]
        v = rearrange(v, 'b s c -> b c s').contiguous() # [B, inner_dim, seq_len]

        attn = torch.softmax(torch.matmul(q, k) * self.scale, dim=-1)  # [B, H*W, seq_len]
        out = torch.matmul(attn.contiguous(), v.transpose(1,2).contiguous()).contiguous()  # [B, H*W, inner_dim]

        out = rearrange(out, 'b (h w) c -> b c h w', h=H, w=W).contiguous()
        return self.to_out(out).contiguous()

class PosCrossAttn(nn.Module):
    def __init__(self,
                 query_dim: int,
                 context_dim: int = 65,
                 heads: int = 8,
                 dim_head: int = 8,
                 dropout: int = 0.1) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.query_dim = query_dim
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(
            in_channels = query_dim,
            out_channels = inner_dim,
            kernel_size = 1,
            bias = False
        )
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, query_dim, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self,
                img_feats: Tensor,
                context: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
        """
        :param img_feats: [B, C, H, W] (Query)
        :param context: [B, N, context_dim]
        :return: [B, C, H, W]
        """
        img_feats = img_feats.contiguous()
        context = context.contiguous()

        h = self.heads

        q = self.to_q(img_feats).contiguous() # [B, inner_dim, H, W]
        q = rearrange(q, 'b (h d) x y -> (b h) (x y) d', h=h).contiguous() # [B*h, H*W, d]

        k = self.to_k(context).contiguous()             # [B, N, inner_dim]
        v = self.to_v(context).contiguous()             # [B, N, inner_dim]

        k = rearrange(k, 'b n (h d) -> (b h) n d', h=h).contiguous()  # [B*h, N, d]
        v = rearrange(v, 'b n (h d) -> (b h) n d', h=h).contiguous()

        sim: Tensor = einsum('b i d, b j d -> b i j', q, k).contiguous() * self.scale  # [B*h, H*W, N]

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)').contiguous()
            max_neg_value = max_neg_value(sim)
            mask = repeat(mask, 'b j -> (b h) () j', h=h).contiguous()
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v).contiguous()  # [B*h, H*W, d]

        _, _, H, W = img_feats.shape
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', 
                        h=h, x=H, y=W).contiguous()  # [B, inner_dim, H, W]

        return self.to_out(out).contiguous()  # [B, C, H, W]

class ComplexChunkedPosCrossAttn(nn.Module):
    def __init__(self,
                 query_dim: int,
                 context_dim: int = 65,
                 heads: int = 8,
                 dim_head: int = 4,
                 dropout: float = 0.1,
                 chunk_size: int = 8) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.query_dim = query_dim
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.chunk_size = chunk_size

        self.softmax = ComplexSoftmax()

        self.to_q = nn.Conv2d(
            in_channels=query_dim,
            out_channels=inner_dim,
            kernel_size=1,
            bias=False,
            dtype=complex64
        )
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, query_dim, kernel_size=1, dtype=complex64),
            ComplexDropout(dropout)
        )
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for all projection layers."""
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        nn.init.xavier_uniform_(self.to_out[0].weight)

    def forward(self,
                img_feats: Tensor,
                context: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
        """
        :param complex img_feats: [B, query_dim, H, W] (Query)
        :param real context: [B, N, context_dim]
        :return: [B, C, H, W]
        """
        _, _, H, W = img_feats.shape
        chunk_size = self.chunk_size

        assert H % chunk_size == 0 and W % chunk_size == 0, \
            f"Dimensions must be divisible by {chunk_size}, got H:{H} W:{W}"

        num_blocks_h = H // chunk_size
        num_blocks_w = W // chunk_size

        img_feats = rearrange(img_feats, 
                            'b c (nh bh) (nw bw) -> (b nh nw) c bh bw', 
                            bh=chunk_size, bw=chunk_size).contiguous()

        q = self.to_q(img_feats)
        q = rearrange(q, 'b (h d) bh bw -> (b h) (bh bw) d', 
                      h=self.heads).contiguous()

        k = self.to_k(context)
        k = rearrange(k, 'b n (h d) -> (b h) n d', h=self.heads).contiguous()
        k = repeat(k, 'bh n d -> (bh nh nw) n d', 
                   nh=num_blocks_h, nw=num_blocks_w).contiguous()

        v = self.to_v(context)
        v = rearrange(v, 'b n (h d) -> (b h) n d', h=self.heads).contiguous()
        v = repeat(v, 'bh n d -> (bh nh nw) n d', 
                   nh=num_blocks_h, nw=num_blocks_w).contiguous()

        sim = einsum('b i d, b j d -> b i j', q, k.to(dtype = q.dtype)).contiguous() * self.scale

        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = max_neg_value(sim)
            mask = repeat(mask, 
                         'b j -> (b h nh nw) () j', 
                         h=self.heads, 
                         nh=num_blocks_h, 
                         nw=num_blocks_w)
            sim.masked_fill_(~mask, max_neg_value)

        attn = self.softmax(sim)
        out = einsum('b i j, b j d -> b i d', attn, v.to(dtype = attn.dtype)).contiguous()

        out = rearrange(out, 
                       '(b nh nw h) (bh bw) d -> b (h d) (nh bh) (nw bw)', 
                       h=self.heads,
                       nh=num_blocks_h,
                       nw=num_blocks_w,
                       bh=chunk_size,
                       bw=chunk_size).contiguous()

        return self.to_out(out)

class ComplexChunkedValueCrossAttn(nn.Module):
    def __init__(self,
                 query_dim: int,
                 context_dim: int = 16,
                 heads: int = 8,
                 dim_head: int = 4,
                 dropout: float = 0.1,
                 chunk_size: int = 32) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.chunk_size = chunk_size

        self.softmax = ComplexSoftmax()

        self.to_q = nn.Conv2d(query_dim, inner_dim, kernel_size=1, bias=False, dtype=complex64)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, query_dim, kernel_size=1, dtype=complex64),
            ComplexDropout(dropout)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for all projection layers."""
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        nn.init.xavier_uniform_(self.to_out[0].weight)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        """
        :param complex-valued x: [B, query_dim, H, W] (Query)
        :param context: [B, context_dim]
        :return: [B, C, H, W]
        """
        _, _, H, W = x.shape
        cs = self.chunk_size

        assert H % cs == 0 and W % cs == 0, \
            f"Dimensions must be divisible by {cs}, got H:{H} W:{W}"

        x_chunks = rearrange(x, 
            'b c (nh h) (nw w) -> (b nh nw) c h w', 
            h=cs, w=cs).contiguous()  # [B*N, C, cs, cs]
        
        q = self.to_q(x_chunks)  # [B*N, inner_dim, cs, cs]
        q = rearrange(q, 'b c h w -> b (h w) c').contiguous()  # [B*N, cs^2, inner_dim]

        num_blocks = (H//cs) * (W//cs)
        context = context.repeat_interleave(num_blocks, dim=0).contiguous()  # [B*N, context_dim]
        context = context.unsqueeze(1)  # [B*N, 1, context_dim]
        
        k = self.to_k(context)  # [B*N, 1, inner_dim]
        v = self.to_v(context)  # [B*N, 1, inner_dim]
        k = rearrange(k, 'b s c -> b c s').contiguous()  # [B*N, inner_dim, 1]
        v = rearrange(v, 'b s c -> b c s').contiguous()  # [B*N, inner_dim, 1]

        attn = self.softmax(torch.matmul(q, k.to(dtype = q.dtype)) * self.scale)  # [B*N, cs^2, 1]
        out = torch.matmul(attn, v.transpose(1, 2).contiguous().to(dtype = attn.dtype))  # [B*N, cs^2, inner_dim]
        out = rearrange(out, 'b (h w) c -> b c h w', h=cs, w=cs).contiguous()  # [B*N, inner_dim, cs, cs]
        out = rearrange(out, 
            '(b nh nw) c h w -> b c (nh h) (nw w)', 
            nh=H//cs, nw=W//cs, h=cs, w=cs).contiguous()  # [B, C, H, W]

        return self.to_out(out)

def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max

if __name__ == '__main__':
    attn = ComplexChunkedValueCrossAttn(64)
    x = torch.randn(2, 64, 256, 256, dtype=complex64)
    text = torch.randn(2, 16)
    y = attn(x, text)
    print(y.shape)
