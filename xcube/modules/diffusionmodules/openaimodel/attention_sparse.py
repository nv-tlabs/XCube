from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from fvdb.nn import ElementwiseMixin, VDBTensor

from xcube.modules.diffusionmodules.openaimodel.util import checkpoint, conv_nd

import fvdb
import fvdb.nn as fvnn
from fvdb.nn import VDBTensor

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

class GELU(ElementwiseMixin, nn.GELU):
    pass

class LayerNorm(nn.LayerNorm):
    def forward(self, input: VDBTensor) -> VDBTensor:
        num_channels = input.feature.jdata.size(1)
        num_batches = input.grid.grid_count

        flat_data, flat_offsets = input.feature.jdata, input.feature.joffsets

        result_data = torch.empty_like(flat_data)

        for b in range(num_batches):
            feat = flat_data[flat_offsets[b, 0]:flat_offsets[b, 1]]
            if feat.size(0) != 0:
                feat = feat.reshape(1, -1, num_channels)
                feat = super().forward(feat)
                feat = feat.reshape(-1, num_channels)

                result_data[flat_offsets[b, 0]:flat_offsets[b, 1]] = feat

        return VDBTensor(input.grid, input.grid.jagged_like(result_data), input.kmap)

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, data: VDBTensor):
        x = data.feature.jdata
        x, gate = self.proj(x).chunk(2, dim=-1)
        out = x * F.gelu(gate)
        return VDBTensor(data.grid, data.grid.jagged_like(out), data.kmap)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            fvnn.Linear(dim, inner_dim),
            GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            fvnn.Dropout(dropout),
            fvnn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.scale = dim_head ** -0.5
        
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads

        self.to_q = fvnn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = fvnn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = fvnn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            fvnn.Linear(inner_dim, query_dim),
            fvnn.Dropout(dropout)
        )

    def forward(self, x: VDBTensor):
        q = self.to_q(x)
        context = x
        k = self.to_k(context)
        v = self.to_v(context)
        
        out = []
        for batch_idx in range(q.grid.grid_count):
            out.append(self._attention(q.feature[batch_idx].jdata, k.feature[batch_idx].jdata, v.feature[batch_idx].jdata))
        out = fvdb.JaggedTensor(out)
        out = VDBTensor(x.grid, out, x.kmap)
        return self.to_out(out)

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # q: (N, C)
        # k: (77, C)
        # v: (77, C)
        # mask: (1, 77)
        h = self.heads
        q, k, v = map(lambda t: rearrange(t, '(b n) (h d) -> b h n d', h=h, b=1), (q, k, v))
        with torch.backends.cuda.sdp_kernel(enable_math=False):
            out = F.scaled_dot_product_attention(q, k, v)[0] # h, n, d
        out = rearrange(out, 'h n d -> n (h d)')
        return out


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.scale = dim_head ** -0.5
        
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads

        self.to_q = fvnn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            fvnn.Linear(inner_dim, query_dim),
            fvnn.Dropout(dropout)
        )

    def forward(self, x: VDBTensor, context: torch.Tensor=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        out = []
        for batch_idx in range(q.grid.grid_count):
            if exists(mask):
                mask = rearrange(mask, 'b ... -> b (...)')
                out.append(self._attention(q.feature[batch_idx].jdata, k[batch_idx], v[batch_idx], mask=mask[batch_idx:batch_idx+1]))
            else:
                out.append(self._attention(q.feature[batch_idx].jdata, k[batch_idx], v[batch_idx]))
        out = fvdb.JaggedTensor(out)
        out = VDBTensor(x.grid, out, x.kmap)
        return self.to_out(out)

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
        # q: (N, C)
        # k: (77, C)
        # v: (77, C)
        # mask: (1, 77)
        if q.size(0) == 0:
            return q

        h = self.heads
        q, k, v = map(lambda t: rearrange(t, '(b n) (h d) -> b h n d', h=h, b=1), (q, k, v))
        
        if exists(mask):
            mask = repeat(mask, 'b s -> b h l s', h=h, l=q.shape[2])
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)[0] # h, n, d
        else:
            with torch.backends.cuda.sdp_kernel(enable_math=False):
                out = F.scaled_dot_product_attention(q, k, v)[0] # h, n, d
        out = rearrange(out, 'h n d -> n (h d)')
        return out



class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True, disable_sa=False):
        super().__init__()
        if not disable_sa:
            self.attn1 = Attention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        if not disable_sa:
            self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.norm3 = LayerNorm(dim)
        self.checkpoint = checkpoint
        self.disable_sa = disable_sa

    def forward(self, x, context=None, mask=None):
        if not self.disable_sa:        
            x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context, mask=mask) + x
        x = self.ff(self.norm3(x)) + x
        return x

class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, disable_sa=False):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = fvnn.GroupNorm(32, in_channels)

        self.proj_in = fvnn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim, disable_sa=disable_sa)
                for d in range(depth)]
        )

        self.proj_out = zero_module(fvnn.Linear(inner_dim, in_channels))

    def forward(self, x: VDBTensor, context=None, mask=None):
        # if x is empty
        if x.grid.ijk.jdata.size(0) == 0:
            return x
        
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        for block in self.transformer_blocks:
            x = block(x, context=context, mask=mask)
        x = self.proj_out(x)
        return x + x_in