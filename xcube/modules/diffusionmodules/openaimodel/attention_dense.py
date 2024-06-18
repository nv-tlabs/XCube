from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from xcube.modules.diffusionmodules.openaimodel.util import checkpoint, conv_nd

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


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
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


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.scale = dim_head ** -0.5
        
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            mask = repeat(mask, 'b s -> b h l s', h=h, l=q.shape[2]) # q.shape[2] is the length of the query
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask) # (b h) n d
        else:
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask) # (b h) n d
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)
    
    # def forward(self, x, context=None, mask=None):
    #     h = self.heads

    #     q = self.to_q(x)
    #     context = default(context, x)
    #     k = self.to_k(context)
    #     v = self.to_v(context)

    #     q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        
    #     sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
    #     if exists(mask):
    #         mask = rearrange(mask, 'b ... -> b (...)')
    #         max_neg_value = -torch.finfo(sim.dtype).max
    #         mask = repeat(mask, 'b j -> (b h) () j', h=h)
    #         sim.masked_fill_(~mask, max_neg_value)
        
    #     attn = sim.softmax(dim=-1)

    #     out = einsum('b i j, b j d -> b i d', attn, v)
    #     out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    #     return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True, disable_sa=False):
        super().__init__()
        if not disable_sa:
            self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        
        if not disable_sa:
            self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
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
        self.norm = Normalize(in_channels)

        self.proj_in = conv_nd(3, in_channels, 
                               inner_dim, 
                               kernel_size=1,
                               stride=1,
                               padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim, disable_sa=disable_sa)
                for d in range(depth)]
        )

        self.proj_out = zero_module(conv_nd(3, inner_dim, 
                                            in_channels, 
                                            kernel_size=1,
                                            stride=1,
                                            padding=0))

    def forward(self, x, context=None, mask=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w, d = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w d -> b (h w d) c')
        for block in self.transformer_blocks:
            x = block(x, context=context, mask=mask)
        x = rearrange(x, 'b (h w d) c -> b c h w d', h=h, w=w, d=d)
        x = self.proj_out(x)
        return x + x_in

