import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import fvdb
import fvdb.nn as fvnn
from fvdb.nn import VDBTensor

from xcube.modules.diffusionmodules.openaimodel.attention_sparse import SpatialTransformer

def zero_module(module: nn.Module):
    """Zero out the parameters of a module."""
    for p in module.parameters():
        p.data.zero_()
    return module


class TimestepModule(nn.Module):
    def forward(self, x, emb, target_tensor = None):
        raise NotImplementedError


class TimestepSequential(nn.Sequential):
    def forward(self, x, emb, target_tensor: Optional[VDBTensor] = None, context=None, mask=None):
        for layer in self:
            if isinstance(layer, TimestepModule):
                x = layer(x, emb, target_tensor)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context, mask)
            else:
                x = layer(x)
        return x


class ResBlock(TimestepModule):
    def __init__(self, channels: int, emb_channels: int, dropout: float,
                 out_channels: Optional[int] = None,
                 up: bool = False, down: bool = False, stride: int = 1):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.stride = stride

        # Nonlinear operations to time/class embeddings
        #   (added between in_layers and out_layers in the res branch)
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.emb_channels, 2 * self.out_channels)
        )

        self.in_layers = nn.Sequential(
            fvnn.GroupNorm(num_groups=32, num_channels=channels),
            fvnn.SiLU(),
            fvnn.SparseConv3d(self.channels, self.out_channels, 3, bias=True)
        )

        self.up, self.down = up, down
        if self.up:
            self.up_module = fvnn.UpsamplingNearest(self.stride)
        elif self.down:
            self.down_module = fvnn.AvgPool(self.stride)

        self.out_layers = nn.Sequential(
            fvnn.GroupNorm(num_groups=32, num_channels=self.out_channels),
            fvnn.SiLU(),
            fvnn.Dropout(p=self.dropout),
            # Zero out res output since this is the residual
            zero_module(fvnn.SparseConv3d(self.out_channels, self.out_channels, 3, bias=True))
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = fvnn.SparseConv3d(channels, self.out_channels, 1, bias=True)

    def forward(self, data: VDBTensor, emb: torch.Tensor,
                target_tensor: Optional[VDBTensor] = None):
        if self.up or self.down:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            data_h = in_rest(data)
            data_h = self.up_module(data_h, ref_fine_data=target_tensor) \
                if self.up else self.down_module(data_h, ref_coarse_data=target_tensor)
            data_h = in_conv(data_h)
            data = self.up_module(data, ref_fine_data=data_h) \
                if self.up else self.down_module(data, ref_coarse_data=data_h)
        else:
            data_h = self.in_layers(data)

        assert isinstance(data_h, VDBTensor)

        emb_h = self.emb_layers(emb)    # (B, 2C)
        scale, shift = emb_h.chunk(2, dim=-1)   # (B, C), (B, C)
        batch_idx = data_h.jidx.long()

        out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        data_h = out_norm(data_h) * (1 + scale[batch_idx]) + shift[batch_idx]
        data_h = out_rest(data_h)

        data = data_h + self.skip_connection(data)
        return data

class UNetModel(nn.Module):
    def __init__(self, 
                 num_input_channels: int, 
                 model_channels: int, 
                 num_res_blocks: int,
                 out_channels: Optional[int] = None, 
                 dropout: float = 0.0,
                 channel_mult: Tuple = (1, 2, 4, 8), 
                 num_classes: Optional[int] = None, 
                 attention_resolutions: list = [],
                 num_heads: int = 8,
                 transformer_depth: int = 1,
                 context_dim: int = 1024,
                 **kwargs):
        super().__init__()

        in_channels = num_input_channels
        self.in_channels = in_channels
        self.model_channels = model_channels

        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        transformer_depth_middle = transformer_depth[-1]

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "provide num_res_blocks either as an int (globally constant) or "
                    "as a list/tuple (per-level) with the same length as channel_mult"
                )
            self.num_res_blocks = num_res_blocks

        self.out_channels = out_channels or in_channels
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_classes = num_classes
        
        self.attention_resolutions = attention_resolutions
        self.num_heads = num_heads

        time_emb_dim = 4 * self.model_channels
        self.time_emb = nn.Sequential(
            nn.Linear(self.model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        if self.num_classes is not None:
            self.label_emb = nn.Linear(self.num_classes, time_emb_dim)

        # Encoder
        self.encoder_blocks = nn.ModuleList([TimestepSequential(
            fvnn.SparseConv3d(self.in_channels, self.model_channels, 3, bias=True),
        )])

        encoder_channels = [self.model_channels]
        current_channels = self.model_channels
        ds = 1
        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks[level]):
                layers = [ResBlock(
                    current_channels, time_emb_dim, self.dropout,
                    out_channels=self.model_channels * mult
                )]
                current_channels = self.model_channels * mult
                # Attention
                if ds in attention_resolutions:
                    # enable self-attention
                    disable_sa = False
                else:
                    disable_sa = True
                dim_head = current_channels // num_heads
                layers.append(
                    SpatialTransformer(current_channels, num_heads, dim_head, depth=transformer_depth[level], context_dim=context_dim, disable_sa=disable_sa)
                    )
                self.encoder_blocks.append(TimestepSequential(*layers))
                encoder_channels.append(current_channels)
            # Downsample for all but the last block
            if level < len(self.channel_mult) - 1:
                layers = [ResBlock(
                    current_channels, time_emb_dim, self.dropout,
                    out_channels=current_channels,
                    down=True, stride=2
                )]
                self.encoder_blocks.append(TimestepSequential(*layers))
                encoder_channels.append(current_channels)
                ds *= 2

        # Middle block (won't change dimension)
        self.middle_block = TimestepSequential(
            ResBlock(current_channels, time_emb_dim, self.dropout),
            SpatialTransformer(current_channels, num_heads, dim_head, depth=transformer_depth_middle, context_dim=context_dim, disable_sa=False),
            ResBlock(current_channels, time_emb_dim, self.dropout)
        )

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            # Use one more block for decoder
            for i in range(self.num_res_blocks[level] + 1):
                skip_channels = encoder_channels.pop()
                layers = [ResBlock(
                    current_channels + skip_channels,
                    time_emb_dim, self.dropout,
                    out_channels=self.model_channels * mult
                )]
                current_channels = self.model_channels * mult
                # Attention
                if ds in attention_resolutions:
                    # enable self-attention
                    disable_sa = False
                else:
                    disable_sa = True
                layers.append(
                    SpatialTransformer(current_channels, num_heads, dim_head, depth=transformer_depth[level], context_dim=context_dim, disable_sa=disable_sa)
                    )
                # Upsample for all but the finest block
                if level > 0 and i == self.num_res_blocks[level]:
                    layers.append(ResBlock(
                        current_channels, time_emb_dim, self.dropout,
                        out_channels=current_channels,
                        up=True, stride=2
                    ))
                    ds //= 2
                self.decoder_blocks.append(TimestepSequential(*layers))

        # Output block
        assert current_channels == self.model_channels
        self.out_block = nn.Sequential(
            fvnn.GroupNorm(num_groups=32, num_channels=current_channels),
            fvnn.SiLU(),
            zero_module(fvnn.SparseConv3d(current_channels, self.out_channels, 3, bias=True))
        )

    def timestep_encoding(self, timesteps: torch.Tensor, max_period: int = 10000):
        dim = self.model_channels
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, data: VDBTensor, timesteps: torch.Tensor,
                y: Optional[torch.Tensor] = None, context: torch.Tensor = None, mask: torch.Tensor = None):
        assert (y is not None) == (self.num_classes is not None), \
            "Must provide labels if num_classes is not None"
        if timesteps.dim() == 0:
            timesteps = timesteps.expand(1).repeat(data.grid.grid_count).to(data.device)
        
        t_emb = self.timestep_encoding(timesteps)
        emb = self.time_emb(t_emb)
        if y is not None:
            emb += self.label_emb(y)

        hs = []
        for block in self.encoder_blocks:
            data = block(data, emb, context=context, mask=mask)
            hs.append(data)
        data = self.middle_block(data, emb, context=context, mask=mask)
        for block in self.decoder_blocks:
            pop_data = hs.pop()
            data = VDBTensor.cat([pop_data, data], dim=1)
            data = block(data, emb, hs[-1] if len(hs) > 0 else None, context=context, mask=mask)

        data = self.out_block(data)
        return data