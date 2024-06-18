import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import fvdb
import fvdb.nn as fvnn
from fvdb.nn import VDBTensor


def zero_module(module: nn.Module):
    """Zero out the parameters of a module."""
    for p in module.parameters():
        p.data.zero_()
    return module


class TimestepModule(nn.Module):
    def forward(self, x, emb, target_tensor = None):
        raise NotImplementedError


class TimestepSequential(nn.Sequential):
    def forward(self, x, emb, target_tensor: Optional[VDBTensor] = None):
        for layer in self:
            if isinstance(layer, TimestepModule):
                x = layer(x, emb, target_tensor)
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
    
    
class AttentionBlock(nn.Module):
    """
    A for loop version with flash attention
    """
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = fvnn.GroupNorm(32, channels)
        self.qkv = fvnn.Linear(channels, channels * 3)
        self.proj_out = zero_module(fvnn.Linear(channels, channels))
        
    def _attention(self, qkv: torch.Tensor):
        # conduct attention for each batch
        length, width = qkv.shape
        assert width % (3 * self.num_heads) == 0
        ch = width // (3 * self.num_heads)
        qkv = qkv.reshape(length, self.num_heads, 3 * ch).unsqueeze(0)
        qkv = qkv.permute(0, 2, 1, 3) # (1, num_heads, length, 3 * ch)
        q, k, v = qkv.chunk(3, dim=-1) # (1, num_heads, length, ch)
        with torch.backends.cuda.sdp_kernel(enable_math=False):
            values = F.scaled_dot_product_attention(q, k, v)[0] # (1, num_heads, length, ch)
        values = values.permute(1, 0, 2) # (length, num_heads, ch)
        values = values.reshape(length, -1)
        return values
        
    def attention(self, qkv: VDBTensor):
        values = []
        for batch_idx in range(qkv.grid.grid_count):
            values.append(self._attention(qkv.feature[batch_idx].jdata))            
        return fvdb.JaggedTensor(values)

    def forward(self, x: VDBTensor):
        return self._forward(x)

    def _forward(self, x: VDBTensor):
        qkv = self.qkv(self.norm(x))
        feature = self.attention(qkv)
        feature = VDBTensor(x.grid, feature, x.kmap)
        feature = self.proj_out(feature)
        return feature + x


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
                 use_middle_attention: bool = False,
                 **kwargs):
        super().__init__()

        in_channels = num_input_channels
        self.in_channels = in_channels
        self.model_channels = model_channels
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
            for _ in range(self.num_res_blocks):
                layers = [ResBlock(
                    current_channels, time_emb_dim, self.dropout,
                    out_channels=self.model_channels * mult
                )]
                current_channels = self.model_channels * mult
                # Attention
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(current_channels, num_heads=self.num_heads)
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
        if use_middle_attention:
            self.middle_block = TimestepSequential(
                ResBlock(current_channels, time_emb_dim, self.dropout),
                AttentionBlock(current_channels, num_heads=self.num_heads),
                ResBlock(current_channels, time_emb_dim, self.dropout)
            )
        else:
            self.middle_block = TimestepSequential(
                ResBlock(current_channels, time_emb_dim, self.dropout),
                ResBlock(current_channels, time_emb_dim, self.dropout)
            )

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            # Use one more block for decoder
            for i in range(self.num_res_blocks + 1):
                skip_channels = encoder_channels.pop()
                layers = [ResBlock(
                    current_channels + skip_channels,
                    time_emb_dim, self.dropout,
                    out_channels=self.model_channels * mult
                )]
                current_channels = self.model_channels * mult
                # Attention
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(current_channels, num_heads=self.num_heads)
                        )
                # Upsample for all but the finest block
                if level > 0 and i == self.num_res_blocks:
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
                y: Optional[torch.Tensor] = None):
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
            data = block(data, emb)
            hs.append(data)
        data = self.middle_block(data, emb)
        for block in self.decoder_blocks:
            pop_data = hs.pop()
            data = VDBTensor.cat([pop_data, data], dim=1)
            data = block(data, emb, hs[-1] if len(hs) > 0 else None)

        data = self.out_block(data)
        return data