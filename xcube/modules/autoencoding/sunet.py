# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import fvdb
import fvdb.nn as fvnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from fvdb.nn import VDBTensor
from loguru import logger
from torch.autograd import Variable


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps


class depth_wrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    def forward(self, *args):
        return self.module(*args), 0


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, order: str, num_groups: int):
        super().__init__()
        for i, char in enumerate(order):
            if char == 'r':
                self.add_module('ReLU', fvnn.ReLU(inplace=True))
            elif char == 's':
                self.add_module('SiLU', fvnn.SiLU(inplace=True))
            elif char == 'c':
                self.add_module('Conv', fvnn.SparseConv3d(
                    in_channels, out_channels, 3, 1, bias='g' not in order))
            elif char == 'g':
                num_channels = in_channels if i < order.index('c') else out_channels
                if num_channels < num_groups:
                    num_groups = 1
                self.add_module('GroupNorm', fvnn.GroupNorm(
                    num_groups=num_groups, num_channels=num_channels, affine=True))
            else:
                raise NotImplementedError


class SparseHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, order, num_groups, enhanced="None"):
        super().__init__()
        self.add_module('SingleConv', ConvBlock(in_channels, in_channels, order, num_groups))
        mid_channels = in_channels
        if out_channels > mid_channels:
            mid_channels = out_channels

        if enhanced == "None":
            self.add_module('OutConv', fvnn.SparseConv3d(in_channels, out_channels, 1, bias=True))
        else:
            raise NotImplementedError


class SparseResBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 order: str,
                 num_groups: int,
                 encoder: bool,
                 pooling = None,
                 use_checkpoint: bool = False,
                 pooling_factor = [2, 2, 2],
                 ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.use_pooling = pooling is not None and encoder

        if encoder:
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
            if pooling == 'max':
                self.maxpooling = fvnn.MaxPool(pooling_factor)
        else:
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        self.conv1 = ConvBlock(conv1_in_channels, conv1_out_channels, order, num_groups)
        self.conv2 = ConvBlock(conv2_in_channels, conv2_out_channels, order, num_groups)

        if conv1_in_channels != conv2_out_channels:
            self.skip_connection = fvnn.SparseConv3d(conv1_in_channels, conv2_out_channels, 1, 1)
        else:
            self.skip_connection = nn.Identity()
    
    def _forward(self, input, hash_tree = None, feat_depth: int = 0):
        if self.use_pooling:
            if hash_tree is not None:
                feat_depth += 1
                input = self.maxpooling(input, hash_tree[feat_depth])
            else:
                input = self.maxpooling(input)
        
        h = input
        h = self.conv1(h)
        h = self.conv2(h)
        input = self.skip_connection(input)

        return h + input, feat_depth
    
    def forward(self, input, hash_tree = None, feat_depth: int = 0):
        if self.use_checkpoint:
            input, feat_depth = checkpoint.checkpoint(self._forward, input, hash_tree, feat_depth, use_reentrant=False) 
        else:
            input, feat_depth = self._forward(input, hash_tree, feat_depth)
        return input, feat_depth


class SparseDoubleConv(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 order: str,
                 num_groups: int,
                 encoder: bool,
                 pooling = None,
                 use_checkpoint: bool = False,
                 pooling_factor = [2, 2, 2],
                 ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        if encoder:
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
            if pooling == 'max':
                self.add_module('MaxPool', fvnn.MaxPool(pooling_factor))
        else:
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        self.add_module('SingleConv1', ConvBlock(conv1_in_channels, conv1_out_channels, order, num_groups))
        self.add_module('SingleConv2', ConvBlock(conv2_in_channels, conv2_out_channels, order, num_groups))
    
    def _forward(self, input, hash_tree = None, feat_depth: int = 0):
        for module in self:
            if module._get_name() == 'MaxPool' and hash_tree is not None:
                feat_depth += 1
                input = module(input, hash_tree[feat_depth])
            else:
                input = module(input)
        return input, feat_depth
    
    def forward(self, input, hash_tree = None, feat_depth: int = 0):
        if self.use_checkpoint:
            input, feat_depth = checkpoint.checkpoint(self._forward, input, hash_tree, feat_depth, use_reentrant=False) 
        else:
            input, feat_depth = self._forward(input, hash_tree, feat_depth)
        return input, feat_depth


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
        self.proj_out = fvnn.Linear(channels, channels)
        
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
        return self._forward(x), None # !: return None for feat_depth

    def _forward(self, x: VDBTensor):
        qkv = self.qkv(self.norm(x))
        feature = self.attention(qkv)
        feature = VDBTensor(x.grid, feature, x.kmap)
        feature = self.proj_out(feature)
        return feature + x


class StructPredictionNet(nn.Module):
    def __init__(self, in_channels, num_blocks, f_maps=64, order='gcr', num_groups=8,
                 pooling='max', pooling_level=[], neck_dense_type="UNCHANGED", cut_ratio=1, neck_bound=4, 
                 with_color_branch=False, with_normal_branch=False,
                 with_semantic_branch=False, num_semantic_classes=23,
                 use_attention=False, use_residual=True, num_res_blocks=1, is_add_dec=True,
                 unstable_cutoff=False, unstable_cutoff_threshold=0.5, 
                 use_checkpoint=False, **kwargs):
        super().__init__()
        n_features = [in_channels] + [f_maps * 2 ** k for k in range(num_blocks)]
        logger.info("latent dim: {}".format(int(n_features[-1] / cut_ratio)))
        self.encoders = nn.ModuleList()
        self.pre_kl_bottleneck = nn.ModuleList()
        self.post_kl_bottleneck = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.struct_convs = nn.ModuleList()
        self.num_blocks = num_blocks

        self.unstable_cutoff = unstable_cutoff
        self.unstable_cutoff_threshold = unstable_cutoff_threshold
        self.is_add_dec = is_add_dec

        if not use_residual:
            basic_block = SparseDoubleConv
        else:
            basic_block = SparseResBlock

        # Attention setup
        self.use_attention = use_attention

        # remain_h setup
        if pooling_level == []:
            # then pooling every level
            pooling_level = list(range(num_blocks))
            
        # Encoder
        self.pre_conv = fvnn.SparseConv3d(in_channels, in_channels, 1, 1) # a MLP to smooth the input
        for layer_idx in range(num_blocks):
            block_in = n_features[layer_idx]
            block_out = n_features[layer_idx + 1]

            if layer_idx in pooling_level:
                pooling_factor = [2, 2, 2]
            else:
                pooling_factor = [2, 2, 1]

            if is_add_dec:
                for i_block in range(num_res_blocks):
                    if i_block == 0:
                        cur_pooling = pooling
                    else:
                        cur_pooling = None
                    block = basic_block(
                        block_in, 
                        block_out, 
                        order, 
                        num_groups, 
                        True, # if encoder branch
                        cur_pooling if layer_idx > 0 else None,
                        use_checkpoint,
                        pooling_factor=pooling_factor,
                        )
                    block_in = block_out
                    self.encoders.add_module(f'Enc{layer_idx}-Block{i_block}', block)
            else:
                block = basic_block(
                    block_in, 
                    block_out, 
                    order, 
                    num_groups, 
                    True, # if encoder branch
                    pooling if layer_idx > 0 else None,
                    use_checkpoint,
                    pooling_factor=pooling_factor,
                    )
                self.encoders.add_module(f'Enc{layer_idx}', block)

        # Bottleneck
        self.pre_kl_bottleneck.add_module(f'pre_kl_bottleneck_0', basic_block(
            n_features[-1], n_features[-1], order, num_groups, False, use_checkpoint=use_checkpoint))  
        if use_attention:
            self.pre_kl_bottleneck.add_module(f'pre_kl_attention', AttentionBlock(
                n_features[-1], use_checkpoint=use_checkpoint))
        if not use_residual:
            self.pre_kl_bottleneck.add_module(f'pre_kl_bottleneck_1', basic_block(
                n_features[-1], int(n_features[-1] / cut_ratio) * 2, order, num_groups, False, use_checkpoint=use_checkpoint))
        else:
            self.pre_kl_bottleneck.add_module(f'pre_kl_bottleneck_1', basic_block(
                n_features[-1], n_features[-1], order, num_groups, False, use_checkpoint=use_checkpoint))
            self.pre_kl_bottleneck.add_module(f'pre_kl_bottleneck_gn', depth_wrapper(
                fvnn.GroupNorm(num_groups=num_groups, num_channels=n_features[-1], affine=True)))
            self.pre_kl_bottleneck.add_module(f'pre_kl_bottleneck_2', depth_wrapper(
                fvnn.SparseConv3d(n_features[-1], int(n_features[-1] / cut_ratio) * 2, 3, 1, bias=True)))

        self.post_kl_bottleneck.add_module(f'post_kl_bottleneck_0', basic_block(
            int(n_features[-1] / cut_ratio), n_features[-1], order, num_groups, False, use_checkpoint=use_checkpoint))
        if use_attention:
            self.post_kl_bottleneck.add_module(f'post_kl_attention', AttentionBlock(
                n_features[-1], use_checkpoint=use_checkpoint))
        self.post_kl_bottleneck.add_module(f'post_kl_bottleneck_1', basic_block(
            n_features[-1], n_features[-1], order, num_groups, False, use_checkpoint=use_checkpoint))
    
        # Decoder
        subdived_level = [-num_blocks - 1 + l for l in pooling_level]
        for layer_idx in range(-1, -num_blocks - 1, -1):
            self.struct_convs.add_module(f'Struct{layer_idx}', SparseHead(
                n_features[layer_idx], 2, order, num_groups))
            if layer_idx < -1:
                if is_add_dec:
                    block = nn.ModuleList()
                    block_in = n_features[layer_idx + 1]
                    block_out = n_features[layer_idx]
                    for i_block in range(num_res_blocks + 1):
                        block.append(basic_block(
                            block_in, 
                            block_out, 
                            order, 
                            num_groups, 
                            False, # if decoder branch
                            None,
                            use_checkpoint))
                        block_in = block_out
                    self.decoders.add_module(f'Dec{layer_idx}', block)
                else:
                    self.decoders.add_module(f'Dec{layer_idx}', basic_block(
                        n_features[layer_idx + 1], 
                        n_features[layer_idx], 
                        order, 
                        num_groups, 
                        False, 
                        None,
                        use_checkpoint))

                if layer_idx in subdived_level:
                    subdived_factor = [2, 2, 2]
                else:
                    subdived_factor = [2, 2, 1]

                self.upsamplers.add_module(f'Up{layer_idx}', fvnn.UpsamplingNearest(subdived_factor))
        self.up_sample0 = fvnn.UpsamplingNearest(1)

        # check the type of neck_bound
        if isinstance(neck_bound, int):
            self.low_bound = [-neck_bound] * 3
            self.voxel_bound = [neck_bound * 2] * 3
        else:        
            self.low_bound = [-res for res in neck_bound]
            self.voxel_bound = [res * 2 for res in neck_bound]
        self.neck_dense_type = neck_dense_type
        
        # add new branchs
        self.with_normal_branch = with_normal_branch
        if with_normal_branch:
            self.normal_head = SparseHead(n_features[1], 3, order, num_groups)
        self.with_semantic_branch = with_semantic_branch
        if with_semantic_branch:
            self.semantic_head = SparseHead(n_features[1], num_semantic_classes, order, num_groups)
        self.with_color_branch = with_color_branch
        if with_color_branch:
            self.color_head = SparseHead(n_features[1], 3, order, num_groups)
        
    @classmethod
    def sparse_zero_padding(cls, in_x: fvnn.VDBTensor, target_grid: fvdb.GridBatch):
        source_grid = in_x.grid
        source_feature = in_x.feature.jdata
        assert torch.allclose(source_grid.origins, target_grid.origins)
        assert torch.allclose(source_grid.voxel_sizes, target_grid.voxel_sizes)
        out_feat = torch.zeros((target_grid.total_voxels, source_feature.size(1)),
                               device=source_feature.device, dtype=source_feature.dtype)
        in_idx = source_grid.ijk_to_index(target_grid.ijk).jdata
        in_mask = in_idx != -1
        out_feat[in_mask] = source_feature[in_idx[in_mask]]
        return fvnn.VDBTensor(target_grid, target_grid.jagged_like(out_feat))
    
    @classmethod
    def struct_to_mask(cls, struct_pred: fvnn.VDBTensor):
        # 0 is exist, 1 is non-exist
        mask = struct_pred.feature.jdata[:, 0] > struct_pred.feature.jdata[:, 1]
        return struct_pred.grid.jagged_like(mask)

    @classmethod
    def cat(cls, x: fvnn.VDBTensor, y: fvnn.VDBTensor):
        assert x.grid == y.grid
        return fvnn.VDBTensor(x.grid, x.grid.jagged_like(torch.cat([x.feature.jdata, y.feature.jdata], dim=1)))
    
    class FeaturesSet:
        def __init__(self):
            self.encoder_features = {}
            self.structure_features = {}
            self.structure_grid = {}
            self.color_features = {}
            self.normal_features = {}
            self.semantic_features = {}

    def _encode(self, x: fvnn.VDBTensor, hash_tree: dict, is_forward: bool = True, neck_bound=None):
        feat_depth = 0
        res = self.FeaturesSet()
        x = self.pre_conv(x)
        for module in self.encoders:
            x, feat_depth = module(x, hash_tree, feat_depth)
        if self.neck_dense_type == "UNCHANGED":
            pass
        elif self.neck_dense_type == "HAND_CRAFTED":
            voxel_size = x.grid.voxel_sizes[0] # !: modify for remain h
            origins = x.grid.origins[0] # !: modify for remain h
            if neck_bound is None:
                voxel_bound = self.voxel_bound
                low_bound = self.low_bound
            else:
                if isinstance(neck_bound, int):
                    low_bound = [-neck_bound] * 3
                    voxel_bound = [neck_bound * 2] * 3
                else:        
                    low_bound = [-res for res in neck_bound]
                    voxel_bound = [res * 2 for res in neck_bound]
            neck_grid = fvdb.sparse_grid_from_dense(
                x.grid.grid_count, 
                voxel_bound, 
                low_bound,
                device="cpu",
                voxel_sizes=voxel_size,
                origins=origins).to(x.device)
            x = fvnn.VDBTensor(neck_grid, neck_grid.fill_to_grid(x.feature, x.grid, 0.0))
        else:
            raise NotImplementedError

        for module in self.pre_kl_bottleneck:
            x, _ = module(x)
        dec_main_feature = x.feature.jdata
        mu, log_sigma = torch.chunk(dec_main_feature, 2, dim=1)

        return res, x, mu, log_sigma
    
    def encode(self, x: fvnn.VDBTensor, hash_tree: dict, neck_bound=None):
        return self._encode(x, hash_tree, True, neck_bound)
    
    def decode(self, res: FeaturesSet, x: fvnn.VDBTensor, is_testing=False):
        for module in self.post_kl_bottleneck:
            x, _ = module(x)
        
        struct_decision = None
        feat_depth = self.num_blocks - 1
        for block, upsampler, struct_conv in zip(
                [None] + list(self.decoders), [None] + list(self.upsamplers), self.struct_convs):  
            if block is not None:
                x = upsampler(x, struct_decision)

                if self.is_add_dec:
                    for module in block:
                        x, _ = module(x)
                else:
                    x, _ = block(x)
            res.structure_features[feat_depth] = struct_conv(x)
            struct_decision = self.struct_to_mask(res.structure_features[feat_depth])
            if feat_depth != self.num_blocks - 1 and self.unstable_cutoff:
                # compute the dense resolution
                current_voxel_bound = [res * 2 ** (self.num_blocks - 1 - feat_depth) for res in self.voxel_bound]
                max_voxel_count = np.prod(current_voxel_bound)

                current_ratio = struct_decision.jdata.sum() / (max_voxel_count * x.grid.grid_count)
                if current_ratio > self.unstable_cutoff_threshold and not is_testing:
                    logger.info("cut off at depth %d with ratio %03f" % (feat_depth, current_ratio))
                    struct_decision = struct_decision.jagged_like(torch.zeros_like(struct_decision.jdata))
                    
            res.structure_grid[feat_depth] = self.up_sample0(x, struct_decision).grid
            feat_depth -= 1
        x = self.up_sample0(x, struct_decision)
        
        if self.with_normal_branch:
            # check if there is activate features
            if x.grid.total_voxels > 0:
                res.normal_features[feat_depth] = self.normal_head(x)
        if self.with_semantic_branch:
            # check if there is activate features
            if x.grid.total_voxels > 0:
                res.semantic_features[feat_depth] = self.semantic_head(x)
        if self.with_color_branch:
            # check if there is activate features
            if x.grid.total_voxels > 0:
                res.color_features[feat_depth] = self.color_head(x)
        
        return res, x
    
    def forward(self, x: fvnn.VDBTensor, hash_tree: dict, noise_step: int = 0, noise_scheduler = None, neck_bound = None):
        dist_features = []
        res, x, mu, log_sigma = self.encode(x, hash_tree, neck_bound=neck_bound)
        dist_features.append((mu, log_sigma))
        posterior = reparametrize(mu, log_sigma)

        if noise_step > 0:
            noise = torch.randn_like(posterior)
            posterior = noise_scheduler.add_noise(posterior, noise, torch.tensor([noise_step]))

        x = fvnn.VDBTensor(x.grid, x.grid.jagged_like(posterior))
        res, x = self.decode(res, x)
        return res, x, dist_features