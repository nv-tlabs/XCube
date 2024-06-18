# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import fvdb
import fvdb.nn as fvnn
from fvdb import JaggedTensor
from torch_scatter import scatter_max, scatter_mean

from xcube.modules.autoencoding.sunet import SparseDoubleConv


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class', crossattn=False):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.crossattn = crossattn

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        if self.crossattn:
            # this is for use in crossattn
            c = batch[key][:, None]
        else:
            c = batch[key]
        c = self.embedding(c)
        return c

class SemanticEncoder(nn.Module):
    def __init__(self, num_semantic, dim_semantic):
        super().__init__()
        self.model = nn.Embedding(num_semantic, dim_semantic)
        
    def forward(self, x):
        return self.model(x)

class StructEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, f_maps=64, order='gcr', num_groups=8,
                 pooling='max', use_checkpoint=False, **kwargs):
        super().__init__()
        n_features = [in_channels] + [f_maps * 2 ** k for k in range(num_blocks)]

        self.encoders = nn.ModuleList()
        self.num_blocks = num_blocks
        # Encoder
        self.pre_conv = fvnn.SparseConv3d(in_channels, in_channels, 1, 1) # a MLP to smooth the input
        for layer_idx in range(num_blocks):
            self.encoders.add_module(f'Enc{layer_idx}', SparseDoubleConv(
                n_features[layer_idx], 
                n_features[layer_idx + 1], 
                order, 
                num_groups,
                True, # if encoder branch
                pooling if layer_idx > 0 else None,
                use_checkpoint
            ))
        self.out_conv = nn.Linear(n_features[-1], out_channels)
            
    def encode(self, x: fvnn.VDBTensor):
        x = self.pre_conv(x)
        for module in self.encoders:
            x, _ = module(x)
        return x
    
    def forward(self, x: fvnn.VDBTensor):
        x = self.encode(x)
        out, _ = scatter_max(x.feature.jdata, x.feature.jidx.long(), dim=0)
        out = self.out_conv(out)
        return out
    
class StructEncoder3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, f_maps=64, order='gcr', num_groups=8,
                 pooling='max', use_checkpoint=False, in_channels_pre=None, **kwargs):
        super().__init__()
        n_features = [in_channels] + [f_maps * 2 ** k for k in range(num_blocks)]
        if in_channels_pre is None:
            in_channels_pre = in_channels

        self.encoders = nn.ModuleList()
        self.num_blocks = num_blocks
        # Encoder
        self.pre_conv = fvnn.SparseConv3d(in_channels_pre, in_channels, 1, 1) # a MLP to smooth the input
        for layer_idx in range(num_blocks):
            self.encoders.add_module(f'Enc{layer_idx}', SparseDoubleConv(
                n_features[layer_idx], 
                n_features[layer_idx + 1], 
                order, 
                num_groups,
                True, # if encoder branch
                pooling if layer_idx > 0 else None,
                use_checkpoint
            ))
        self.out_conv = fvnn.Linear(n_features[-1], out_channels)
            
    def encode(self, x: fvnn.VDBTensor, hash_tree: dict=None):
        x = self.pre_conv(x)
        for module in self.encoders:
            x, _ = module(x)
        return x
    
    def forward(self, x: fvnn.VDBTensor, hash_tree: dict=None):
        x = self.encode(x, hash_tree)
        out = self.out_conv(x)
        return out
    
class StructEncoder3D_v2(StructEncoder3D):
    def encode(self, x: fvnn.VDBTensor, hash_tree: dict):
        feat_depth = 0
        x = self.pre_conv(x)
        for module in self.encoders:
            x, feat_depth = module(x, hash_tree, feat_depth)
        return x
    
class StructEncoder3D_remain_h(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, f_maps=64, order='gcr', num_groups=8,
                 pooling='max', pooling_level=[], use_checkpoint=False, in_channels_pre=None, **kwargs):
        super().__init__()
        n_features = [in_channels] + [f_maps * 2 ** k for k in range(num_blocks)]
        if in_channels_pre is None:
            in_channels_pre = in_channels

        self.encoders = nn.ModuleList()
        self.num_blocks = num_blocks
        # Encoder
        self.pre_conv = fvnn.SparseConv3d(in_channels_pre, in_channels, 1, 1) # a MLP to smooth the input
        for layer_idx in range(num_blocks):
            if layer_idx in pooling_level:
                pooling_factor = [2, 2, 2]
            else:
                pooling_factor = [2, 2, 1]
            
            self.encoders.add_module(f'Enc{layer_idx}', SparseDoubleConv(
                n_features[layer_idx], 
                n_features[layer_idx + 1], 
                order, 
                num_groups,
                True, # if encoder branch
                pooling=pooling if layer_idx > 0 else None,
                pooling_factor=pooling_factor,
                use_checkpoint=use_checkpoint
            ))
        self.out_conv = fvnn.Linear(n_features[-1], out_channels)
            
    def encode(self, x: fvnn.VDBTensor):
        x = self.pre_conv(x)
        for module in self.encoders:
            x, _ = module(x)
        return x
    
    def forward(self, x: fvnn.VDBTensor, hash_tree: dict=None):
        x = self.encode(x)
        out = self.out_conv(x)
        return out