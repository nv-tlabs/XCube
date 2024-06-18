# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn as nn
import fvdb
from fvdb import JaggedTensor, GridBatch

from xcube.utils.embedder_util import get_embedder
from xcube.data.base import DatasetSpec as DS

class Encoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        encoder_input_dims = 3
        if self.hparams.use_input_normal:
            encoder_input_dims += 3
        if self.hparams.use_input_semantic:
            encoder_input_dims += self.hparams.dim_semantic
            self.semantic_embed_fn = nn.Embedding(self.hparams.num_semantic, self.hparams.dim_semantic)
        if self.hparams.use_input_color:
            encoder_input_dims += 3

        embed_fn, input_ch = get_embedder(5)
        self.pos_embedder = embed_fn
        
        input_dim = 0
        input_dim += input_ch
        if self.hparams.use_input_normal:
            input_dim += 3 # normal
        if self.hparams.use_input_intensity:
            input_dim += 1
        if self.hparams.use_input_semantic:
            input_dim += self.hparams.dim_semantic
        if self.hparams.use_input_color:
            input_dim += 3 # color
            
        self.mix_fc = nn.Linear(input_dim, self.hparams.network.encoder.c_dim)

    def forward(self, grid: GridBatch, batch) -> torch.Tensor:
        input_normal = batch[DS.TARGET_NORMAL] if DS.TARGET_NORMAL in batch.keys() else None
        if self.hparams.use_input_color:
            input_color = batch[DS.INPUT_COLOR]
        else:
            input_color = None          

        coords = grid.grid_to_world(grid.ijk.float()).jdata
        unet_feat = self.pos_embedder(coords)
        
        if self.hparams.use_input_normal:
            ref_grid = batch['input_grid']
            ref_xyz = ref_grid.grid_to_world(ref_grid.ijk.float()) 
            # splatting normal
            input_normal = grid.splat_trilinear(ref_xyz, fvdb.JaggedTensor(input_normal))
            # normalize normal
            input_normal.jdata /= (input_normal.jdata.norm(dim=1, keepdim=True) + 1e-6)
            unet_feat = torch.cat([unet_feat, input_normal.jdata], dim=1)

        if self.hparams.use_input_semantic:
            input_semantic = fvdb.JaggedTensor(batch[DS.GT_SEMANTIC])
            input_semantic_embed = self.semantic_embed_fn(input_semantic.jdata.long())
            unet_feat = torch.cat([unet_feat, input_semantic_embed], dim=1)

        if self.hparams.use_input_intensity:
            input_intensity = fvdb.JaggedTensor(batch[DS.INPUT_INTENSITY])
            unet_feat = torch.cat([unet_feat, input_intensity.jdata], dim=1)
            
        if self.hparams.use_input_color:
            input_color = fvdb.JaggedTensor(batch[DS.INPUT_COLOR])
            unet_feat = torch.cat([unet_feat, input_color.jdata], dim=1)

        unet_feat = self.mix_fc(unet_feat)
        return unet_feat