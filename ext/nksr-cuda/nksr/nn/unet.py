# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import torch
import torch.nn as nn
from nksr.svh import SparseFeatureHierarchy, VoxelStatus

import fvdb
from fvdb import JaggedTensor, GridBatch
import fvdb.nn as fvnn
from fvdb.nn import VDBTensor


class SparseConvBlock(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 order: str,
                 num_groups: int,
                 kernel_size: int = 3):
        super().__init__()
        for i, char in enumerate(order):
            if char == 'r':
                self.add_module('ReLU', fvnn.ReLU(inplace=True))
            elif char == 'l':
                self.add_module('LeakyReLU', fvnn.LeakyReLU(negative_slope=0.1, inplace=True))
            elif char == 'c':
                # add learnable bias only in the absence of batchnorm/groupnorm
                self.add_module(
                    'Conv', fvnn.SparseConv3d(in_channels, out_channels, kernel_size, 1, bias='g' not in order, transposed=False))
            elif char == 'g':
                if i < order.index('c'):
                    num_channels = in_channels
                else:
                    num_channels = out_channels

                # use only one group if the given number of groups is greater than the number of channels
                if num_channels < num_groups:
                    num_groups = 1

                assert num_channels % num_groups == 0, \
                    f'Expected number of channels in input to be divisible by num_groups. ' \
                    f'num_channels={num_channels}, num_groups={num_groups}'

                self.add_module('GroupNorm', fvnn.GroupNorm(num_groups=num_groups, num_channels=num_channels))

            else:
                raise NotImplementedError


class SparseDoubleConv(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 order: str,
                 num_groups: int,
                 encoder: bool):
        super().__init__()
        if encoder:
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        self.add_module('SingleConv1',
                        SparseConvBlock(conv1_in_channels, conv1_out_channels, order, num_groups))
        self.add_module('SingleConv2',
                        SparseConvBlock(conv2_in_channels, conv2_out_channels, order, num_groups))


class SparseHead(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 order: str,
                 num_groups: int,
                 enhanced: bool = False):
        super().__init__()
        self.add_module('SingleConv', SparseConvBlock(in_channels, in_channels, order, num_groups))
        if enhanced:
            mid_channels = min(64, in_channels)
            self.add_module('SingleConv2', SparseConvBlock(in_channels, mid_channels, order, num_groups))
            self.add_module('OneConv0', SparseConvBlock(mid_channels, mid_channels, order, num_groups,
                                                        kernel_size=1))
            self.add_module('OutConv', fvnn.SparseConv3d(mid_channels, out_channels, 1, bias=True))
        else:
            self.add_module('OutConv', fvnn.SparseConv3d(in_channels, out_channels, 1, bias=True))


class FeaturesSet:
    def __init__(self):
        self.structure_features = {}
        self.normal_features = {}
        self.basis_features = {}
        self.udf_features = {}

    # def populate_empty(self, depth: int, device, dtype,
    #                    structure_dim: int, normal_dim: int, basis_dim: int, udf_dim: int):
    #     for d in range(depth):
     #         if d not in self.structure_features and structure_dim > 0:
    #             self.structure_features[d] = torch.zeros((0, structure_dim), device=device, dtype=dtype)
    #         if d not in self.normal_features and normal_dim > 0:
    #             self.normal_features[d] = torch.zeros((0, normal_dim), device=device, dtype=dtype)
    #         if d not in self.basis_features and basis_dim > 0:
    #             self.basis_features[d] = torch.zeros((0, basis_dim), device=device, dtype=dtype)
    #         if d not in self.udf_features and udf_dim > 0:
    #             self.udf_features[d] = torch.zeros((0, udf_dim), device=device, dtype=dtype)


class SparseStructureNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_blocks: int,
                 basis_channels: int,
                 normal_channels: int = 3,
                 f_maps: int = 64,
                 order: str = 'gcr',
                 num_groups: int = 8,
                 neck_type: str = "dense",
                 neck_expand: int = 1,
                 udf_branch_dim: int = 16):

        super().__init__()

        n_features = [in_channels] + [f_maps * 2 ** k for k in range(num_blocks)]
        self.num_blocks = num_blocks
        self.neck_type = neck_type
        self.neck_expand = neck_expand
        self.basis_channels = basis_channels
        self.normal_channels = normal_channels
        self.udf_branch_dim = udf_branch_dim

        self.downsamplers = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.struct_heads = nn.ModuleList()
        self.normal_heads = nn.ModuleList()
        self.basis_heads = nn.ModuleList()

        if self.udf_branch_dim == 0:
            self.udf_heads = [None for _ in range(num_blocks)]
        else:
            self.udf_heads = nn.ModuleList()

        for layer_idx in range(num_blocks):
            self.encoders.add_module(f'Enc{layer_idx}', SparseDoubleConv(
                n_features[layer_idx],
                n_features[layer_idx + 1], order, num_groups, True
            ))

        for layer_idx in range(1, num_blocks):
            self.downsamplers.add_module(f'Down{layer_idx}', fvnn.MaxPool(kernel_size=2))

        for layer_idx in range(-1, -num_blocks - 1, -1):
            self.struct_heads.add_module(f'Struct{layer_idx}', SparseHead(
                n_features[layer_idx], 3, order, num_groups))

            if self.udf_branch_dim > 0:
                self.udf_heads.add_module(f'UDF{layer_idx}', SparseHead(
                    n_features[layer_idx],
                    self.udf_branch_dim,
                    order, num_groups
                ))
            if self.normal_channels == 0:
                self.normal_heads.add_module(f'Normal{layer_idx}', None)
            else:
                self.normal_heads.add_module(f'Normal{layer_idx}', SparseHead(
                    n_features[layer_idx],
                    self.normal_channels,
                    order, num_groups
                ))
            self.basis_heads.add_module(f'Basis{layer_idx}', SparseHead(
                n_features[layer_idx],
                self.basis_channels,
                order, num_groups, enhanced=True
            ))

            if layer_idx < -1:
                self.decoders.add_module(f'Dec{layer_idx}', SparseDoubleConv(
                    n_features[layer_idx + 1] + n_features[layer_idx],
                    n_features[layer_idx], order, num_groups, False
                ))
                up_module = fvnn.UpsamplingNearest(2)
                self.upsamplers.add_module(f'Up{layer_idx}', up_module)

        self.padding = fvnn.FillToGrid()

    def build_neck_grid(self, sparse_grid: GridBatch, dense_svh: SparseFeatureHierarchy):
        sparse_coords = sparse_grid.ijk
        n_padding = (self.neck_expand - 1) // 2

        if self.neck_type == "dense":
            all_coords = []
            for b in range(sparse_grid.grid_count):
                min_bound = torch.min(sparse_coords[b].jdata, dim=0).values.cpu().numpy() - n_padding
                max_bound = torch.max(sparse_coords[b].jdata, dim=0).values.cpu().numpy() + 1 + n_padding
                cx = torch.arange(min_bound[0], max_bound[0], dtype=torch.int32, device=sparse_coords.device)
                cy = torch.arange(min_bound[1], max_bound[1], dtype=torch.int32, device=sparse_coords.device)
                cz = torch.arange(min_bound[2], max_bound[2], dtype=torch.int32, device=sparse_coords.device)
                coords = torch.stack(torch.meshgrid(cx, cy, cz, indexing='ij'), dim=3).view(-1, 3)
                all_coords.append(coords)
            all_coords = JaggedTensor(all_coords)
            dense_svh.build_from_grid_coords(dense_svh.depth - 1, all_coords)

        else:
            dense_svh.build_from_grid_coords(dense_svh.depth - 1, sparse_coords,
                                             [-n_padding, -n_padding, -n_padding],
                                             [n_padding, n_padding, n_padding])

    def forward(self,
                feat: JaggedTensor,
                encoder_svh: SparseFeatureHierarchy,
                adaptive_depth: int,
                gt_decoder_svh: SparseFeatureHierarchy = None):

        res = FeaturesSet()
        feat_depth = 0
        vdb_tensor = VDBTensor(encoder_svh.grids[feat_depth], feat)

        # Down-sample
        encoder_features = {}
        for module, downsampler in zip(self.encoders, [None] + list(self.downsamplers)):
            if downsampler is not None:
                vdb_tensor = downsampler(vdb_tensor, ref_coarse_data=encoder_svh.grids[feat_depth + 1])
                feat_depth += 1
            vdb_tensor = module(vdb_tensor)
            encoder_features[feat_depth] = vdb_tensor

        # Bottleneck processing
        decoder_svh = SparseFeatureHierarchy(
            encoder_svh.voxel_size, encoder_svh.depth, encoder_svh.device
        )
        decoder_tmp_svh = SparseFeatureHierarchy(
            encoder_svh.voxel_size, encoder_svh.depth, encoder_svh.device
        )

        self.build_neck_grid(encoder_svh.grids[feat_depth], decoder_tmp_svh)
        vdb_tensor = self.padding(vdb_tensor, decoder_tmp_svh.grids[feat_depth])

        # Up-sample
        upsample_mask = None
        for module, upsampler, struct_conv, normal_conv, basis_conv, udf_conv in \
                zip([None] + list(self.decoders), [None] + list(self.upsamplers),
                    self.struct_heads, self.normal_heads, self.basis_heads, self.udf_heads):

            if module is not None:
                vdb_tensor = upsampler(vdb_tensor, mask=upsample_mask)
                feat_depth -= 1
                decoder_tmp_svh.build_from_grid(feat_depth, vdb_tensor.grid)

                enc_feat = self.padding(encoder_features[feat_depth], vdb_tensor)
                vdb_tensor = VDBTensor.cat([enc_feat, vdb_tensor], dim=1)

                vdb_tensor = module(vdb_tensor)

            # Do structure inference.
            res.structure_features[feat_depth] = struct_conv(vdb_tensor).feature

            if udf_conv is not None:
                res.udf_features[feat_depth] = udf_conv(vdb_tensor).feature

            if gt_decoder_svh is None:
                struct_decision = torch.argmax(res.structure_features[feat_depth].jdata, dim=1).byte()
            else:
                struct_decision = gt_decoder_svh.evaluate_voxel_status(
                    decoder_tmp_svh.grids[feat_depth], feat_depth)

            exist_mask = struct_decision != VoxelStatus.VS_NON_EXIST.value

            # If the predicted structure is empty, then stop early.
            #   (Related branch will not have gradient)
            if not torch.any(exist_mask):
                break

            dec_ijk = decoder_tmp_svh.grids[feat_depth].ijk.r_masked_select(exist_mask)
            decoder_svh.build_from_grid_coords(feat_depth, dec_ijk)

            vdb_tensor = self.padding(vdb_tensor, decoder_svh.grids[feat_depth])
            upsample_mask = decoder_svh.grids[feat_depth].fill_to_grid(
                (struct_decision == VoxelStatus.VS_EXIST_CONTINUE.value).float(),
                decoder_tmp_svh.grids[feat_depth]
            ).type(torch.bool)

            # Do normal&basis prediction.
            if feat_depth < adaptive_depth and normal_conv is not None:
                res.normal_features[feat_depth] = normal_conv(vdb_tensor).feature
            res.basis_features[feat_depth] = basis_conv(vdb_tensor).feature

            # If next level is for sure empty, then stop earlier
            if not torch.any(upsample_mask.jdata):
                break

        # res.populate_empty(
        #     self.num_blocks, feat.device, feat.dtype,
        #     3, self.normal_channels, self.basis_channels, self.udf_branch_dim
        # )
        return res, decoder_svh, decoder_tmp_svh
