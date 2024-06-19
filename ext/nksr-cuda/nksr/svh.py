# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import torch
import numpy as np
from typing import Dict, Tuple, List, Union, Optional
from enum import Enum

import fvdb
from fvdb import GridBatch
from nksr.utils import get_device, Device
from pycg.isometry import Isometry


class VoxelStatus(Enum):
    # Voxel Status: 0-NE, 1-E&-, 2-E&v
    VS_NON_EXIST = 0
    VS_EXIST_STOP = 1
    VS_EXIST_CONTINUE = 2


class SparseFeatureHierarchy:
    """ A hierarchy of batched sparse grids, where voxel corners align with the origin """
    def __init__(self, voxel_size: float, depth: int, device):
        self.device = device
        self.voxel_size = voxel_size
        self.depth = depth
        self.grids: List[GridBatch] = [fvdb.GridBatch(self.device) for _ in range(self.depth)]

    @staticmethod
    def from_grid_list(grids: List[GridBatch]) -> "SparseFeatureHierarchy":
        voxel_size = grids[0].voxel_sizes[0, 0].item()
        depth = len(grids)
        svh = SparseFeatureHierarchy(voxel_size, depth, grids[0].device)
        svh.grids = [grids[d] for d in range(depth)]
        return svh

    def __repr__(self):
        text = f"SparseFeatureHierarchy - {self.depth} layers, Voxel size = {self.voxel_size}"
        text += '\n'
        for d, d_grid in enumerate(self.grids):
            text += f"\t[{d} {d_grid.total_voxels} voxels in {d_grid.grid_count} batch elements]"
        return text + "\n"

    def get_grid_voxel_size_origin(self, depth: int):
        assert 0 <= depth < self.depth, f"Invalid depth {depth}."
        return self.voxel_size * (2 ** depth), 0.5 * self.voxel_size * (2 ** depth)

    def get_voxel_centers(self, depth: int):
        grid = self.grids[depth]
        return grid.grid_to_world(grid.ijk.float())

    def get_f_bound(self):
        grid = self.grids[self.depth - 1]
        assert grid.grid_count == 1, "Only support single batch element!"
        grid_coords = grid.ijk.float().jdata
        min_extent = grid.grid_to_world(torch.min(grid_coords, dim=0).values.unsqueeze(0) - 1.5).jdata[0]
        max_extent = grid.grid_to_world(torch.max(grid_coords, dim=0).values.unsqueeze(0) + 1.5).jdata[0]
        return min_extent, max_extent

    def evaluate_voxel_status(self, grid: GridBatch, depth: int):
        """
        Evaluate the voxel status of given coordinates
        :param grid: Featuregrid Grid
        :param depth: int
        :return: (N, ) byte tensor, with value 0,1,2
        """
        status = torch.full((grid.total_voxels,), VoxelStatus.VS_NON_EXIST.value,
                            dtype=torch.uint8, device=self.device)

        exist_idx = grid.ijk_to_index(self.grids[depth].ijk).jdata
        status[exist_idx[exist_idx != -1]] = VoxelStatus.VS_EXIST_STOP.value

        if depth > 0:
            child_grid = self.grids[depth - 1]
            child_coords = torch.div(child_grid.ijk.jdata, 2, rounding_mode='floor')
            child_idx = grid.ijk_to_index(child_grid.jagged_like(child_coords)).jdata
            status[child_idx[child_idx != -1]] = VoxelStatus.VS_EXIST_CONTINUE.value

        return status

    def get_test_grid(self, depth: int = 0, resolution: int = 2):
        grid = self.grids[depth]
        primal_coords = grid.ijk
        box_coords = torch.linspace(-0.5, 0.5, resolution, device=self.device)
        box_coords = torch.stack(torch.meshgrid(box_coords, box_coords, box_coords, indexing='ij'), dim=3)
        box_coords = box_coords.view(-1, 3)
        query_pos = primal_coords.jdata.unsqueeze(1) + box_coords.unsqueeze(0)
        query_pos = fvdb.JaggedTensor.from_data_and_offsets(
            query_pos.view(-1, 3), primal_coords.joffsets * box_coords.size(0))
        return grid.grid_to_world(query_pos), primal_coords

    def get_visualization(self, batch_idx: int = 0):
        from pycg import vis

        wire_blocks = []
        for d in range(self.depth):
            target_grid = self.grids[d][batch_idx]
            primal_coords = target_grid.ijk.float()
            is_lowest = len(wire_blocks) == 0
            wire_blocks.append(vis.wireframe_bbox(
                target_grid.grid_to_world(primal_coords - (0.45 if is_lowest else 0.5)).jdata,
                target_grid.grid_to_world(primal_coords + (0.45 if is_lowest else 0.5)).jdata,
                ucid=d, solid=is_lowest
            ))
        return wire_blocks

    @classmethod
    def joined(cls, svhs: List["SparseFeatureHierarchy"], transforms: List[Isometry]):
        ref_svh = svhs[0]
        inst = cls(ref_svh.voxel_size, ref_svh.depth, ref_svh.device)
        for d in range(ref_svh.depth):
            d_samples = []
            vs, orig = inst.get_grid_voxel_size_origin(d)
            for svh, iso in zip(svhs, transforms):
                grid = svh.grids[d]
                assert grid.grid_count == 1, "Only support single batch element!"
                test_pos = torch.linspace(-0.5, 0.5, 3, device=ref_svh.device)
                test_pos = torch.stack(torch.meshgrid(test_pos, test_pos, test_pos, indexing='ij'), dim=3)
                test_pos = test_pos.view(-1, 3) * 0.99
                test_pos = grid.ijk.jdata.unsqueeze(1) + test_pos.unsqueeze(0)
                test_pos = iso @ grid.grid_to_world(test_pos.view(-1, 3)).jdata
                test_pos = ((test_pos - orig) / vs).round().long()
                test_pos = torch.unique(test_pos, dim=0)
                d_samples.append(test_pos)
            d_samples = torch.unique(torch.cat(d_samples, dim=0), dim=0)
            inst.build_from_grid_coords(d, d_samples)
        return inst

    def to_(self, device: Device):
        device = torch.device(device)
        if device == self.device:
            return
        self.device = device
        self.grids = [v.to(device) for v in self.grids]

    def build_iterative_coarsening(self, pts: fvdb.JaggedTensor):
        assert pts.device == self.device, f"Device not match {pts.device} vs {self.device}."
        vs, vo = self.get_grid_voxel_size_origin(0)
        self.grids[0] = fvdb.sparse_grid_from_points(
            pts, voxel_sizes=[vs] * 3, origins=[vo] * 3)
        for d in range(1, self.depth):
            self.grids[d] = self.grids[d - 1].coarsened_grid(2)

    def build_point_splatting(self, pts: fvdb.JaggedTensor):
        assert pts.device == self.device, f"Device not match {pts.device} vs {self.device}."
        for d in range(self.depth):
            vs, vo = self.get_grid_voxel_size_origin(d)
            self.grids[d] = fvdb.sparse_grid_from_nearest_voxels_to_points(
                pts, voxel_sizes=[vs] * 3, origins=[vo] * 3)
            
    def build_grid_splatting(self, grid: fvdb.GridBatch):
        assert grid.device == self.device, f"Device not match {grid.device} vs {self.device}."
        pts = grid.grid_to_world(grid.ijk.float())
        for d in range(self.depth):
            vs, vo = self.get_grid_voxel_size_origin(d)
            if d == 0:
                # Make sure no excessive grids are generated.
                self.grids[d] = fvdb.sparse_grid_from_points(
                    pts, voxel_sizes=[vs] * 3, origins=[vo] * 3)
            else:
                self.grids[d] = fvdb.sparse_grid_from_nearest_voxels_to_points(
                    pts, voxel_sizes=[vs] * 3, origins=[vo] * 3)

    def build_adaptive_normal_variation(self, pts: fvdb.JaggedTensor, normal: fvdb.JaggedTensor,
                                        tau: float = 0.2, adaptive_depth: int = 100):
        from nksr.scatter import scatter_std
        
        assert pts.device == normal.device == self.device, "Device not match"
        inv_mapping = None
        for d in range(self.depth - 1, -1, -1):
            # Obtain points & normals for this level
            if inv_mapping is not None:
                pts, normal, jidx = pts.jdata, normal.jdata, pts.jidx
                nx, ny, nz = torch.abs(normal[:, 0]), torch.abs(normal[:, 1]), torch.abs(normal[:, 2])
                vnx = scatter_std(nx, inv_mapping)
                vny = scatter_std(ny, inv_mapping)
                vnz = scatter_std(nz, inv_mapping)
                pts_mask = ((vnx + vny + vnz) > tau)[inv_mapping]
                pts, normal, jidx = pts[pts_mask], normal[pts_mask], jidx[pts_mask]
                pts = fvdb.JaggedTensor.from_data_and_offsets(pts, jidx)
                normal = fvdb.JaggedTensor.from_data_and_offsets(normal, jidx)

            if pts.jdata.size(0) == 0:
                return

            vs, vo = self.get_grid_voxel_size_origin(d)
            self.grids[d] = fvdb.sparse_grid_from_nearest_voxels_to_points(
                pts, voxel_sizes=[vs] * 3, origins=[vo] * 3)

            if 0 < d < adaptive_depth:
                inv_mapping = self.grids[d].ijk_to_index(self.grids[d].world_to_grid(pts).round().int()).jdata

    def build_from_grid_coords(self, depth: int, grid_coords: fvdb.JaggedTensor,
                               pad_min: list = None, pad_max: list = None):
        if pad_min is None:
            pad_min = [0, 0, 0]

        if pad_max is None:
            pad_max = [0, 0, 0]

        assert grid_coords.device == self.device, "Device not match"
        assert self.grids[depth].total_voxels == 0, "Grid is not empty"
        vs, vo = self.get_grid_voxel_size_origin(depth)
        self.grids[depth] = fvdb.sparse_grid_from_ijk(
            grid_coords, pad_min, pad_max, voxel_sizes=[vs] * 3, origins=[vo] * 3)

    def build_from_grid(self, depth: int, grid: GridBatch):
        assert self.grids[depth].total_voxels == 0, "Grid is not empty"
        self.grids[depth] = grid
