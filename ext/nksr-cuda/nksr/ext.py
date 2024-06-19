# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from typing import List

import torch
import fvdb
from fvdb import GridBatch

from nksr import _C

kernel_eval = _C.kernel_eval
sparse_solve = _C.sparse_solve
meshing = _C.meshing
pcproc = _C.pcproc


def _build_joint_dual_grid(primal_grids: List[GridBatch]):
    grid_scales = [1]
    for primal_grid in primal_grids[1:]:
        cur_scale = primal_grid.voxel_sizes / primal_grids[0].voxel_sizes
        assert cur_scale.size(0) == 1, "Only support single batch element!"
        cur_scale = round(cur_scale[0, 0].item())
        grid_scales.append(cur_scale)

    dual_ijks = []
    for primal_grid, primal_scale in zip(primal_grids, grid_scales):
        ijk = primal_grid.ijk.jdata
        ijk = (ijk.reshape(-1, 1, 3) + torch.tensor([[
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
        ]], device=primal_grid.device, dtype=ijk.dtype)) * primal_scale
        dual_ijks.append(ijk.view(-1, 3))

    return fvdb.sparse_grid_from_ijk(
        torch.cat(dual_ijks, dim=0),
        voxel_sizes=primal_grids[0].voxel_sizes,
        origins=primal_grids[0].origins - 0.5 * primal_grids[0].voxel_sizes,
    )


meshing.build_joint_dual_grid = _build_joint_dual_grid
