# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


from typing import Union

import torch.nn
from omegaconf.omegaconf import DictConfig
from pycg import exp
import fvdb
from fvdb import GridBatch, JaggedTensor

from nksr.ext import kernel_eval, sparse_solve
from nksr.fields.base_field import BaseField, EvaluationResult
from nksr.solver import SparseMatrix, SparseMatrixBatch
from nksr.svh import SparseFeatureHierarchy, GridBatch
from nksr.utils import points_voxel_downsample
from nksr.scatter import scatter_sum


class QGMatrix:
    """ Used to hold pre-computed Q&G matrices: row->pts, col->vx """
    def __init__(self, values: torch.Tensor, indexer: torch.Tensor, grid: GridBatch):
        assert values.size(0) == indexer.size(0)
        assert values.size(1) == indexer.size(1) == 27

        self.grid = grid
        self.n_row = values.size(0)
        self.n_col = self.grid.num_voxels

        # Compress matrix (also indices using 32-bit)
        pts_inds, vx_local_inds = torch.where(indexer != -1)
        vx_inds = indexer[pts_inds, vx_local_inds]
        self.values = values[pts_inds, vx_local_inds]
        self.row_ptr = sparse_solve.ind2ptr(pts_inds, self.n_row).int()
        self.col_inds = vx_inds

    def transposed_vecmul(self, target: torch.Tensor):
        row_inds = sparse_solve.ptr2ind(self.row_ptr, self.col_inds.size(0)).long()
        return scatter_sum(
            torch.sum(target[row_inds] * self.values, dim=1),
            self.col_inds.long(), dim_size=self.n_col
        )

    @classmethod
    def adjoint_matmul(cls, lhs: "QGMatrix", rhs: "QGMatrix", out_indexer: torch.Tensor, num_entries: int):
        return kernel_eval.csr_matrix_multiplication(
            lhs.grid._grid, rhs.grid._grid,
            lhs.grid.active_grid_coords(), rhs.grid.active_grid_coords(),
            lhs.values, rhs.values,
            lhs.row_ptr, rhs.row_ptr,
            lhs.col_inds, rhs.col_inds,
            out_indexer, num_entries
        )[0]


class KernelField(BaseField):
    """ A continuous field represented by kernels """
    def __init__(self,
                 svh: SparseFeatureHierarchy,
                 interpolator: torch.nn.ModuleDict = None,
                 features: dict = None,
                 approx_kernel_grad: bool = False,
                 balanced_kernel: bool = False,
                 solver_max_iter: int = 2000,
                 solver_tol: float = 1.0e-5):
        super().__init__(svh)
        self.interpolator = interpolator
        self.features = features
        if self.features is not None:
            assert len(self.features) == len(self.interpolator) == self.svh.depth
        self.approx_kernel_grad = approx_kernel_grad
        self.balanced_kernel = balanced_kernel
        self.solutions = {}

        # Pre-evaluate grid kernel to speed up.
        if self.balanced_kernel:
            self.grid_kernel = {
                d: self.evaluate_kernel(
                    self.svh.get_voxel_centers(d), d) for d in range(self.svh.depth)
            }
        else:
            # Add more representation power using non-linear kernels
            self.grid_kernel = {
                d: self.features[d] if self.features is not None else
                torch.zeros((self.svh.grids[d].num_voxels, 0), **self.torch_kwargs)
                for d in range(self.svh.depth)
            }

        self.solver_config = DictConfig({
            'max_iter': solver_max_iter, 'tol': solver_tol, 'verbose': False
        })

    @property
    def torch_kwargs(self):
        return {'dtype': torch.float32, 'device': self.svh.device}

    def evaluate_kernel(self, xyz: JaggedTensor, depth: int, grad: bool = False):
        grid = self.svh.grids[depth]

        do_compute_grad = grad and not self.approx_kernel_grad
        if self.interpolator is not None:
            interp_res = self.interpolator[str(depth)].interpolate(
                xyz, grid, self.features[depth], do_compute_grad)
        else:
            interp_res = grid.jagged_like(torch.zeros((xyz.size(0), 0), **self.torch_kwargs))
        if isinstance(interp_res, JaggedTensor) and grad:
            approx_grad_kernel_pts = JaggedTensor([
                torch.zeros((0, interp_res.jdata.size(1), 3), **self.torch_kwargs)])
            return interp_res, approx_grad_kernel_pts
        else:
            return interp_res

    def to_(self, device: Union[torch.device, str]):
        super().to_(device)
        if self.features is not None:
            self.features = {k: v.to(device) for k, v in self.features.items()}
        self.solutions = {k: v.to(device) for k, v in self.solutions.items()}
        self.grid_kernel = {k: v.to(device) if v is not None else None
                            for k, v in self.grid_kernel.items()}

    def evaluate_f_depth(self, xyz: JaggedTensor, depth: int, grad: bool = False):
        grid = self.svh.grids[depth]
        if grid is None:
            return EvaluationResult.zero(grad)
        if grad:
            xyz_kernel, grad_kernel_xyz = self.evaluate_kernel(xyz, depth, grad=True)
            f_depth, f_depth_grad = kernel_eval.kernel_evaluation_grad(
                grid, xyz, xyz_kernel, self.grid_kernel[depth], self.solutions[depth], grad_kernel_xyz)
        else:
            xyz_kernel = self.evaluate_kernel(xyz, depth, grad=False)
            f_depth, f_depth_grad = kernel_eval.kernel_evaluation(
                grid, xyz, xyz_kernel, self.grid_kernel[depth], self.solutions[depth]), None
        return EvaluationResult(f_depth, f_depth_grad)

    def evaluate_f(self, xyz: JaggedTensor, grad: bool = False):
        f = EvaluationResult.zero(grad)
        for d in range(self.svh.depth):
            f_depth = self.evaluate_f_depth(xyz, d, grad)
            f = f + f_depth
        return f

    def solve_non_fused(self, pos_xyz: torch.Tensor, normal_xyz: torch.Tensor, normal_value: torch.Tensor,
                        pos_weight: float = 1.0, normal_weight: float = 1.0, reg_weight: float = 0.0):

        assert pos_weight > 0.0 and normal_weight > 0.0, "data weights have to be >0!"
        assert normal_xyz.size(0) == normal_value.size(0)

        self.solutions = {}
        grids = self.svh.grids

        # Evaluate kernels at custom positions
        pos_kernel = {d: self.evaluate_kernel(pos_xyz, d) for d in range(self.svh.depth)}
        normal_kernel = {d: self.evaluate_kernel(normal_xyz, d, True) for d in range(self.svh.depth)}

        # Pre-build Q and G matrices
        q_matrices, g_matrices = {}, {}
        for d in range(self.svh.depth):
            if grids[d] is None:
                continue

            g_value, g_indexer = kernel_eval.qg_building(
                grids[d]._grid, pos_xyz,
                pos_kernel[d], self.grid_kernel[d],
                torch.zeros((0, 0, 3), **self.torch_kwargs), False)
            g_matrices[d] = QGMatrix(g_value, g_indexer, grids[d])

            q_value, q_indexer = kernel_eval.qg_building(
                grids[d]._grid, normal_xyz,
                normal_kernel[d][0], self.grid_kernel[d],
                normal_kernel[d][1], True)
            q_matrices[d] = QGMatrix(q_value, q_indexer, grids[d])

        lhs_mat = SparseMatrix(self.svh.depth)
        rhs_vec = {}

        for d in range(self.svh.depth - 1, -1, -1):
            if grids[d] is None:
                continue

            rhs_vec[d] = normal_weight * q_matrices[d].transposed_vecmul(normal_value)
            for dd in range(self.svh.depth - 1, d - 1, -1):
                if grids[dd] is None:
                    continue

                mat_indexer = kernel_eval.build_coo_indexer(grids[d]._grid, grids[dd]._grid)

                with exp.pt_profile_named("mat-indexer"):
                    d_inds, dd_local_inds = torch.where(mat_indexer != -1)
                    dd_inds = mat_indexer[d_inds, dd_local_inds]
                    mat_indexer_type = torch.long if d_inds.size(0) > 2e9 else torch.int32
                    mat_indexer = mat_indexer.type(mat_indexer_type)
                    mat_indexer[d_inds, dd_local_inds] = torch.arange(
                        d_inds.size(0), device=self.svh.device, dtype=mat_indexer_type)
                    del dd_local_inds
                    d_inds = d_inds.int()

                gtg = QGMatrix.adjoint_matmul(
                    g_matrices[d], g_matrices[dd], mat_indexer, d_inds.size(0)
                )
                qtq = QGMatrix.adjoint_matmul(
                    q_matrices[d], q_matrices[dd], mat_indexer, d_inds.size(0)
                )

                lhs = pos_weight * gtg + normal_weight * qtq
                if d == dd and reg_weight > 0.0:
                    lhs += reg_weight * kernel_eval.k_building(
                        grids[d]._grid, self.grid_kernel[d], mat_indexer, d_inds.size(0)
                    )[0]
                lhs_mat.add_block(
                    d, dd, grids[d].num_voxels, grids[dd].num_voxels,
                    d_inds, dd_inds, lhs)

        self.solutions = lhs_mat.solve(rhs_vec, self.solver_config)

    def solve(self, pos_xyz: JaggedTensor, normal_xyz: JaggedTensor, normal_value: JaggedTensor,
              pos_weight: torch.Tensor, normal_weight: torch.Tensor, reg_weight: float = 0.0,
              nystrom_min_depth: int = 100):

        assert torch.all(pos_weight > 0.0) and torch.all(normal_weight > 0.0), "data weights have to be >0!"
        assert torch.all(normal_xyz.joffsets == normal_value.joffsets)

        self.solutions = {}
        grids = self.svh.grids

        # Evaluate kernels at custom positions
        pos_kernel = {d: self.evaluate_kernel(pos_xyz, d) for d in range(self.svh.depth)}
        normal_kernel = {d: self.evaluate_kernel(normal_xyz, d, True) for d in range(self.svh.depth)}

        lhs_mat = SparseMatrixBatch(grids[-1].grid_count, self.svh.depth)
        rhs_vec = {}
        for d in range(self.svh.depth - 1, -1, -1):

            rhs_vec[d] = kernel_eval.rhs_evaluation(
                grids[d], normal_xyz, normal_kernel[d][0],
                self.grid_kernel[d], normal_kernel[d][1], normal_value)
            rhs_vec[d].jdata *= normal_weight[rhs_vec[d].jidx.int()]

            # Allow for nystrom sampling (keep 2^3 points in each voxel)
            if d >= nystrom_min_depth:
                pos_xyz_d = points_voxel_downsample(pos_xyz, grids[d].voxel_size / 2)
                pos_kernel_d = {dd: self.evaluate_kernel(pos_xyz_d, dd) for dd in range(d, self.svh.depth)}
                pos_weight_d = pos_weight * pos_xyz.size(0) / pos_xyz_d.size(0)
            else:
                pos_xyz_d, pos_kernel_d, pos_weight_d = pos_xyz, pos_kernel, pos_weight

            for dd in range(self.svh.depth - 1, d - 1, -1):

                with exp.pt_profile_named("coo-indexer"):
                    mat_indexer = kernel_eval.build_coo_indexer(grids[d], grids[dd])

                with exp.pt_profile_named("mat-indexer"):
                    mat_mask = mat_indexer.jagged_like(mat_indexer.jdata != -1)
                    nnz_count = [mat_mask[b].jdata.sum().item() for b in range(grids[d].grid_count)]
                    ref_mat = fvdb.JaggedTensor([torch.empty(t, 0, device=grids[d].device) for t in nnz_count])
                    d_inds, dd_local_inds = torch.where(mat_mask.jdata)
                    del mat_mask, nnz_count

                    dd_inds = mat_indexer.jdata[d_inds, dd_local_inds]
                    if d_inds.size(0) > 2e9 or self.device.type == "cpu":
                        # Use 64-bit indexing on CPU or if the matrix is too large
                        mat_indexer_type = torch.long
                    else:
                        mat_indexer_type = torch.int32
                    mat_indexer = mat_indexer.type(mat_indexer_type)
                    mat_indexer.jdata[d_inds, dd_local_inds] = torch.arange(
                        d_inds.size(0), device=self.svh.device, dtype=mat_indexer_type)
                    del dd_local_inds
                    d_inds = d_inds.int()

                gtg = kernel_eval.matrix_building(
                    grids[d], grids[dd],
                    pos_xyz_d, pos_kernel_d[d], pos_kernel_d[dd],
                    self.grid_kernel[d], self.grid_kernel[dd],
                    JaggedTensor([torch.zeros((0, 0, 3), **self.torch_kwargs)]),
                    JaggedTensor([torch.zeros((0, 0, 3), **self.torch_kwargs)]),
                    mat_indexer,
                    False, ref_mat
                )

                qtq = kernel_eval.matrix_building(
                    grids[d], grids[dd],
                    normal_xyz, normal_kernel[d][0], normal_kernel[dd][0],
                    self.grid_kernel[d], self.grid_kernel[dd],
                    normal_kernel[d][1], normal_kernel[dd][1],
                    mat_indexer,
                    True, ref_mat
                )

                lhs = gtg * pos_weight_d[gtg.jidx.int()] + qtq * normal_weight[qtq.jidx.int()]
                if d == dd and reg_weight > 0.0:
                    lhs += kernel_eval.k_building(
                        grids[d], self.grid_kernel[d], mat_indexer, ref_mat
                    ) * reg_weight
                lhs_mat.add_block(
                    d, dd,
                    grids[d].num_voxels.tolist(), 
                    grids[dd].num_voxels.tolist(),
                    d_inds, dd_inds, lhs)

        self.solutions = lhs_mat.solve(rhs_vec, self.solver_config)
