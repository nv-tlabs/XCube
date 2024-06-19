# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import tqdm
import math
from pathlib import Path
from typing import Any, List, Union, Mapping, Optional, Callable

import fvdb
from fvdb import JaggedTensor
import torch
from pycg.isometry import Isometry
from pycg.exp import logger

from nksr.configs import get_hparams, load_checkpoint_from_url
from nksr.nn.unet import SparseStructureNet
from nksr.nn.encdec import PointEncoder, MultiscalePointDecoder
from nksr.interpolator import MLPFeatureInterpolator
from nksr.svh import SparseFeatureHierarchy
from nksr.fields import KernelField, NeuralField, FusedField, LayerField
from nksr.utils import split_into_chunks, get_device, Device

__version__ = '1.0.3'
__version_info__ = (1, 0, 3)


# Handle import * logic
__all__ = [
    'Reconstructor', 'Isometry',
    'KernelField', 'NeuralField', 'FusedField', 'LayerField',
    'split_into_chunks'
]


class NKSRNetwork(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        if 'encoder_dim' in hparams:
            encoder_dim = hparams.encoder_dim
        else:
            encoder_dim = 3 if hparams.feature == 'none' else 6

        self.encoder = PointEncoder(dim=encoder_dim)

        if 'normal_dim' in hparams:
            normal_channels = hparams.normal_dim
        else:
            if self.hparams.geometry == 'kernel':
                self.interpolators = torch.nn.ModuleDict({
                    str(d): MLPFeatureInterpolator(
                        theta_dim=self.hparams.kernel_dim,
                        n_hidden=self.hparams. interpolator.n_hidden,
                        hidden_dim=self.hparams.interpolator.hidden_dim
                    )
                    for d in range(self.hparams.tree_depth)
                })
                normal_channels = 3
            elif self.hparams.geometry == 'neural':
                self.sdf_decoder = MultiscalePointDecoder(
                    c_each_dim=self.hparams.kernel_dim,
                    multiscale_depths=self.hparams.tree_depth,
                    coords_depths=[2, 3]
                )
                normal_channels = 0
            else:
                raise NotImplementedError

        if 'neck_pad' in hparams:
            neck_pad = hparams.neck_pad
            neck_type = "sparse"
        else:
            neck_pad, neck_type = 1, "dense"

        self.unet = SparseStructureNet(
            in_channels=32,
            num_blocks=self.hparams.tree_depth,
            basis_channels=self.hparams.kernel_dim,
            normal_channels=normal_channels,
            f_maps=self.hparams.unet.f_maps,
            udf_branch_dim=16 if self.hparams.udf.enabled else 0,
            neck_type=neck_type, neck_expand=neck_pad
        )

        if self.hparams.udf.enabled:
            self.udf_decoder = MultiscalePointDecoder(
                c_each_dim=16,
                multiscale_depths=self.hparams.tree_depth,
                out_init=5 * self.hparams.voxel_size,
                coords_depths=[2, 3]
            )

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        new_state_dict = {}
        # Compatibility with old weights
        for key, value in state_dict.items():
            if key.endswith(".kernel"):
                key = key.replace(".kernel", ".weight")
                if value.size(0) == 27:
                    d1, d2 = value.size(1), value.size(2)
                    value = value.view(3, 3, 3, d1, d2).permute(4, 3, 2, 1, 0)
                elif value.ndim == 2:
                    value = value.permute(1, 0)
            new_state_dict[key] = value
        return super().load_state_dict(new_state_dict, strict)


def default_preprocess(xyz: torch.Tensor, normal: torch.Tensor, sensor: torch.Tensor):
    return xyz, normal, sensor


def get_estimate_normal_preprocess_fn(knn: int, drop_threshold_degrees: float = 90.0) -> Callable:
    """

    Obtain a functor that estimates per-point normals given sensor positions.
    The function should be used along with the :code:`nksr.Reconstructor` class, for example:

    >>> reconstructor = nksr.Reconstructor(..., preprocess_fn=get_estimate_normal_preprocess_fn(knn=64))

    Sensor origins will be automatically converted into normals using this functor.

    Args:
        knn (int): the number of nearest neighbour used to estimate normals
        drop_threshold_degrees (float): maximum angles (in degrees) between point-sensor and normal to be
            considered as inliers. Points with angles larger than this threshold will be disgarded.

    Returns:
        fn (Callable): preprocessing functor.
    """
    assert knn > 0, "normal estimation knn should be positive integer!"

    def func(xyz: torch.Tensor, normal: torch.Tensor, sensor: torch.Tensor):
        assert normal is None, "normal already exists"
        assert sensor is not None, "please provide sensor positions for consistent orientations"

        if knn > xyz.size(0):
            return None

        normal, sub_indices = utils.estimate_normals(
            xyz, sensor,
            knn=knn, drop_threshold_degrees=drop_threshold_degrees
        )
        xyz = xyz[sub_indices]
        return xyz, normal, None

    return func


class Reconstructor:
    """
    Main Reconstructor class
    """
    def __init__(self, device: Device, config: Union[str, Mapping] = 'ks'):
        """
        Args:
            device (torch.device): device to run the reconstructor on.
            config (str, dict): name of the network configuration.
        """
        self.device = get_device(device)
        self.chunk_tmp_device = self.device
        self.hparams = get_hparams(config)
        self.network = NKSRNetwork(self.hparams).to(self.device).eval().requires_grad_(False)
        ckpt_data = load_checkpoint_from_url(self.hparams.url)
        self.network.load_state_dict(ckpt_data['state_dict'])

    def set_chunk_tmp_device(self, device: Device) -> None:
        """
        If you choose to reconstruct by chunks, the finished chunks can be temporarily stored in another :attr:`device`
        (usually CPU) to save the memory of the device for the working chunk.

        Args:
            device (Device): The temporary device used to store reconstructed chunks.

        """
        self.chunk_tmp_device = get_device(device)

    def reconstruct(self,
                    xyz: Optional[torch.Tensor],
                    normal: Optional[torch.Tensor] = None,
                    sensor: Optional[torch.Tensor] = None,
                    detail_level: Optional[float] = 0.0,
                    voxel_size: Optional[float] = None,
                    chunk_size: float = -1,
                    overlap_ratio: float = 0.05,
                    approx_kernel_grad: bool = False,
                    solver_max_iter: int = 2000,
                    solver_tol: float = 1.0e-5,
                    nystrom_min_depth: int = 100,
                    fused_mode: bool = True,
                    preprocess_fn: Optional[Callable] = None):
        """

        Args:
            xyz (torch.Tensor): (N, 3) input point positions
            normal (torch.Tensor): (N, 3) input point normals
            sensor (torch.Tensor): (N, 3) input per-point sensor positions
            detail_level (float): the level of detail to reconstruct, the recommended value is from 0.0 to 1.0
                The higher the value, the more details but longer time and (maybe) worse completeness.
                None indicates using the original scale the model is trained on.
            voxel_size (float): force override the predefined voxel size in the pretrained weights.
                If specified, then `detailed_level` would be invalidated.
            chunk_size (float): size of each chunk, -1 indicates not chunking
            overlap_ratio (float): the ratio of overlapping regions between chunks to create a smooth transition
            approx_kernel_grad (bool): approximate kernel gradient using only of its spatial component
            solver_max_iter (int): the maximum number of iterations for the PCG-solver
            solver_tol (float): the convergence tolerance for the PCG-solver
            nystrom_min_depth (int): low-rank approximation of :math:`G^T G` matrix.
            fused_mode (bool): whether to use fused kernel (False -> faster, True -> more memory efficient)
            preprocess_fn (Callable): a functor that preprocesses the input point cloud before reconstruction.

        Returns:
            field (Field): the implicit field to extract mesh from.
        """

        if chunk_size > 0.0:
            transform_list, xyz_list, feat_list = split_into_chunks(
                xyz, chunk_size, overlap_ratio, normal=normal, sensor=sensor)
            # Not supporting detail_level or voxel_size for chunked reconstruction
            return self.reconstruct_by_chunk(
                transform_list, xyz_list,
                normal_list=feat_list['normal'],
                sensor_list=feat_list['sensor'],
                detail_level=None,
                approx_kernel_grad=approx_kernel_grad,
                nystrom_min_depth=nystrom_min_depth,
                fused_mode=fused_mode,
                preprocess_fn=preprocess_fn
            )

        if preprocess_fn is None:
            preprocess_fn = default_preprocess

        res = preprocess_fn(xyz, normal, sensor)
        if res is None:
            return None
        xyz, normal, sensor = res
        if xyz is None or xyz.size(0) == 0:
            return None

        global_scale = 1.0

        if voxel_size is not None:
            global_scale = voxel_size / self.hparams.voxel_size

        elif detail_level is not None and self.hparams.density_range is not None:
            vox_ijk = torch.unique(
                torch.div(xyz, self.hparams.voxel_size, rounding_mode='floor').long(), dim=0)
            cur_density = xyz.size(0) / vox_ijk.size(0)
            min_density, max_density = self.hparams.density_range
            target_density = min_density + (max_density - min_density) * (1.0 - detail_level)
            target_density = max(target_density, 0.01)
            global_scale = math.sqrt(target_density / cur_density)

        if global_scale != 1.0:
            logger.info(f"Input scale factor: {1.0 / global_scale:.4f}")
            xyz = xyz / global_scale
            if sensor is not None:
                sensor = sensor / global_scale

        if self.hparams.feature == 'normal':
            assert normal is not None, "normal is needed. please refer to example scripts to add a preprocessor!"
            feat = normal

        elif self.hparams.feature == 'sensor':
            assert sensor is not None, "sensor must be provided in this config!"
            view_dir = sensor - xyz
            del sensor
            view_dir = view_dir / (torch.linalg.norm(view_dir, dim=-1, keepdim=True) + 1e-6)
            feat = view_dir

        else:
            feat = None

        xyz = JaggedTensor([xyz])
        if feat is not None:
            feat = JaggedTensor([feat])

        svh = SparseFeatureHierarchy(
            voxel_size=self.hparams.voxel_size,
            depth=self.hparams.tree_depth,
            device=self.device
        )
        svh.build_point_splatting(xyz)
        feat = self.network.encoder(xyz, feat, svh, 0)
        feat, svh, udf_svh = self.network.unet(
            feat, svh,
            adaptive_depth=self.hparams.adaptive_depth
        )

        if self.hparams.geometry == 'kernel':
            output_field = KernelField(
                svh=svh,
                interpolator=self.network.interpolators,
                features=feat.basis_features,
                approx_kernel_grad=approx_kernel_grad,
                solver_max_iter=solver_max_iter,
                solver_tol=solver_tol
            )
            normal_xyz = fvdb.cat([svh.get_voxel_centers(d) for d in range(self.hparams.adaptive_depth)], dim=1)
            normal_value = fvdb.cat([feat.normal_features[d] for d in range(self.hparams.adaptive_depth)], dim=1)
            normal_weight = self.hparams.solver.normal_weight * (self.hparams.voxel_size ** 2) / (normal_xyz.joffsets[:, 1] - normal_xyz.joffsets[:, 0])
            solve_kwargs = {
                'pos_xyz': xyz, 'normal_xyz': normal_xyz, 'normal_value': -normal_value,
                'pos_weight': self.hparams.solver.pos_weight / (xyz.joffsets[:, 1] - xyz.joffsets[:, 0]),
                'normal_weight': normal_weight,
                'reg_weight': 1.0
            }
            if not fused_mode:
                output_field.solve_non_fused(**solve_kwargs)
            else:
                output_field.solve(
                    **solve_kwargs,
                    nystrom_min_depth=nystrom_min_depth
                )

        elif self.hparams.geometry == 'neural':
            output_field = NeuralField(
                svh=svh,
                decoder=self.network.sdf_decoder,
                features=feat.basis_features
            )

        else:
            raise NotImplementedError

        if self.hparams.udf.enabled:
            mask_field = NeuralField(
                svh=udf_svh,
                decoder=self.network.udf_decoder,
                features=feat.udf_features
            )
            mask_field.set_level_set(2 * self.hparams.voxel_size)
        else:
            mask_field = LayerField(svh, self.hparams.adaptive_depth)

        output_field.set_mask_field(mask_field)
        output_field.set_scale(global_scale)

        return output_field

    def reconstruct_by_chunk(self,
                             transform_list: List[Isometry],
                             xyz_list: List[torch.Tensor],
                             normal_list: List[torch.Tensor] = None,
                             sensor_list: List[torch.Tensor] = None,
                             **kwargs):

        n_chunks = len(transform_list)

        if normal_list is None:
            normal_list = [None for _ in range(n_chunks)]
        else:
            assert len(normal_list) == n_chunks, "normal list length does not agree!"

        if sensor_list is None:
            sensor_list = [None for _ in range(n_chunks)]
        else:
            assert len(sensor_list) == n_chunks, "sensor list length does not agree!"

        all_fields = []
        all_transforms = []
        iter_tqdm = tqdm.tqdm(
            zip(transform_list, xyz_list, normal_list, sensor_list),
            desc='nksr.chunk',
            total=n_chunks
        )

        for transform, xyz, normal, sensor in iter_tqdm:
            chunk_field = self.reconstruct(xyz, normal=normal, sensor=sensor, **kwargs)
            if chunk_field is None:
                continue
            chunk_field.to_(self.chunk_tmp_device)
            all_fields.append(chunk_field)
            all_transforms.append(transform)

        [f.to_(self.device) for f in all_fields]
        return FusedField(all_fields, all_transforms)

