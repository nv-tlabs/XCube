# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import gc
import random
import omegaconf
import importlib
from typing import Optional

import polyscope as ps
import fvdb
import numpy as np
import torch
import torch.nn.functional as F
from fvdb import JaggedTensor, GridBatch
from nksr import NKSRNetwork, SparseFeatureHierarchy
from nksr.configs import load_checkpoint_from_url
from nksr.fields import KernelField, LayerField, NeuralField
from nksr.fields.base_field import BaseField, EvaluationResult
# from pycg import vis # ! required for visualization
from pycg.isometry import ScaledIsometry
from xcube.utils import wandb_util

from xcube.models.base_model import BaseModel
from xcube.data.base import DatasetSpec as DS
from xcube.data.base import list_collate
from xcube.utils import exp


class UDFPtsField(BaseField):

    def __init__(self, xyz: JaggedTensor, normal: JaggedTensor, hparams):
        svh = SparseFeatureHierarchy(
            voxel_size=hparams.voxel_size,
            depth=hparams.tree_depth,
            device=xyz.device
        )
        svh.build_point_splatting(xyz)
        super().__init__(svh)
        self.xyz = xyz
        self.normal = normal
        self.hparams = hparams

    def evaluate_f(self, xyz: JaggedTensor, grad: bool = False):
        assert not grad, "PCNNField does not support gradient!"
        from xcube.modules.autoencoding.losses.nksr_loss import UDFLoss
        gt_sdf = UDFLoss.compute_gt_chi_from_pts(self.hparams, xyz, self.xyz, self.normal, transform=False)
        return EvaluationResult(torch.abs(gt_sdf.jdata))



class Model(BaseModel):
    def __init__(self, hparams):

        if hparams.dataset_global_scale != 1.0:
            omegaconf.OmegaConf.resolve(hparams)
            transform_kwargs = {
                'name': 'FixedScale',
                'args': {'scale': hparams.dataset_global_scale}
            }
            self.add_transforms_if_needed(hparams.train_kwargs.transforms, transform_kwargs)
            self.add_transforms_if_needed(hparams.val_kwargs.transforms, transform_kwargs)
            self.add_transforms_if_needed(hparams.test_kwargs.transforms, transform_kwargs)

        super().__init__(hparams)
        self.network = NKSRNetwork(self.hparams)        
        self.mismatched_voxel_size = True
        self.input_is_points = False

        if self.hparams.args_ckpt == 'none':
            self.vae = None

    @classmethod
    def add_transforms_if_needed(cls, target_kwargs, transform_kwargs):
        transform_name = transform_kwargs['name']
        if len(target_kwargs) == 0 or target_kwargs[-1]['name'] != transform_name:
            target_kwargs.append(transform_kwargs)

    @exp.mem_profile(every=1)
    def forward(self, out: dict):
        enc_svh = SparseFeatureHierarchy(
            voxel_size=self.hparams.voxel_size,
            depth=self.hparams.tree_depth,
            device=self.device
        )

        if self.input_is_points:
            input_xyz = out['in_xyz']
            enc_svh.build_point_splatting(input_xyz)

        else:
            in_grid = out['in_grid']
            if self.mismatched_voxel_size:
                enc_svh.build_point_splatting(in_grid.grid_to_world(in_grid.ijk.float()))
            else:
                enc_svh.build_grid_splatting(in_grid)
            input_xyz = in_grid.grid_to_world(in_grid.ijk.float())

        input_feature = []
        if 'normal' in self.hparams.feature:
            input_feature.append(out['in_normal'])
        if 'semantics' in self.hparams.feature:
            sem: JaggedTensor = out['in_semantics']
            input_feature.append(sem.jagged_like(torch.softmax(sem.jdata, dim=1)))
        input_feature = fvdb.cat(input_feature, dim=-1) if len(input_feature) > 0 else None

        feat = self.network.encoder(
            input_xyz, input_feature,
            enc_svh, 0
        )
        feat, dec_svh, udf_svh = self.network.unet(
            feat, enc_svh,
            adaptive_depth=self.hparams.adaptive_depth,
            gt_decoder_svh=out.get('gt_svh', None)
        )

        if all([dec_svh.grids[d].total_voxels == 0 for d in range(self.hparams.adaptive_depth)]):
            if self.trainer.training or self.trainer.validating:
                # In case training data is corrupted (pd & gt not aligned)...
                exp.logger.warning("Empty grid detected during training/validation.")
                return None

        if self.hparams.feature == 'none':
            pass
        elif self.hparams.feature == 'normal':
            out_normal_features = feat.normal_features
            out['out_normal'] = out_normal_features[0]
        elif self.hparams.feature == 'normal-semantics':
            out_data = {d: f.jdata for d, f in feat.normal_features.items()}
            out_normal_features = {d: feat.normal_features[d].jagged_like(f[:, :3]) for d, f in out_data.items()}
            out_semantics_features = {d: feat.normal_features[d].jagged_like(f[:, 3:]) for d, f in out_data.items()}
            out['out_normal'] = out_normal_features[0]
            out['out_semantics'] = out_semantics_features[0]
        else:
            raise NotImplementedError

        out.update({'enc_svh': enc_svh, 'dec_svh': dec_svh, 'dec_tmp_svh': udf_svh})
        out.update({
            'structure_features': feat.structure_features
        })

        out['out_grid'] = dec_svh.grids[0]

        if self.hparams.finetune_kernel_sdf:
            output_field = KernelField(
                svh=dec_svh,
                interpolator=self.network.interpolators,
                features=feat.basis_features,
                approx_kernel_grad=False
            )

            normal_xyz = fvdb.cat(
                [dec_svh.get_voxel_centers(d) for d in range(self.hparams.adaptive_depth)], dim=1
            )
            normal_value = fvdb.cat(
                [out_normal_features[d] for d in range(self.hparams.adaptive_depth)], dim=1
            )

            normal_weight = self.hparams.solver.normal_weight * (self.hparams.voxel_size ** 2) / \
                (normal_xyz.joffsets[:, 1] - normal_xyz.joffsets[:, 0])
            output_field.solve(
                pos_xyz=input_xyz,
                normal_xyz=normal_xyz,
                normal_value=-normal_value,
                pos_weight=self.hparams.solver.pos_weight / \
                    (input_xyz.joffsets[:, 1] - input_xyz.joffsets[:, 0]),
                normal_weight=normal_weight,
                reg_weight=1.0
            )

            out['kernel_sdf'] = output_field

        if self.hparams.finetune_neural_udf:
            mask_field = NeuralField(
                svh=udf_svh,
                decoder=self.network.udf_decoder,
                features=feat.udf_features
            )
            mask_field.set_level_set(1.0 * self.hparams.voxel_size)

            out['neural_udf'] = mask_field

        return out

    def compute_gt_svh(self, batch, out):
        if 'gt_svh' in out.keys():
            return out['gt_svh']

        gt_svh = SparseFeatureHierarchy(
            voxel_size=self.hparams.voxel_size,
            depth=self.hparams.tree_depth,
            device=self.device
        )

        if self.input_is_points:
            if DS.GT_GEOMETRY in batch.keys():
                ref_xyz = JaggedTensor([r.torch_attr()[0] for r in batch[DS.GT_GEOMETRY]])
            else:
                ref_xyz = JaggedTensor(batch[DS.GT_DENSE_PC])

            gt_svh.build_point_splatting(ref_xyz)
        else:
            gt_grid = out['gt_grid']
            if self.mismatched_voxel_size:
                gt_svh.build_point_splatting(gt_grid.grid_to_world(gt_grid.ijk.float()))
            else:
                gt_svh.build_grid_splatting(out['gt_grid'])
        out['gt_svh'] = gt_svh
        return gt_svh

    @exp.mem_profile(every=1)
    def compute_loss(self, batch, out, compute_metric: bool):
        loss_dict = exp.TorchLossMeter()
        metric_dict = exp.TorchLossMeter()

        from xcube.modules.autoencoding.losses.nksr_loss import StructureLoss, SpatialLoss, GTSurfaceLoss, UDFLoss
        self.compute_gt_svh(batch, out)
        StructureLoss.apply(self.hparams, loss_dict, metric_dict, batch, out, compute_metric)

        if 'in_normal' in out.keys() and 'gt_normal' in out.keys() and not self.mismatched_voxel_size:
            gt_grid, gt_normal = out['gt_grid'], out['gt_normal'].jdata
            out_grid, out_normal = out['out_grid'], out['out_normal'].jdata
            gt_index = gt_grid.ijk_to_index(out_grid.ijk)

            loss_mask = gt_index.jdata != -1
            out_normal = out_normal[loss_mask]
            gt_normal = gt_normal[gt_index.jdata[loss_mask]]
            normal_loss = F.l1_loss(out_normal, gt_normal, reduction='mean')

            loss_dict.add_loss('normal', normal_loss, 100.0)

        if 'in_semantics' in out.keys() and 'gt_semantics' in out.keys():
            assert not self.mismatched_voxel_size, "If voxel size mismatched, semantics cannot be used!"
            gt_grid, gt_semantics = out['gt_grid'], out['gt_semantics'].jdata
            out_grid, out_semantics = out['out_grid'], out['out_semantics'].jdata
            gt_index = gt_grid.ijk_to_index(out_grid.ijk)

            loss_mask = gt_index.jdata != -1
            out_semantics = out_semantics[loss_mask]
            gt_semantics = gt_semantics[gt_index.jdata[loss_mask]]
            # Try Focal loss?
            semantics_loss = F.cross_entropy(out_semantics, gt_semantics, reduction='mean')

            loss_dict.add_loss('semantics', semantics_loss, 5.0)

        if self.hparams.finetune_kernel_sdf:
            # Even this we don't suppress normal loss with the hope that it should help reconstruction.
            assert 'kernel_sdf' in out.keys()
            SpatialLoss.apply(self.hparams, loss_dict, metric_dict, batch, {'field': out['kernel_sdf']}, compute_metric)
            GTSurfaceLoss.apply(self.hparams, loss_dict, metric_dict, batch, {'field': out['kernel_sdf']}, compute_metric)

        if self.hparams.finetune_neural_udf:
            assert 'neural_udf' in out.keys()
            UDFLoss.apply(self.hparams, loss_dict, metric_dict, batch, out, compute_metric)

        return loss_dict, metric_dict

    def extract_vae_out(self, vae_out_dict):
        extracted_dict = {}

        out_grid = vae_out_dict['tree'][0]
        aug_grid = None

        if self.hparams.vae_crop_count > 0:
            crop_count = self.hparams.vae_crop_count * self.hparams.batch_size
            crop_indices = torch.randint(0, out_grid.total_voxels, (crop_count,))
            out_grid_ijk = out_grid.ijk
            rm_ijk, rm_jidx = [], []
            for crop_ind in crop_indices:
                center_ijk, center_batch = out_grid_ijk.jdata[crop_ind], out_grid_ijk.jidx[crop_ind]
                crop_size = random.randint(self.hparams.vae_crop_size_min, self.hparams.vae_crop_size_max)
                delta_ijk = torch.arange(-crop_size // 2 + 1, crop_size // 2 + 1, device=self.device)
                delta_ijk = torch.stack(torch.meshgrid(delta_ijk, delta_ijk, delta_ijk, indexing='ij'), dim=-1).reshape(-1, 3)
                crop_ijk = center_ijk + delta_ijk
                crop_jidx = torch.full_like(crop_ijk[:, 0], center_batch)
                rm_ijk.append(crop_ijk)
                rm_jidx.append(crop_jidx)
            rm_ijk, rm_jidx = torch.cat(rm_ijk), torch.cat(rm_jidx).short()
            rm_ijk = JaggedTensor.from_data_and_jidx(rm_ijk, rm_jidx, out_grid.grid_count)
            rm_inds = out_grid.ijk_to_index(rm_ijk).jdata
            exist_mask = torch.ones(out_grid.total_voxels, dtype=torch.bool, device=self.device)
            exist_mask[rm_inds[rm_inds != -1]] = False
            aug_ijk = out_grid_ijk.r_masked_select(exist_mask)
            aug_grid = fvdb.sparse_grid_from_ijk(aug_ijk, voxel_sizes=out_grid.voxel_sizes, origins=out_grid.origins)

        extracted_dict['grid'] = aug_grid if aug_grid is not None else out_grid

        if 'normal_features' in vae_out_dict:
            normal_feature = vae_out_dict['normal_features'][-1].feature
            if aug_grid is not None:
                normal_feature = aug_grid.fill_to_grid(normal_feature, out_grid)
            extracted_dict['normal_features'] = normal_feature

        if 'semantic_features' in vae_out_dict:
            semantic_feature = vae_out_dict['semantic_features'][-1].feature
            if aug_grid is not None:
                semantic_feature = aug_grid.fill_to_grid(semantic_feature, out_grid)
            extracted_dict['semantic_features'] = semantic_feature

        return extracted_dict

    # @exp.mem_profile(every=1)
    def train_val_step(self, batch, batch_idx, is_val):
        if batch_idx % 100 == 0:
            gc.collect()

        out = {'idx': batch_idx}

        vae_out = {}
        vae_noise_step = random.randint(
            self.hparams.vae_noise_step_min,
            self.hparams.vae_noise_step_max
        )
        with torch.no_grad():
            vae_out = self.vae(batch, vae_out, noise_step=vae_noise_step)

        # Random crop augmentation
        vae_out = self.extract_vae_out(vae_out)

        out['in_grid'] = vae_out['grid']
        out['gt_grid']: GridBatch = fvdb.cat(batch[DS.INPUT_PC])

        if 'normal' in self.hparams.feature:
            out['in_normal']: JaggedTensor = vae_out['normal_features']
            out['gt_normal']: JaggedTensor = JaggedTensor(batch[DS.TARGET_NORMAL])

        if 'semantics' in self.hparams.feature:
            out['in_semantics']: JaggedTensor = vae_out['semantic_features']
            out['gt_semantics']: JaggedTensor = JaggedTensor(batch[DS.GT_SEMANTIC])

        if not self.should_use_pd_structure(is_val):
            self.compute_gt_svh(batch, out)

        with exp.pt_profile_named("forward"):
            out = self(out)

        from xcube.modules.autoencoding.losses.nksr_loss import grid_iou
        self.log("iou_before", np.mean(grid_iou(out['in_grid'], out['gt_grid'])))
        self.log("iou_after", np.mean(grid_iou(out['out_grid'], out['gt_grid'])))

        if self.hparams.runtime_visualize:
            in_grid, out_grid, gt_grid = out['in_grid'][0], out['out_grid'][0], out['gt_grid'][0]
            if 'normal' in self.hparams.feature:
                in_normal, out_normal, gt_normal = out['in_normal'][0].jdata, out['out_normal'][0].jdata, out['gt_normal'][0].jdata
            else:
                in_normal, out_normal, gt_normal = None, None, None

            if 'semantics' in self.hparams.feature:
                in_semantics, out_semantics, gt_semantics = \
                    out['in_semantics'][0].jdata, out['out_semantics'][0].jdata, out['gt_semantics'][0].jdata
                in_semantics = torch.argmax(in_semantics, dim=1)
                out_semantics = torch.argmax(out_semantics, dim=1)
            else:
                in_semantics, out_semantics, gt_semantics = None, None, None

            if self.hparams.finetune_kernel_sdf or self.hparams.finetune_neural_udf:
                ref_xyz, ref_normal = JaggedTensor(batch[DS.GT_DENSE_PC]), JaggedTensor(batch[DS.GT_DENSE_NORMAL])
                ref_pcd = vis.pointcloud(ref_xyz[0].jdata, normal=ref_normal[0].jdata)
                
                # Show reference pcd.
                vis.show_3d([ref_pcd], [vis.pointcloud(in_grid.grid_to_world(in_grid.ijk.float()).jdata)])

                # Show pd and gt mesh.
                mesh_key = 'kernel_sdf' if self.hparams.finetune_kernel_sdf else 'neural_udf'
                pd_mesh = out[mesh_key].extract_dual_mesh()
                gt_field = UDFPtsField(ref_xyz[0], ref_normal[0], self.hparams)
                gt_field.set_level_set(2 * self.hparams.voxel_size)
                gt_mesh = gt_field.extract_dual_mesh()
                vis.show_3d([vis.mesh(pd_mesh.v, pd_mesh.f)], [vis.mesh(gt_mesh.v, gt_mesh.f)])

            ps.init()
            ps.set_ground_plane_mode("none")
            ps.set_up_dir("z_up")
            ps_in = ps.register_point_cloud("in", in_grid.grid_to_world(in_grid.ijk.float()).jdata.cpu().numpy())
            ps_out = ps.register_point_cloud("out", out_grid.grid_to_world(out_grid.ijk.float()).jdata.cpu().numpy())
            ps_gt = ps.register_point_cloud("gt", gt_grid.grid_to_world(gt_grid.ijk.float()).jdata.cpu().numpy())

            if in_normal is not None:
                ps_in.add_color_quantity("normal", in_normal.detach().cpu().numpy() / 2 + 0.5)
                ps_out.add_color_quantity("normal", out_normal.detach().cpu().numpy() / 2 + 0.5)
                ps_gt.add_color_quantity("normal", gt_normal.detach().cpu().numpy() / 2 + 0.5)

            if in_semantics is not None:
                ps_in.add_scalar_quantity("semantics", in_semantics.cpu().numpy(), enabled=True)
                ps_out.add_scalar_quantity("semantics", out_semantics.cpu().numpy(), enabled=True)
                ps_gt.add_scalar_quantity("semantics", gt_semantics.cpu().numpy(), enabled=True) 

            ps.show()

        # OOM Guard.
        if out is None:
            fake_loss = 0.0
            for r in self.parameters():
                if r.requires_grad:
                    fake_loss += r.sum() * 0.0
            return fake_loss

        with exp.pt_profile_named("loss"):
            loss_dict, metric_dict = self.compute_loss(batch, out, compute_metric=is_val)

        if not is_val:
            self.log_dict_prefix('train_loss', loss_dict)
        else:
            self.log_dict_prefix('val_metric', metric_dict)
            self.log_dict_prefix('val_loss', loss_dict)

        loss_sum = loss_dict.get_sum()
        if is_val and torch.any(torch.isnan(loss_sum)):
            exp.logger.warning("Get nan val loss during validation. Setting to 0.")
            loss_sum = 0
        self.log('val_loss' if is_val else 'train_loss/sum', loss_sum)

        return loss_sum

    def should_use_pd_structure(self, is_val):
        # In case this returns True:
        #   - The tree generation would completely rely on prediction, so does the supervision signal.
        prob = (self.trainer.global_step - self.hparams.structure_schedule.start_step) / \
               (self.hparams.structure_schedule.end_step - self.hparams.structure_schedule.start_step)
        prob = min(max(prob, 0.0), 1.0)
        if not is_val:
            self.log("pd_struct_prob", prob, prog_bar=True, on_step=True, on_epoch=False)
        return random.random() < prob

    def test_step(self, batch, batch_idx):
        out = {'idx': batch_idx}
        vae_out = {}
        with torch.no_grad():
            vae_out = self.vae(batch, vae_out,  noise_step=600)

        out['in_normal']: JaggedTensor = vae_out['normal_features'][-1].feature
        out['in_grid'] = vae_out['tree'][0]

        out['gt_normal']: JaggedTensor = JaggedTensor(batch[DS.TARGET_NORMAL])     # Currently wrong
        out['gt_grid']: GridBatch = fvdb.cat(batch[DS.INPUT_PC])
        batch_size = out['gt_grid'].grid_count

        out = self(out)
        # self.compute_gt_svh(batch, out)

        for b in range(batch_size):
            exp.logger.info(f"Now visualizing data {b + 1} of {batch_size}...")
            self._test_metric_and_visualize(b, batch, out)

    def _test_metric_and_visualize(self, batch_idx: int, batch, out: dict):

        in_grid, out_grid, gt_grid = out['in_grid'][batch_idx], out['out_grid'][batch_idx], out['gt_grid'][batch_idx]
        
        if not self.mismatched_voxel_size:
            from xcube.modules.autoencoding.losses.nksr_loss import grid_iou
            print("Before", grid_iou(in_grid, gt_grid), "After", grid_iou(out_grid, gt_grid))

        if self.hparams.feature == 'normal':
            in_normal, out_normal, gt_normal = out['in_normal'][batch_idx].jdata, out['out_normal'][batch_idx].jdata, out['gt_normal'][batch_idx].jdata
        else:
            in_normal, out_normal, gt_normal = None, None, None

        if self.hparams.finetune_kernel_sdf or self.hparams.finetune_neural_udf:
            mesh_key = 'kernel_sdf' if self.hparams.finetune_kernel_sdf else 'neural_udf'
            pd_mesh = out[mesh_key].extract_dual_mesh(batch_idx=batch_idx)
            out_geom = vis.mesh(pd_mesh.v, pd_mesh.f)
        
        else:
            out_geom = vis.pointcloud(out_grid.grid_to_world(out_grid.ijk.float()).jdata, normal=out_normal)

        vis.show_3d([vis.pointcloud(in_grid.grid_to_world(in_grid.ijk.float()).jdata, normal=in_normal)],
                    [out_geom],
                    [vis.pointcloud(gt_grid.grid_to_world(gt_grid.ijk.float()).jdata, normal=gt_normal)])

    def get_dataset_spec(self):
        return [DS.SHAPE_NAME, DS.INPUT_PC,
                DS.GT_DENSE_PC, DS.GT_DENSE_NORMAL, 
                DS.TARGET_NORMAL, DS.GT_SEMANTIC,
                DS.INPUT_INTENSITY, DS.GT_GEOMETRY]

    def get_collate_fn(self):
        return list_collate

    def get_hparams_metrics(self):
        return [('val_loss', True)]
