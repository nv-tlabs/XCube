# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import fvdb
from fvdb import JaggedTensor, GridBatch
import torch
import torch.nn.functional as F
from nksr.svh import SparseFeatureHierarchy

import ext
from xcube.data.base import DatasetSpec as DS


def grid_iou(gt_grid: GridBatch, pd_grid: GridBatch):
    assert gt_grid.grid_count == pd_grid.grid_count
    idx = pd_grid.ijk_to_index(gt_grid.ijk)
    upi = (pd_grid.num_voxels + gt_grid.num_voxels).cpu().numpy().tolist()
    ious = []
    for i in range(len(upi)):
        inter = torch.sum(idx[i].jdata >= 0).item()
        ious.append(inter / (upi[i] - inter + 1.0e-6))
    return ious


class KitchenSinkMetricLoss:
    @classmethod
    def apply(cls, hparams, loss_dict, metric_dict, batch, out, compute_metric):
        raise NotImplementedError

    @classmethod
    def _get_svh_samples(cls, svh: SparseFeatureHierarchy,
                         n_samples: int, expand: int = 0, expand_top: int = 0):
        """
        Get random samples, across all layers of the decoder hierarchy
        :param n_samples: int, number of total samples
        :param expand: size of expansion
        :param expand_top: size of expansion of the coarsest level.
        :return: (n_samples, 3)
        """
        base_coords, base_scales = [], []

        for d in range(svh.depth):
            grid = svh.grids[d]
            ijk_coords = grid.ijk
            d_expand = expand if d != svh.depth - 1 else expand_top
            if d_expand >= 3:
                ijk_coords = fvdb.sparse_grid_from_ijk(
                    ijk_coords, 
                    pad_min=[-d_expand // 2] * 3, pad_max=[d_expand // 2] * 3,
                    voxel_sizes=grid.voxel_sizes, origins=grid.origins).ijk
            base_coords.append(grid.grid_to_world(ijk_coords.float()))
            base_scales.append(ijk_coords.jagged_like(grid.voxel_sizes[ijk_coords.jidx.int()]))

        base_coords = fvdb.cat(base_coords, dim=1)
        base_scales = fvdb.cat(base_scales, dim=1)

        local_ids = (torch.rand((n_samples, ), device=svh.device) * base_coords.jdata.size(0)).long()
        local_coords = (torch.rand((n_samples, 3), device=svh.device) - 0.5) * base_scales.jdata[local_ids]
        query_jidx = base_coords.jidx[local_ids]
        query_pos = base_coords.jdata[local_ids] + local_coords
        return JaggedTensor.from_data_and_jidx(query_pos, query_jidx, svh.grids[-1].grid_count)

    @classmethod
    def _get_samples(cls, hparams, configs, svh, ref_xyz, ref_normal):
        all_samples = []
        for config in configs:
            if config.type == "uniform":
                all_samples.append(
                    cls._get_svh_samples(svh, config.n_samples, config.expand, config.expand_top)
                )
            elif config.type == "band":
                band_inds = (torch.rand((config.n_samples, ), device=ref_xyz.device) * ref_xyz.jdata.size(0)).long()
                sample_jidx = ref_xyz.jidx[band_inds]
                eps = config.eps * hparams.voxel_size
                band_pos = ref_xyz.jdata[band_inds] + \
                    ref_normal.jdata[band_inds] * \
                        torch.randn((config.n_samples, 1), device=ref_xyz.jdata.device) * eps
                all_samples.append(JaggedTensor.from_data_and_jidx(band_pos, sample_jidx, svh.grids[-1].grid_count))
        return fvdb.cat(all_samples, dim=1)

    @classmethod
    def transform_field(cls, hparams, field: JaggedTensor):
        spatial_config = hparams.supervision.spatial
        assert spatial_config.gt_type != "binary"
        truncation_size = spatial_config.gt_band * hparams.voxel_size

        field_data = field.jdata
        # non-binary supervision (made sure derivative norm at 0 if 1)
        if spatial_config.gt_soft:
            field_data = torch.tanh(field_data / truncation_size) * truncation_size
        else:
            field_data = torch.clone(field_data)
            field_data[field_data > truncation_size] = truncation_size
            field_data[field_data < -truncation_size] = -truncation_size
        return field.jagged_like(field_data)

    @classmethod
    def compute_gt_chi_from_pts(cls, hparams, query_pos: JaggedTensor, ref_xyz: JaggedTensor, ref_normal: JaggedTensor,
                                transform: bool = True):
        mc_query_sdfs = []
        for b in range(query_pos.joffsets.size(0)):
            q_pos, r_xyz, r_normal = query_pos[b].jdata, ref_xyz[b].jdata, ref_normal[b].jdata
            stdv = 0.2 if hparams.voxel_size > 0.004 else 0.002
            mc_query_sdfs.append(-ext.sdfgen.sdf_from_points(q_pos, r_xyz, r_normal, 8, stdv, False)[0])
        mc_query_sdf = JaggedTensor(mc_query_sdfs)
        if transform:
            return cls.transform_field(hparams, mc_query_sdf)
        else:
            return mc_query_sdf


class ShapeNetIoUMetric(KitchenSinkMetricLoss):
    """
    Will only output for ShapeNet data.
    """
    @classmethod
    def apply(cls, hparams, loss_dict, metric_dict, batch, out, compute_metric):
        if compute_metric:
            if DS.GT_ONET_SAMPLE not in batch.keys():
                return
            with torch.no_grad():
                onet_sample = JaggedTensor(batch[DS.GT_ONET_SAMPLE][0])
                onet_gt = JaggedTensor(batch[DS.GT_ONET_SAMPLE][1])
                iou_pd = out['field'].evaluate_f_bar(onet_sample).jdata > 0
                iou_gt = onet_gt.jdata > 0
            iou = torch.sum(torch.logical_and(iou_pd, iou_gt)) / (
                    torch.sum(torch.logical_or(iou_pd, iou_gt)) + 1.0e-6)
            metric_dict.add_loss('iou', iou)


class UDFLoss(KitchenSinkMetricLoss):
    """
    UDF Loss for supervising the UDF branch
    """
    @classmethod
    def compute_gt_tudf(cls, chi_pos, hparams, ref_xyz, ref_normal, ref_geometry):
        if ref_geometry is not None:
            gt_tsdf = cls.transform_field(hparams, JaggedTensor([ref_geometry.query_sdf(chi_pos.jdata)]))
        else:
            gt_tsdf = cls.compute_gt_chi_from_pts(hparams, chi_pos, ref_xyz, ref_normal)
        gt_tudf = torch.abs(gt_tsdf.jdata)
        return gt_tudf

    @classmethod
    def apply(cls, hparams, loss_dict, metric_dict, batch, out, compute_metric):
        udf_config = hparams.supervision.udf
        if hparams.udf.enabled and udf_config.weight > 0.0:

            if DS.GT_GEOMETRY not in batch.keys():
                ref_geometry = None
                ref_xyz, ref_normal = JaggedTensor(batch[DS.GT_DENSE_PC]), JaggedTensor(batch[DS.GT_DENSE_NORMAL])
            else:
                assert len(batch[DS.GT_GEOMETRY]) == 1, "Only supports single batch for now"
                ref_geometry = batch[DS.GT_GEOMETRY][0]
                ref_xyz, ref_normal, _ = ref_geometry.torch_attr()
                ref_xyz, ref_normal = JaggedTensor([ref_xyz]), JaggedTensor([ref_normal])

            if 'field' in out.keys():
                field_svh = out['field'].svh
                udf_field = out['field'].mask_field

            else:
                udf_field = out['neural_udf']
                field_svh = out['neural_udf'].svh

            chi_pos = cls._get_samples(hparams, udf_config.samplers, field_svh, ref_xyz, ref_normal)
            pd_chi = udf_field.evaluate_f(chi_pos).value

            gt_tudf = cls.compute_gt_tudf(chi_pos, hparams, ref_xyz, ref_normal, ref_geometry)
            pd_tudf = cls.transform_field(hparams, pd_chi).jdata
            udf_loss_normalized = torch.mean(torch.abs(pd_tudf - gt_tudf) / hparams.voxel_size)

            loss_dict.add_loss(f"udf", udf_loss_normalized, udf_config.weight)


class StructureLoss(KitchenSinkMetricLoss):
    """
    Cross entropy of the voxel classification
    (will also output accuracy metric)
    """
    @classmethod
    def apply(cls, hparams, loss_dict, metric_dict, batch, out, compute_metric):
        if hparams.supervision.structure_weight > 0.0:
            gt_svh = out['gt_svh']
            for feat_depth, struct_feat in out['structure_features'].items():
                if struct_feat.jdata.size(0) == 0:
                    continue
                gt_status = gt_svh.evaluate_voxel_status(out['dec_tmp_svh'].grids[feat_depth], feat_depth)
                loss_dict.add_loss(f"struct-{feat_depth}", F.cross_entropy(struct_feat.jdata, gt_status),
                                   hparams.supervision.structure_weight)
                if compute_metric:
                    metric_dict.add_loss(f"struct-acc-{feat_depth}",
                                         torch.mean((struct_feat.jdata.argmax(dim=1) == gt_status).float()))


class GTSurfaceLoss(KitchenSinkMetricLoss):
    """
    1. L1 Loss on the surface
    2. Dot-product loss on the surface normals
    """
    @classmethod
    def apply(cls, hparams, loss_dict, metric_dict, batch, out, compute_metric):
        gt_surface_config = hparams.supervision.gt_surface
        field = out['field']

        if gt_surface_config.value > 0.0 or gt_surface_config.normal > 0.0:

            if DS.GT_GEOMETRY not in batch.keys():
                ref_xyz, ref_normal = JaggedTensor(batch[DS.GT_DENSE_PC]), JaggedTensor(batch[DS.GT_DENSE_NORMAL])
            else:
                ref_xyz_normal = [r.torch_attr() for r in batch[DS.GT_GEOMETRY]]
                ref_xyz, ref_normal = JaggedTensor([t[0] for t in ref_xyz_normal]), \
                    JaggedTensor([t[1] for t in ref_xyz_normal])

            n_subsample = gt_surface_config.subsample
            if 0 < n_subsample < ref_xyz.jdata.size(0):
                ref_xyz_inds = (torch.rand((n_subsample,), device=ref_xyz.device) *
                                ref_xyz.jdata.size(0)).long()
            else:
                ref_xyz_inds = torch.arange(ref_xyz.jdata.size(0), device=ref_xyz.device)

            compute_grad = gt_surface_config.normal > 0.0

            batch_size = ref_xyz.joffsets.size(0)
            ref_jidx = ref_xyz.jidx[ref_xyz_inds]
            ref_xyz = JaggedTensor.from_data_and_jidx(ref_xyz.jdata[ref_xyz_inds], ref_jidx, batch_size)

            eval_res = field.evaluate_f(ref_xyz, grad=compute_grad)

            if compute_grad:
                ref_normal = JaggedTensor.from_data_and_jidx(ref_normal.jdata[ref_xyz_inds], ref_jidx, batch_size)
                pd_grad = eval_res.gradient.jdata
                pd_grad = -pd_grad / (torch.linalg.norm(pd_grad, dim=-1, keepdim=True) + 1.0e-6)
                loss_dict.add_loss('gt-surface-normal',
                                   1.0 - torch.sum(pd_grad * ref_normal.jdata, dim=-1).mean(),
                                   gt_surface_config.normal)

            loss_dict.add_loss('gt-surface-value', torch.abs(eval_res.value.jdata).mean(), gt_surface_config.value)


class SpatialLoss(KitchenSinkMetricLoss):
    """
    1. TSDF-Loss:
        - Near Surface: L1 of TSDF
        - Far Surface: (ShapeNet does not contain this region) exp
    2. RegSDF-Loss
    """

    @classmethod
    def apply(cls, hparams, loss_dict, metric_dict, batch, out, compute_metric):
        opt = hparams.supervision.spatial
        field = out['field']

        if DS.GT_GEOMETRY not in batch.keys():
            ref_geometry = None
            ref_xyz, ref_normal = JaggedTensor(batch[DS.GT_DENSE_PC]), JaggedTensor(batch[DS.GT_DENSE_NORMAL])
        else:
            assert len(batch[DS.GT_GEOMETRY]) == 1, "Only supports single batch for now"
            ref_geometry = batch[DS.GT_GEOMETRY][0]
            ref_xyz, ref_normal, _ = ref_geometry.torch_attr()
            ref_xyz, ref_normal = JaggedTensor([ref_xyz]), JaggedTensor([ref_normal])

        if opt.weight > 0.0:
            chi_pos = cls._get_samples(hparams, opt.samplers, field.svh, ref_xyz, ref_normal)

            # Note: If expand <= 3 then chi_mask will always be valid.
            pd_chi = field.evaluate_f(chi_pos).value

            if ref_geometry is not None:
                gt_sdf = ref_geometry.query_sdf(chi_pos.jdata)
                gt_tsdf = cls.transform_field(hparams, JaggedTensor([gt_sdf]))

                gt_cls = ref_geometry.query_classification(chi_pos.jdata)
                near_surface_mask = gt_cls == 0
                empty_space_mask = gt_cls == 1

            else:
                gt_tsdf = cls.compute_gt_chi_from_pts(hparams, chi_pos, ref_xyz, ref_normal)

                near_surface_mask = torch.ones(chi_pos.jdata.size(0), dtype=bool, device=chi_pos.device)
                empty_space_mask = ~near_surface_mask

            pd_tsdf = cls.transform_field(hparams, pd_chi)
            near_surface_l1 = torch.abs(
                (pd_tsdf.jdata[near_surface_mask] - gt_tsdf.jdata[near_surface_mask]) / hparams.voxel_size)

            # Empty space: value as small as possible.
            empty_scale = 2.0 * hparams.voxel_size
            empty_space_loss = 0.1 * torch.exp(pd_chi.jdata[empty_space_mask] / empty_scale)
            mixed_loss = (torch.sum(near_surface_l1) + torch.sum(empty_space_loss)) / chi_pos.jdata.size(0)
            loss_dict.add_loss(f"spatial", mixed_loss, opt.weight)

            # RegSDF Loss:
            if opt.reg_sdf_weight > 0.0:
                reg_sdf_eps = 0.5
                reg_sdf_loss = torch.mean(reg_sdf_eps / (pd_chi.jdata ** 2 + reg_sdf_eps ** 2))
                loss_dict.add_loss(f"msa", reg_sdf_loss, opt.reg_sdf_weight)