# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn as nn
import torch.nn.functional as F
import fvdb
import fvdb.nn as fvnn
import numpy as np

from xcube.utils.color_util import color_from_points, semantic_from_points
from xcube.utils.loss_util import TorchLossMeter
from xcube.data.base import DatasetSpec as DS
    
class Loss(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

    def transform_field(self, field: torch.Tensor):
        gt_band = 1.0 # not sure if this will be changed
        truncation_size = gt_band * self.hparams.voxel_size
        # non-binary supervision (made sure derivative norm at 0 if 1)
        field = torch.tanh(field / truncation_size) * truncation_size
        return field
    
    def cross_entropy(self, pd_struct: fvnn.VDBTensor, gt_grid: fvdb.GridBatch, dynamic_grid: fvdb.GridBatch = None):
        assert torch.allclose(pd_struct.grid.origins, gt_grid.origins)
        assert torch.allclose(pd_struct.grid.voxel_sizes, gt_grid.voxel_sizes)
        idx_mask = gt_grid.ijk_to_index(pd_struct.grid.ijk).jdata == -1
        idx_mask = idx_mask.long()
        if dynamic_grid is not None:
            dynamic_mask = dynamic_grid.ijk_to_index(pd_struct.grid.ijk).jdata == -1
            loss = F.cross_entropy(pd_struct.feature.jdata, idx_mask, reduction='none') * dynamic_mask.float()
            loss = loss.mean()
        else:
            loss = F.cross_entropy(pd_struct.feature.jdata, idx_mask)
        return 0.0 if idx_mask.size(0) == 0 else loss
    
    def struct_acc(self, pd_struct: fvnn.VDBTensor, gt_grid: fvdb.GridBatch):
        assert torch.allclose(pd_struct.grid.origins, gt_grid.origins)
        assert torch.allclose(pd_struct.grid.voxel_sizes, gt_grid.voxel_sizes)
        idx_mask = gt_grid.ijk_to_index(pd_struct.grid.ijk).jdata == -1
        idx_mask = idx_mask.long()
        return torch.mean((pd_struct.feature.jdata.argmax(dim=1) == idx_mask).float())
    
    def grid_iou(self, gt_grid: fvdb.GridBatch, pd_grid: fvdb.GridBatch):
        assert gt_grid.grid_count == pd_grid.grid_count
        idx = pd_grid.ijk_to_index(gt_grid.ijk)
        upi = (pd_grid.num_voxels + gt_grid.num_voxels).cpu().numpy().tolist()
        ious = []
        for i in range(len(upi)):
            inter = torch.sum(idx[i].jdata >= 0).item()
            ious.append(inter / (upi[i] - inter + 1.0e-6))
        return np.mean(ious)

    def normal_loss(self, batch, normal_feats: fvnn.VDBTensor, eps=1e-6):
        if self.hparams.use_fvdb_loader:
            ref_grid = batch['input_grid']
            ref_xyz = ref_grid.grid_to_world(ref_grid.ijk.float()) 
        else:
            ref_xyz = fvdb.JaggedTensor(batch[DS.INPUT_PC])
        
        gt_normal = normal_feats.grid.splat_trilinear(ref_xyz, fvdb.JaggedTensor(batch[DS.TARGET_NORMAL]))
        # normalize normal
        gt_normal.jdata /= (gt_normal.jdata.norm(dim=1, keepdim=True) + eps)
        normal_loss = F.l1_loss(gt_normal.jdata, normal_feats.feature.jdata)
        return normal_loss
    
    def color_loss(self, batch, color_feats: fvnn.VDBTensor):
        assert self.hparams.use_fvdb_loader is True
        # check if color_feats is empty
        if color_feats.grid.total_voxels == 0:
            return 0.0
        ref_grid = batch['input_grid']
        ref_xyz = ref_grid.grid_to_world(ref_grid.ijk.float())
        ref_color = fvdb.JaggedTensor(batch[DS.INPUT_COLOR])
        
        target_xyz = color_feats.grid.grid_to_world(color_feats.grid.ijk.float())
        target_color = []
        slect_color_feats = []
        for batch_idx in range(ref_grid.grid_count):
            ref_color_i = ref_color[batch_idx].jdata
            target_color.append(color_from_points(target_xyz[batch_idx].jdata, ref_xyz[batch_idx].jdata, ref_color_i, k=1))
            slect_color_feats.append(color_feats.feature[batch_idx].jdata)
            
        if len(target_color) == 0 or len(slect_color_feats) == 0: # to avoid JaggedTensor build from empty list
            return 0.0  
        
        target_color = fvdb.JaggedTensor(target_color)
        slect_color_feats = fvdb.JaggedTensor(slect_color_feats)
        color_loss = F.l1_loss(slect_color_feats.jdata, target_color.jdata)
        return color_loss
    
    def semantic_loss(self, batch, semantic_feats: fvnn.VDBTensor):
        assert self.hparams.use_fvdb_loader is True
        # check if semantic_feats is empty
        if semantic_feats.grid.total_voxels == 0:
            return 0.0
        ref_grid = batch['input_grid']
        ref_xyz = ref_grid.grid_to_world(ref_grid.ijk.float())
        ref_semantic = fvdb.JaggedTensor(batch[DS.GT_SEMANTIC])
        if ref_semantic.jdata.size(0) == 0: # if all samples in this batch is without semantic
            return 0.0
                
        target_xyz = semantic_feats.grid.grid_to_world(semantic_feats.grid.ijk.float())       
        target_semantic = []
        slect_semantic_feats = []
        for batch_idx in range(ref_grid.grid_count):
            ref_semantic_i = ref_semantic[batch_idx].jdata
            if ref_semantic_i.size(0) == 0:
                continue
            target_semantic.append(semantic_from_points(target_xyz[batch_idx].jdata, ref_xyz[batch_idx].jdata, ref_semantic_i))
            slect_semantic_feats.append(semantic_feats.feature[batch_idx].jdata)
                    
        if len(target_semantic) == 0 or len(slect_semantic_feats) == 0: # to avoid JaggedTensor build from empty list
            return 0.0

        target_semantic = fvdb.JaggedTensor(target_semantic)
        slect_semantic_feats = fvdb.JaggedTensor(slect_semantic_feats)
        
        if slect_semantic_feats.jdata.size(0) == 0: # to aviod cross_entropy take empty tensor
            return 0.0
        
        semantic_loss = F.cross_entropy(slect_semantic_feats.jdata, target_semantic.jdata.long())
        return semantic_loss
    
    def get_kl_weight(self, global_step):
        # linear annealing the kl weight
        if global_step > self.hparams.anneal_star_iter:
            if global_step < self.hparams.anneal_end_iter:
                kl_weight = self.hparams.kl_weight_min + \
                                         (self.hparams.kl_weight_max - self.hparams.kl_weight_min) * \
                                         (global_step - self.hparams.anneal_star_iter) / \
                                         (self.hparams.anneal_end_iter - self.hparams.anneal_star_iter)
            else:
                kl_weight = self.hparams.kl_weight_max
        else:
            kl_weight = self.hparams.kl_weight_min

        return kl_weight

    def forward(self, batch, out, compute_metric: bool, global_step, current_epoch, optimizer_idx=0):
        loss_dict = TorchLossMeter()
        metric_dict = TorchLossMeter()
        latent_dict = TorchLossMeter()

        dynamic_grid = None

        if not self.hparams.use_hash_tree:
            gt_grid = out['gt_grid']
            if self.hparams.supervision.structure_weight > 0.0:
                for feat_depth, pd_struct_i in out['structure_features'].items():
                    downsample_factor = 2 ** feat_depth
                    if self.hparams.remain_h:
                        pd_voxel_size = pd_struct_i.grid.voxel_sizes[0]
                        h_factor = pd_voxel_size[0] // pd_voxel_size[2]
                        downsample_factor = [downsample_factor, downsample_factor, downsample_factor // h_factor]
                    if downsample_factor != 1:             
                        gt_grid_i = gt_grid.coarsened_grid(downsample_factor)
                        dyn_grid_i = dynamic_grid.coarsened_grid(downsample_factor) if dynamic_grid is not None else None
                    else:
                        gt_grid_i = gt_grid
                        dyn_grid_i = dynamic_grid
                    loss_dict.add_loss(f"struct-{feat_depth}", self.cross_entropy(pd_struct_i, gt_grid_i, dyn_grid_i),
                                    self.hparams.supervision.structure_weight)
                    if compute_metric:
                        with torch.no_grad():
                            metric_dict.add_loss(f"struct-acc-{feat_depth}", self.struct_acc(pd_struct_i, gt_grid_i))
        else:
            if self.hparams.supervision.structure_weight > 0.0:
                gt_tree = out['gt_tree']
                for feat_depth, pd_struct_i in out['structure_features'].items():
                    gt_grid_i = gt_tree[feat_depth]
                    # get dynamic grid
                    dyn_grid_i = dynamic_grid.coarsened_grid(2 ** feat_depth) if dynamic_grid is not None else None
                    loss_dict.add_loss(f"struct-{feat_depth}", self.cross_entropy(pd_struct_i, gt_grid_i, dyn_grid_i),
                                    self.hparams.supervision.structure_weight)
                    if compute_metric:
                        with torch.no_grad():
                            metric_dict.add_loss(f"struct-acc-{feat_depth}", self.struct_acc(pd_struct_i, gt_grid_i))
        
        # compute normal loss
        if self.hparams.with_normal_branch:
            if out['normal_features'] == {}:
                normal_loss = 0.0
            else:
                feat_depth = min(out['normal_features'].keys())
                normal_loss = self.normal_loss(batch, out['normal_features'][feat_depth])
                    
            loss_dict.add_loss(f"normal", normal_loss, self.hparams.supervision.normal_weight)
        
        # compute semantic loss
        if self.hparams.with_semantic_branch:
            for feat_depth, pd_semantic_i in out['semantic_features'].items():
                semantic_loss = self.semantic_loss(batch, pd_semantic_i)
                if semantic_loss == 0.0: # do not take empty into log
                    continue
                loss_dict.add_loss(f"semantic_{feat_depth}", semantic_loss, self.hparams.supervision.semantic_weight)
                
        # compute color loss
        if self.hparams.with_color_branch:
            for feat_depth, pd_color_i in out['color_features'].items():
                color_loss = self.color_loss(batch, pd_color_i)
                if color_loss == 0.0:
                    continue
                loss_dict.add_loss(f"color_{feat_depth}", color_loss, self.hparams.supervision.color_weight)

        # compute KL divergence
        if "dist_features" in out:
            dist_features = out['dist_features']
            kld = 0.0
            for latent_id, (mu, logvar) in enumerate(dist_features):
                num_voxel = mu.size(0)
                kld_temp = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kld_total = kld_temp.item()
                if self.hparams.normalize_kld:
                    kld_temp /= num_voxel

                kld += kld_temp
                latent_dict.add_loss(f"mu-{latent_id}", mu.mean())
                latent_dict.add_loss(f"logvar-{latent_id}", logvar.mean())
                latent_dict.add_loss(f"kld-true-{latent_id}", kld_temp.item())
                latent_dict.add_loss(f"kld-total-{latent_id}", kld_total)

            if self.hparams.enable_anneal:
                loss_dict.add_loss("kld", kld, self.get_kl_weight(global_step))
            else:
                loss_dict.add_loss("kld", kld, self.hparams.kl_weight)
            
        return loss_dict, metric_dict, latent_dict