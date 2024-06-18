# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import gc

import fvdb
import fvdb.nn as fvnn
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from loguru import logger

from xcube.models.base_model import BaseModel
from xcube.data.base import DatasetSpec as DS
from xcube.data.base import list_collate
from xcube.modules.autoencoding.hparams import hparams_handler
from xcube.modules.autoencoding.base_encoder import Encoder
from xcube.modules.autoencoding.losses.base_loss import Loss
from xcube.modules.autoencoding.sunet import StructPredictionNet

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

def lambda_lr_wrapper(it, lr_config, batch_size, accumulate_grad_batches=1):
    return max(
        lr_config['decay_mult'] ** (int(it * batch_size * accumulate_grad_batches / lr_config['decay_step'])),
        lr_config['clip'] / lr_config['init'])

class Model(BaseModel):
    def __init__(self, hparams):
        hparams = hparams_handler(hparams) # set up hparams automatically
        super().__init__(hparams)
        self.encoder = Encoder(self.hparams)           
        self.unet = eval(self.hparams.network.unet.target)(cut_ratio=self.hparams.cut_ratio, 
                                                           with_normal_branch=self.hparams.with_normal_branch,
                                                           with_semantic_branch=self.hparams.with_semantic_branch,
                                                           **self.hparams.network.unet.params)
        self.loss = Loss(self.hparams)
        
        # load pretrained weight
        if self.hparams.pretrained_weight is not None:
            logger.info(f"load pretrained weight from {self.hparams.pretrained_weight}")
            checkpoint = torch.load(self.hparams.pretrained_weight, map_location='cpu')
            missing_keys, unexpected_keys = self.load_state_dict(checkpoint['state_dict'], strict=False)
            logger.info(f"missing_keys: {missing_keys}")
            logger.info(f"unexpected_keys: {unexpected_keys}")

        # using for testing time
        self.reconstructor = None
    
    def build_hash_tree(self, input_xyz):
        if self.hparams.use_fvdb_loader:
            if isinstance(input_xyz, dict):
                return input_xyz
            return self.build_hash_tree_from_grid(input_xyz)
        
        return self.build_hash_tree_from_points(input_xyz)
    
    def build_hash_tree_from_points(self, input_xyz):
        if isinstance(input_xyz, torch.Tensor):
            input_xyz = fvdb.JaggedTensor(input_xyz)
        elif isinstance(input_xyz, fvdb.JaggedTensor):
            pass
        else:
            raise NotImplementedError
        
        hash_tree = {}
        for depth in range(self.hparams.tree_depth):
            if depth != 0 and not self.hparams.use_hash_tree:
                break
            voxel_size = [sv * 2 ** depth for sv in self.hparams.voxel_size]
            origins = [sv / 2. for sv in voxel_size]            
            hash_tree[depth] = fvdb.sparse_grid_from_nearest_voxels_to_points(input_xyz, 
                                                                              voxel_sizes=voxel_size, 
                                                                              origins=origins)
        return hash_tree
    
    def build_hash_tree_from_grid(self, input_grid):
        hash_tree = {}
        input_xyz = input_grid.grid_to_world(input_grid.ijk.float())
        
        for depth in range(self.hparams.tree_depth):
            if depth != 0 and not self.hparams.use_hash_tree:
                break            
            voxel_size = [sv * 2 ** depth for sv in self.hparams.voxel_size]
            origins = [sv / 2. for sv in voxel_size]
            
            if depth == 0:
                hash_tree[depth] = input_grid
            else:
                hash_tree[depth] = fvdb.sparse_grid_from_nearest_voxels_to_points(input_xyz, 
                                                                                  voxel_sizes=voxel_size, 
                                                                                  origins=origins)
        return hash_tree

    def forward(self, batch, out: dict):
        input_xyz = batch[DS.INPUT_PC]
        hash_tree = self.build_hash_tree(input_xyz)
        input_grid = hash_tree[0]
        batch.update({'input_grid': input_grid})

        if not self.hparams.use_hash_tree:
            hash_tree = None
                
        unet_feat = self.encoder(input_grid, batch)
        unet_feat = fvnn.VDBTensor(input_grid, input_grid.jagged_like(unet_feat))
        unet_res, unet_output, dist_features = self.unet(unet_feat, hash_tree)

        out.update({'tree': unet_res.structure_grid})
        out.update({
            'structure_features': unet_res.structure_features,
            'dist_features': dist_features,
        })
        out.update({'gt_grid': input_grid})
        out.update({'gt_tree': hash_tree})
        
        if self.hparams.with_normal_branch:
            out.update({
                'normal_features': unet_res.normal_features,
            })
        if self.hparams.with_semantic_branch:
            out.update({
                'semantic_features': unet_res.semantic_features,
            })
        if self.hparams.with_color_branch:
            out.update({
                'color_features': unet_res.color_features,
            })
        return out

    def on_validation_epoch_start(self):
        pass

    def train_val_step(self, batch, batch_idx, is_val):
        if batch_idx % 1 == 0:
            # Squeeze memory really hard :)
            # This is a trade-off between memory and speed
            gc.collect()
            torch.cuda.empty_cache()

        out = {'idx': batch_idx}
        out = self(batch, out)
        
        if out is None and not is_val:
            return None

        loss_dict, metric_dict, latent_dict = self.loss(batch, out, 
                                                        compute_metric=is_val, 
                                                        global_step=self.global_step,
                                                        current_epoch=self.current_epoch)

        if not is_val:
            self.log_dict_prefix('train_loss', loss_dict)
            self.log_dict_prefix('train_loss', latent_dict)
            if self.hparams.enable_anneal:
                self.log('anneal_kl_weight', self.loss.get_kl_weight(self.global_step))
        else:
            self.log_dict_prefix('val_metric', metric_dict)
            self.log_dict_prefix('val_loss', loss_dict)
            self.log_dict_prefix('val_loss', latent_dict)

        loss_sum = loss_dict.get_sum()
        self.log('val_loss' if is_val else 'train_loss/sum', loss_sum)
        self.log('val_step', self.global_step)

        return loss_sum

    def test_step(self, batch, batch_idx):        
        self.log('source', batch[DS.SHAPE_NAME][0])
        out = {'idx': batch_idx}
        out = self(batch, out)
        loss_dict, metric_dict, latent_dict = self.loss(batch, out, 
                                                        compute_metric=True, 
                                                        global_step=self.trainer.global_step,
                                                        current_epoch=self.current_epoch)
        self.log_dict(loss_dict)
        self.log_dict(metric_dict)
        self.log_dict(latent_dict)
        
    def get_dataset_spec(self):
        all_specs = [DS.SHAPE_NAME, DS.INPUT_PC,
                     DS.GT_DENSE_PC, DS.GT_GEOMETRY]
        if self.hparams.use_input_normal:
            all_specs.append(DS.TARGET_NORMAL)
            all_specs.append(DS.GT_DENSE_NORMAL)
        if self.hparams.use_input_semantic or self.hparams.with_semantic_branch:
            all_specs.append(DS.GT_SEMANTIC)
        if self.hparams.use_input_intensity:
            all_specs.append(DS.INPUT_INTENSITY)
        return all_specs

    def get_collate_fn(self):
        return list_collate

    def get_hparams_metrics(self):
        return [('val_loss', True)]

    def configure_optimizers(self):
        # overwrite this from base model to fix pretrained vae layer
        lr_config = self.hparams.learning_rate
        # parameters = list(self.parameters())
        parameters = list(self.encoder.parameters()) + list(self.unet.parameters())

        if self.hparams.optimizer == 'SGD':
            optimizer = torch.optim.SGD(parameters, lr=lr_config['init'], momentum=0.9,
                                        weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'Adam':
            # AdamW corrects the bad weight dacay implementation in Adam.
            # AMSGrad also do some corrections to the original Adam.
            # The learning rate here is the maximum rate we can reach for each parameter.
            optimizer = torch.optim.AdamW(parameters, lr=lr_config['init'],
                                          weight_decay=self.hparams.weight_decay, amsgrad=True)        
        else:
            raise NotImplementedError

        # build scheduler
        import functools

        from torch.optim.lr_scheduler import LambdaLR
        scheduler = LambdaLR(optimizer,
                             lr_lambda=functools.partial(
                                 lambda_lr_wrapper, lr_config=lr_config, batch_size=self.hparams.batch_size, accumulate_grad_batches=self.trainer.accumulate_grad_batches))

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    @torch.no_grad()
    def _encode(self, batch, use_mode=False):
        input_xyz = batch[DS.INPUT_PC]
        hash_tree = self.build_hash_tree(input_xyz)
        input_grid = hash_tree[0]
        batch.update({'input_grid': input_grid})

        if not self.hparams.use_hash_tree:
            hash_tree = None

        unet_feat = self.encoder(input_grid, batch)
        unet_feat = fvnn.VDBTensor(input_grid, input_grid.jagged_like(unet_feat))
        _, x, mu, log_sigma = self.unet.encode(unet_feat, hash_tree=hash_tree)
        if use_mode:
            sparse_feature = mu
        else:
            sparse_feature = reparametrize(mu, log_sigma)
        
        return fvnn.VDBTensor(x.grid, x.grid.jagged_like(sparse_feature))