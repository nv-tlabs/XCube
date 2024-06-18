# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import gc
import importlib
from contextlib import contextmanager
import os

import fvdb
from fvdb.nn import VDBTensor
from fvdb import GridBatch
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
from pathlib import Path
from pytorch_lightning.utilities.distributed import rank_zero_only

from xcube.models.base_model import BaseModel
from xcube.data.base import DatasetSpec as DS
from xcube.data.base import list_collate
from xcube.utils import exp
from xcube.utils import wandb_util
from xcube.utils.wandb_util import find_mismatched_keys, load_state_dict_into_model
from xcube.utils.embedder_util import get_embedder
from xcube.utils.vis_util import vis_pcs
from xcube.modules.diffusionmodules.ema import LitEma
from xcube.modules.diffusionmodules.schedulers.scheduling_ddim import DDIMScheduler
from xcube.modules.diffusionmodules.schedulers.scheduling_ddpm import DDPMScheduler
from xcube.modules.diffusionmodules.schedulers.scheduling_dpmpp_2m import DPMSolverMultistepScheduler
from xcube.modules.diffusionmodules.openaimodel.unet_dense import UNetModel as UNetModel_Dense
from xcube.modules.diffusionmodules.openaimodel.unet_sparse import UNetModel as UNetModel_Sparse
from xcube.modules.diffusionmodules.openaimodel.unet_sparse_crossattn import UNetModel as UNetModel_Sparse_CrossAttn

from xcube.modules.encoders import (SemanticEncoder, ClassEmbedder, PointNetEncoder,
                                    StructEncoder, StructEncoder3D, StructEncoder3D_remain_h, StructEncoder3D_v2)

def is_rank_node_zero():
    return os.environ.get('NODE_RANK', '0') == '0'

def lambda_lr_wrapper(it, lr_config, batch_size):
    return max(
        lr_config['decay_mult'] ** (int(it / lr_config['decay_step'])),
        lr_config['clip'] / lr_config['init'])

class Model(BaseModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        if not hasattr(self.hparams, 'ema'):
            self.hparams.ema = False
        if not hasattr(self.hparams, 'use_ddim'):
            self.hparams.use_ddim = False
        if not hasattr(self.hparams, 'scale_by_std'):
            self.hparams.scale_by_std = False
        if not hasattr(self.hparams, 'scale_factor'):
            self.hparams.scale_factor = 1.0
        if not hasattr(self.hparams, 'num_inference_steps'):
            self.hparams.num_inference_steps = 1000
        if not hasattr(self.hparams, 'conditioning_key'):
            self.hparams.conditioning_key = "none"
        if not hasattr(self.hparams, 'log_image'):
            self.hparams.log_image = True

        # position embedding
        if not hasattr(self.hparams, 'use_pos_embed'):
            self.hparams.use_pos_embed = False
        if not hasattr(self.hparams, 'use_pos_embed_high'):
            self.hparams.use_pos_embed_high = False
        if not hasattr(self.hparams, 'use_pos_embed_world'):
            self.hparams.use_pos_embed_world = False
        if not hasattr(self.hparams, 'use_pos_embed_world_high'):
            self.hparams.use_pos_embed_world_high = False

        # get vae model
        self.vae = self.load_first_stage_from_pretrained().eval()
        self.vae.requires_grad_(False)

        # setup diffusion unet
        unet_num_blocks = self.vae.hparams.network.unet.params.num_blocks
        num_input_channels = self.vae.hparams.network.unet.params.f_maps * 2 ** (unet_num_blocks - 1) # Fix by using VAE hparams
        num_input_channels = int(num_input_channels / self.vae.hparams.cut_ratio)

        out_channels = num_input_channels
        num_classes = None
        use_spatial_transformer = False
        context_dim=None
        concat_dim=None

        if self.hparams.conditioning_key == "concat":
            num_input_channels += self.hparams.num_classes
        elif self.hparams.conditioning_key == "adm":
            num_classes = self.hparams.num_classes
        elif self.hparams.conditioning_key == "none":
            pass # do nothing
        elif self.hparams.conditioning_key == "crossattn":
            use_spatial_transformer = True
            context_dim=self.hparams.context_dim
        elif self.hparams.conditioning_key == "c_crossattn":
            use_spatial_transformer = True
            context_dim=self.hparams.context_dim
            num_input_channels += self.hparams.num_classes 
        elif self.hparams.conditioning_key == "c_adm":
            num_input_channels += self.hparams.concat_dim
            concat_dim = self.hparams.concat_dim
            num_classes = self.hparams.num_classes
        else:
            raise NotImplementedError

        if self.hparams.use_pos_embed:
            num_input_channels += 3
        elif self.hparams.use_pos_embed_high:
            embed_fn, input_ch = get_embedder(5)
            self.pos_embedder = embed_fn
            num_input_channels += input_ch
        elif self.hparams.use_pos_embed_world:
            num_input_channels += 3
        elif self.hparams.use_pos_embed_world_high:
            embed_fn, input_ch = get_embedder(5)
            self.pos_embedder = embed_fn
            num_input_channels += input_ch

        logger.info(f"num_input_channels: {num_input_channels}, out_channels: {out_channels}, \
                    num_classes: {num_classes}, context_dim: {context_dim}, concat_dim: {concat_dim} \
                    conditioning_key: {self.hparams.conditioning_key}")
        self.unet = eval(self.hparams.network.diffuser_name)(num_input_channels=num_input_channels, 
                                                             out_channels=out_channels, 
                                                             num_classes=num_classes,
                                                             use_spatial_transformer=use_spatial_transformer,
                                                             context_dim=context_dim,
                                                             **self.hparams.network.diffuser)
                
        # get the scheduler
        self.noise_scheduler = DDPMScheduler(**self.hparams.network.scheduler)
        self.ddim_scheduler = DDIMScheduler(**self.hparams.network.scheduler)

        # setup diffusion condition
        if not hasattr(self.hparams, 'use_mask_cond'):
            self.hparams.use_mask_cond = False
        if not hasattr(self.hparams, 'use_point_cond'):
            self.hparams.use_point_cond = False
        if not hasattr(self.hparams, 'use_semantic_cond'):
            self.hparams.use_semantic_cond = False
        if not hasattr(self.hparams, 'use_normal_concat_cond'):
            self.hparams.use_normal_concat_cond = False 
            
        if not hasattr(self.hparams, 'use_single_scan_concat_cond'):
            self.hparams.use_single_scan_concat_cond = False
        if not hasattr(self.hparams, 'encode_single_scan_by_points'):
            self.hparams.encode_single_scan_by_points = False
            
        if not hasattr(self.hparams, 'use_class_cond'):
            self.hparams.use_class_cond = False
        if not hasattr(self.hparams, 'use_micro_cond'):
            self.hparams.use_micro_cond = False
        if not hasattr(self.hparams, 'use_text_cond'):
            self.hparams.use_text_cond = False

        # noise offset config
        if not hasattr(self.hparams, 'use_noise_offset'):
            self.hparams.use_noise_offset = False
            
        # classifier-free config
        if not hasattr(self.hparams, 'use_classifier_free'):
            self.hparams.use_classifier_free = False # text cond in not influenced by this flag
        if not hasattr(self.hparams, 'classifier_free_prob'):
            self.hparams.classifier_free_prob = 0.1 # prob to drop the label
            
        # finetune config
        if not hasattr(self.hparams, 'pretrained_model_name_or_path'):
            self.hparams.pretrained_model_name_or_path = None
        if not hasattr(self.hparams, 'ignore_mismatched_size'):
            self.hparams.ignore_mismatched_size = False

        # mask or point or semantic condition
        if self.hparams.use_mask_cond or self.hparams.use_point_cond or self.hparams.use_semantic_cond or self.hparams.use_class_cond:
            self.cond_stage_model = eval(self.hparams.network.cond_stage_model.target)(**self.hparams.network.cond_stage_model.params)
        
        # micro condition
        if self.hparams.use_micro_cond:
            micro_dim = len(self.hparams.micro_key)
            embed_fn, input_micro_ch = get_embedder(6, input_dims=micro_dim)
            self.micro_pos_embedder = embed_fn
            self.micro_cond_model = nn.Linear(input_micro_ch, self.hparams.num_classes)
            
        # single scan concat condition
        if self.hparams.use_single_scan_concat_cond:
            # concat to the latent
            if self.hparams.encode_single_scan_by_points:
                self.single_scan_pos_embedder = PointNetEncoder(**self.hparams.network.single_scan_cond_model.embedder_params)
                input_ch = self.hparams.network.single_scan_cond_model.embedder_params.c_dim
            else:
                embed_fn, input_ch = get_embedder(5)
                self.single_scan_pos_embedder = embed_fn
            self.single_scan_cond_model = eval(self.hparams.network.single_scan_cond_model.target)(in_channels_pre=input_ch,
                                                                                                   **self.hparams.network.single_scan_cond_model.params)
        
        # load pretrained unet weight (ema version)
        if self.hparams.pretrained_model_name_or_path is not None:
            logger.info(f"Loading pretrained weight from {self.hparams.pretrained_model_name_or_path}")
            wdb_run, args_ckpt = wandb_util.get_wandb_run(self.hparams.pretrained_model_name_or_path, wdb_base=self.hparams.wandb_base, default_ckpt="test_auto")
            assert args_ckpt is not None, "Please specify checkpoint version!"
            assert args_ckpt.exists(), "Selected checkpoint does not exist!"     
            state_dict_all = torch.load(args_ckpt, map_location='cpu')['state_dict']
            
            # set scale_factor from pretrained weight
            if self.hparams.scale_by_std:
                self.hparams.scale_factor = state_dict_all['scale_factor'].item()
            
            # create an temporal ema model
            unet_ema = LitEma(self.unet, decay=self.hparams.ema_decay)
            state_dict_unet_ema = {k.replace("unet_ema.", ""): v for k, v in state_dict_all.items() if 'unet_ema' in k} # remove the prefix
            loaded_keys = [k for k in state_dict_unet_ema.keys()]
            # allow misalign size
            mismatched_keys = find_mismatched_keys(
                state_dict_unet_ema,
                unet_ema.state_dict(),
                loaded_keys,
                ignore_mismatched_sizes=self.hparams.ignore_mismatched_size,
            )            
            logger.info(f"Found {len(mismatched_keys)} mismatched keys: {mismatched_keys}")
            # unet_ema.load_state_dict(state_dict_unet_ema)
            error_msgs = load_state_dict_into_model(unet_ema, state_dict_unet_ema)
            if len(error_msgs) > 0:
                error_msg = "\n\t".join(error_msgs)
                raise RuntimeError(f"Error(s) in loading state_dict for {unet_ema.__class__.__name__}:\n\t{error_msg}")
            unet_ema.copy_to(self.unet)
            del unet_ema
                        
            state_dict_cond = {k.replace("cond_stage_model.", ""): v for k, v in state_dict_all.items() if 'cond_stage_model' in k} # remove the prefix
            # if have cond_stage_model
            if len(state_dict_cond) > 0:
                self.cond_stage_model.load_state_dict(state_dict_cond)
                logger.info(f"Loaded cond_stage_model from {args_ckpt}")
            
        # build ema
        if self.hparams.ema:
            self.unet_ema = LitEma(self.unet, decay=self.hparams.ema_decay)
            logger.info(f"Keeping EMAs of {len(list(self.unet_ema.buffers()))}.")
            
        # scale by std
        if not self.hparams.scale_by_std:
            self.scale_factor = self.hparams.scale_factor
            assert self.scale_factor == 1., 'when not using scale_by_std, scale_factor should be 1.'
        else:
            self.register_buffer('scale_factor', torch.tensor(self.hparams.scale_factor).float())
            logger.info(f"Manully setting scale_factor to {self.scale_factor}")

    @torch.no_grad()
    def load_first_stage_from_pretrained(self):
        model_yaml_path = Path(self.hparams.vae_config)
        model_args = exp.parse_config_yaml(model_yaml_path)
        net_module = importlib.import_module("xcube.models." + model_args.model).Model
        args_ckpt = Path(self.hparams.vae_checkpoint)
        assert args_ckpt.exists(), "Selected VAE checkpoint does not exist!"
        return net_module.load_from_checkpoint(args_ckpt, hparams=model_args)

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        # only for very first batch
        if self.hparams.scale_by_std and self.global_step == 0 and batch_idx == 0 and self.scale_factor == 1.:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            logger.info("### USING STD-RESCALING ###")
            latents = self.extract_latent(batch)
            z = latents.feature.jdata.detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            logger.info(f"setting self.scale_factor to {self.scale_factor}")
            logger.info("### USING STD-RESCALING ###")

    def on_train_batch_end(self, *args, **kwargs):
        if self.hparams.ema:
            self.unet_ema(self.unet)

    @contextmanager
    def ema_scope(self, context=None):
        if self.hparams.ema:
            self.unet_ema.store(self.unet.parameters())
            self.unet_ema.copy_to(self.unet)
            if context is not None:
                logger.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.hparams.ema:
                self.unet_ema.restore(self.unet.parameters())
                if context is not None:
                    logger.info(f"{context}: Restored training weights")
    
    @torch.no_grad()
    def extract_latent(self, batch):
        return self.vae._encode(batch, use_mode=False)
    
    def get_pos_embed(self, h):
        return h[:, :3]
    
    def get_pos_embed_high(self, h):
        xyz = h[:, :3] # N, 3
        xyz = self.pos_embedder(xyz) # N, C
        return xyz
    
    def conduct_classifier_free(self, cond, batch_size, device, is_testing=False):
        if isinstance(cond, VDBTensor):
            cond = cond.feature
        assert isinstance(cond, fvdb.JaggedTensor), "cond should be JaggedTensor"

        mask = torch.rand(batch_size, device=device) < self.hparams.classifier_free_prob 
        new_cond = []
        for idx in range(batch_size):
            if mask[idx] or is_testing:
                # during testing, use this function to zero the condition
                new_cond.append(torch.zeros_like(cond[idx].jdata))
            else:
                new_cond.append(cond[idx].jdata)
        new_cond = fvdb.JaggedTensor(new_cond)
        return new_cond
        
    def _forward_cond(self, noisy_latents: VDBTensor, timesteps: torch.Tensor, 
                      batch = None, cond_dict = None, is_testing=False, guidance_scale=1.0) -> VDBTensor:
        do_classifier_free_guidance = guidance_scale != 1.0
        # ! adm part
        # mask condition
        if self.hparams.use_mask_cond:
            coords = noisy_latents.grid.grid_to_world(noisy_latents.grid.ijk.float())
            coords = VDBTensor(noisy_latents.grid, coords)
            cond = self.cond_stage_model(coords)
        # point condition
        if self.hparams.use_point_cond:
            coords = noisy_latents.grid.grid_to_world(noisy_latents.grid.ijk.float()) # JaggedTensor
            if self.hparams.network.cond_stage_model.use_normal:
                if batch is not None: # training-time: get normal from batch
                    ref_xyz = fvdb.JaggedTensor(batch[DS.INPUT_PC])
                    # splatting normal
                    input_normal = noisy_latents.grid.splat_trilinear(ref_xyz, fvdb.JaggedTensor(batch[DS.TARGET_NORMAL]))
                    # normalize normal
                    input_normal.jdata /= (input_normal.jdata.norm(dim=1, keepdim=True) + 1e-6) # avoid nan
            else:
                input_normal = None
            cond = self.cond_stage_model(coords, input_normal)
        # class condition:
        if self.hparams.use_class_cond:
            if batch is not None:
                cond = self.cond_stage_model(batch, key=DS.CLASS)
            else:
                cond = self.cond_stage_model(cond_dict, key="class") # not checked yet
        # micro condition
        if self.hparams.use_micro_cond:
            if batch is not None:
                micro = batch[DS.MICRO]
                micro = torch.stack(micro).float()
            else:
                micro = cond_dict['micro']
            micro = self.micro_pos_embedder(micro)
            cond = self.micro_cond_model(micro)
        
        # ! concat part            
        concat_list = []        
        # semantic condition
        if self.hparams.use_semantic_cond:
            # traing-time: get semantic from batch
            if batch is not None:
                input_semantic = fvdb.JaggedTensor(batch[DS.LATENT_SEMANTIC])
            else:
                input_semantic = cond_dict['semantics']
            semantic_cond = self.cond_stage_model(input_semantic.jdata.long())
            if not is_testing and self.hparams.use_classifier_free: # if VDBtensor, convert to JaggedTensor
                semantic_cond = self.conduct_classifier_free(semantic_cond, noisy_latents.grid.grid_count, noisy_latents.grid.device)  
            concat_list.append(semantic_cond) # ! tensor type
        # single scan concat condition
        if self.hparams.use_single_scan_concat_cond:
            # traing-time: get single scan crop from batch
            if batch is not None:
                single_scan = fvdb.JaggedTensor(batch[DS.SINGLE_SCAN_CROP])
                single_scan_intensity = fvdb.JaggedTensor(batch[DS.SINGLE_SCAN_INTENSITY_CROP])
            else:
                single_scan = cond_dict['single_scan']
                single_scan_intensity = cond_dict['single_scan_intensity']
                
            # here use splatting to build the single scan grid tree
            single_scan_hash_tree = self.vae.build_normal_hash_tree(single_scan)
            single_scan_grid = single_scan_hash_tree[0]            
            if self.hparams.encode_single_scan_by_points:
                single_scan_feat = self.single_scan_pos_embedder(single_scan, single_scan_intensity, single_scan_grid)
                single_scan_feat = VDBTensor(single_scan_grid, single_scan_feat)
            else:
                single_scan_coords = single_scan_grid.grid_to_world(single_scan_grid.ijk.float()).jdata
                single_scan_feat = self.single_scan_pos_embedder(single_scan_coords)
                single_scan_feat = VDBTensor(single_scan_grid, single_scan_grid.jagged_like(single_scan_feat))
            single_scan_cond = self.single_scan_cond_model(single_scan_feat, single_scan_hash_tree)
            # align this feature to the latent
            single_scan_cond = noisy_latents.grid.fill_to_grid(single_scan_cond.feature, single_scan_cond.grid, 0.0)
            if not is_testing and self.hparams.use_classifier_free:
                single_scan_cond = self.conduct_classifier_free(single_scan_cond, noisy_latents.grid.grid_count, noisy_latents.grid.device)             
            concat_list.append(single_scan_cond)
        if self.hparams.use_normal_concat_cond:
            # traing-time: get single scan crop from batch
            if batch is not None:
                # assert self.hparams.use_fvdb_loader is True, "use_fvdb_loader should be True for normal concat condition"
                ref_grid = fvdb.cat(batch[DS.INPUT_PC])    
                ref_xyz = ref_grid.grid_to_world(ref_grid.ijk.float()) 
                concat_normal = noisy_latents.grid.splat_trilinear(ref_xyz, fvdb.JaggedTensor(batch[DS.TARGET_NORMAL]))
            else:
                concat_normal = cond_dict['normal']
            concat_normal.jdata /= (concat_normal.jdata.norm(dim=1, keepdim=True) + 1e-6) # avoid nan
            if not is_testing and self.hparams.use_classifier_free:
                concat_normal = self.conduct_classifier_free(concat_normal, noisy_latents.grid.grid_count, noisy_latents.grid.device)            
            concat_list.append(concat_normal)

        if do_classifier_free_guidance and len(concat_list) > 0: # ! not tested yet
            if not self.hparams.use_classifier_free:
                logger.info("Applying classifier-free guidance without doing it for concat condition")
                concat_list_copy = concat_list
            else:
                # logger.info("Applying classifier-free guidance for concat condition")    
                # assert self.hparams.use_classifier_free, "do_classifier_free_guidance should be used with use_classifier_free"
                concat_list_copy = []
                for cond in concat_list:
                    cond = self.conduct_classifier_free(cond, noisy_latents.grid.grid_count, noisy_latents.grid.device, is_testing=True)
                    concat_list_copy.append(cond)
        
        # ! corssattn part
        # text condition
        if self.hparams.use_text_cond:
            # traing-time: get text from batch
            if batch is not None:
                text_emb = torch.stack(batch[DS.TEXT_EMBEDDING]) # B, 77, 1024
                mask = torch.stack(batch[DS.TEXT_EMBEDDING_MASK]) # B, 77
            else:
                text_emb = cond_dict['text_emb']
                mask = cond_dict['text_emb_mask']                
            context = text_emb
            if do_classifier_free_guidance:
                context_copy = cond_dict['text_emb_null']
                mask_copy = cond_dict['text_emb_mask_null']

        # concat pos_emb
        if self.hparams.use_pos_embed:
            pos_embed = noisy_latents.grid.ijk
            noisy_latents = VDBTensor.cat([noisy_latents, pos_embed], dim=1)
        elif self.hparams.use_pos_embed_high:
            pos_embed = self.get_pos_embed_high(noisy_latents.grid.ijk.jdata)
            noisy_latents = VDBTensor.cat([noisy_latents, pos_embed], dim=1)
        elif self.hparams.use_pos_embed_world:
            pos_embed = noisy_latents.grid.grid_to_world(noisy_latents.grid.ijk.float())
            noisy_latents = VDBTensor.cat([noisy_latents, pos_embed], dim=1)
        elif self.hparams.use_pos_embed_world_high:
            pos_embed = noisy_latents.grid.grid_to_world(noisy_latents.grid.ijk.float())
            pos_embed = self.get_pos_embed_high(pos_embed.jdata)
            noisy_latents = VDBTensor.cat([noisy_latents, pos_embed], dim=1)

        if self.hparams.conditioning_key == 'none':
            model_pred = self.unet(noisy_latents, timesteps)
        elif self.hparams.conditioning_key == 'concat':
            assert len(concat_list) > 0, "concat_list should not be empty"
            noisy_latents_in = VDBTensor.cat([noisy_latents] + concat_list, dim=1)
            model_pred = self.unet(noisy_latents_in, timesteps)
            
            if do_classifier_free_guidance:
                noisy_latents_in_copy = VDBTensor.cat([noisy_latents] + concat_list_copy, dim=1)
                model_pred_copy = self.unet(noisy_latents_in_copy, timesteps)
                model_pred = VDBTensor(model_pred.grid, model_pred.grid.jagged_like(model_pred.feature.jdata + guidance_scale * (model_pred.feature.jdata - model_pred_copy.feature.jdata)))
        elif self.hparams.conditioning_key == 'adm':
            assert cond is not None, "cond should not be None"
            model_pred = self.unet(noisy_latents, timesteps, y=cond)
        elif self.hparams.conditioning_key == 'crossattn':
            assert context is not None, "context should not be None"
            model_pred = self.unet(noisy_latents, timesteps, context=context, mask=mask)
            
            if do_classifier_free_guidance:
                model_pred_copy = self.unet(noisy_latents, timesteps, context=context_copy, mask=mask_copy)
                model_pred = VDBTensor(model_pred.grid, model_pred.grid.jagged_like(model_pred.feature.jdata + guidance_scale * (model_pred.feature.jdata - model_pred_copy.feature.jdata)))
        elif self.hparams.conditioning_key == 'c_crossattn':
            assert len(concat_list) > 0, "concat_list should not be empty"
            assert context is not None, "context should not be None"
            noisy_latents_in = VDBTensor.cat([noisy_latents] + concat_list, dim=1)
            model_pred = self.unet(noisy_latents_in, timesteps, context=context, mask=mask)
            
            if do_classifier_free_guidance:
                noisy_latents_in_copy = VDBTensor.cat([noisy_latents] + concat_list_copy, dim=1)
                model_pred_copy = self.unet(noisy_latents_in_copy, timesteps, context=context_copy, mask=mask_copy)
                model_pred = VDBTensor(model_pred.grid, model_pred.grid.jagged_like(model_pred.feature.jdata + guidance_scale * (model_pred.feature.jdata - model_pred_copy.feature.jdata)))
        else:
            raise NotImplementedError

        return model_pred

    @exp.mem_profile(every=1)
    def forward(self, batch, out: dict):
        # first get latent from vae
        with torch.no_grad():
            latents = self.extract_latent(batch)

        # To Do: scale the latent
        if self.hparams.scale_by_std:
            latents = latents * self.scale_factor

        # then get the noise
        latent_data = latents.feature.jdata
        noise = torch.randn_like(latent_data) # N, C

        bsz = latents.grid.grid_count
        if self.hparams.use_noise_offset:
            noise_offset = torch.randn(bsz, noise.shape[1], device=noise.device) * self.hparams.noise_offset_scale
            noise += noise_offset[latents.feature.jidx.long()]

        # Sample a random timestep for each latent
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device) # B
        timesteps_sparse = timesteps.long()
        timesteps_sparse = timesteps_sparse[latents.feature.jidx.long()] # N, 1

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latent_data, noise, timesteps_sparse)

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latent_data, noise, timesteps_sparse)
        elif self.noise_scheduler.config.prediction_type == "sample":
            target = latent_data
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        # Predict the noise residual and compute loss
        # forward_cond function use batch-level timesteps
        noisy_latents = VDBTensor(grid=latents.grid, feature=latents.grid.jagged_like(noisy_latents))
        model_pred = self._forward_cond(noisy_latents, timesteps, batch)

        out.update({'pred': model_pred.feature.jdata})
        out.update({'target': target})

        return out
    
    def get_random_sample_pcs(self, ijk: fvdb.JaggedTensor, batch_size=1, M=3, use_center=False):
        # M: sample per point
        # !: be careful about the batch_size
        output_ijk = []
        for idx in range(batch_size):
            current_ijk = ijk[idx].jdata.float() # N, 3
            if use_center:
                output_ijk.append(current_ijk)
            else:            
                N = current_ijk.shape[0]
                # create offsets of size M*N x 3 with values in range [-0.5, 0.5]
                offsets = torch.FloatTensor(N * M, 3).uniform_(-0.5, 0.5).to(current_ijk.device)
                # duplicate your original point cloud M times
                expanded_point_cloud = current_ijk.repeat(M, 1)
                # add offsets to duplicated points
                expanded_point_cloud += offsets
                output_ijk.append(expanded_point_cloud)
        return fvdb.JaggedTensor(output_ijk)
            
    def decode_to_points(self, latents):
        res = self.vae.unet.FeaturesSet()
        res, output_x = self.vae.unet.decode(res, latents, is_testing=True)
        grid_tree = res.structure_grid
        fine_list = []
        coarse_list = []
        for batch_idx in range(output_x.grid.grid_count):
            # fineest level
            pd_grid_0 = grid_tree[0]
            pd_xyz_0 = pd_grid_0.grid_to_world(self.get_random_sample_pcs(pd_grid_0.ijk[batch_idx], M=8))
            fine_list.append(pd_xyz_0.jdata.cpu().numpy())
            # coarsest level
            pd_grid_1 = grid_tree[len(grid_tree.keys()) - 1]
            pd_xyz_1 = pd_grid_1.grid_to_world(self.get_random_sample_pcs(pd_grid_1.ijk[batch_idx], M=3))
            coarse_list.append(pd_xyz_1.jdata.cpu().numpy())
        # plot the fine and coarse level
        viz_fine = vis_pcs(fine_list)
        viz_coarse = vis_pcs(coarse_list)
        decode_results = np.concatenate([viz_fine, viz_coarse], axis=0)
        return decode_results
    
    @exp.mem_profile(every=1)
    def compute_loss(self, batch, out, compute_metric: bool):
        loss_dict = exp.TorchLossMeter()
        metric_dict = exp.TorchLossMeter()

        # compute the MSE loss
        if self.hparams.supervision.mse_weight > 0.0:
            loss_dict.add_loss("mse", F.mse_loss(out["pred"], out["target"]), self.hparams.supervision.mse_weight)
        if compute_metric: # currently use MSE as metric
            metric_dict.add_loss("mse", F.mse_loss(out["pred"], out["target"]))

        return loss_dict, metric_dict 
    
    def train_val_step(self, batch, batch_idx, is_val):
        if batch_idx % 100 == 0:
            # Squeeze memory really hard :)
            gc.collect()
            torch.cuda.empty_cache()

        out = {'idx': batch_idx}
        with exp.pt_profile_named("forward"):
            out = self(batch, out)

        if out is None and not is_val:
            return None

        with exp.pt_profile_named("loss"):
            loss_dict, metric_dict = self.compute_loss(batch, out, compute_metric=is_val)

        if not is_val:
            self.log_dict_prefix('train_loss', loss_dict)
        else:
            self.log_dict_prefix('val_metric', metric_dict)
            self.log_dict_prefix('val_loss', loss_dict)
            if self.hparams.log_image:
                cond_dict = {}
                if self.hparams.use_single_scan_concat_cond:
                    pass
                
                if self.hparams.use_text_cond:
                    text_emb = torch.stack(batch[DS.TEXT_EMBEDDING]) # B, 77, 1024
                    mask = torch.stack(batch[DS.TEXT_EMBEDDING_MASK]) # B, 77
                    cond_dict['text_emb'] = text_emb
                    cond_dict['text_emb_mask'] = mask
                
                if self.trainer.global_rank == 0: # only log the image on rank 0
                    if batch_idx == 0:
                        logger.info("running visualisation on rank 0...")
                        with self.ema_scope("Plotting"):
                            # first extract latent
                            clean_latents = self.extract_latent(batch)
                            grids = clean_latents.grid
                            sample_latents = self.random_sample_latents(grids, use_ddim=self.hparams.use_ddim, ddim_step=100, cond_dict=cond_dict) # TODO: change this ddim_step to variable
                            # decode clean latents first
                            decode_clean = self.decode_to_points(clean_latents)
                            decode_sample = self.decode_to_points(sample_latents)
                            sample = np.concatenate([decode_clean, decode_sample], axis=0)
                            self.log_image("img/sample", sample)
                            # clean matplotlib opens
                            plt.close('all')
                else:
                    if batch_idx == 0:
                        clean_latents = self.extract_latent(batch)
                        grids = clean_latents.grid
                        _ = self.random_sample_latents(grids, use_ddim=self.hparams.use_ddim, ddim_step=100, cond_dict=cond_dict) # TODO: change this ddim_step to variable

        loss_sum = loss_dict.get_sum()
        self.log('val_loss' if is_val else 'train_loss/sum', loss_sum)
        self.log('val_step', self.global_step)

        return loss_sum
    
    def evaluation_api(self, batch = None, grids: GridBatch = None, batch_size: int = None, latent_prev: VDBTensor = None, 
                       use_ddim=False, ddim_step=100, use_ema=True, use_dpm=False, use_karras=False, solver_order=3,
                       h_stride=1, guidance_scale: float = 1.0, 
                       cond_dict=None, res_coarse=None, guided_grid=None):
        """
        * @param grids: GridBatch from previous stage for conditional diffusion
        * @param batch_size: batch_size for unconditional diffusion
        * @param latent_prev: previous stage latent for conditional diffusion; not implemented yet
        * @param use_ddim: use DDIM or not
        * @param ddim_step: number of steps for DDIM
        * @param use_dpm: use DPM++ solver or not
        * @param use_karras: use Karras noise schedule or not 
        * @param solver_order: order of the solver; 3 for unconditional diffusion, 2 for guided sampling
        * @param use_ema: use EMA or not
        * @param h_stride: flag for remain_h VAE to create a anisotropic latent grid
        * @param cond_dict: conditional dictionary -> only pass if manully effort needed
        * @param res_coarse: previous stage result (semantics, normals, etc) for conditional diffusion
        """
        if grids is None: 
            if batch is not None:
                latents = self.extract_latent(batch)
                grids = latents.grid
            else:
                # use dense diffusion
                # create a dense grid
                assert batch_size is not None, "batch_size should be provided"
                feat_depth = self.vae.hparams.tree_depth - 1
                gap_stride = 2 ** feat_depth
                gap_strides = [gap_stride, gap_stride, gap_stride // h_stride]
                if isinstance(self.hparams.network.diffuser.image_size, int):
                    neck_bound = int(self.hparams.network.diffuser.image_size / 2)
                    low_bound = [-neck_bound] * 3
                    voxel_bound = [neck_bound * 2] * 3
                else:        
                    voxel_bound = self.hparams.network.diffuser.image_size
                    low_bound = [- int(res / 2) for res in self.hparams.network.diffuser.image_size]
                
                voxel_sizes = [sv * gap for sv, gap in zip(self.vae.hparams.voxel_size, gap_strides)] # !: carefully setup
                origins = [sv / 2. for sv in voxel_sizes]
                grids = fvdb.sparse_grid_from_dense(
                                batch_size, 
                                voxel_bound, 
                                low_bound, 
                                device="cpu", # hack to fix bugs
                                voxel_sizes=voxel_sizes,
                                origins=origins).to(self.device)
        # parse the cond_dict
        if cond_dict is None:
            cond_dict = {}
        if self.hparams.use_semantic_cond:
            # check if semantics is in cond_dict
            if 'semantics' not in cond_dict:
                if batch is not None:
                    cond_dict['semantics'] = fvdb.JaggedTensor(batch[DS.LATENT_SEMANTIC])
                elif res_coarse is not None:
                    cond_semantic = res_coarse.semantic_features[-1].feature.jdata # N, class_num
                    cond_semantic = torch.argmax(cond_semantic, dim=1)
                    cond_dict['semantics'] = grids.jagged_like(cond_semantic)
                else:
                    raise NotImplementedError("No semantics provided")
        if self.hparams.use_normal_concat_cond:
            # traing-time: get single scan crop from batch
            if batch is not None:
                ref_grid = fvdb.cat(batch[DS.INPUT_PC])    
                ref_xyz = ref_grid.grid_to_world(ref_grid.ijk.float()) 
                concat_normal = grids.splat_trilinear(ref_xyz, fvdb.JaggedTensor(batch[DS.TARGET_NORMAL]))
            elif res_coarse is not None:
                concat_normal = res_coarse.normal_features[-1].feature # N, 3
                concat_normal.jdata /= (concat_normal.jdata.norm(dim=1, keepdim=True) + 1e-6) # avoid nan
            else:
                raise NotImplementedError("No normal provided")
            cond_dict['normal'] = concat_normal                
        
        # diffusion process
        if use_ema:
            with self.ema_scope("Evaluation API"):
                latents = self.random_sample_latents(grids, use_ddim=use_ddim, ddim_step=ddim_step, use_dpm=use_dpm, use_karras=use_karras, solver_order=solver_order,
                                                     cond_dict=cond_dict, guidance_scale=guidance_scale)
        else:
            latents = self.random_sample_latents(grids, use_ddim=use_ddim, ddim_step=ddim_step, use_dpm=use_dpm, use_karras=use_karras, solver_order=solver_order,
                                                     cond_dict=cond_dict, guidance_scale=guidance_scale)
        # decode
        res = self.vae.unet.FeaturesSet()
        if guided_grid is None:
            res, output_x = self.vae.unet.decode(res, latents, is_testing=True)
        else:
            res, output_x = self.vae.unet.decode(res, latents, guided_grid)
        # TODO: add SDF output
        return res, output_x
    
    def random_sample_latents(self, grids: GridBatch, generator: torch.Generator = None, 
                              use_ddim=False, ddim_step=None, use_dpm=False, use_karras=False, solver_order=3,
                              cond_dict=None, guidance_scale=1.0) -> VDBTensor:
        if use_ddim:
            if ddim_step is None:
                ddim_step = self.hparams.num_inference_steps
            self.ddim_scheduler.set_timesteps(ddim_step, device=grids.device)
            timesteps = self.ddim_scheduler.timesteps
            scheduler = self.ddim_scheduler
        elif use_dpm:
            logger.info("Using DPM++ solver with order %d and karras %s" % (solver_order, use_karras))
            if ddim_step is None:
                ddim_step = self.hparams.num_inference_steps
            try:
                self.dpm_scheduler.set_timesteps(ddim_step, device=grids.device)
            except:
                # create a new dpm scheduler
                self.dpm_scheduler = DPMSolverMultistepScheduler(
                    num_train_timesteps=self.hparams.network.scheduler.num_train_timesteps,
                    beta_start=self.hparams.network.scheduler.beta_start,
                    beta_end=self.hparams.network.scheduler.beta_end,
                    beta_schedule=self.hparams.network.scheduler.beta_schedule,
                    solver_order=solver_order,
                    prediction_type=self.hparams.network.scheduler.prediction_type,
                    algorithm_type="dpmsolver++",
                    use_karras_sigmas=use_karras,
                )
                self.dpm_scheduler.set_timesteps(ddim_step, device=grids.device)
            timesteps = self.dpm_scheduler.timesteps
            scheduler = self.dpm_scheduler
        else:
            timesteps = self.noise_scheduler.timesteps
            scheduler = self.noise_scheduler
        
        # prepare the latents
        latents = torch.randn(grids.total_voxels, self.unet.out_channels, device=grids.device, generator=generator)
        
        for i, t in tqdm(enumerate(timesteps)):
            latent_model_input = latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            latent_model_input = VDBTensor(grid=grids, feature=grids.jagged_like(latent_model_input))
            noise_pred = self._forward_cond(latent_model_input, t, cond_dict=cond_dict, is_testing=True, guidance_scale=guidance_scale) # TODO: cond
            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred.feature.jdata, t, latents).prev_sample # TODO: when there is scale model input, why there is latents
            
        # scale the latents to the original scale
        if self.hparams.scale_by_std:
            latents = 1. / self.scale_factor * latents
        
        return VDBTensor(grid=grids, feature=grids.jagged_like(latents))
    
    def get_dataset_spec(self):
        all_specs = self.vae.get_dataset_spec()
        # further add new specs
        if self.hparams.use_semantic_cond:
            all_specs.append(DS.LATENT_SEMANTIC)
        if self.hparams.use_single_scan_concat_cond:
            all_specs.append(DS.SINGLE_SCAN_CROP)
            all_specs.append(DS.SINGLE_SCAN)
        if self.hparams.use_class_cond:
            all_specs.append(DS.CLASS)
        if self.hparams.use_text_cond:
            all_specs.append(DS.TEXT_EMBEDDING)
            all_specs.append(DS.TEXT_EMBEDDING_MASK)
        if self.hparams.use_micro_cond:
            all_specs.append(DS.MICRO)
        return all_specs
    
    def get_collate_fn(self):
        return list_collate
    
    def get_hparams_metrics(self):
        return [('val_loss', True)]

    def train_dataloader(self):
        import xcube.data as dataset
        from torch.utils.data import DataLoader

        shuffle = True
        train_set = dataset.build_dataset(
            self.hparams.train_dataset, self.get_dataset_spec(), self.hparams, self.hparams.train_kwargs, duplicate_num=self.hparams.duplicate_num) # !: A change here for adding duplicate num for trainset without lantet

        batch_size = self.hparams.batch_size
        
        return DataLoader(train_set, batch_size=batch_size, shuffle=shuffle,
                          num_workers=self.hparams.train_val_num_workers, collate_fn=self.get_collate_fn())

    def val_dataloader(self):
        import xcube.data as dataset
        from torch.utils.data import DataLoader
        val_set = dataset.build_dataset(
            self.hparams.val_dataset, self.get_dataset_spec(), self.hparams, self.hparams.val_kwargs)
        return DataLoader(val_set, batch_size=self.hparams.batch_size_val, shuffle=False,
                          num_workers=self.hparams.train_val_num_workers, collate_fn=self.get_collate_fn())
    
    def test_dataloader(self):
        import xcube.data as dataset
        from torch.utils.data import DataLoader

        test_set = dataset.build_dataset(
            self.hparams.test_dataset, self.get_dataset_spec(), self.hparams, self.hparams.test_kwargs)
        if self.hparams.test_set_shuffle:
            torch.manual_seed(0)
        if not hasattr(self.hparams, 'batch_len'):
            self.hparams.batch_len = 1
        return DataLoader(test_set, batch_size=self.hparams.batch_len, shuffle=self.hparams.test_set_shuffle, 
                          num_workers=0, collate_fn=self.get_collate_fn())

    def configure_optimizers(self):
        # overwrite this from base model to double make sure vae is fixed
        lr_config = self.hparams.learning_rate
        if self.hparams.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.trainer.model.parameters(), lr=lr_config['init'], momentum=0.9,
                                        weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'Adam':
            # AdamW corrects the bad weight dacay implementation in Adam.
            # AMSGrad also do some corrections to the original Adam.
            # The learning rate here is the maximum rate we can reach for each parameter.                   
            optimizer = torch.optim.AdamW(self.trainer.model.parameters(), lr=lr_config['init'],
                                        weight_decay=self.hparams.weight_decay, amsgrad=True)
        else:
            raise NotImplementedError

        # build scheduler
        import functools
        from torch.optim.lr_scheduler import LambdaLR
        scheduler = LambdaLR(optimizer,
                             lr_lambda=functools.partial(
                                 lambda_lr_wrapper, lr_config=lr_config, batch_size=self.hparams.batch_size))

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]