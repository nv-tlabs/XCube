# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

def hparams_handler(hparams):
    # !: anisotropic setting
    if not hasattr(hparams, 'remain_h'):
        hparams.remain_h = False
    if isinstance(hparams.voxel_size, int) or isinstance(hparams.voxel_size, float):
        hparams.voxel_size = [hparams.voxel_size] * 3

    # !: pretrain weight
    if not hasattr(hparams, 'pretrained_weight'):
        hparams.pretrained_weight = None
    
    hparams.use_input_color = False
    hparams.with_color_branch = False
    hparams.supervision.color_weight = 0.0
        
    hparams.with_normal_branch = False
    if not hasattr(hparams.supervision, 'normal_weight'):
        hparams.supervision.normal_weight = 0.0
    if hparams.supervision.normal_weight > 0:
        hparams.with_normal_branch = True
        
    hparams.with_semantic_branch = False
    if not hasattr(hparams.supervision, 'semantic_weight'):
        hparams.supervision.semantic_weight = 0.0
    if hparams.supervision.semantic_weight > 0:
        hparams.with_semantic_branch = True

    return hparams