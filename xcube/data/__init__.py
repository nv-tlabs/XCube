# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from .shapenet import ShapeNetDataset
from .waymo import WaymoDataset
from .objaverse import ObjaverseDataset

def build_dataset(name: str, spec, hparams, kwargs: dict, duplicate_num=1):
    return eval(name)(**kwargs, spec=spec, hparams=hparams, duplicate_num=duplicate_num)