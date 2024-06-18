# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
from pathlib import Path

import fvdb
import torch
from loguru import logger

from xcube.data.base import DatasetSpec as DS
from xcube.data.base import RandomSafeDataset

class ShapeNetDataset(RandomSafeDataset):
    def __init__(self, onet_base_path, spec, split, resolution, categories=None,
                 random_seed=0, hparams=None, skip_on_error=False, custom_name="shapenet", duplicate_num=1,
                 **kwargs):
        if isinstance(random_seed, str):
            super().__init__(0, True, skip_on_error)
        else:
            super().__init__(random_seed, False, skip_on_error)

        self.resolution = resolution
        onet_base_path = os.path.join(onet_base_path, str(resolution))
             
        self.skip_on_error = skip_on_error
        self.custom_name = custom_name

        self.split = split
        self.spec = spec

        # If categories is None, use all sub-folders
        if categories is None:
            base_path = Path(onet_base_path)
            categories = os.listdir(base_path)
            categories = [c for c in categories if (base_path / c).is_dir()]
        self.categories = categories

        # Get all models
        self.models = []
        self.onet_base_paths = {}
        for c in categories:
            self.onet_base_paths[c] = Path(onet_base_path + "/" + c)
            split_file = self.onet_base_paths[c] / (split + '.lst')
            with split_file.open('r') as f:
                models_c = f.read().split('\n')
            if '' in models_c:
                models_c.remove('')
            self.models += [{'category': c, 'model': m} for m in models_c]
        self.hparams = hparams

        self.duplicate_num = duplicate_num

    def __len__(self):
        return len(self.models) * self.duplicate_num
            
    def get_name(self):
        return f"{self.custom_name}-cat{len(self.categories)}-{self.split}"

    def get_short_name(self):
        return self.custom_name

    def _get_item(self, data_id, rng):
        category = self.models[data_id % len(self.models)]['category']
        model = self.models[data_id % len(self.models)]['model']

        data = {}
        input_data = torch.load(os.path.join(self.onet_base_paths[category], "%s.pkl" % model))

        input_points = input_data['points']
        input_normals = input_data['normals'].jdata
    
        if DS.SHAPE_NAME in self.spec:
            data[DS.SHAPE_NAME] = "/".join([category, model])

        if DS.TARGET_NORMAL in self.spec:
            data[DS.TARGET_NORMAL] = input_normals
    
        if DS.INPUT_PC in self.spec:
            data[DS.INPUT_PC] = input_points
                
        if DS.GT_DENSE_PC in self.spec:
            data[DS.GT_DENSE_PC] = input_points

        if DS.GT_DENSE_NORMAL in self.spec:
            data[DS.GT_DENSE_NORMAL] = input_normals

        return data
