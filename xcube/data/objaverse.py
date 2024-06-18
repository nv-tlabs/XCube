# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import torch
import random

from xcube.data.base import DatasetSpec as DS
from xcube.data.base import RandomSafeDataset

class ObjaverseDataset(RandomSafeDataset):
    def __init__(self, onet_base_path, spec, split, resolution, image_base_path=None,
                 random_seed=0, hparams=None, skip_on_error=False, custom_name="objaverse",
                 text_emb_path="../data/objaverse/objaverse/text_emb", null_embed_path="./assets/null_text_emb.pkl", text_embed_drop_prob=0.0, max_text_len=77,
                 duplicate_num=1, split_base_path=None, **kwargs):
        if isinstance(random_seed, str):
            super().__init__(0, True, skip_on_error)
        else:
            super().__init__(random_seed, False, skip_on_error)

        self.skip_on_error = skip_on_error
        self.custom_name = custom_name
        self.resolution = resolution
        self.split = split
        self.spec = spec
        
        # setup path
        self.onet_base_path = onet_base_path
        if split_base_path is None:
            split_base_path = onet_base_path
        split_file = os.path.join(split_base_path, (split + '.lst'))
        if image_base_path is None:
            image_base_path = onet_base_path
        self.image_base_path = image_base_path
        
        with open(split_file, 'r') as f:
            models_c = f.read().split('\n')
        if '' in models_c:
            models_c.remove('')
        self.models = [{'category': m.split("/")[-2], 'model': m.split("/")[-1]} for m in models_c]
        self.hparams = hparams
        
        # setup text condition
        if DS.TEXT_EMBEDDING in self.spec:
            self.text_emb_path = text_emb_path
            self.null_text_emb = torch.load(null_embed_path)
            self.max_text_len = max_text_len
            self.text_embed_drop_prob = text_embed_drop_prob
        
        self.duplicate_num = duplicate_num

    def __len__(self):
        return len(self.models) * self.duplicate_num
            
    def get_name(self):
        return f"{self.custom_name}-{self.split}"

    def get_short_name(self):
        return self.custom_name
    
    def get_null_text_emb(self):
        null_text_emb = self.null_text_emb['text_embed_sd_model.last_hidden_state'] # 2, 1024
        return self.padding_text_emb(null_text_emb)
        
    def padding_text_emb(self, text_emb):
        padded_text_emb = torch.zeros(self.max_text_len, text_emb.shape[1])
        padded_text_emb[:text_emb.shape[0]] = text_emb
        mask = torch.zeros(self.max_text_len)
        mask[:text_emb.shape[0]] = 1
        return padded_text_emb, mask.bool()
        
    def _get_item(self, data_id, rng):
        category = self.models[data_id % len(self.models)]['category']
        model = self.models[data_id % len(self.models)]['model']
        data = {}
        input_data = torch.load(os.path.join(self.onet_base_path, category, model, "%s.pkl" % self.resolution))
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

        if DS.TEXT_EMBEDDING in self.spec:
            # first sample prob to drop text embedding
            if random.random() < self.text_embed_drop_prob:
                # drop the text
                text_emb, text_mask = self.get_null_text_emb()
                caption = ""
            else:
                text_emb_path = os.path.join(self.text_emb_path, model + ".pkl")
                if os.path.exists(text_emb_path):
                    text_emb_data = torch.load(text_emb_path)
                    text_emb = text_emb_data['text_embed_sd_model.last_hidden_state']
                    text_emb, text_mask = self.padding_text_emb(text_emb)
                    caption = text_emb_data['caption']
                else:
                    text_emb, text_mask = self.get_null_text_emb()
                    caption = ""
            data[DS.TEXT_EMBEDDING] = text_emb.detach()
            data[DS.TEXT_EMBEDDING_MASK] = text_mask.detach()
            data[DS.TEXT] = caption
        
        return data
