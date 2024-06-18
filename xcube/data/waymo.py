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
import math
from loguru import logger

from xcube.data.base import DatasetSpec as DS
from xcube.data.base import RandomSafeDataset

import fvdb
fvdb._Cpp.SparseGridBatch = fvdb._Cpp.GridBatch

import pickle
custom_pickle = pickle
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "featurevdb._Cpp":
            module = "fvdb._Cpp"
        return super().find_class(module, name)
custom_pickle.Unpickler = CustomUnpickler

def random_crop(point_cloud, intensity, crop_size, M=1):
    """
    Removes a region from a point cloud based on a random crop.

    Args:
    - point_cloud (torch.Tensor): A tensor of shape (N, 3) representing the 3D coordinates of the points.
    - crop_size (tuple): A tuple of (dx, dy, dz) indicating the size of the 3D sub-volume to be removed.

    Returns:
    - remaining_points (torch.Tensor): The points outside the removed sub-volume.
    """
    
    # Compute the min and max values for each dimension of the point cloud
    min_vals, _ = torch.min(point_cloud, dim=0)
    max_vals, _ = torch.max(point_cloud, dim=0)
    
    remaining_points = point_cloud
    remaining_intensity = intensity
    for _ in range(M):
        # Compute the starting point of the random crop
        start_x = torch.rand(1) * (max_vals[0] - crop_size[0] - min_vals[0]) + min_vals[0]
        start_y = torch.rand(1) * (max_vals[1] - crop_size[1] - min_vals[1]) + min_vals[1]
        start_z = torch.rand(1) * (max_vals[2] - crop_size[2] - min_vals[2]) + min_vals[2]

        # Define the bounding box of the crop
        crop_min = torch.tensor([start_x, start_y, start_z])
        crop_max = crop_min + torch.tensor(crop_size)
        
        # Create a mask for points that lie inside the bounding box
        mask_inside_bbox = torch.all((remaining_points >= crop_min) & (remaining_points <= crop_max), dim=1)
        
        # Use the mask to get the points outside the bounding box
        remaining_points = remaining_points[~mask_inside_bbox]
        remaining_intensity = remaining_intensity[~mask_inside_bbox]

    return remaining_points, remaining_intensity


class WaymoDataset(RandomSafeDataset):
    def __init__(self, base_path, split, resolution, spec=None,
                 random_seed=0, hparams=None, skip_on_error=False, custom_name="scene", 
                 micro_key=[], voxel_num_interval=25000, car_voxel_num_interval=5000, 
                 single_scan_crop=False, single_scan_crop_size=[10.0, 10.0, 10.0], single_scan_crop_num=[10, 20],
                 duplicate_num=1, **kwargs):
        if isinstance(random_seed, str):
            super().__init__(0, True, skip_on_error)
        else:
            super().__init__(random_seed, False, skip_on_error)
        self.skip_on_error = skip_on_error
        self.custom_name = custom_name
        self.resolution = resolution

        self.split = split
        if spec is None:
            self.spec = [DS.INPUT_PC]
        else:
            self.spec = spec
        
        # Get all items
        self.all_items = []
        split_file = os.path.join(base_path, (split + '.lst'))
        with open(split_file, 'r') as f:
            models_c = f.read().split('\n')
        if '' in models_c:
            models_c.remove('')
        self.all_items += [os.path.join(base_path, str(resolution), "%s.pkl" % m) for m in models_c]
        
        logger.info(f"WaymoDataset: {len(self.all_items)} items")
        self.hparams = hparams
        
        # micro condition        
        self.micro_key = micro_key
        self.voxel_num_interval = voxel_num_interval
        self.car_voxel_num_interval = car_voxel_num_interval
        
        # single scan random crop
        self.single_scan_crop = single_scan_crop
        self.single_scan_crop_size = single_scan_crop_size
        self.single_scan_crop_num_min = single_scan_crop_num[0]
        self.single_scan_crop_num_max = single_scan_crop_num[1]

        self.duplicate_num = duplicate_num

    def __len__(self):
        return len(self.all_items) * self.duplicate_num

    def get_name(self):
        return f"{self.custom_name}-{self.split}"
    
    def get_short_name(self):
        return self.custom_name

    def _get_item(self, data_id, rng):
        data = {}
        input_data = torch.load(self.all_items[data_id % len(self.all_items)], pickle_module=custom_pickle)
        input_points = input_data['points']
        input_normals = input_data['normals'].jdata
        shape_name = self.all_items[data_id % len(self.all_items)]

        if DS.SHAPE_NAME in self.spec:
            data[DS.SHAPE_NAME] = shape_name

        if DS.TARGET_NORMAL in self.spec:
            data[DS.TARGET_NORMAL] = input_normals
        
        if DS.INPUT_PC in self.spec:
            data[DS.INPUT_PC] = input_points
                
        if DS.GT_DENSE_PC in self.spec:
            data[DS.GT_DENSE_PC] = input_points

        if DS.GT_DENSE_NORMAL in self.spec:
            data[DS.GT_DENSE_NORMAL] = input_normals
            
        if DS.GT_DYN_FLAG in self.spec:
            data[DS.GT_DYN_FLAG] = input_data["dynamic_flag"]
            
        if DS.GT_SEMANTIC in self.spec:
            data[DS.GT_SEMANTIC] = input_data["semantics"]
                
        if DS.LATENT_SEMANTIC in self.spec:
            latent_semantic = input_data["latent_semantics"]
            data[DS.LATENT_SEMANTIC] = latent_semantic
            
        if DS.INPUT_INTENSITY in self.spec:
            data[DS.INPUT_INTENSITY] = input_data['intensity'] # N, 1
            
        if DS.SINGLE_SCAN in self.spec:
            data[DS.SINGLE_SCAN] = input_data['cond_xyz']
            data[DS.SINGLE_SCAN_INTENSITY] = input_data['cond_intensity'] / 255.0
        
        if DS.SINGLE_SCAN_CROP in self.spec:
            cond_xyz_crop = input_data['cond_xyz_crop']
            cond_intensity_crop = input_data['cond_intensity_crop'] / 255.0
            
            if self.single_scan_crop: # use as training augumentation
                # random a total_num for crop bbox
                total_num = random.randint(self.single_scan_crop_num_min, self.single_scan_crop_num_max)
                # random crop
                cond_xyz_crop, cond_intensity_crop = random_crop(cond_xyz_crop, cond_intensity_crop, self.single_scan_crop_size, M=total_num)
                
            data[DS.SINGLE_SCAN_CROP] = cond_xyz_crop
            data[DS.SINGLE_SCAN_INTENSITY_CROP] = cond_intensity_crop
            
        if DS.MICRO in self.spec:
            micro = []
            H = (input_points.ijk.jdata[:, 0].max() -  input_points.ijk.jdata[:, 0].min()).item()
            W = (input_points.ijk.jdata[:, 1].max() -  input_points.ijk.jdata[:, 1].min()).item()
            D = (input_points.ijk.jdata[:, 2].max() -  input_points.ijk.jdata[:, 2].min()).item()

            H_ = 2 ** round(math.log2(H))
            W_ = 2 ** round(math.log2(W))
            D_ = 2 ** round(math.log2(D))
            N = 2 ** round(input_points.total_voxels / self.voxel_num_interval)
            
            C = (input_data["semantics"] == 1).sum().item() # car number
            C_ = 2 ** round(C / self.car_voxel_num_interval)
            
            R = 2 ** round((input_points.total_voxels - C) / self.voxel_num_interval)
            
            if "H" in self.micro_key:
                micro.append(H_)
            if "W" in self.micro_key:
                micro.append(W_)
            if "D" in self.micro_key:
                micro.append(D_)
            if "N" in self.micro_key:
                micro.append(N)
            if "C" in self.micro_key:
                micro.append(C_)
            if "R" in self.micro_key: 
                micro.append(R)
                
            data[DS.MICRO] = torch.Tensor(micro)

        return data
