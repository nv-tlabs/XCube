import os

import pytorch_lightning as pl
import argparse
import importlib
import torch
import numpy as np
import pickle
from loguru import logger 
import random
from fvdb.nn import VDBTensor
from pathlib import Path
from datetime import datetime
import trimesh

from xcube.utils.vis_util import random_seed
from xcube.utils import exp


def get_default_parser():
    default_parser = argparse.ArgumentParser(add_help=False)
    default_parser = pl.Trainer.add_argparse_args(default_parser)
    return default_parser

def create_model_from_args(config_path, ckpt_path, strict=True):
    model_yaml_path = Path(config_path)
    model_args = exp.parse_config_yaml(model_yaml_path)
    net_module = importlib.import_module("xcube.models." + model_args.model).Model
    args_ckpt = Path(ckpt_path)
    assert args_ckpt.exists(), "Selected checkpoint does not exist!"
    net_model = net_module.load_from_checkpoint(args_ckpt, hparams=model_args, strict=strict)
    return net_model.eval()

def get_parser():
    parser = exp.ArgumentParserX(base_config_path='../structure-ldm/configs/default/param.yaml', parents=[get_default_parser()])
    parser.add_argument('--category', type=str, required=True, help='ShapeNet category.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--total_len', type=int, default=700, help='Number of samples to generate.')
    parser.add_argument('--batch_len', type=int, default=64, help='Number of samples to generate in each batch.')
    parser.add_argument('--ema', action='store_true', help='Whether to turn on ema option.')
    parser.add_argument('--use_ddim', action='store_true', help='Whether to use ddim during sampling.')
    parser.add_argument('--ddim_step', type=int, default=50, help='Number of steps to increase ddim.')
    parser.add_argument('--use_dpm', action='store_true', help='Whether to use dpm solver during sampling.')
    parser.add_argument('--use_karras', action='store_true', help='Whether to use karras std during sampling.')
    parser.add_argument('--solver_order', type=int, default=3, help='Order of DPM solver.')
    parser.add_argument('--extract_mesh', action='store_true', help='Whether to extract mesh.')
    parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel size used to extract mesh.')
    return parser

known_args = get_parser().parse_known_args()[0]
random_seed(known_args.seed)

# setup model config and path
if known_args.category == "chair":
    config_coarse = "configs/shapenet/chair/train_diffusion_16x16x16_dense.yaml"
    ckpt_coarse = "checkpoints/chair/coarse_diffusion/last.ckpt"

    config_fine = "configs/shapenet/chair/train_diffusion_128x128x128_sparse.yaml"
    ckpt_fine = "checkpoints/chair/fine_diffusion/last.ckpt"

    config_nksr = "configs/shapenet/chair/train_nksr_refine.yaml"
    ckpt_nksr = "checkpoints/chair/nksr_refine/last.ckpt"
elif known_args.category == "car":
    config_coarse = "configs/shapenet/car/train_diffusion_16x16x16_dense.yaml"
    ckpt_coarse = "checkpoints/car/coarse_diffusion/last.ckpt"

    config_fine = "configs/shapenet/car/train_diffusion_128x128x128_sparse.yaml"
    ckpt_fine = "checkpoints/car/fine_diffusion/last.ckpt"

    config_nksr = "configs/shapenet/car/train_nksr_refine.yaml"
    ckpt_nksr = "checkpoints/car/nksr_refine/last.ckpt"
elif known_args.category == "plane":
    config_coarse = "configs/shapenet/plane/train_diffusion_16x16x16_dense.yaml"
    ckpt_coarse = "checkpoints/plane/coarse_diffusion/last.ckpt"

    config_fine = "configs/shapenet/plane/train_diffusion_128x128x128_sparse.yaml"
    ckpt_fine = "checkpoints/plane/fine_diffusion/last.ckpt"

    config_nksr = "configs/shapenet/plane/train_nksr_refine.yaml"
    ckpt_nksr = "checkpoints/plane/nksr_refine/last.ckpt"
else:
    raise ValueError("Unknown category: %s" % known_args.category)

net_model = create_model_from_args(config_coarse, ckpt_coarse).cuda()
net_model_c = create_model_from_args(config_fine, ckpt_fine).cuda()

# setup nksr
if known_args.extract_mesh:
    import nksr
    reconstructor = create_model_from_args(config_nksr, ckpt_nksr, strict=False).cuda()

# begin sample pcs for evaluation
logger.info(f"Sampling from XCube on {known_args.category} ...")
save_folder = f"./results/{known_args.category}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
logger.info(f"Saving results to {save_folder}")
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

if known_args.extract_mesh:
    mesh_folder = os.path.join(save_folder, "mesh")
    if not os.path.exists(mesh_folder):
        os.makedirs(mesh_folder)

with torch.no_grad():
    num_sample = 0
    while True:
        logger.info("Sampling %d / %d" % (num_sample, known_args.total_len))
        
        res_coarse, output_x_coarse = net_model.evaluation_api(batch_size=known_args.batch_len, 
                                                            use_ddim=known_args.use_ddim, 
                                                            ddim_step=known_args.ddim_step,
                                                            use_dpm=known_args.use_dpm, 
                                                            solver_order=known_args.solver_order, 
                                                            use_karras=known_args.use_karras,  
                                                            use_ema=known_args.ema)
            
        res, output_x = net_model_c.evaluation_api(grids=output_x_coarse.grid, 
                                                    use_ddim=known_args.use_ddim, 
                                                    ddim_step=known_args.ddim_step,
                                                    use_dpm=known_args.use_dpm, 
                                                    solver_order=known_args.solver_order, 
                                                    use_karras=known_args.use_karras, 
                                                    use_ema=known_args.ema, 
                                                    res_coarse=res_coarse,
                                                    )
        
        # get result grids
        for batch_idx in range(output_x.grid.grid_count):
            ## coarse stage
            result_dict = {}
            result_dict['coarse_xyz'] = output_x_coarse.grid.grid_to_world(output_x_coarse.grid[batch_idx].ijk.float()).jdata.cpu().numpy()
            result_dict['coarse_normal'] = res_coarse.normal_features[-1].feature[batch_idx].jdata.cpu().numpy() 
            ## fine stage
            result_dict['fine_xyz'] = output_x.grid.grid_to_world(output_x.grid[batch_idx].ijk.float()).jdata.cpu().numpy()
            result_dict['fine_normal'] = res.normal_features[-1].feature[batch_idx].jdata.cpu().numpy()

            # save result_dict
            result_path = os.path.join(save_folder, "result_dict_%d.pkl" % num_sample)
            torch.save(result_dict, result_path)
                    
            # extract mesh from grid
            if known_args.extract_mesh:
                # pd_xyz = output_x.grid.grid_to_world(output_x.grid[batch_idx].ijk.float()).jdata
                pd_grid = output_x.grid[batch_idx]
                pd_normal = res.normal_features[-1].feature[batch_idx].jdata
                with torch.no_grad():
                    field = reconstructor.forward({'in_grid': pd_grid, 'in_normal': pd_normal})
                field = field['kernel_sdf']
                mesh = field.extract_dual_mesh(max_depth=0, grid_upsample=2)
                # save mesh
                mesh_save = trimesh.Trimesh(vertices=mesh.v.cpu().numpy(), faces=mesh.f.cpu().numpy())
                # mesh_save.merge_vertices()
                mesh_save.export(os.path.join(mesh_folder, "mesh_%d.obj" % num_sample))

            num_sample += 1

        if num_sample >= known_args.total_len:
            break