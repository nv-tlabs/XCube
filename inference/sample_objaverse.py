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
from transformers import CLIPModel, AutoProcessor, CLIPTextModel, AutoTokenizer

from xcube.utils.vis_util import random_seed
from xcube.utils import exp

def padding_text_emb(text_emb, max_text_len=77):
        padded_text_emb = torch.zeros(max_text_len, text_emb.shape[1])
        padded_text_emb[:text_emb.shape[0]] = text_emb
        mask = torch.zeros(max_text_len)
        mask[:text_emb.shape[0]] = 1
        return padded_text_emb, mask.bool()

def _setup_clip_model(device, clip_tag='h14'):
    clip_names = {
        'l14': 'openai/clip-vit-large-patch14',
        'h14': 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
    }
    clip_version = clip_names[clip_tag]
    clip_preprocess = AutoProcessor.from_pretrained(clip_version, cache_dir="../cache")
    return clip_preprocess

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

config_coarse = "configs/objaverse/train_diffusion_16x16x16_dense_text_cond.yaml"
ckpt_coarse = "checkpoints/objaverse/coarse_diffusion/last.ckpt"

config_fine = "configs/objaverse/train_diffusion_128x128x128_sparse_text_cond.yaml"
ckpt_fine = "checkpoints/objaverse/fine_diffusion/last.ckpt"

config_nksr = "configs/objaverse/train_nksr_refine.yaml"
ckpt_nksr = "checkpoints/objaverse/nksr_refine/last.ckpt"

net_model = create_model_from_args(config_coarse, ckpt_coarse).cuda()
net_model_c = create_model_from_args(config_fine, ckpt_fine).cuda()

# load null_text_embedding
null_text_emb = torch.load("./assets/null_text_emb.pkl")
null_text_emb = null_text_emb['text_embed_sd_model.last_hidden_state'] # 2, 1024
null_text_emb, null_text_emb_mask = padding_text_emb(null_text_emb, max_text_len=77)

# setup clip model
device = torch.device('cuda')
clip_preprocess = _setup_clip_model(device)
sd_text_encoder = CLIPTextModel.from_pretrained(
    'stabilityai/stable-diffusion-2-1-base',
    subfolder="text_encoder", cache_dir="../cache").to(device)

# setup nksr
if known_args.extract_mesh:
    import nksr
    reconstructor = create_model_from_args(config_nksr, ckpt_nksr, strict=False).cuda()

# begin sample pcs for evaluation
known_args.category = "objaverse"
logger.info(f"Sampling from XCube on {known_args.category} ...")
save_folder = f"./results/{known_args.category}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
logger.info(f"Saving results to {save_folder}")
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

with torch.no_grad():
    while True:
        user_input = input("Please enter a prompt: ")
        strength = float(input("Please enter a strength: "))
        logger.info("Sampling for prompt: %s" % user_input)

        bcaps = [user_input]
        inputs = clip_preprocess(text=bcaps, return_tensors="pt", padding=True,
                                max_length=77, truncation=True)
        inputs.data = {k: v.to(device) for k, v in inputs.data.items()}
        text_embed_sd_model = sd_text_encoder.text_model(**inputs)
        text_emb = text_embed_sd_model.last_hidden_state[0]
        text_emb, mask = padding_text_emb(text_emb, max_text_len=77)

        batch_size = known_args.batch_len
        batch_size_temp = 2
        
        cond_dict = {}
        cond_dict["text_emb"] = text_emb.unsqueeze(0).repeat(batch_size_temp, 1, 1).to(device)
        cond_dict["text_emb_mask"] = mask.unsqueeze(0).repeat(batch_size_temp, 1).to(device)
        cond_dict["text_emb_null"] = null_text_emb.unsqueeze(0).repeat(batch_size_temp, 1, 1).to(device)
        cond_dict["text_emb_mask_null"] = null_text_emb_mask.unsqueeze(0).repeat(batch_size_temp, 1).to(device)

        for _ in range(known_args.batch_len // batch_size_temp):
            res_coarse, output_x_coarse = net_model.evaluation_api(batch_size=batch_size_temp, 
                                                                use_ddim=known_args.use_ddim, 
                                                                ddim_step=known_args.ddim_step,
                                                                use_dpm=known_args.use_dpm, 
                                                                solver_order=known_args.solver_order, 
                                                                use_karras=known_args.use_karras,  
                                                                use_ema=known_args.ema,
                                                                cond_dict=cond_dict,
                                                                guidance_scale=strength)
                
            res, output_x = net_model_c.evaluation_api(grids=output_x_coarse.grid, 
                                                        use_ddim=known_args.use_ddim, 
                                                        ddim_step=known_args.ddim_step,
                                                        use_dpm=known_args.use_dpm, 
                                                        solver_order=known_args.solver_order, 
                                                        use_karras=known_args.use_karras, 
                                                        use_ema=known_args.ema, 
                                                        res_coarse=res_coarse,
                                                        cond_dict=cond_dict)
            
            # get result grids
            cur_save_folder = os.path.join(save_folder, "%s" % user_input.replace(" ", "_"))
            os.makedirs(cur_save_folder, exist_ok=True)
            exist_obj_number = len(os.listdir(cur_save_folder))

            for batch_idx in range(output_x.grid.grid_count):
                ## coarse stage
                result_dict = {}
                result_dict['coarse_xyz'] = output_x_coarse.grid.grid_to_world(output_x_coarse.grid[batch_idx].ijk.float()).jdata.cpu().numpy()
                result_dict['coarse_normal'] = res_coarse.normal_features[-1].feature[batch_idx].jdata.cpu().numpy() 
                ## fine stage
                result_dict['fine_xyz'] = output_x.grid.grid_to_world(output_x.grid[batch_idx].ijk.float()).jdata.cpu().numpy()
                result_dict['fine_normal'] = res.normal_features[-1].feature[batch_idx].jdata.cpu().numpy()

                # save result_dict
                cur_idx = batch_idx + exist_obj_number
                result_path = os.path.join(cur_save_folder, "result_dict_%d.pkl" % cur_idx)
                torch.save(result_dict, result_path)
                        
                # extract mesh from grid
                if known_args.extract_mesh:
                    cur_mesh_folder = os.path.join(cur_save_folder, "mesh")
                    os.makedirs(cur_mesh_folder, exist_ok=True)
                    pd_grid = output_x.grid[batch_idx]
                    pd_normal = res.normal_features[-1].feature[batch_idx].jdata
                    with torch.no_grad():
                        field = reconstructor.forward({'in_grid': pd_grid, 'in_normal': pd_normal})
                    field = field['neural_udf']
                    mesh = field.extract_dual_mesh(grid_upsample=2)
                    # save mesh
                    mesh_save = trimesh.Trimesh(vertices=mesh.v.cpu().numpy(), faces=mesh.f.cpu().numpy())
                    mesh_save.export(os.path.join(cur_mesh_folder, "mesh_%d.obj" % cur_idx))