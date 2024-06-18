import os
import polyscope as ps
import argparse
import torch
import trimesh
import numpy as np
import point_cloud_utils as pcu
from xcube.utils.vis_util import waymo_palette, waymo_mapping

parser = argparse.ArgumentParser()
parser.add_argument('-p', "--path", type=str, required=True)
parser.add_argument('-i', "--id", type=int, default=0)
args = parser.parse_args()

# load result
result_dict_path = os.path.join(args.path, f"result_dict_{args.id}.pkl")
result_dict = torch.load(result_dict_path)

ps.init()
ps.set_up_dir("z_up")
ps.set_ground_plane_mode("none")
# coarse stage
coarse_xyz = result_dict["coarse_xyz"]
coarse_normal = result_dict["coarse_normal"]
coarse_normal_color = coarse_normal * 0.5 + 0.5
coarse_semantic = result_dict["coarse_semantic"]
coarse_semantic = waymo_mapping[coarse_semantic]
coarse_semantic_color = waymo_palette[coarse_semantic]

pc = ps.register_point_cloud(f"Coarse Point", coarse_xyz)
pc.add_color_quantity("semantic", coarse_semantic_color, enabled=True)

# fine stage
fine_xyz = result_dict["fine_xyz"]
fine_normal = result_dict["fine_normal"]
fine_normal = fine_normal / (np.linalg.norm(fine_normal, axis=1, keepdims=True) + 1e-6)
fine_semantic = result_dict["fine_semantic"]
fine_semantic = waymo_mapping[fine_semantic]
fine_semantic_color = waymo_palette[fine_semantic]

pc = ps.register_point_cloud(f"Fine Point", fine_xyz)
fine_normal_color = fine_normal * 0.5 + 0.5
pc.add_color_quantity("semantic", fine_semantic_color, enabled=True)

# Mesh
mesh = trimesh.load(os.path.join(args.path, f"mesh/udf_mesh_{args.id}.obj"))
# mesh_c = result_dict["udf_mesh_color"]
mesh_c = mesh.visual.vertex_colors[:, :3] / 255.0
ps.register_surface_mesh(f"NKSR", mesh.vertices, mesh.faces).add_color_quantity("semantic", mesh_c, enabled=True) 
ps.show()