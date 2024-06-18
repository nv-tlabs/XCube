import os
import polyscope as ps
import argparse
import torch
import trimesh
import numpy as np
import point_cloud_utils as pcu

parser = argparse.ArgumentParser()
parser.add_argument('-p', "--path", type=str, required=True)
parser.add_argument('-i', "--id", type=int, default=0)
args = parser.parse_args()

# load result
result_dict_path = os.path.join(args.path, f"result_dict_{args.id}.pkl")
result_dict = torch.load(result_dict_path)

ps.init()
ps.set_ground_plane_mode("none")
# coarse stage
coarse_xyz = result_dict["coarse_xyz"]
coarse_normal = result_dict["coarse_normal"]
coarse_normal_color = coarse_normal * 0.5 + 0.5

pc = ps.register_point_cloud(f"Coarse Point", coarse_xyz)
pc.add_color_quantity("normal", coarse_normal_color, enabled=True)

# fine stage
fine_xyz = result_dict["fine_xyz"]
fine_normal = result_dict["fine_normal"]
fine_normal = fine_normal / (np.linalg.norm(fine_normal, axis=1, keepdims=True) + 1e-6)

pc = ps.register_point_cloud(f"Fine Point", fine_xyz)
fine_normal_color = fine_normal * 0.5 + 0.5
pc.add_color_quantity("normal", fine_normal_color, enabled=True)

# Mesh
mesh = trimesh.load(os.path.join(args.path, f"mesh/mesh_{args.id}.obj"))
mesh_n = pcu.estimate_mesh_vertex_normals(mesh.vertices, mesh.faces)
mesh_c = (mesh_n + 1) / 2

ps.register_surface_mesh(f"NKSR", mesh.vertices, mesh.faces).add_color_quantity("normal", mesh_c, enabled=True) 
ps.show()