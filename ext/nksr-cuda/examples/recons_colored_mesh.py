import torch
import numpy as np
from pycg import vis, exp
from nksr import Reconstructor, utils, fields

if __name__ == '__main__':
    device = torch.device("cuda:0")

    test_geom = vis.from_file("/workspace/nkf-wild/nvscan/iphone_doggies.ply")
    # test_geom = vis.from_file("/workspace/nkf-wild/nvscan/stanford_burghers.ply")
    # test_geom = vis.from_file("/workspace/nkf-wild/nvscan/stanford_stonewall.ply")
    # test_geom = vis.from_file("/workspace/nkf-wild/nvscan/stanford_totempole.ply")

    input_xyz = torch.from_numpy(np.asarray(test_geom.points)).float().to(device)
    input_normal = torch.from_numpy(np.asarray(test_geom.normals)).float().to(device)
    input_color = torch.from_numpy(np.asarray(test_geom.colors)).float().to(device)

    nksr = Reconstructor(device)

    field = nksr.reconstruct(input_xyz, input_normal, solver_tol=1.0e-4, detail_level=0.8)
    field.set_texture_field(fields.PCNNField(input_xyz, input_color))

    mesh = field.extract_dual_mesh(max_points=2 ** 22, mise_iter=1)
    mesh = vis.mesh(mesh.v, mesh.f, color=mesh.c)

    vis.show_3d([mesh], [test_geom])
