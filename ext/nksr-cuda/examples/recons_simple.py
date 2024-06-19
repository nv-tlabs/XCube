import torch
import numpy as np
import nksr
from pycg import vis
from common import load_bunny_example, warning_on_low_memory


if __name__ == '__main__':
    warning_on_low_memory(1024.0)
    device = torch.device("cuda:0")

    bunny_geom = load_bunny_example()

    input_xyz = torch.from_numpy(np.asarray(bunny_geom.points)).float().to(device)
    input_normal = torch.from_numpy(np.asarray(bunny_geom.normals)).float().to(device)

    reconstructor = nksr.Reconstructor(device)
    field = reconstructor.reconstruct(input_xyz, input_normal, detail_level=1.0)
    mesh = field.extract_dual_mesh(mise_iter=1)

    vis.show_3d([vis.mesh(mesh.v, mesh.f)], [bunny_geom])
