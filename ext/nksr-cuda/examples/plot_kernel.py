import torch
import numpy as np
import nksr
from pycg import vis
from common import load_bunny_example
from sklearn.decomposition import PCA


if __name__ == "__main__":
    device = torch.device("cuda:0")

    bunny_geom = load_bunny_example()

    input_xyz = torch.from_numpy(np.asarray(bunny_geom.points)).float().to(device)
    input_normal = torch.from_numpy(np.asarray(bunny_geom.normals)).float().to(device)

    # Reconstruct the mesh
    reconstructor = nksr.Reconstructor(device)
    field = reconstructor.reconstruct(input_xyz, input_normal, detail_level=0.2)
    mesh = field.extract_dual_mesh(mise_iter=1)

    # Evaluate the kernel features on the mesh
    #   Here, I compute the features for each level individually but you can also concat them.
    pca = PCA(n_components=3)
    for d in range(field.svh.depth):
        feat_d = field.evaluate_kernel(mesh.v, d, grad=False)
        # Do PCA to reduce dimension to 3 using sklearn
        feat_d = pca.fit_transform(feat_d.cpu().numpy())
        feat_d = (feat_d - feat_d.min()) / (feat_d.max() - feat_d.min())
        # Visualize the features
        vis.show_3d([vis.mesh(mesh.v, mesh.f, color=feat_d)])
