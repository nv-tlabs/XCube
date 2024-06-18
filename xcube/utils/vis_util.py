import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from pycg import color

def vis_pcs(pcl_lst, S=3, vis_order=[2,0,1], bound=1):
    fig = plt.figure(figsize=(3 * len(pcl_lst), 3))
    num_col = len(pcl_lst)
    for idx, pts in enumerate(pcl_lst):
        ax1 = fig.add_subplot(1, num_col, 1 + idx, projection='3d')
        rgb = None
        psize = S 
        
        # normalize the points
        if pts.size > 0:
            if np.abs(pts).max() > bound:
                pts = pts / np.abs(pts).max()
        
        ax1.scatter(pts[:, vis_order[0]], -pts[:, vis_order[1]], pts[:, vis_order[2]], s=psize, c=rgb)
        ax1.set_xlim(-bound, bound)
        ax1.set_ylim(-bound, bound)
        ax1.set_zlim(-bound, bound)
        ax1.grid(False)
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig) # close the figure to avoid memory leak
    return image_from_plot


def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


WAYMO_CATEGORY_NAMES = [
    "UNDEFINED", "CAR", "TRUCK", "BUS", "OTHER_VEHICLE", "MOTORCYCLIST", "BICYCLIST", "PEDESTRIAN",
    "SIGN", "TRAFFIC_LIGHT", "POLE", "CONSTRUCTION_CONE", "BICYCLE", "MOTORCYCLE", "BUILDING",
    "VEGETATION", "TREE_TRUNK", "CURB", "ROAD", "LANE_MARKER", "OTHER_GROUND", "WALKABLE", "SIDEWALK"
]

WAYMO_MAPPED = {
    0: ["SIGN", "TRAFFIC_LIGHT", "POLE", "CONSTRUCTION_CONE", "UNDEFINED"],
    1: ["MOTORCYCLIST", "BICYCLIST", "PEDESTRIAN", "BICYCLE", "MOTORCYCLE"],
    3: ["CAR", "TRUCK", "BUS", "OTHER_VEHICLE"],
    5: ["CURB", "LANE_MARKER"],
    4: ["VEGETATION", "TREE_TRUNK"],
    2: ["WALKABLE", "SIDEWALK"],
    6: ["BUILDING"],
    7: ["ROAD", "OTHER_GROUND"],
}

WAYMO_MAPPED_USER_STUDY = {
    0: ["SIGN", "TRAFFIC_LIGHT", "POLE", "CONSTRUCTION_CONE", "UNDEFINED"],
    1: ["MOTORCYCLIST", "BICYCLIST", "PEDESTRIAN", "BICYCLE", "MOTORCYCLE"],
    3: ["CAR", "TRUCK", "BUS", "OTHER_VEHICLE"],
    5: [],
    4: ["VEGETATION", "TREE_TRUNK"],
    2: ["WALKABLE", "SIDEWALK"],
    6: ["BUILDING"],
    7: ["ROAD", "OTHER_GROUND", "CURB", "LANE_MARKER"],
}

waymo_mapping = np.zeros(23, dtype=np.int32)
for wkey, warr in WAYMO_MAPPED.items():
    for w in warr:
        waymo_mapping[WAYMO_CATEGORY_NAMES.index(w)] = wkey

waymo_palette = color.get_cmap_array('Set2')
# Change the purple and green color
waymo_palette[3] = color.get_cmap_array('Set3')[9]
waymo_palette[4] = color.get_cmap_array('Set1')[2]