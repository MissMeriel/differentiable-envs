import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import h5py
import quality_fork as qf
import torch
from pytorch3d.structures import Meshes
import pytorch3d.io
from tqdm import tqdm
import pytorch3d_ext as re
import numpy as np
import random
from run_gqcnn import *
from select_grasp import *
import traceback

class dexnet_db:
    def __init__(self,path):
        self.data_ = h5py.File(path, 'r')

def config_as_torch(configuration, device='cpu'):

    if not isinstance(configuration, np.ndarray) or (configuration.shape[0] != 9 and configuration.shape[0] != 10):
        raise ValueError('Configuration must be numpy ndarray of size 9 or 10')
    if configuration.shape[0] == 9:
        min_grasp_width = 0
    else:
        min_grasp_width = configuration[9]
    if np.abs(np.linalg.norm(configuration[3:6]) - 1.0) > 1e-5:
        raise ValueError('Illegal grasp axis. Must be norm one')
    center3D = torch.from_numpy(configuration[0:3]).float().to(device).unsqueeze(0)
    axis3D = torch.from_numpy(configuration[3:6]).float().to(device).unsqueeze(0)
    max_width = configuration[6]
    # angle, jaw_width, min_width unused
    return center3D, axis3D, max_width


if __name__ == "__main__":
    db = dexnet_db('/mnt/array/Home/Data/HPSTA/dexnet_database/dexnet_2.0_training_database/dexnet_2_database.hdf5')
    device = torch.device("cpu")
    datasets = db.data_['datasets']
    r = re.Renderer(device=device)
    exp_root_dir = 'exp-10-plots-fix-scaling-allow-cf0-january-paper'

    for dataset in datasets:
        object_list = ['8362720e40cf45cdc0d3742cb5ee0e30']
        for object in tqdm(object_list,desc=f'from {dataset}',leave=False):
            try:
                inner = db.data_['datasets'][dataset]['objects'][object]
                for name in db.data_['datasets'][dataset]['objects'][object]['grasps']:
                    verts = db.data_['datasets'][dataset]['objects'][object]['mesh']['vertices'][:]
                    faces = db.data_['datasets'][dataset]['objects'][object]['mesh']['triangles'][:]

                    mesh = r.pre_process_object(verts,faces)
                    pytorch3d.io.save_obj(f'{exp_root_dir}/debug_mesh_attack/{object}.obj', mesh.verts_packed(),mesh.faces_packed())
            except Exception as e:
                pass
