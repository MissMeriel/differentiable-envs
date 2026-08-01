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
    exp_root_dir = 'debug_throwaway'
    adv_grasp_dir = 'adv/adv-grasp/'
    model = KitModel(os.path.join(adv_grasp_dir,"weights.npy"),device=device)
    model.eval()
    config_dict = {
        "torque_scaling":1000,
        "soft_fingers":1,
        "friction_coef": 0.8, # TODO use 0.8 in practice
        "antipodality_pctile": 1.0 
    }

    if not os.path.isdir(f'{exp_root_dir}'):
        os.mkdir(f'{exp_root_dir}')
        os.mkdir(f'{exp_root_dir}/debug_mesh_attack')


    for dataset in datasets:
        object_list = list(db.data_['datasets'][dataset]['objects'])
        random.shuffle(object_list)
        # object_list=['94e289c89059106bd8f74b0004a598cd']
        for object in tqdm(object_list,desc=f'from {dataset}',leave=False):

                inner = db.data_['datasets'][dataset]['objects'][object]
                for name in db.data_['datasets'][dataset]['objects'][object]['grasps']:
                    verts = db.data_['datasets'][dataset]['objects'][object]['mesh']['vertices'][:]
                    faces = db.data_['datasets'][dataset]['objects'][object]['mesh']['triangles'][:]

                    mesh = r.pre_process_object(verts,faces)
                    # pytorch3d.io.save_obj(f'{exp_root_dir}/debug_mesh_attack/{object}.obj', mesh.verts_packed(),mesh.faces_packed())
                    grasp_list = list(db.data_['datasets'][dataset]['objects'][object]['grasps'][name])
                    random.shuffle(grasp_list)
                    grasp_list = grasp_list[:1]
                    # grasp_list = ['grasp_90']
                    for index_grasp, grasp_name in tqdm(enumerate(grasp_list), desc=f'Considering grasps to attack in {object}', leave=False):
                            config = db.data_['datasets'][dataset]['objects'][object]['grasps'][name][grasp_name].attrs['configuration']

                            center3D, axis3D, max_width = config_as_torch(config, device=device)
                            graspObj = qf.GraspTorch(center3D, world_axis=axis3D, width=max_width,
                                                    friction_coef=config_dict["friction_coef"], torque_scaling=config_dict["torque_scaling"])
                            graspObj.make2D(camera_intr=r.camera)

                            graspObj = graspObj.apply_to_mesh(mesh, is_watertight=False, 
                                                            is_inverted=False, use_dexnet_normal=False)
                            dim = r.mesh_to_depth_im(mesh, display=False)
                            pose_batch, image_batch = qf.GQCNNQualityFunction.extract_tensors_batch(grasp=graspObj, d_ims=dim)
                            out_batch = model(pose_batch, image_batch)
                            pose, image = qf.GQCNNQualityFunction.extract_tensors(grasp=graspObj, d_im=dim)
                            out = model(pose, image)
                            # np.savetxt(f'{exp_root_dir}/debug_mesh_attack/{object}_{grasp_name}.csv', image.squeeze().numpy(force=True), delimiter=",")
                            # np.savetxt(f'{exp_root_dir}/debug_mesh_attack/{object}_{grasp_name}_batch_after.csv', image_batch.squeeze().numpy(force=True), delimiter=",")

                            tqdm.write(f'q: {out[0,1]:0.4f}, qb: {out_batch[0,1]:0.4f}  max depth diff {torch.max(torch.abs(image-image_batch)):0.7f} pose: {pose.item():0.3f}, poseb: {pose_batch.item():0.3f}, {object}, {grasp_name}')
                            
            
            

