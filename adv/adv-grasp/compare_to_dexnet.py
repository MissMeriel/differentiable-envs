import h5py
import quality_fork as qf
import torch
from pytorch3d.structures import Meshes
import pytorch3d.io
from tqdm import tqdm
import render as re
import numpy as np
from run_gqcnn import *
from select_grasp import *

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
    gpu_id = 1

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
    else:
        print("cuda not available")
        device = torch.device("cpu")

    r = re.Renderer(device=device)

    config_dict = {
        "torque_scaling":1000,
        "soft_fingers":1,
        "friction_coef": 0.8, # TODO use 0.8 in practice
        "antipodality_pctile": 1.0 
    }
    cf = qf.CannyFerrariQualityFunction(config_dict,min_quality=1)
    mw = qf.minWeightQualityFunction(config_dict,min_quality=1)
    rcf = qf.RobustCannyFerrariQualityFunction(config_dict,min_quality=1)
    rmw = qf.RobustMinWeightQualityFunction(config_dict,min_quality=1)
    r = re.Renderer(device=device)
    adv_grasp_dir = 'adv/adv-grasp/'
    model = KitModel(os.path.join(adv_grasp_dir,"weights.npy"),device=device)
    model.eval()
    datasets = db.data_['datasets']
    print(datasets.keys())

    with open('quality_compare.csv', 'w') as f:
        for dataset in datasets:
            
            for object in tqdm(db.data_['datasets'][dataset]['objects'],desc=f'from {dataset}',leave=False):
                inner = db.data_['datasets'][dataset]['objects'][object]
                for name in db.data_['datasets'][dataset]['objects'][object]['grasps']:
                    verts = db.data_['datasets'][dataset]['objects'][object]['mesh']['vertices'][:]
                    faces = db.data_['datasets'][dataset]['objects'][object]['mesh']['triangles'][:]
                   
                    try:
                        mesh = r.pre_process_object(verts,faces)
                        mesh_props = qf.mesh_properties(mesh)
                    except Exception as e:
                        tqdm.write(f'failed to compute connectivity on {object}, saving for debug')
                        tqdm.write(str(e))
                        pytorch3d.io.save_obj(f'debug_mesh/{object}.obj', mesh.verts_packed(),mesh.faces_packed())
                        f.write(f'{object}, 0, 2, {0}, 0, {0}, 0, {0}, {0}, 0, {0}, {0}\n')

                        break

                    for grasp in tqdm(db.data_['datasets'][dataset]['objects'][object]['grasps'][name],desc=f'grasp from {object} wt: {mesh_props.is_watertight} inv: {mesh_props.is_inverted}',leave=False):
                        try:
                            cf_db = db.data_['datasets'][dataset]['objects'][object]['grasps'][name][grasp]['metrics'].attrs['ferrari_canny'] 
                            rcf_db = db.data_['datasets'][dataset]['objects'][object]['grasps'][name][grasp]['metrics'].attrs['robust_ferrari_canny'] 
                            config = db.data_['datasets'][dataset]['objects'][object]['grasps'][name][grasp].attrs['configuration']
                            center3D, axis3D, max_width = config_as_torch(config, device=device)
                            graspObj = qf.GraspTorch(center3D, axis3D=axis3D, width=max_width,
                                                    friction_coef=config_dict["friction_coef"], torque_scaling=config_dict["torque_scaling"])
                            graspObj = graspObj.apply_to_mesh(mesh, is_watertight=mesh_props.is_watertight, is_inverted=mesh_props.is_inverted, use_dexnet_normal=False)
                            
                            graspObj.make2D(camera_intr=r.camera)
                            grasp_h = Grasp(depth=graspObj.depth.float(),im_center=graspObj.center.float(),
                                    im_angle=graspObj.angle.float(),im_axis=graspObj.axis.float(), 
                                    world_center=graspObj.center3D.float(), world_axis=graspObj.axis3D.float(),
                                    oracle_method='pytorch',device=device)
                            
                            # adv_mesh_clone = mesh.clone()
                            dim = r.mesh_to_depth_im(mesh, display=False)
                            pose, image = grasp_h.extract_tensors_batch(dim)
                            out = model(pose, image)
                            gqcnn_val = out[:,0:1].to(mesh.device)
                            
                            cf_val = cf(mesh, graspObj)
                            rcf_val = rcf(mesh, graspObj)
                            mw_val = mw(mesh,graspObj)
                            rmw_val = rmw(mesh,graspObj)

                            tqdm.write(f'cf: db: {cf_db:.4f} ours: {cf_val.item():.4f} rcf: db: {rcf_db:.4f}, ours: {rcf_val.item():.4f} mw: {mw_val.item():.4f} rmw: {rmw_val.item():.4f} gqcnn {gqcnn_val.item():.4f}')
                            f.write(f'{object}, {grasp}, 0, {cf_db:.8f}, {cf_val.item():.8f}, {rcf_db:.8f}, {rcf_val.item():.8f}, {mw_val.item():.8f}, {rmw_val.item():.8f}, {gqcnn_val.item():.8f}, {mesh_props.is_watertight}, {mesh_props.is_inverted}\n')
                        except Exception as e:
                            tqdm.write(str(e))
                            tqdm.write(f'failed to compute quality on {object} grasp {grasp}, saving for debug')
                            pytorch3d.io.save_obj(f'debug_mesh/{object}.obj', mesh.verts_packed(),mesh.faces_packed())
                            graspObj.write_obj(f'debug_mesh/{object}_{grasp}.obj', include_coordinate=True, include_line_o_action=True)
                            f.write(f'{object}, {grasp}, 1, {cf_db:.8f}, 0, {rcf_db:.8f}, 0, 0, 0, 0, {mesh_props.is_watertight}, {mesh_props.is_inverted}\n')

