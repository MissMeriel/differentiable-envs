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
import gc

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
    gpu_id = 0

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
    mom=0
    lr0 = 1e-4
    cf = qf.CannyFerrariQualityFunction(config_dict,min_quality=1)
    rcf = qf.RobustCannyFerrariQualityFunction(config_dict,min_quality=1)
    datasets = db.data_['datasets']
    print(datasets.keys())
    adv_grasp_dir = 'adv/adv-grasp/'
    exp_root_dir = 'dexnet-loop-lower-collision-saturate-higher-lr/'

    model = KitModel(os.path.join(adv_grasp_dir,"weights.npy"),device=device)
    model.eval()
    run1 = Attack(num_plots=20, steps_per_plot=25, model=model, renderer=r, oracle_method="pytorch")

    os.mkdir(f'{exp_root_dir}')
    os.mkdir(f'{exp_root_dir}/debug_mesh_attack')

    for dataset in datasets:
        object_list = list(db.data_['datasets'][dataset]['objects'])
        # object_list = ['bc910044930d827950a69e53f7a35c05']
        random.shuffle(object_list)
        for object in tqdm(object_list,desc=f'from {dataset}',leave=False):
            inner = db.data_['datasets'][dataset]['objects'][object]
            for name in db.data_['datasets'][dataset]['objects'][object]['grasps']:
                verts = db.data_['datasets'][dataset]['objects'][object]['mesh']['vertices'][:]
                faces = db.data_['datasets'][dataset]['objects'][object]['mesh']['triangles'][:]
                
                try:
                    # gc.collect()
                    mesh = r.pre_process_object(verts,faces)
                    mesh_props = re.mesh_properties(mesh)
                    minDist = mesh_props.self_collision_min(mesh,forceNormalDist=0).item()
                except Exception as e:
                    tqdm.write(f'failed to compute connectivity on {object}, saving for debug faces: {faces.shape} verts: {verts.shape}')
                    tqdm.write(str(e))
                    #
                    log = open(f'{exp_root_dir}/debug_mesh_attack/{object}_crash.txt', 'a')
                    log.write(str(e))
                    log.close()
                    #pytorch3d.io.save_obj(f'debug_mesh_attack/{object}.obj', mesh.verts_packed(),mesh.faces_packed())

                    break
                try:
                    graspObj = random.choice(list(db.data_['datasets'][dataset]['objects'][object]['grasps'][name]))
                except Exception as e:
                    tqdm.write(f'failed to random choice on {object}, saving for debug faces: {faces.shape} verts: {verts.shape}')
                    tqdm.write(str(e))
                    #
                    log = open(f'{exp_root_dir}/debug_mesh_attack/{object}_empty.txt', 'a')
                    log.write(str(e))
                    log.close()
                    break
                #grasp_name = 'grasp_21'
                if minDist < 1e-9:
                    open(f'{exp_root_dir}/debug_mesh_attack/{object}_collision.txt', 'a').close()
                else:
                    try:
                        # gc.collect()
                        cf_db = db.data_['datasets'][dataset]['objects'][object]['grasps'][name][graspObj]['metrics'].attrs['ferrari_canny'] 
                        rcf_db = db.data_['datasets'][dataset]['objects'][object]['grasps'][name][graspObj]['metrics'].attrs['robust_ferrari_canny'] 
                        config = db.data_['datasets'][dataset]['objects'][object]['grasps'][name][graspObj].attrs['configuration']
                        center3D, axis3D, max_width = config_as_torch(config, device=device)
                        graspObj = qf.GraspTorch(center3D, world_axis=axis3D, width=max_width,
                                                friction_coef=config_dict["friction_coef"], torque_scaling=config_dict["torque_scaling"])
                        graspObj.make2D(camera_intr=r.camera)
                        # grasp_h = GraspTorch(depth=graspObj.depth.float(),im_center=graspObj.im_center.float(),
                        #                 im_angle=graspObj.im_angle.float(),im_axis=graspObj.im_axis.float(), 
                        #                 world_center=graspObj.world_center.float(), world_axis=graspObj.world_axis.float(),
                        #                 oracle_method='pytorch')
                        graspObj = graspObj.apply_to_mesh(mesh, is_watertight=mesh_props.is_watertight, 
                                                          is_inverted=mesh_props.is_inverted, use_dexnet_normal=False)
                        # cf_val = cf(mesh, graspObj)
                        # rcf_val = rcf(mesh, graspObj)
                        
                        run1.attack(mesh=mesh, grasp=graspObj, dir=f"{exp_root_dir}/{object}/cf-UP-mw-DOWN-coll-up/{graspObj}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.CF_UP,AttackMethod.MW_DOWN,AttackMethod.SELF_COLLISION_UP])
                        run1.attack(mesh=mesh, grasp=graspObj, dir=f"{exp_root_dir}/{object}/cf-DOWN-mw-UP-coll-up/{graspObj}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.CF_DOWN,AttackMethod.MW_UP,AttackMethod.SELF_COLLISION_UP])
                                
                        run1.attack(mesh=mesh, grasp=graspObj, dir=f"{exp_root_dir}/{object}/cf-GQCNN-DIFF-coll-up/{graspObj}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.GQCNN_CF_DIFF,AttackMethod.SELF_COLLISION_UP])
                        run1.attack(mesh=mesh, grasp=graspObj, dir=f"{exp_root_dir}/{object}/cf-UP-GQCNN-DOWN-coll-up/{graspObj}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.CF_UP,AttackMethod.GQCNN_DOWN,AttackMethod.SELF_COLLISION_UP])
                        run1.attack(mesh=mesh, grasp=graspObj, dir=f"{exp_root_dir}/{object}/cf-DOWN-GQCNN-UP-coll-up/{graspObj}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.CF_DOWN,AttackMethod.GQCNN_UP,AttackMethod.SELF_COLLISION_UP])

                        # run1.attack(mesh=mesh, grasp=grasp_h, dir=f"{exp_root_dir}/{object}/minweight-grad/{grasp_name}", lr=lr0, momentum=mom, loss_alpha=None, method=AttackMethod.MINWEIGHT)
                        # run1.attack(mesh=mesh, grasp=grasp_h, dir=f"{exp_root_dir}/{object}/minweight-grad-DOWN/{grasp_name}", lr=lr0, momentum=mom, loss_alpha=None, method=AttackMethod.MINWEIGHT_DOWN)
                        # run1.attack(mesh=mesh, grasp=grasp_h, dir=f"{exp_root_dir}/{object}/minweight-grad-UP/{grasp_name}", lr=lr0, momentum=mom, loss_alpha=None, method=AttackMethod.MINWEIGHT_UP)
                        # tqdm.write(f'cf: db: {cf_db:.4f} ours: {cf_val.item():.4f} rcf: db: {rcf_db:.4f}, ours: {rcf_val.item():.4f}')
                        # f.write(f'{object}, {grasp_name}, 0, {cf_db:.8f}, {cf_val.item():.8f}, {rcf_db:.8f}, {rcf_val.item():.8f}, {mesh_props.is_watertight}, {mesh_props.is_inverted}\n')
                    except Exception as e:
                        tqdm.write(str(e))
                        tqdm.write(f'failed to compute quality on {object} grasp {graspObj}, saving for debug')
                        pytorch3d.io.save_obj(f'{exp_root_dir}/debug_mesh_attack/{object}.obj', mesh.verts_packed(),mesh.faces_packed())
                        log = open(f'debug_mesh_attack/{object}_{graspObj}_grasp.txt', 'a')
                        log.write(str(e))
                        log.close()
                        graspObj.write_obj(f'debug_mesh_attack/{object}_{graspObj}.obj', include_coordinate=True, include_line_o_action=True)

