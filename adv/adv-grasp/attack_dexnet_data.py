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
    gpu_id = 0

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
    else:
        print("cuda not available")
        device = torch.device("cpu")

    r = re.Renderer(device=device)

    objects_skip_file = 'bad_objs.txt'
    objects_white_list = 'badStack.txt'
    if objects_skip_file is not None:
        with open(objects_skip_file) as file:
            objects_skip = set([line.rstrip() for line in file])
    else:
        objects_skip = set([])

    if objects_white_list is not None:
        with open(objects_white_list) as file:
            objects_allow = set([line.rstrip() for line in file])
    else:
        objects_allow = None

    config_dict = {
        "torque_scaling":1000,
        "soft_fingers":1,
        "friction_coef": 0.8, # TODO use 0.8 in practice
        "antipodality_pctile": 1.0 
    }
    mom=0
    lr0 = 1e-4
    n_consider = 50
    n_attack = 1
    cf = qf.CannyFerrariQualityFunction(config_dict,min_quality=1)
    rcf = qf.RobustCannyFerrariQualityFunction(config_dict,min_quality=1)
    datasets = db.data_['datasets']
    print(datasets.keys())
    adv_grasp_dir = 'adv/adv-grasp/'
    exp_root_dir = 'exp-25-plots-fix-qp-feasible-fix-ref-dist-january-paper'
    #exp_root_dir = 'throwaway-debug'

    model = KitModel(os.path.join(adv_grasp_dir,"weights.npy"),device=device)
    model.eval()
    run1 = Attack(num_plots=25, steps_per_plot=10, model=model, renderer=r, oracle_method="pytorch")

    if not os.path.isdir(f'{exp_root_dir}'):
        os.mkdir(f'{exp_root_dir}')
        os.mkdir(f'{exp_root_dir}/debug_mesh_attack')

    for dataset in datasets:
        object_list = list(db.data_['datasets'][dataset]['objects'])
        object_list = list(set(object_list).difference(objects_skip))
        if objects_allow is not None:
            object_list = list(set(object_list).intersection(objects_allow))
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
                    dist, bary = mesh_props.self_collision(mesh_props, mesh)
                    minDist = re.mesh_properties.self_collision_min(dist, bary).item()
                except Exception as e:
                    tqdm.write(f'failed to compute connectivity on {object}, saving for debug faces: {faces.shape} verts: {verts.shape}')
                    tqdm.write(str(e))
                    #
                    log = open(f'{exp_root_dir}/debug_mesh_attack/{object}_crash.txt', 'a')
                    log.write(str(e))
                    log.write(traceback.format_exc())
                    log.close()
                    #pytorch3d.io.save_obj(f'debug_mesh_attack/{object}.obj', mesh.verts_packed(),mesh.faces_packed())
                    break
                if minDist < 5e-9 or not math.isfinite(minDist):
                    pytorch3d.io.save_obj(f'{exp_root_dir}/debug_mesh_attack/{object}.obj', mesh.verts_packed(),mesh.faces_packed())
                    bad_faces = mesh_props.get_colliding_faces(dist, bary, min_dist = 1e-8)
                    mesh = r.pre_process_object(mesh.verts_packed(), mesh.faces_packed(), faces_to_remove=bad_faces)
                    pytorch3d.io.save_obj(f'{exp_root_dir}/debug_mesh_attack/{object}_stripped.obj', mesh.verts_packed(),mesh.faces_packed())
                    open(f'{exp_root_dir}/debug_mesh_attack/{object}_collision.txt', 'a').close()
                    tqdm.write(f'{object} is in collision, skipping')
                    bad_faces = None
                    break
                else:
                    # check that grasps exist at all
                    grasp_list = list(db.data_['datasets'][dataset]['objects'][object]['grasps'][name])
                    if len(grasp_list) == 0:
                        tqdm.write(f'{object} has no grasps but mesh has: {faces.shape} verts: {verts.shape}')
                        tqdm.write(str(e))
                        log = open(f'{exp_root_dir}/debug_mesh_attack/{object}_empty.txt', 'a')
                        log.write(str(e))
                        log.close()
                        break
                    
                    random.shuffle(grasp_list)
                    grasp_list = grasp_list[:n_consider]
                    stats = np.zeros((len(grasp_list),4))
                    # find N grasps with interesting properties
                    
                        # try:
                        # gc.collect()
                        
                    try:
                        for index_grasp, grasp_name in tqdm(enumerate(grasp_list), desc=f'Considering grasps to attack in {object}', leave=False):
                            cf_db = db.data_['datasets'][dataset]['objects'][object]['grasps'][name][grasp_name]['metrics'].attrs['ferrari_canny'] 
                            rcf_db = db.data_['datasets'][dataset]['objects'][object]['grasps'][name][grasp_name]['metrics'].attrs['robust_ferrari_canny'] 
                            config = db.data_['datasets'][dataset]['objects'][object]['grasps'][name][grasp_name].attrs['configuration']
                            center3D, axis3D, max_width = config_as_torch(config, device=device)
                            graspObj = qf.GraspTorch(center3D, world_axis=axis3D, width=max_width,
                                                    friction_coef=config_dict["friction_coef"], torque_scaling=config_dict["torque_scaling"])
                            graspObj.make2D(camera_intr=r.camera)

                            graspObj = graspObj.apply_to_mesh(mesh, is_watertight=mesh_props.is_watertight, 
                                                            is_inverted=mesh_props.is_inverted, use_dexnet_normal=False)
                            dim = r.mesh_to_depth_im(mesh, display=False)
                            pose, image = qf.GQCNNQualityFunction.extract_tensors_batch(grasp=graspObj, d_ims=dim)
                            out = model(pose, image)
                            if graspObj.count_misses == 0:
                                gqcnn_val = out[:,1:2].item()
                                cf_val = cf(mesh, graspObj).item()
                                rcf_val = rcf(mesh, graspObj).item()
                                stats[index_grasp,:] = (gqcnn_val, cf_val, rcf_val, 1)
                            tqdm.write(f'{index_grasp} {stats[index_grasp,:]}')
                    except Exception as e:
                        tqdm.write(f'failed to compute quality on {object} grasp {grasp_name}, saving for debug')
                        log = open(f'{exp_root_dir}/debug_mesh_attack/{object}_{grasp_name}_grasp.txt', 'a')
                        log.write(str(e))
                        log.write(traceback.format_exc())
                        log.close()
                        break
                    
                       
                    # sort them
                    indices_interesting = np.argsort(stats[:,0])[::-1][:n_attack]
                    stats_sort = stats[indices_interesting,:]
                    indices_interesting = indices_interesting[np.logical_and(stats_sort[:,1]>0,stats_sort[:,2]>0,stats_sort[:,3]>0)] # must be in collision AND rcf be non-zero. rcf derivative undefined if 0

                    minDist, dist, bary, mesh_props = None, None, None, None
                    try:
                        for grasp_ind in tqdm(indices_interesting, desc=f'attacking grasps in {object} with gqcnn of {stats_sort[:,0]}',leave=False):
                            grasp_name = grasp_list[grasp_ind]

                            config = db.data_['datasets'][dataset]['objects'][object]['grasps'][name][grasp_name].attrs['configuration']
                            center3D, axis3D, max_width = config_as_torch(config, device=device)
                            graspObj = qf.GraspTorch(center3D, world_axis=axis3D, width=max_width,
                                                    friction_coef=config_dict["friction_coef"], torque_scaling=config_dict["torque_scaling"])
                            graspObj.make2D(camera_intr=r.camera)
                            # loop
                            #run1.attack(mesh=mesh, grasp=graspObj, dir=f"{exp_root_dir}/{object}/cf-UP-mw-DOWN-coll-up/{grasp_name}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.CF_UP,AttackMethod.MW_DOWN,AttackMethod.SELF_COLLISION_UP])
                            
                            #run1.attack(mesh=mesh, grasp=graspObj, dir=f"{exp_root_dir}/{object}/cf-DOWN-mw-UP-coll-up/{grasp_name}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.CF_DOWN,AttackMethod.MW_UP,AttackMethod.SELF_COLLISION_UP])
                                    
                            #run1.attack(mesh=mesh, grasp=graspObj, dir=f"{exp_root_dir}/{object}/cf-GQCNN-DIFF-coll-up/{grasp_name}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.GQCNN_CF_DIFF,AttackMethod.SELF_COLLISION_UP])
                            _,attack_failed = run1.attack(mesh=mesh, grasp=graspObj, dir=f"{exp_root_dir}/{object}/cf-UP-GQCNN-DOWN-coll-up/{grasp_name}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.CF_UP,AttackMethod.GQCNN_DOWN,AttackMethod.SELF_COLLISION_UP])
                            if attack_failed is not None:
                                raise Exception(attack_failed)
                            _,attack_failed = run1.attack(mesh=mesh, grasp=graspObj, dir=f"{exp_root_dir}/{object}/cf-DOWN-GQCNN-UP-coll-up/{grasp_name}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.CF_DOWN,AttackMethod.GQCNN_UP,AttackMethod.SELF_COLLISION_UP])

                            _,attack_failed = run1.attack(mesh=mesh, grasp=graspObj, dir=f"{exp_root_dir}/{object}/cf-UP-GQCNN-DOWN/{grasp_name}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.CF_UP,AttackMethod.GQCNN_DOWN])
                            _,attack_failed = run1.attack(mesh=mesh, grasp=graspObj, dir=f"{exp_root_dir}/{object}/cf-DOWN-GQCNN-UP/{grasp_name}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.CF_DOWN,AttackMethod.GQCNN_UP])

                            _,attack_failed = run1.attack(mesh=mesh, grasp=graspObj, dir=f"{exp_root_dir}/{object}/GQCNN-DOWN-coll-up/{grasp_name}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.GQCNN_DOWN,AttackMethod.SELF_COLLISION_UP])
                            _,attack_failed = run1.attack(mesh=mesh, grasp=graspObj, dir=f"{exp_root_dir}/{object}/GQCNN-UP-coll-up/{grasp_name}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.GQCNN_UP,AttackMethod.SELF_COLLISION_UP])

                            _,attack_failed = run1.attack(mesh=mesh, grasp=graspObj, dir=f"{exp_root_dir}/{object}/GQCNN-DOWN/{grasp_name}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.GQCNN_DOWN])
                            _,attack_failed = run1.attack(mesh=mesh, grasp=graspObj, dir=f"{exp_root_dir}/{object}/GQCNN-UP/{grasp_name}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.GQCNN_UP])
                            torch.cuda.synchronize()
                                # run1.attack(mesh=mesh, grasp=grasp_h, dir=f"{exp_root_dir}/{object}/minweight-grad/{grasp_name}", lr=lr0, momentum=mom, loss_alpha=None, method=AttackMethod.MINWEIGHT)
                                # run1.attack(mesh=mesh, grasp=grasp_h, dir=f"{exp_root_dir}/{object}/minweight-grad-DOWN/{grasp_name}", lr=lr0, momentum=mom, loss_alpha=None, method=AttackMethod.MINWEIGHT_DOWN)
                                # run1.attack(mesh=mesh, grasp=grasp_h, dir=f"{exp_root_dir}/{object}/minweight-grad-UP/{grasp_name}", lr=lr0, momentum=mom, loss_alpha=None, method=AttackMethod.MINWEIGHT_UP)
                                # tqdm.write(f'cf: db: {cf_db:.4f} ours: {cf_val.item():.4f} rcf: db: {rcf_db:.4f}, ours: {rcf_val.item():.4f}')
                                # f.write(f'{object}, {grasp_name}, 0, {cf_db:.8f}, {cf_val.item():.8f}, {rcf_db:.8f}, {rcf_val.item():.8f}, {mesh_props.is_watertight}, {mesh_props.is_inverted}\n')

                    except Exception as e:
                        tqdm.write(str(e))
                        tqdm.write(f'failed to compute quality on {object} grasp {grasp_name}, saving for debug')
                        pytorch3d.io.save_obj(f'{exp_root_dir}/debug_mesh_attack/{object}.obj', mesh.verts_packed(),mesh.faces_packed())
                        log = open(f'{exp_root_dir}/debug_mesh_attack/{object}_{grasp_name}_grasp.txt', 'a')
                        log.write(str(e))
                        log.write(traceback.format_exc())
                        log.close()
                        graspObj.write_obj(f'{exp_root_dir}/debug_mesh_attack/{object}_{grasp_name}.obj', include_coordinate=True, include_line_o_action=True)

