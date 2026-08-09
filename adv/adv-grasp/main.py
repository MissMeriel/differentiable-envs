# main file for grasp selection/attack loop
import matplotlib
matplotlib.use('agg')

from run_gqcnn import *
import select_grasp as sg
import quality_fork as qf
from pytorch3d_ext import Renderer
import os

import pytorch3d.ops as ops
from tqdm import tqdm

from torch.profiler import profile, record_function, ProfilerActivity

import torch

adv_grasp_dir = 'adv/adv-grasp/'
DATA_FILE = os.path.join(adv_grasp_dir,"data/new_barclamp.obj")
gpu_id = 0

if torch.cuda.is_available():
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
else:
    print("cuda not available")
    device = torch.device("cpu")

r = Renderer(device=device)
mesh, _ = r.render_object(DATA_FILE, display=False)
# divide = ops.SubdivideMeshes()
# mesh = divide(mesh)
g = GraspTorch.read(os.path.join(adv_grasp_dir,"grasp-dataset2/grasp-batch.json"),device=device)
g2 = GraspTorch.read(os.path.join(adv_grasp_dir,"grasp-dataset/grasp-batch.json"),device=device)

grasps = [g[i] for i in range(len(g))]
#grasps = [g[i] for i in range(len(g))]


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

model = KitModel(os.path.join(adv_grasp_dir,"weights.npy"),device=device)
model.eval()
run1 = Attack(num_plots=25, steps_per_plot=10, model=model, renderer=r, oracle_method="pytorch")
# run2 = Attack(num_plots=50, steps_per_plot=2, model=model, renderer=r, oracle_method="pytorch")

#lr_lst = [(1e-5, 0.0), (1e-5, 0.9), (1e-5, 0.99), (1e-4, 0.0), (1e-4, 0.9)]
lr_lst = [(1e-4, 0)]

#grasps = [grasps[0]]
#exp_root_dir = os.path.join(adv_grasp_dir,'test_cf-mw-scale-grad-no-collision-projection')
exp_root_dir = os.path.join(adv_grasp_dir,'feb-experiments-gpu-bar-clamp')

for idx,graspObj in tqdm(enumerate(grasps),desc='N grasps',leave=False):
    graspObj.c0 = None
    graspObj.c1 = None
    graspObj.torque_scaling = config_dict['torque_scaling']
    graspObj.friction_coef = config_dict['friction_coef']

    graspObj.make2D(camera_intr=r.camera)
    graspObj = graspObj.apply_to_mesh(mesh, is_watertight=True, is_inverted=False, use_dexnet_normal=False)
    dim = r.mesh_to_depth_im(mesh, display=False)
    pose, image = qf.GQCNNQualityFunction.extract_tensors_batch(grasp=graspObj, d_ims=dim)
    out = model(pose, image)
    gqcnn_val = out[:,1:2].to(mesh.device)
    cf_val = cf(mesh, graspObj)
    rcf_val = rcf(mesh, graspObj)
    mw_val = mw(mesh,graspObj)
    rmw_val = rmw(mesh,graspObj)
    in_contact = torch.any(graspObj.contact_mask)

    tqdm.write(f'{idx} cf ours: {cf_val.item():.4f} rcf ours: {rcf_val.item():.4f} mw: {mw_val.item():.4f} rmw: {rmw_val.item():.4f} gqcnn {gqcnn_val.item():.4f} contact {in_contact}')

    for lr in tqdm(lr_lst,desc='M lr',leave=False):
        lr0, mom = lr[0], lr[1]
        # try:
        # run1.attack(mesh=mesh, grasp=graspObj, dir=f"{exp_root_dir}/cf-UP-GQCNN-DOWN-coll-up/grasp_{idx}/lr-{lr_lst.index(lr)}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.CF_UP,AttackMethod.GQCNN_DOWN,AttackMethod.SELF_COLLISION_LOSS_DOWN])
        # # except:
        # #     pass
        # try:
        #     run1.attack(mesh=mesh, grasp=graspObj, dir=f"{exp_root_dir}/cf-DOWN-GQCNN-UP-coll-up/grasp_{idx}/lr-{lr_lst.index(lr)}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.CF_DOWN,AttackMethod.GQCNN_UP,AttackMethod.SELF_COLLISION_LOSS_DOWN])
        # except:
        #     pass
        # try:
        #     run1.attack(mesh=mesh, grasp=graspObj, dir=f"{exp_root_dir}/GQCNN-DOWN-coll-up/grasp_{idx}/lr-{lr_lst.index(lr)}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.CF_UP,AttackMethod.GQCNN_DOWN,AttackMethod.SELF_COLLISION_LOSS_DOWN])
        # except:
        #     pass
        # try:
        #     run1.attack(mesh=mesh, grasp=graspObj, dir=f"{exp_root_dir}/GQCNN-UP-coll-up/grasp_{idx}/lr-{lr_lst.index(lr)}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.CF_DOWN,AttackMethod.GQCNN_UP,AttackMethod.SELF_COLLISION_LOSS_DOWN])
        # except:
        #     pass
        # try:
        #     run1.attack(mesh=mesh, grasp=graspObj, dir=f"{exp_root_dir}/cf-UP-GQCNN-DOWN/grasp_{idx}/lr-{lr_lst.index(lr)}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.CF_UP,AttackMethod.GQCNN_DOWN])
        # except:
        #     pass
        # try:
        #     run1.attack(mesh=mesh, grasp=graspObj, dir=f"{exp_root_dir}/cf-DOWN-GQCNN-UP/grasp_{idx}/lr-{lr_lst.index(lr)}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.CF_DOWN,AttackMethod.GQCNN_UP])
        # except:
        #     pass
        # _,attack_failed = run1.attack(mesh=mesh, grasp=graspObj, dir=f"{exp_root_dir}/Random_Fuzz/seed_0/grasp_{idx}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.RANDOM_FUZZ])
        # _,attack_failed = run1.attack(mesh=mesh, grasp=graspObj, dir=f"{exp_root_dir}/Random_Fuzz/seed_1/grasp_{idx}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.RANDOM_FUZZ])
        # _,attack_failed = run1.attack(mesh=mesh, grasp=graspObj, dir=f"{exp_root_dir}/Random_Fuzz/seed_2/grasp_{idx}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.RANDOM_FUZZ])
        # _,attack_failed = run1.attack(mesh=mesh, grasp=graspObj, dir=f"{exp_root_dir}/cf-DOWN-GQCNN-UP-laplace-down/grasp_{idx}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.CF_DOWN,AttackMethod.GQCNN_UP,AttackMethod.LAPLACIAN_SMOOTHING_DOWN])
        # _,attack_failed = run1.attack(mesh=mesh, grasp=graspObj, dir=f"{exp_root_dir}/cf-UP-GQCNN-DOWN-laplace-down/grasp_{idx}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.CF_UP,AttackMethod.GQCNN_DOWN,AttackMethod.LAPLACIAN_SMOOTHING_DOWN])
        # _,attack_failed = run1.attack(mesh=mesh, grasp=graspObj, dir=f"{exp_root_dir}/cf-DOWN-GQCNN-UP-l2-down/grasp_{idx}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.CF_DOWN,AttackMethod.GQCNN_UP,AttackMethod.L2_NORM_DOWN])
        # _,attack_failed = run1.attack(mesh=mesh, grasp=graspObj, dir=f"{exp_root_dir}/cf-UP-GQCNN-DOWN-l2-down/grasp_{idx}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.CF_UP,AttackMethod.GQCNN_DOWN,AttackMethod.L2_NORM_DOWN])
        run1.attack(mesh=mesh, grasp=graspObj, dir=f"{exp_root_dir}/GQCNN-DOWN/grasp_{idx}/lr-{lr_lst.index(lr)}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.GQCNN_DOWN])
        run1.attack(mesh=mesh, grasp=graspObj, dir=f"{exp_root_dir}/GQCNN-UP/grasp_{idx}/lr-{lr_lst.index(lr)}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.GQCNN_UP])

        # try:
        #     run1.attack(mesh=mesh, grasp=graspObj, dir=f"{exp_root_dir}/cf-UP-mw-DOWN-coll-up/grasp_{idx}/lr-{lr_lst.index(lr)}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.CF_UP,AttackMethod.MW_DOWN,AttackMethod.SELF_COLLISION_UP])
        # except:
        #     pass
        # try:
        #     run1.attack(mesh=mesh, grasp=graspObj, dir=f"{exp_root_dir}/cf-DOWN-mw-UP-coll-up/grasp_{idx}/lr-{lr_lst.index(lr)}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.CF_DOWN,AttackMethod.MW_UP,AttackMethod.SELF_COLLISION_UP])
        # except:
        #     pass
        # try:        
        #     run1.attack(mesh=mesh, grasp=graspObj, dir=f"{exp_root_dir}/oracle-grad-coll-up/grasp_{idx}/lr-{lr_lst.index(lr)}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.GQCNN_CF_DIFF,AttackMethod.SELF_COLLISION_LOSS_DOWN])
        # except:
        #     pass
            
        # run1.attack(mesh=mesh, grasp=grasp_name, dir=f"{exp_root_dir}/oracle-grad-DOWN-coll-up/grasp_{idx}/lr-{lr_lst.index(lr)}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.GQCNN_UP_CF_DOWN,AttackMethod.SELF_COLLISION_UP])
        # run1.attack(mesh=mesh, grasp=grasp_name, dir=f"{exp_root_dir}/oracle-grad-UP-coll-up/grasp_{idx}/lr-{lr_lst.index(lr)}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.GQCNN_DOWN_CF_UP,AttackMethod.SELF_COLLISION_UP])
