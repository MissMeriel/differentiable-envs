# main file for grasp selection/attack loop
from run_gqcnn import *
from select_grasp import *
import os
import matplotlib
import pytorch3d.ops as ops
matplotlib.use('agg')
from torch.profiler import profile, record_function, ProfilerActivity


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
g = Grasp.read(os.path.join(adv_grasp_dir,"grasp-dataset2/grasp-batch.json"),device=device)
g2 = Grasp.read(os.path.join(adv_grasp_dir,"grasp-dataset/grasp-batch.json"),device=device)
grasps = [g[2], g[5], g[7]]

model = KitModel(os.path.join(adv_grasp_dir,"weights.npy"),device=device)
model.eval()
run1 = Attack(num_plots=20, steps_per_plot=25, model=model, renderer=r, oracle_method="pytorch")
# run2 = Attack(num_plots=50, steps_per_plot=2, model=model, renderer=r, oracle_method="pytorch")

#lr_lst = [(1e-5, 0.0), (1e-5, 0.9), (1e-5, 0.99), (1e-4, 0.0), (1e-4, 0.9)]
lr_lst = [(1e-5, 0)]

#grasps = [grasps[0]]
exp_root_dir = os.path.join(adv_grasp_dir,'test_cf-mw-scale-grad-no-collision-projection')

for idx,grasp_name in enumerate(grasps):
    grasp_name.c0 = None
    grasp_name.c1 = None
    for lr in lr_lst:
        lr0, mom = lr[0], lr[1]
        try:
            run1.attack(mesh=mesh, grasp=grasp_name, dir=f"{exp_root_dir}/cf-UP-GQCNN-DOWN-coll-up/grasp{idx}/lr-{lr_lst.index(lr)}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.CF_UP,AttackMethod.GQCNN_DOWN,AttackMethod.SELF_COLLISION_UP])
        except:
            pass
        try:
            run1.attack(mesh=mesh, grasp=grasp_name, dir=f"{exp_root_dir}/cf-DOWN-GQCNN-UP-coll-up/grasp{idx}/lr-{lr_lst.index(lr)}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.CF_DOWN,AttackMethod.GQCNN_UP,AttackMethod.SELF_COLLISION_UP])
        except:
            pass
        try:
            run1.attack(mesh=mesh, grasp=grasp_name, dir=f"{exp_root_dir}/cf-UP-mw-DOWN-coll-up/grasp{idx}/lr-{lr_lst.index(lr)}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.CF_UP,AttackMethod.MW_DOWN,AttackMethod.SELF_COLLISION_UP])
        except:
            pass
        try:
            run1.attack(mesh=mesh, grasp=grasp_name, dir=f"{exp_root_dir}/cf-DOWN-mw-UP-coll-up/grasp{idx}/lr-{lr_lst.index(lr)}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.CF_DOWN,AttackMethod.MW_UP,AttackMethod.SELF_COLLISION_UP])
        except:
            pass
        try:        
            run1.attack(mesh=mesh, grasp=grasp_name, dir=f"{exp_root_dir}/oracle-grad-coll-up/grasp{idx}/lr-{lr_lst.index(lr)}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.GQCNN_CF_DIFF,AttackMethod.SELF_COLLISION_UP])
        except:
            pass
            
        # run1.attack(mesh=mesh, grasp=grasp_name, dir=f"{exp_root_dir}/oracle-grad-DOWN-coll-up/grasp{idx}/lr-{lr_lst.index(lr)}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.GQCNN_UP_CF_DOWN,AttackMethod.SELF_COLLISION_UP])
        # run1.attack(mesh=mesh, grasp=grasp_name, dir=f"{exp_root_dir}/oracle-grad-UP-coll-up/grasp{idx}/lr-{lr_lst.index(lr)}", lr=lr0, momentum=mom, loss_alpha=None, method=[AttackMethod.GQCNN_DOWN_CF_UP,AttackMethod.SELF_COLLISION_UP])
