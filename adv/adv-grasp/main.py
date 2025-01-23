# main file for grasp selection/attack loop
from run_gqcnn import *
from select_grasp import *
import os
import matplotlib
import pytorch3d.ops as ops
matplotlib.use('agg')

adv_grasp_dir = 'adv/adv-grasp/'
DATA_FILE = os.path.join(adv_grasp_dir,"data/new_barclamp.obj")
gpu_id = 1

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
run1 = Attack(num_plots=100, steps_per_plot=5, model=model, renderer=r, oracle_method="pytorch")

#lr_lst = [(1e-5, 0.0), (1e-5, 0.9), (1e-5, 0.99), (1e-4, 0.0), (1e-4, 0.9)]
lr_lst = [(1e-5, 0), (5e-6, 0)]

#grasps = [grasps[0]]
exp_root_dir = os.path.join(adv_grasp_dir,'test_normals_fixed_1-5_5-6_non_normal_non_tri')

for idx,grasp in enumerate(grasps):
    grasp.c0 = None
    grasp.c1 = None
    for lr in lr_lst:
        lr0, mom = lr[0], lr[1]
         
        run1.attack(mesh=mesh, grasp=grasp, dir=f"{exp_root_dir}/oracle-grad/grasp{idx}/lr-{lr_lst.index(lr)}", lr=lr0, momentum=mom, loss_alpha=None, method=AttackMethod.ORACLE_GRAD)
        run1.attack(mesh=mesh, grasp=grasp, dir=f"{exp_root_dir}/oracle-grad-DOWN/grasp{idx}/lr-{lr_lst.index(lr)}", lr=lr0, momentum=mom, loss_alpha=None, method=AttackMethod.ORACLE_GRAD_DOWN)
        run1.attack(mesh=mesh, grasp=grasp, dir=f"{exp_root_dir}/oracle-grad-UP/grasp{idx}/lr-{lr_lst.index(lr)}", lr=lr0, momentum=mom, loss_alpha=None, method=AttackMethod.ORACLE_GRAD_UP)
        run1.attack(mesh=mesh, grasp=grasp, dir=f"{exp_root_dir}/no-oracle/grasp{idx}/lr-{lr_lst.index(lr)}", lr=lr0, momentum=mom, loss_alpha=None, method=AttackMethod.NO_ORACLE)
        run1.attack(mesh=mesh, grasp=grasp, dir=f"{exp_root_dir}/no-oracle-grad/grasp{idx}/lr-{lr_lst.index(lr)}", lr=lr0, momentum=mom, loss_alpha=None, method=AttackMethod.NO_ORACLE_GRAD)
        run1.attack(mesh=mesh, grasp=grasp, dir=f"{exp_root_dir}/random-fuzz/grasp{idx}/lr-{lr_lst.index(lr)}", lr=lr0, momentum=mom, loss_alpha=None, method=AttackMethod.RANDOM_FUZZ)
