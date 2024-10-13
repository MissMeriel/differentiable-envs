# main file for grasp selection/attack loop
from run_gqcnn import *
from select_grasp import *
import os
import matplotlib
matplotlib.use('agg')

DATA_FILE = "data/new_barclamp.obj"

r = Renderer()
mesh, _ = r.render_object(DATA_FILE, display=False)
g = Grasp.read("grasp-dataset2/grasp-batch.json")
g2 = Grasp.read("grasp-dataset/grasp-batch.json")
grasps = g # [g[2], g[5], g[7]]

model = KitModel("weights.npy")
model.eval()
run1 = Attack(num_plots=10, steps_per_plot=2, model=model, renderer=r, oracle_method="pytorch")

#lr_lst = [(1e-5, 0.0), (1e-5, 0.9), (1e-5, 0.99), (1e-4, 0.0), (1e-4, 0.9)]
lr_lst = [(1e-3, 0.0), (1e-4, 0.0)]

#grasps = [grasps[0]]
lr_lst = lr_lst[0:3]

for idx,grasp in enumerate(grasps):
    grasp.c0 = None
    grasp.c1 = None
    for lr in lr_lst:
        lr0, mom = lr[0], lr[1]
        
        run1.attack(mesh=mesh, grasp=grasp, dir=f"test/oracle-grad/grasp{idx}/lr-{lr_lst.index(lr)}", lr=lr0, momentum=mom, loss_alpha=None, method=AttackMethod.ORACLE_GRAD)
        run1.attack(mesh=mesh, grasp=grasp, dir=f"test/no-oracle/grasp{idx}/lr-{lr_lst.index(lr)}", lr=lr0, momentum=mom, loss_alpha=None, method=AttackMethod.NO_ORACLE)
        run1.attack(mesh=mesh, grasp=grasp, dir=f"test/no-oracle-grad/grasp{idx}/lr-{lr_lst.index(lr)}", lr=lr0, momentum=mom, loss_alpha=None, method=AttackMethod.NO_ORACLE_GRAD)
        run1.attack(mesh=mesh, grasp=grasp, dir=f"test/random-fuzz/grasp{idx}/lr-{lr_lst.index(lr)}", lr=lr0, momentum=mom, loss_alpha=None, method=AttackMethod.RANDOM_FUZZ)
