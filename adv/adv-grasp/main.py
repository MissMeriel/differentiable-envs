# main file for grasp selection/attack loop
from run_gqcnn import *
from select_grasp import *
import os

DATA_FILE = "data/new_barclamp.obj"

r = Renderer()
mesh, _ = r.render_object(DATA_FILE, display=False)
g = Grasp.read("grasp-dataset2/grasp-batch.json")
g2 = Grasp.read("grasp-dataset/grasp-batch.json")
grasps = [g[2], g[5], g[7]]

model = KitModel("weights.npy")
model.eval()
run1 = Attack(num_plots=10, steps_per_plot=10, model=model, renderer=r, oracle_method="pytorch")

lr_lst = [(1e-5, 0.0), (1e-5, 0.9), (1e-5, 0.99), (1e-4, 0.0), (1e-4, 0.9)]

grasps = [grasps[0]]
lr_lst = lr_lst[0:3]

for grasp in grasps:
    grasp.c0 = None
    grasp.c1 = None
    for lr in lr_lst:
        lr0, mom = lr[0], lr[1]
        dir = f"test/oracle-grad/lr-{lr_lst.index(lr)}"
        run1.attack(mesh=mesh, grasp=grasp, dir=dir, lr=lr0, momentum=mom, loss_alpha=None, method=AttackMethod.ORACLE_GRAD)
        run1.attack(mesh=mesh, grasp=grasp, dir="test/no-oracle", lr=lr0, momentum=mom, loss_alpha=None, method=AttackMethod.NO_ORACLE)
        run1.attack(mesh=mesh, grasp=grasp, dir="test/no-oracle-grad", lr=lr0, momentum=mom, loss_alpha=None, method=AttackMethod.NO_ORACLE_GRAD)
        run1.attack(mesh=mesh, grasp=grasp, dir="test/random-fuzz", lr=lr0, momentum=mom, loss_alpha=None, method=AttackMethod.RANDOM_FUZZ)
