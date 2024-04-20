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

# qual = grasp.oracle_eval("data/new_barclamp.obj", "pytorch", renderer=r)
# print(qual.item(), "\n", qual.requires_grad)
# dim = r.mesh_to_depth_im(mesh, display=False)
# pose, processed_dim = grasp.extract_tensors_batch(dim)
# out = run1.run(pose, processed_dim)
# print(out[:,0:1].requires_grad)

for grasp in grasps:
    grasp.c0 = None
    grasp.c1 = None
    for lr in lr_lst:
        lr0, mom = lr[0], lr[1]
        dir = f"exp-results2/oracle-grad/lr-{lr_lst.index(lr)}/grasp-{grasps.index(grasp)}"
        run1.attack(mesh=mesh, grasp=grasp, dir=dir, lr=lr0, momentum=mom, loss_alpha=None, method="oracle-grad")


# for i, grasp in enumerate(g):
#     scaled_qual = Attack.scale_oracle(grasp.quality).item()
#     print(f"input: {grasp.quality.item():.5f} \toutput: {scaled_qual:.5f}")

# print(f"\ninput: 0.0 \toutput {Attack.scale_oracle(0.0).item():.5f}")
# print(f"input: 0.001 \toutput {Attack.scale_oracle(0.001):.5f}")
# print(f"input: 0.00001 \toutput {Attack.scale_oracle(0.00001):.5f}")
# print(f"input: 0.002 \toutput {Attack.scale_oracle(0.002):.5f}")


# d = "exp-results/random-fuzz/"
# grasp_dirs = {d+"grasp0/": g[0], d+"grasp1/": g[1], d+"grasp3/": g[2], d+"grasp4/": g[3], d+"grasp5/": g[4], d+"grasp6/": g[5]}

# oracle_grad_lr = [(1e-6, 0.0), (1e-6, 0.99), (1e-5, 0.0), (1e-5, 0.9), (1e-5, 0.99), (1e-4, 0.0), (1e-4, 0.9)]	# learning rate and momentum combinations
# oracle_grad_alpha = [0.1, 0.3, 0.5, 0.7]

# learning rate and momentum combinations
# random_fuzz_lr = {"lr0": 1e-6, "lr1":1e-5, "lr3": 1e-4}

# for d in grasp_dirs.keys():
# 	for ext in random_fuzz_lr.keys():
# 		save = d + ext + "/"
# 		lr = random_fuzz_lr[ext]
# 		grasp = grasp_dirs[d]
# 		print(f"saving to: {save} with learning rate {lr}")
# 		run1.attack(mesh=mesh, grasp=grasp, dir=save, lr=lr, momentum=None, loss_alpha=None, method="random-fuzz")