# main file for grasp selection/attack loop
from run_gqcnn import *

DATA_FILE = "data/new_barclamp.obj"

r = Renderer()
mesh, _ = r.render_object(DATA_FILE, display=False)
g = Grasp.read("grasp-batch.json")
grasps = [g[0], g[1], g[3], g[4], g[5], g[6]]

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

model = KitModel("weights.npy")
model.eval()
run1 = Attack(num_plots=10, steps_per_plot=10, model=model, renderer=r, oracle_method="pytorch")

# for d in grasp_dirs.keys():
# 	for ext in random_fuzz_lr.keys():
# 		save = d + ext + "/"
# 		lr = random_fuzz_lr[ext]
# 		grasp = grasp_dirs[d]
# 		print(f"saving to: {save} with learning rate {lr}")
# 		run1.attack(mesh=mesh, grasp=grasp, dir=save, lr=lr, momentum=None, loss_alpha=None, method="random-fuzz")

# print("ATTACK SET 1\n")
grasp = grasps[1]
dir = "test/oracle-grad/"
run1.attack(mesh=mesh, grasp=grasp, dir=dir, lr=1e-5, momentum=0.9, loss_alpha=0.3, method="oracle-grad")