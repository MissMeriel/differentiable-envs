# main file for grasp selection/attack loop

from run_gqcnn import *

# sample grasps
renderer = Renderer()
mesh, _ = renderer.render_object("data/bar_clamp.obj", display=False)
grasps = Grasp.sample_grasps("data/new_barclamp.obj", 1, renderer=renderer)

# initialize model and attacks
model = KitModel("weights.npy")
model.eval()
run1 = Attack(num_plots=10, steps_per_plot=50, model=model, renderer=renderer, oracle_method="dexnet")
run2 = Attack(num_plots=10, steps_per_plot=50, model=model, renderer=renderer, oracle_method="pytorch")

# run attacks on each grasp
print("num grasps:", len(grasps))
for i in range(len(grasps)):
    grasp = grasps[i]
    save_path = "experiment-results/main/grasp-" + str(i)
    adv_mesh, final_pic = run1.attack(mesh, grasp, save_path, lr=1e-5, momentum=0.9)