# main file for grasp selection/attack loop

from run_gqcnn import *

ORACLE_ROBUST = False
DATA_FILE = "data/Waterglass_800_tex.obj"

# sample grasps
renderer = Renderer()
mesh, _ = renderer.render_object(DATA_FILE, display=True)
grasps = Grasp.sample_grasps(DATA_FILE, 1, renderer=renderer, min_qual=0.002, oracle_robust=ORACLE_ROBUST)

# save grasp sphere/visualizations
title = "quality: " + str(grasps.quality.item())
gsphere = renderer.grasp_sphere([grasps[0].c0, grasps[0].c1], mesh, title=title, save="grasp-dataset/waterglass.png")
print(title)
qual = grasps.oracle_eval(DATA_FILE, robust=ORACLE_ROBUST, renderer=renderer)
print("recalculated quality:", qual.item(), qual)

# random grasp variations 
grasps.random_grasps(num_samples=25, camera=renderer.camera)
grasps.oracle_eval(DATA_FILE, robust=ORACLE_ROBUST, renderer=renderer)
print("qualities:", grasps.quality)

# initialize model and attacks
model = KitModel("weights.npy")
model.eval()
run1 = Attack(num_plots=10, steps_per_plot=10, model=model, renderer=renderer, oracle_method="dexnet", oracle_robust=ORACLE_ROBUST)

# run model
dim = renderer.mesh_to_depth_im(mesh, display=False)
pose, image = grasps.extract_tensors_batch(dim)
print("pose:", type(pose), pose.shape)
print("image:", type(image), image.shape)
pred = run1.run(pose, image)
print("model prediction:", type(pred), pred.shape)
print(pred)

grasps.prediction = pred
grasps.save("grasp-dataset/waterglass-random.json")

# # run attacks on each grasp
# print("num grasps:", len(grasps))
# for i in range(len(grasps)):
#     grasp = grasps[i]
#     save_path = "experiment-results/main/grasp-" + str(i)
#     adv_mesh, final_pic = run1.attack(mesh, grasp, save_path, lr=1e-5, momentum=0.9)