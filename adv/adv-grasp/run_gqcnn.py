import shutil
import logging
import numpy as np
from torchviz import make_dot
from pytorch3d.io import save_obj
from pytorch3d.loss import mesh_edge_loss, mesh_normal_consistency, mesh_laplacian_smoothing

from render import *
from gqcnn_pytorch import KitModel
from select_grasp import *

SHARED_DIR = "/home/hmitchell/pytorch3d/dex_shared_dir"

class Attack: 

	# SET UP LOGGING
	logger = logging.getLogger('run_gqcnn')
	logger.setLevel(logging.INFO)
	if not logger.handlers:
		ch = logging.StreamHandler()
		ch.setLevel(logging.INFO)
		formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		ch.setFormatter(formatter)
		logger.addHandler(ch)

	def __init__(self, num_plots=0, steps_per_plot=0, model=None, renderer=None, oracle_method="dexnet", oracle_robust=True):
		self.num_plots = num_plots
		self.steps_per_plot = steps_per_plot
		self.num_steps = num_plots * steps_per_plot

		self.model = model
		if model == None:
			model = KitModel("weights.npy")
			model.eval()
			self.model = model

		self.renderer = renderer
		if renderer == None:
			self.renderer = Renderer()

		self.oracle_method = oracle_method
		self.oracle_robust = oracle_robust
		if oracle_method not in ["dexnet", "pytorch"]:
			Attack.logger.debug("Attack oracle evaluation method must be 'dexnet' or 'pytorch' -- defaulting to 'dexnet'.")
			self.oracle_method = "dexnet"

		self.loss_weights = {
			"edge": 5,
			"normal": 0.4,
			"smooth": 5
		}
		self.losses = None

	def run(self, pose_tensor, image_tensor):
		"""
		Run the model and return the prediction

		Parameters
		----------
		pose_tensor: {batch_size} x 1 torch.tensor
			pose array input for model
		image_tensor: {batch_size} x 1 x 32 x 32 torch.tensor
			image array input for model
		Returns
		-------
		torch.tensor: {batch_size} where entries indicate grasp quality prediction by model
		"""

		# check for bad output from Grasp.extract_tensors
		if pose_tensor==None or image_tensor==None:
			Attack.logger.error("Pose tensor and/or image tensor input is bad.")
			return 0

		return self.model(pose_tensor, image_tensor)

	@staticmethod
	def plot_losses(losses, dir):
		fig = plt.figure(figsize=(13, 5))
		ax = fig.gca()
		for k, l in losses.items():
			ax.plot(l, label=k + " loss")
		ax.legend(fontsize="16")
		ax.set_xlabel("Iteration", fontsize="16")
		ax.set_ylabel("Loss", fontsize="16")
		ax.set_title("Loss vs iterations", fontsize="18")
		# plt.show()
		plt.savefig(dir+"losses.png")

	def calc_loss(self, adv_mesh, grasp):
		"""
		Calculates loss for the adversarial attack to maximize difference between oracle and model prediction

		Parameters
		----------
		adv_mesh:
		grasp:
		Returns
		-------
		float: calculated loss
		"""

		# CHECK CURRENT MODEL PREDICTION
		adv_mesh_clone = adv_mesh.clone()
		dim = self.renderer.mesh_to_depth_im(adv_mesh_clone, display=False)
		pose, image = grasp.extract_tensors_batch(dim)
		out = self.run(pose, image)
		cur_pred = out[:, 0].unsqueeze(1).to(adv_mesh.device)
		oracle_pred = torch.clone(grasp.quality) / 0.002	# scale to model range
		assert oracle_pred.shape[0] == 10

		# MAXIMIZE DIFFERENCE BETWEEN CURRENT PREDICTION AND ORACLE PREDICTION
		# max = torch.ones(oracle_pred.shape).to(adv_mesh.device)
		loss = torch.sub(1.0, torch.abs(torch.sub(torch.mean(cur_pred), torch.mean(oracle_pred))))

		# WEIGHTED LOSS WITH PYTORCH3D.LOSS FUNCS
		edge_loss = mesh_edge_loss(adv_mesh) * self.loss_weights["edge"]
		normal_loss = mesh_normal_consistency(adv_mesh) * self.loss_weights["normal"]
		smooth_loss = mesh_laplacian_smoothing(adv_mesh) * self.loss_weights["smooth"]
		weighted_loss = loss + edge_loss + normal_loss + smooth_loss
		self.losses["prediction"].append(loss.item())
		self.losses["edge"].append(edge_loss.item())
		self.losses["normal"].append(normal_loss.item())
		self.losses["smoothing"].append(smooth_loss.item())

		data = ("\n\tModel prediction mean: " + str(torch.mean(cur_pred).item()) + "\n\tModel prediction: " + str(cur_pred.detach().cpu().numpy().tolist()) 
			+ f"\n\tPrediction loss: {loss}\n\tNormal consistency loss: {normal_loss}\n\tLaplacian smoothing loss: {smooth_loss}\n\tMesh edge loss: {edge_loss}\n\tWeighted loss: {weighted_loss}")

		return weighted_loss, torch.mean(cur_pred), data

	def perturb(self, mesh, param, grasp):
		"""
		Perturb the mesh for the adversarial attack

		Parameters
		----------
		mesh: pytorch3d.structures.Meshes
			Mesh to be perturbed
		param: torch.tensor
		grasp:
		Returns
		-------
		float: loss, pytorch.3d.structures.Meshes: adv_mesh
			loss: loss calculated by calc_loss
			adv_mesh: perturbed mesh structure
		"""

		# PERTURB VERTICES
		# current_faces = mesh.faces_packed()
		# adv_mesh = Meshes(verts=[param], faces=[current_faces])
		adv_mesh = mesh.offset_verts(param)

		loss, model_pred, data = self.calc_loss(adv_mesh, grasp)

		return loss, adv_mesh, model_pred, data

	def attack_setup(self, dir, logfile, mesh, grasp, lr, momentum):
		"""Set up attack by saving info and resetting losses"""

		# save attack info
		if not os.path.exists(dir):
			os.mkdir(dir)
		save_obj(dir+"initial-mesh.obj", verts=mesh.verts_list()[0], faces=mesh.faces_list()[0])
		grasp.save(dir+"grasp.json")
		self.renderer.draw_grasp(mesh, grasp.c0, grasp.c1, title="grasp-vis", save=dir+"graps-vis.png", display=False)

		data = {
			"lr": lr,
			"momentum": momentum,
			"optimizer": "SGD",
			"loss weights": list(self.loss_weights.items()), 
			"original oracle ferrari_canny quality": str(grasp.quality.detach().cpu().numpy().tolist()), 
			"oracle robust": self.oracle_robust
		}
		with open(logfile, "w") as f:
			json.dump(data, f, indent=4)

		# reset loss tracking to plot at the end
		self.losses = {
			"prediction": [],
			"edge": [],
			"normal": [],
			"smoothing": []
		}

	def attack(self, mesh, grasp, dir, lr, momentum):
		"""
		Run an attack on the model for number of steps specified in self.num_steps

		Parameters
		----------
		mesh: pytorch3d.structures.Meshes
			Mesh to perturb for adversarial attack
		Returns
		-------
		pytorch3d.structures.Meshes: adv_mesh
			Final adversarial mesh
		"""

		# set up attack: save info + reset losses
		if dir[-1] != "/":
			dir = dir+"/"
		logfile = dir+"logfile.txt"
		self.attack_setup(dir=dir, logfile=logfile, mesh=mesh, grasp=grasp, lr=lr, momentum=momentum)
		original_quality = torch.mean(grasp.quality).item()

		param = torch.zeros(mesh.verts_packed().shape, device=mesh.device, requires_grad=True)
		optimizer = torch.optim.SGD([param], lr=lr, momentum=momentum)

		adv_mesh = mesh.clone()

		num_steps = 0
		prev = 0.002
		for i in range(self.num_steps):
			optimizer.zero_grad()
			loss, adv_mesh, model_pred, data = self.perturb(mesh, param, grasp)
			loss.backward()
			optimizer.step()

			if i % self.steps_per_plot == 0:
				# check oracle prediction
				mesh_file = dir+"cur-mesh.obj"
				save_obj(mesh_file, verts=adv_mesh.verts_list()[0], faces=adv_mesh.faces_list()[0])
				oracle_full = grasp.oracle_eval(mesh_file, oracle_method=self.oracle_method, robust=self.oracle_robust)
				# oracle = torch.mean(oracle_full)
				oracle = oracle_full[0]
				grasp.qualily = oracle_full
				oracle_scaled = oracle / (0.002)	# scale to 1.0
				oracle_scaled = oracle_scaled.item()

				# plot
				print(f"step {i}\t{loss.item()=:.4f}")
				title="step " + str(i) + "\nmodel pred: " + str(model_pred.item()) + "\noriginal oracle:" + str(original_quality) + "\noracle: " + str(oracle.item()) + "\noracle scaled: " + str(oracle_scaled)
					# " loss: " + str(loss.item()) + 
				filename = dir + "step" + str(i) + ".png"
				dim = self.renderer.mesh_to_depth_im(adv_mesh, display=True, title=title, save=filename)

				# log
				data = (f"\n\nIteration{i}\n\tOracle quality mean: " + str(oracle.item()) + f"\n\tOracle scaled: {oracle_scaled}" + data)
				with open(logfile, "a") as f:
					f.write(data)

				if ((oracle_scaled < 0.35 and model_pred > 0.5) or (oracle_scaled > 0.5 and model_pred < 0.5)): # and (oracle - prev <= 0.0005) and i>1:
					num_steps = i+1
					break
			else:
				if (model_pred < 0.5) and i > 1:
					num_steps = i+1
					break

			prev = oracle

		# save final object
		final_mesh = mesh.offset_verts(param)
		final_mesh_file = dir+"final-mesh.obj"
		save_obj(final_mesh_file, verts=mesh.verts_list()[0], faces=mesh.faces_list()[0])

		# get final oracle prediction and save
		# oracle = torch.mean(grasp.oracle_eval(final_mesh_file, oracle_method=self.oracle_method, robust=self.oracle_robust)).item()
		oracle = grasp.oracle_eval(final_mesh_file, oracle_method=self.oracle_method, robust=self.oracle_robust)[0].item()
		data = f"\nFinal oracle mean: {oracle}"
		with open(logfile, "a") as f:
			f.write(data)

		# save final image and attack info
		title = "step" + str(num_steps) + " loss " + str(loss.item()) + "\nmodel pred: " + str(model_pred.item()) + "\noracle pred: " + str(oracle)
		if num_steps == 0:
			filename = dir + "step" + str(self.num_steps) + ".png"
		else:
			filename = dir + "step" + str(num_steps) + ".png"
		self.renderer.mesh_to_depth_im(final_mesh, display=True, title=title, save=filename)
		
		# plot losses
		Attack.plot_losses(self.losses, dir)

		_, image = grasp.extract_tensors_batch(dim)
		return final_mesh, image[0]


def test_run():
	"""Test prediction of gqcnn_pytorch model"""

	Attack.logger.info("Running test_run...")

	if torch.cuda.is_available():
		device = torch.device("cuda:0")
		torch.cuda.set_device(device)
	else:
		print("cuda not available")
		device = torch.device("cpu")

	depth0 = np.load("/home/hmitchell/pytorch3d/dex_shared_dir/depth_0.npy")
	grasp = Grasp(depth=0.607433762324266, im_center=(416, 286), im_angle=-2.896613990462929)

	# load input tensors from gqcnn library for prediction
	pose0 = torch.from_numpy(np.load("data/pose_tensor1_raw.npy")).float().to(device)
	image0 = torch.from_numpy(np.load("data/image_tensor1_raw.npy")).float().permute(0,3,1,2).to(device)

	# tensors from pytorch extraction
	pose1, image1 = grasp.extract_tensors(depth0)

	# tensors from pytorch extraction & pytorch depth image
	renderer = Renderer()
	mesh, _ = renderer.render_object("data/bar_clamp.obj", display=False)
	dim = renderer.mesh_to_depth_im(mesh, display=False)
	pose2, image2 = grasp.extract_tensors(dim)

	# testing with new barclamp object
	mesh2, _ = renderer.render_object("data/new_barclamp.obj", display=False)
	dim2 = renderer.mesh_to_depth_im(mesh2, display=False)
	pose3, image3 = grasp.extract_tensors(dim2)

	# instantiate GQCNN PyTorch model
	model = KitModel("weights.npy")
	model.eval()

	# instantiate Attack class and run prediction
	run1 = Attack(model=model)
	print(run1.run(pose0, image0)[0][0].item())
	print(run1.run(pose1, image1)[0][0].item())
	print(run1.run(pose2, image2)[0][0].item())	# original barclamp object
	print(run1.run(pose3, image3)[0][0].item())	# new barclamp object

	# test model with varying batch sizes
	pose4 = torch.cat([pose1, pose2, pose3], 0)
	image4 = torch.cat([image1, image2, image3], 0)
	print("\n")
	print(pose4.shape, image4.shape)
	print(run1.run(pose4, image4))

	pose5 = torch.cat([pose1, pose2, pose2, pose3, pose1, pose3, pose3, pose1, pose2])
	image5 = torch.cat([image1, image2, image2, image3, image1, image3, image3, image1, image2])
	print(pose5.shape, image5.shape)
	print(run1.run(pose5, image5))

	pose6 = torch.cat([pose4, pose5], dim=0)
	image6 = torch.cat([image4, image5], dim=0)
	print(pose6.shape, image6.shape)
	print(run1.run(pose6, image6))
	
	Attack.logger.info("Finished test_run.")

def test_attack():
	"""Test attack on gqcnn_pytorch model"""

	Attack.logger.info("Running test_attack...")

	# RENDER MESH AND DEPTH IMAGE
	renderer = Renderer()
	mesh, _ = renderer.render_object("data/bar_clamp.obj", display=False)
	dim = renderer.mesh_to_depth_im(mesh, display=False)

	# FIXED GRASP TO ATTACK
	# grasp = Grasp(
	# 	quality=(1, 0.00039880830039262474), 
	# 	depth=0.5824155807495117, 
	# 	world_center=torch.tensor([ 2.7602e-02,  1.7584e-02, -9.2734e-05], device='cuda:0'), 
	# 	world_axis=torch.tensor([-0.9385,  0.2661, -0.2201], device='cuda:0'), 
	# 	c0=torch.tensor([0.0441, 0.0129, 0.0038], device='cuda:0'), 
	# 	c1=torch.tensor([ 0.0112,  0.0222, -0.0039], device='cuda:0'))
	# grasp = Grasp.read("example-grasps/grasp_3.json")
	grasp = Grasp.sample_grasps(obj_f="data/new_barclamp.obj", num_samples=2, renderer=renderer, min_qual=0.001, save_grasp = "experiment-results/ex03/grasp-orig")
	grasp.random_grasps(num_samples=10, camera=renderer.camera)
	assert grasp.num_grasps() == 10
	grasp.trans_world_to_im(renderer.camera)
	grasp.oracle_eval("data/new_barclamp.obj")
	print(grasp.quality)

	pose, image = grasp.extract_tensors_batch(dim)

	model = KitModel("weights.npy")
	model.eval()
	run1 = Attack(num_plots=10, steps_per_plot=10, model=model, renderer=renderer, oracle_method=grasp.oracle_method, oracle_robust=grasp.oracle_robust)

	# MODEL VISUALIZATIONS
	# print("model print:")
	# print(model)

	Attack.logger.info("ATTACK")
	adv_mesh, final_pic = run1.attack(mesh, grasp, "experiment-results/ex03/", lr=1e-5, momentum=0.0)
	renderer.display(final_pic, title="final_grasp", save="experiment-results/ex03/final-grasp.png")

	Attack.logger.info("Finished running test_attack.")

if __name__ == "__main__":
	# test_run()
	test_attack()
