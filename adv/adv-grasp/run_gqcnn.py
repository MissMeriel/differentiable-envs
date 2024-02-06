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
			"edge": 10,
			"normal": 5,
			"smooth": 10
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
		plt.show()
		plt.savefig(dir+"losses.png")

	def calc_loss(self, adv_mesh, grasp):
		"""
		Calculates loss for the adversarial attack to maximize difference between oracle and model prediction

		Parameters
		----------
		pose_tensor: {batch_size} x 1 torch.tensor
		image_tensor: {batch_size} x 32 x 32 x 1 torch.tensor
		oracle_pred: List of size {batch_size}
			Oracle predicted values of each grasp
		Returns
		-------
		float: calculated loss
		"""

		# CHECK CURRENT MODEL PREDICTION
		adv_mesh_clone = adv_mesh.clone()
		dim = self.renderer.mesh_to_depth_im(adv_mesh_clone, display=False)
		pose, image = grasp.extract_tensors(dim)
		out = self.run(pose, image)
		cur_pred = out[0][0]
		cur_pred = cur_pred.to(adv_mesh.device)
		oracle_pred = torch.tensor([float(grasp.fc_quality)], requires_grad=False, device=adv_mesh.device)

		# MAXIMIZE DIFFERENCE BETWEEN CURRENT PREDICTION AND ORACLE PREDICTION
		loss = torch.sub(1.0, torch.abs(torch.sub(cur_pred, oracle_pred)))

		# WEIGHTED LOSS WITH PYTORCH3D.LOSS FUNCS
		edge_loss = mesh_edge_loss(adv_mesh) * self.loss_weights["edge"]
		normal_loss = mesh_normal_consistency(adv_mesh) * self.loss_weights["normal"]
		smooth_loss = mesh_laplacian_smoothing(adv_mesh) * self.loss_weights["smooth"]
		weighted_loss = loss + edge_loss + normal_loss + smooth_loss
		self.losses["prediction"].append(loss.item())
		self.losses["edge"].append(edge_loss.item())
		self.losses["normal"].append(normal_loss.item())
		self.losses["smoothing"].append(smooth_loss.item())

		return weighted_loss, cur_pred

	def perturb(self, mesh, param, grasp):
		"""
		Perturb the mesh for the adversarial attack

		Parameters
		----------
		mesh: pytorch3d.structures.Meshes
			Mesh to be perturbed
		param: torch.tensor
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

		loss, model_pred = self.calc_loss(adv_mesh, grasp)

		return loss, adv_mesh, model_pred

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

		# start by saving attack information
		if dir[-1] != "/":
			dir = dir+"/"
		if not os.path.exists(dir):
			os.mkdir(dir)
		save_obj(dir+"initial-mesh.obj", verts=mesh.verts_list()[0], faces=mesh.faces_list()[0])
		grasp.save(dir+"grasp.json")

		# reset loss tracking to plot at the end
		self.losses = {
			"prediction": [],
			"edge": [],
			"normal": [],
			"smoothing": []
		}

		# param = mesh.verts_packed().clone().detach().requires_grad_(True)
		param = torch.zeros(mesh.verts_packed().shape, device=mesh.device, requires_grad=True)
		optimizer = torch.optim.SGD([param], lr=lr, momentum=momentum)

		adv_mesh = mesh.clone()

		num_steps = 0
		for i in range(self.num_steps):
			optimizer.zero_grad()
			loss, adv_mesh, model_pred = self.perturb(mesh, param, grasp)
			loss.backward()
			optimizer.step()

			if i % self.steps_per_plot == 0:
				print(f"step {i}\t{loss.item()=:.4f}")
				title="step " + str(i) + " loss: " + str(loss.item()) + "\nmodel pred: " + str(model_pred.item())
				filename = dir + "step" + str(i) + ".png"
				dim = self.renderer.mesh_to_depth_im(adv_mesh, display=True, title=title, save=True, fname=filename)

			if model_pred < 0.5:
				num_steps = i+1
				break

		# save final object
		final_mesh = mesh.offset_verts(param)
		final_mesh_file = dir+"final-mesh.obj"
		save_obj(final_mesh_file, verts=final_mesh.verts_list()[0], faces=final_mesh.faces_list()[0])

		# get final oracle prediction
		oracle = grasp.oracle_eval(final_mesh_file, oracle_method=self.oracle_method, robust=self.oracle_robust)

		# save final image and attack info, plot losses
		title = "step" + str(self.num_steps) + " loss " + str(loss.item()) + "\nmodel pred: " + str(model_pred.item()) + "\noracle pred: " + str(oracle)
		if num_steps == 0:
			filename = dir + "step" + str(self.num_steps) + ".png"
		else:
			filename = dir + "step" + str(num_steps) + ".png"
		self.renderer.mesh_to_depth_im(final_mesh, display=True, title=title, save=True, fname=filename)
		Attack.plot_losses(self.losses, dir)
		data = {
			"lr": lr,
			"momentum": momentum,
			"optimizer": "SGD",
			"loss weights": list(self.loss_weights.items()), 
			"oracle ferrari_canny quality": oracle, 
			"oracle robust": self.oracle_robust
		}
		with open(dir+"setup.txt", "w") as f:
			json.dump(data, f, indent=4)

		_, image = grasp.extract_tensors(dim)
		return final_mesh, image

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
	grasp = Grasp(
		quality=(1, 0.00039880830039262474), 
		depth=0.5824155807495117, 
		world_center=torch.tensor([ 2.7602e-02,  1.7584e-02, -9.2734e-05], device='cuda:0'), 
		world_axis=torch.tensor([-0.9385,  0.2661, -0.2201], device='cuda:0'), 
		c0=torch.tensor([0.0441, 0.0129, 0.0038], device='cuda:0'), 
		c1=torch.tensor([ 0.0112,  0.0222, -0.0039], device='cuda:0'))
	grasp.trans_world_to_im(renderer.camera)

	pose, image = grasp.extract_tensors(dim)

	model = KitModel("weights.npy")
	model.eval()
	run1 = Attack(num_plots=10, steps_per_plot=50, model=model, renderer=renderer)

	# MODEL VISUALIZATIONS
	print("model print:")
	print(model)

	Attack.logger.info("ATTACK")
	adv_mesh, final_pic = run1.attack(mesh, grasp, "experiment-results/ex07/", lr=1e-5, momentum=0.9)
	renderer.display(final_pic, title="final_grasp", save=True, fname="experiment-results/ex07/final-grasp.png")

	Attack.logger.info("Finished running test_attack.")

if __name__ == "__main__":
	test_run()
	test_attack()
