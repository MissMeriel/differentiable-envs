# import shutil
import logging
import numpy as np
# from torchviz import make_dot
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

	def __init__(self, num_plots=0, steps_per_plot=0, model=None, renderer=None, oracle_method="pytorch"):
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
		if oracle_method not in ["dexnet", "pytorch"]:
			Attack.logger.debug("Attack oracle evaluation method must be 'dexnet' or 'pytorch' -- defaulting to 'pytorch'.")
			self.oracle_method = "pytorch"

		# self.loss_weights = {
		# 	"edge": 5,
		# 	"normal": 0.4,
		# 	"smooth": 5
		# }

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
			return None

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

		# # NO ORACLE
		# adv_mesh_clone = adv_mesh.clone()
		# dim = self.renderer.mesh_to_depth_im(adv_mesh_clone, display=False)
		# pose, image = grasp.extract_tensors_batch(dim)
		# out = self.run(pose, image)
		# cur_pred = out[:,0:1].to(adv_mesh.device)
		# loss = cur_pred
		# return loss

		# NO ORACLE GRADIENT - check current model prediction
		adv_mesh_clone = adv_mesh.clone()
		dim = self.renderer.mesh_to_depth_im(adv_mesh_clone, display=False)
		pose, image = grasp.extract_tensors_batch(dim)
		out = self.run(pose, image)
		cur_pred = out[:,0:1].to(adv_mesh.device)
		if isinstance(grasp.quality, torch.Tensor):
			oracle_pred = torch.clone(grasp.quality) / 0.004	# scale to model range
		else:
			oracle_pred = torch.zeros(1, 1).to(adv_mesh.device)
		loss = torch.sub(1.0, torch.abs(torch.sub(cur_pred, oracle_pred)))
		self.losses["prediction"].append(loss.item())
		return loss

		# MAXIMIZE DIFFERENCE BETWEEN CURRENT PREDICTION AND ORACLE PREDICTION
		# loss = torch.sub(1.0, torch.abs(torch.sub(torch.mean(cur_pred), torch.mean(oracle_pred))))
		loss = torch.sub((self.loss_alpha * oracle_pred), ((1.0 - self.loss_alpha) * cur_pred))		# use means for batches

		# ignore regularization losses for now
		# # WEIGHTED LOSS WITH PYTORCH3D.LOSS FUNCS
		# edge_loss = mesh_edge_loss(adv_mesh) * self.loss_weights["edge"]
		# normal_loss = mesh_normal_consistency(adv_mesh) * self.loss_weights["normal"]
		# smooth_loss = mesh_laplacian_smoothing(adv_mesh) * self.loss_weights["smooth"]
		# weighted_loss = loss + edge_loss + normal_loss + smooth_loss
		# self.losses["prediction"].append(loss.item())
		# self.losses["edge"].append(edge_loss.item())
		# self.losses["normal"].append(normal_loss.item())
		# self.losses["smoothing"].append(smooth_loss.item())

		# data = ("\n\tModel prediction: " + str(cur_pred.detach().cpu().numpy().tolist()) + f"\n\tPrediction loss: {loss}")	
		#\n\tNormal consistency loss: {normal_loss}\n\tLaplacian smoothing loss: {smooth_loss}\n\tMesh edge loss: {edge_loss}\n\tWeighted loss: {weighted_loss} + "\n\tModel prediction mean: " + str(torch.mean(cur_pred).item()) + 

		# return weighted_loss, torch.mean(cur_pred), data
		return loss, cur_pred

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

		# # RANDOM FUZZ PERTURBATION
		# v = normal sample(dims)
		# v = v / torch.linalg.norm(v)
		# random_step = self.learning_rate * v
		# adv_mesh = mesh.offset_verts(param)
		# return adv_mesh
  
		# # NO ORACLE
		# adv_mesh = mesh.offset_verts(param)
		# loss = self.calc_loss(adv_mesh, grasp)
		# return loss, adv_mesh

		# NO ORACLE GRADIENT - perturb vertices
		adv_mesh = mesh.offset_verts(param)
		loss = self.calc_loss(adv_mesh, grasp)

		return loss, adv_mesh

	def attack_setup(self, dir, logfile, grasp, method):
		"""Set up attack by saving info and resetting losses"""

		# save attack info
		if not os.path.exists(dir):
			os.mkdir(dir)
		grasp.save(dir+"grasp.json")

		data = {
			"learning_rate": self.learning_rate,
			"momentum": self.momentum,
			"optimizer": "SGD",
			"loss weight (alpha)": self.loss_alpha, 
			"method": method
		}
		with open(logfile, "w") as f:
			json.dump(data, f, indent=4)

		# reset loss tracking to plot at the end
		self.losses = {
			"prediction": [],
			# "edge": [],
			# "normal": [],
			# "smoothing": []
		}

	def attack(self, mesh, grasp, dir, lr, momentum, loss_alpha, method=""):
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

		# set attack parameters
		self.learning_rate = lr
		self.momentum = momentum
		self.loss_alpha = loss_alpha

		# set up attack: save info + reset losses
		if dir[-1] != "/": dir = dir+"/"
		logfile = dir+"logfile.txt"
		self.attack_setup(dir=dir, logfile=logfile, grasp=grasp, method=method)
		dim = self.renderer.mesh_to_depth_im(mesh, display=False)
		_, orig_pdim = grasp.extract_tensors_batch(dim)

		# SNAPSHOT
		self.snapshot(mesh=mesh, grasp=grasp, dir=dir, iteration=0, orig_pdim=orig_pdim)

		param = torch.zeros(mesh.verts_packed().shape, device=mesh.device, requires_grad=True)
		optimizer = torch.optim.SGD([param], lr=lr, momentum=momentum)

		adv_mesh = mesh.clone()

		for i in range(1, self.num_steps):
			optimizer.zero_grad()
			loss, adv_mesh = self.perturb(mesh, param, grasp)
			loss.backward()
			optimizer.step()

			if i % self.steps_per_plot == 0:
				# SNAPSHOT
				self.snapshot(mesh=adv_mesh, grasp=grasp, dir=dir, iteration=i, orig_pdim=orig_pdim)

		# save final object
		final_mesh = mesh.offset_verts(param)
		self.snapshot(mesh=final_mesh, grasp=grasp, dir=dir, iteration=i+1, orig_pdim=orig_pdim, save_mesh=True)
		
		# plot losses
		Attack.plot_losses(self.losses, dir)

		# _, image = grasp.extract_tensors_batch(dim)
		return final_mesh	#, image[0]

	def snapshot(self, mesh, grasp, dir, iteration, orig_pdim, save_mesh=False):
		"""Save information at current point in attack."""
		# TODO: Implement for batch of grasps in attack

		mesh_file = dir + f"it-{iteration}.obj"
		save_obj(mesh_file, verts=mesh.verts_list()[0], faces=mesh.faces_list()[0])
		image = self.renderer.render_mesh(mesh, display=False)

		dim = self.renderer.mesh_to_depth_im(mesh, display=False)
		pose, processed_dim = grasp.extract_tensors_batch(dim)
		model_pred = self.run(pose, processed_dim)[:,0:1]
		oracle_qual = grasp.oracle_eval(mesh_file, renderer=self.renderer)
		if isinstance(oracle_qual, torch.Tensor): oracle_qual = oracle_qual.item()
		if orig_pdim is not None: depth_diff = orig_pdim - processed_dim

		title = f"Iteration {iteration}: oracle quality {oracle_qual:.4f}, gqcnn prediction {model_pred.item():.4f}"
		fname = dir + "it-" + str(iteration) + ".png"
		self.renderer.display(images=[image, dim, processed_dim, depth_diff], shape=(1,4), title=title, save=fname)

		if not save_mesh:
			os.remove(mesh_file)

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
	mesh, _ = renderer.render_object("data/new_barclamp.obj", display=False)
	dim = renderer.mesh_to_depth_im(mesh, display=False)

	# FIXED GRASP TO ATTACK
	grasp = Grasp.read("grasp-batch.json")[0]
	print("oracle quality:", grasp.quality.item())

	# SET UP ATTACK
	model = KitModel("weights.npy")
	model.eval()
	run1 = Attack(num_plots=10, steps_per_plot=10, model=model, renderer=renderer, oracle_method=grasp.oracle_method)

	# RUN INITIAL MODEL PREDICTION
	pose, image = grasp.extract_tensors_batch(dim)
	pred = run1.run(pose, image)
	print("initial model prediction:", pred[0].item())

	Attack.logger.info("ATTACK")
	adv_mesh, final_pic = run1.attack(mesh, grasp, "test-attack", lr=1e-5, momentum=0.0)
	renderer.display(final_pic, title="final_grasp", save="test-attack/final-grasp.png")

	Attack.logger.info("Finished running test_attack.")

def test_run2():
	Attack.logger.info("Running test_run2...")

	# set up
	gqcnn_model = KitModel("weights.npy")
	gqcnn_model.eval()
	r = Renderer()
	model = Attack(model=gqcnn_model, renderer=r)
	mesh, _ = r.render_object("data/new_barclamp.obj", display=False)
	depth_im = r.mesh_to_depth_im(mesh, display=False)

	# run model on batch of grasps
	grasp = Grasp.read("grasp-batch.json")
	grasp.trans_world_to_im(camera=r.camera)
	poses, images = grasp.extract_tensors_batch(depth_im)
	out = model.run(poses, images)
	out = out[:,0:1]
	print(f"prediction: {out.shape} {out.device}\n{out}")

	grasp.prediction = out
	grasp.save("grasp-batch.json")

	# run model on individual grasps
	for i, g in enumerate(grasp):
		pose, image = g.extract_tensors(depth_im)
		pred = model.run(pose, image)[:, 0:1]
		print(f"\nGrasp {i}: \n\tModel prediction: {pred.item()}\n\tOriginal prediction: {out[i].item()}\n\tOracle quality: {g.quality.item()}")
		if not out[i].item() == pred.item():
			print(f"\tPrediction diff: {torch.sub(pred, out[i]).item()}")

	Attack.logger.info("Done running test_run2.")

if __name__ == "__main__":
	# test_run2()

	r = Renderer()
	mesh, _ = r.render_object("data/new_barclamp.obj", display=False)
	g = Grasp.read("grasp-batch.json")
	grasps = [g[0], g[1], g[3], g[4], g[5], g[6]]

	model = KitModel("weights.npy")
	model.eval()
	run1 = Attack(num_plots=10, steps_per_plot=20, model=model, renderer=r, oracle_method="pytorch")

	print("ATTACK 1\n")
	run1.attack(mesh=mesh, grasp=grasps[5], dir="exp-results/no-oracle-grad/grasp6/lr0/", lr=1e-5, momentum=0.0, loss_alpha=None, method="no-oracle-grad")
	print("ATTACK 2")
	run1.attack(mesh=mesh, grasp=grasps[5], dir="exp-results/no-oracle-grad/grasp6/lr1/", lr=1e-5, momentum=0.9, loss_alpha=None, method="no-oracle-grad")
	print("ATTACK 3\n")
	run1.attack(mesh=mesh, grasp=grasps[5], dir="exp-results/no-oracle-grad/grasp6/lr2/", lr=1e-5, momentum=0.99, loss_alpha=None, method="no-oracle-grad")
	print("ATTACK 4\n")
	run1.attack(mesh=mesh, grasp=grasps[5], dir="exp-results/no-oracle-grad/grasp6/lr3/", lr=1e-4, momentum=0.0, loss_alpha=None, method="no-oracle-grad")
	print("ATTACK 5\n")
	run1.attack(mesh=mesh, grasp=grasps[5], dir="exp-results/no-oracle-grad/grasp6/lr4/", lr=1e-4, momentum=0.9, loss_alpha=None, method="no-oracle-grad")
	print("ATTACK 6\n")
	run1.attack(mesh=mesh, grasp=grasps[5], dir="exp-results/no-oracle-grad/grasp6/lr5/", lr=1e-4, momentum=0.99, loss_alpha=None, method="no-oracle-grad")