import logging
import numpy as np
from torchviz import make_dot
from pytorch3d.io import save_obj
from pytorch3d.loss import mesh_edge_loss, mesh_normal_consistency, mesh_laplacian_smoothing

from render import *
from gqcnn_pytorch import KitModel
from select_grasp import *

class Attack: 

	def __init__(self, num_plots=0, steps_per_plot=0, model=None, renderer=None):
		self.num_plots = num_plots
		self.steps_per_plot = steps_per_plot
		self.num_steps = num_plots * steps_per_plot
		self.model = model
		self.renderer = renderer

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

		return self.model(pose_tensor, image_tensor)

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
		pose, image = extract_tensors(dim, grasp, logger)
		out = self.run(pose, image)
		cur_pred = out[0][0]
		cur_pred = cur_pred.to(adv_mesh.device)
		oracle_pred = torch.tensor([float(grasp.fc_quality)], requires_grad=False, device=adv_mesh.device)

		# MAXIMIZE DIFFERENCE BETWEEN CURRENT PREDICTION AND ORACLE PREDICTION
		loss = torch.sub(1.0, torch.abs(torch.sub(cur_pred, oracle_pred)))

		# WEIGHTED LOSS WITH PYTORCH3D.LOSS FUNCS
		edge_loss = mesh_edge_loss(adv_mesh)
		normal_loss = mesh_normal_consistency(adv_mesh)
		smooth_loss = mesh_laplacian_smoothing(adv_mesh)
		#print("\nedge_loss:", edge_loss.item(), "\nnormal_loss:", normal_loss.item(), "\nsmooth_loss", smooth_loss.item())
		weighted_loss = loss + 10*edge_loss + 10*normal_loss + 10*smooth_loss

		return weighted_loss

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

		loss = self.calc_loss(adv_mesh, grasp)

		return loss, adv_mesh

	def attack(self, mesh, grasp, dir):
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
		save_obj(dir+"initial-mesh.obj", verts=mesh.verts_list()[0], faces=mesh.faces_list()[0])
		grasp.save(dir+"grasp.json")

		# param = mesh.verts_packed().clone().detach().requires_grad_(True)
		param = torch.zeros(mesh.verts_packed().shape, device=mesh.device, requires_grad=True)
		print("param before:", param)
		print("param grad before:", param.grad)
		print("\n\n")
		optimizer = torch.optim.SGD([param], lr=1e-5, momentum=0.99)

		adv_mesh = mesh.clone()

		for i in range(self.num_steps):
			optimizer.zero_grad()
			loss, adv_mesh = self.perturb(mesh, param, grasp)
			loss.backward()
			optimizer.step()
			# print("\n")
			print(f"step {i}\t{loss.item()=:.4f}")
			# print("param:", param)
			# print("param grad:", param.grad)

			if i % self.steps_per_plot == 0:
				title="step " + str(i) + " loss " + str(loss.item())
				filename = dir + "step" + str(i) + ".png"
				dim = self.renderer.mesh_to_depth_im(adv_mesh, display=True, title=title, save=True, fname=filename)

		# save final image and object
		final_mesh = mesh.offset_verts(param)
		save_obj(dir+"final-mesh.obj", verts=final_mesh.verts_list()[0], faces=final_mesh.faces_list()[0])
		title = "step" + str(self.num_steps) + " loss " + str(loss.item())
		filename = dir + "step" + str(self.num_steps) + ".png"
		self.renderer.mesh_to_depth_im(final_mesh, display=True, title=title, save=True, fname=filename)

def test_run(logger):
	"""Test prediction of gqcnn_pytorch model"""

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
	pose1, image1 = extract_tensors(depth0, grasp, logger)

	# tensors from pytorch extraction & pytorch depth image
	renderer = Renderer()
	mesh, _ = renderer.render_object("data/bar_clamp.obj", display=False)
	dim = renderer.mesh_to_depth_im(mesh, display=False)
	pose2, image2 = extract_tensors(dim, grasp, logger)

	# instantiate GQCNN PyTorch model
	model = KitModel("weights.npy")
	model.eval()

	# instantiate Attack class and run prediction
	run1 = Attack(model=model)
	print(run1.run(pose0, image0)[0])
	print(run1.run(pose1, image1)[0])
	print(run1.run(pose2, image2)[0])
	
	return "success"

def test_attack(logger):
	"""Test attack on gqcnn_pytorch model"""

	# RENDER MESH AND DEPTH IMAGE
	renderer = Renderer()
	mesh, _ = renderer.render_object("data/bar_clamp.obj", display=False)
	dim = renderer.mesh_to_depth_im(mesh, display=False)
	# dim = dim[:, :, np.newaxis]

	# FIXED GRASP TO ATTACK
	grasp = Grasp(
		quality=(1, 0.00039880830039262474), 
		depth=0.5824155807495117, 
		world_center=torch.tensor([ 2.7602e-02,  1.7584e-02, -9.2734e-05], device='cuda:0'), 
		world_axis=torch.tensor([-0.9385,  0.2661, -0.2201], device='cuda:0'), 
		c0=torch.tensor([0.0441, 0.0129, 0.0038], device='cuda:0'), 
		c1=torch.tensor([ 0.0112,  0.0222, -0.0039], device='cuda:0'))
	grasp.trans_world_to_im(renderer.camera)

	pose, image = extract_tensors(dim, grasp, logger)

	model = KitModel("weights.npy")
	model.eval()
	run1 = Attack(num_plots=10, steps_per_plot=50, model=model, renderer=renderer)

	# MODEL VISUALIZATIONS
	print("model print:")
	print(model)

	# yhat = model(pose, image)
	# make_dot(yhat, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")

	# print("oracle rfc value:", grasp.rfc_quality)
	# print("oracle fc value:", grasp.fc_quality)
	# print("prediction:", run1.run(pose, image)[0][0].item())

	print("\nattack")
	run1.attack(mesh, grasp, "experiment-results/ex02/")

	return "success"



if __name__ == "__main__":
	# SET UP LOGGING
	logger = logging.getLogger('run_gqcnn')
	logger.setLevel(logging.INFO)
	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	ch.setFormatter(formatter)
	logger.addHandler(ch)

	# print(test_run(logger))
	print(test_attack(logger))


