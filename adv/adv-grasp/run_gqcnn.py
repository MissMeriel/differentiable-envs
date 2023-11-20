import logging
import numpy as np
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

		# check current model prediction
		dim = self.renderer.mesh_to_depth_im(adv_mesh, display=False)
		dim = dim[:, :, np.newaxis]
		pose, image = extract_tensors(dim, grasp, logger)
		cur_pred = self.run(pose, image)[0][0].item()

		# maximize difference between cur_pred and oracle prediction
		loss = 1 - abs(cur_pred - grasp.fc_quality)
		loss = torch.tensor([loss], requires_grad=True)

		return loss

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

		# perturb vertices
		# current_faces = mesh.faces_packed()
		# adv_mesh = Meshes(verts=[param], faces=[current_faces])
		adv_mesh = mesh.offset_verts(param)

		loss = self.calc_loss(adv_mesh, grasp)

		return loss, adv_mesh

	def attack(self, mesh, grasp):
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

		# param = mesh.verts_packed().clone().detach().requires_grad_(True)
		param = torch.zeros(mesh.verts_packed().shape, device=mesh.device, requires_grad=True)
		optimizer = torch.optim.SGD([param], lr=1e-4, momentum=0.99)

		adv_mesh = mesh

		for i in range(self.num_steps):
			optimizer.zero_grad()
			loss, adv_mesh = self.perturb(adv_mesh, param, grasp)
			loss.backward()
			optimizer.step()
			print(f"step {i}\t{loss.item()=:.4f}")

			# if i % self.steps_per_plot == 0:
			# 	title="step " + str(i) + " loss " + str(loss.item())
			# 	dim = self.renderer.mesh_to_depth_im(adv_mesh, display=True, title=title)

def test_run(logger):
	"""Test prediction of gqcnn_pytorch model"""

	depth0 = np.load("/home/hmitchell/pytorch3d/dex_shared_dir/depth_0.npy")
	grasp = Grasp(depth=0.607433762324266, im_center=(416, 286), im_angle=-2.896613990462929)

	# load input tensors from gqcnn library for prediction
	pose0 = torch.from_numpy(np.load("data/pose_tensor1_raw.npy")).float()
	image0 = torch.from_numpy(np.load("data/image_tensor1_raw.npy")).float().permute(0,3,1,2)

	# tensors from pytorch extraction
	pose1, image1 = extract_tensors(depth0, grasp, logger)

	# tensors from pytorch extraction & pytorch depth image
	renderer = Renderer()
	mesh, _ = renderer.render_object("data/bar_clamp.obj", display=False)
	dim = renderer.mesh_to_depth_im(mesh, display=False)
	dim = dim[:, :, np.newaxis]
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

	# render mesh and depth image
	renderer = Renderer()
	mesh, _ = renderer.render_object("data/bar_clamp.obj", display=False)
	dim = renderer.mesh_to_depth_im(mesh, display=False)
	dim = dim[:, :, np.newaxis]

	# fixed grasp to attack
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
	run1 = Attack(num_plots=3, steps_per_plot=100, model=model, renderer=renderer)

	print("oracle rfc value:", grasp.rfc_quality)
	print("oracle fc value:", grasp.fc_quality)
	print("prediction:", run1.run(pose, image)[0][0].item())

	print("\nattack")
	run1.attack(mesh, grasp)

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


