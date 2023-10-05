import numpy as np
from render import *
from gqcnn_pytorch import KitModel

class Attack: 

	def __init__(self, num_plots=0, steps_per_plot=0, model=None):
		self.num_plots = num_plots
		self.steps_per_plot = steps_per_plot
		self.num_steps = num_plots * steps_per_plot
		self.model = model

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

		return self.model(pose_tensor, image_tensor)[:, -1] 

	def calc_loss(self, pose_tensor, image_tensor, oracle_pred):
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

		return 0

	def perturb(self, mesh, param):
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

		return 0, None

	def attack(self, mesh):
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

def test_run():
	"""Test prediction of gqcnn_pytorch model"""

	# load input tensors for prediction
	input_pose = torch.from_numpy(np.load("data/pose_tensor1_raw.npy")).float()
	input_image = torch.from_numpy(np.load("data/image_tensor1_raw.npy")).float().permute(0,3,1,2)
	print("image shape:", input_image.shape)	

	# instantiate GQCNN PyTorch model
	model = KitModel("weights.npy")
	model.eval()

	# instantiate Attack class
	run1 = Attack(model=model)
	return run1.run(input_pose, input_image)	
	# return run1.run(input_pose[0], input_image[0].unsqueeze(0))


if __name__ == "__main__":
	print(test_run())



