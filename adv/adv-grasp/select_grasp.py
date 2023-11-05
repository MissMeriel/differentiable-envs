import Pyro4
import os
import sys
import math
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

from pytorch3d.ops import sample_points_from_meshes

from render import *
from run_gqcnn import *

SHARED_DIR = "/home/hmitchell/pytorch3d/dex_shared_dir"

def save_nparr(image, filename):
	""" 
	Save numpy.ndarray image in the shared dir
	Parameters
	----------
	image: numpy.ndarray
		Numpy array to be saved
	filename: String
		Name to save numpy file as
	Returns
	-------
	None
	"""
	
	filepath = os.path.join(SHARED_DIR, filename)
	np.save(filepath, image)

def preprocess_dim(d_im):
	"""
	Pre-processes a numpy depth image before grasp selection (if necessary?)
	Parameters
	----------
	d_im: numpy.ndarray
		Depth image to be pre-processed before grasp selection (particular size??)
	Returns
	-------
	numpy.ndarray
		Processed depth image
	"""

	return None	

def sample_grasps(mesh, num_samples, camera, min_qual=0.5, max_qual=1.0):
	"""
	Samples grasps using dexnet oracle and returns grasps with qualities between min_qual and max_qual
	  inclusive. Contains RPC to oracle.py in running docker container.
	Parameters
	----------
	dim_file: String
		Name of depth image file saved in shared directory
	num_samples: int
		Total number of grasps to evaluate
	min_qual: float
		Float between 0.0 and 1.0, minimum quality for returned grasps, defaults to 0.5
	max_qual: float
		Float between 0.0 and 1.0, maximum quality for returned grasps, defaults to 1.0
	camera: pytorch3d.renderer.cameras.CameraBase
		Camera to convert grasp info to image space	
	Returns
	-------
	List of lists containing grasps of the object with quality between min_qual and max_qual inclusive
	[[float: grasp_quality, float: grasp_center, float: grasp_depth, float: grasp_angle, float:
	   grasp_depth, float: gripper_width], ...]
	"""

	g = []

	while len(g) < num_samples:

		# randomly get two surface points to test for grasping
		samples = sample_points_from_meshes(mesh, 2)[0] 
		p0 = samples[0] 
		p1 = samples[1]

		v_norm = torch.linalg.vector_norm(p1 - p0)	
		if v_norm.item() > 0.0 and v_norm.item() < 0.05: # does grasp fit in gripper?

			print("samples:", samples)
	
			world_center = (p0 + p1) / 2
			world_axis = (p1 - p0) / v_norm
			world_points = torch.stack((world_center, world_axis))
			# print("\nworld points:\n", world_points)

			# convert to camera space
			im_points = camera.transform_points(world_points)
			for i in range(im_points.shape[0]):		# fix depth value
				im_points[i][2] = 1/im_points[i][2]
			im_center = im_points[0]
			im_axis = im_points[1]
			
			"""
			# ORIGIN TEST FOR DEBUGGING
			origin_test = camera.transform_points(torch.zeros([2,3]).to(camera.device))[0]
			origin_test[2] = 1/origin_test[2]
			# print("\norigin test:", origin_test)  
			# print(camera.get_full_projection_transform().get_matrix())
			# print("transform  points:", camera.get_full_projection_transform().transform_points(torch.zeros([2,3]).to(camera.device)))
			# tens = torch.tensor([[0.0], [0.0], [0.0], [1.0]]).to(camera.device)
			# print(torch.matmul(camera.get_full_projection_transform().get_matrix(), tens))
			# print("world to view transform:", camera.get_world_to_view_transform().transform_points(torch.zeros([2,3]).to(camera.device)))
			"""

			# create grasp
			angle = 0
			grasp = [(im_center[0].item(), im_center[1].item()), angle, im_center[2].item()]  
			print("grasp:", grasp)

			# can fingers close?
			# 1) create lines of action (two total - one for each gripper finger) 
			# 2) find contacts along lines of action - approx method from gqcnn or implement exact method with COM 

			g.append([0])

	return None 


def extract_tensors(d_im, grasp):
	"""
	Use grasp information and depth image to get image and pose tensors in form of GQCNN input
	Parameters
	----------
	d_im: numpy.ndarray
		Numpy array depth image of object being grasped
	grasp: List containing grasp information: [(int, int): grasp_center, float: grasp_angle, float:
	   grasp_depth]
		grasp_center: Grasp center used to center depth image
		grasp_angle: Angle used to rotate depth image to align with middle row of pixels
		grasp_depth: Depth of grasp to create pose array
	Returns
	-------
	torch.tensor: pose_tensor, torch.tensor: image_tensor
		pose_tensor: 1 x 1 tensor of grasp pose
		image_tensor: 1 x 1 x 32 x 32 tensor of depth image processed for grasp
	"""

	# check type of input_dim
	if isinstance(d_im, np.ndarray):
		torch_dim = torch.tensor(d_im, dtype=torch.float32).permute(2, 0, 1)

	# construct pose tensor from grasp depth
	pose_tensor = torch.zeros([1, 1])
	pose_tensor[0] = grasp[2]

	# process depth image wrt grasp (steps 1-3) 
	
	# 1 - resize image tensor
	out_shape = torch.tensor([torch_dim.shape], dtype=torch.float32)
	out_shape *= (1/3)
	out_shape = tuple(out_shape.type(torch.int)[0][1:].numpy())

	torch_transform = transforms.Resize(out_shape, antialias=False) 
	torch_image_tensor = torch_transform(torch_dim)


	# 2 - translate wrt to grasp angle and grasp center 
	theta = -1 * math.degrees(grasp[1])
	
	dim_cx = torch_dim.shape[2] // 2
	dim_cy = torch_dim.shape[1] // 2	
	
	translate = ((dim_cx - grasp[0][0]) / 3, (dim_cy - grasp[0][1]) / 3)

	cx = torch_image_tensor.shape[2] // 2
	cy = torch_image_tensor.shape[1] // 2

	torch_translated = transforms.functional.affine(
		torch_image_tensor,
		0,		# angle of rotation in degrees clockwise, between -180 and 180 inclusive
		translate,
		scale=1,	# no scale
		shear=0,	# no shear 
		interpolation=transforms.InterpolationMode.NEAREST,
		center=(cx, cy)	
	)
	
	torch_translated = transforms.functional.affine(
		torch_translated,
		theta,
		translate=(0, 0),
		scale=1,
		shear=0,
		interpolation=transforms.InterpolationMode.NEAREST,
		center=(cx, cy)
	)


	# 3 - crop image to size (32, 32)
	torch_cropped = transforms.functional.crop(torch_translated, cy-17, cx-17, 32, 32)
	image_tensor = torch_cropped.unsqueeze(0)

	return pose_tensor, image_tensor  
	
if __name__ == "__main__":
	renderer1 = Renderer()
	mesh, image = renderer1.render_object("data/bar_clamp.obj", display=False, title="imported renderer")
	d_im = renderer1.mesh_to_depth_im(mesh, display=False)

	# Pyro4.config.COMMTIMEOUT = None
	# server = Pyro4.Proxy("PYRO:Server@localhost:5000")
	# print(server.test_extraction("depth_0.npy"))
	# print(server.gqcnn_sample_grasps("depth_0.npy", 100))

	# testing tensor extraction and gqcnn prediction
	depth0 = np.load("/home/hmitchell/pytorch3d/dex_shared_dir/depth_0.npy")
	grasp = [(416, 286), -2.896613990462929, 0.607433762324266]	
	pose, image = extract_tensors(depth0, grasp)	# tensor extraction
	model = KitModel("weights.npy")
	model.eval()
	run1 = Attack(model=model)
	print("prediction:", run1.run(pose, image))		# gqcnn prediction
	
	# sample_grasps(mesh, 1, camera=renderer1.camera)


