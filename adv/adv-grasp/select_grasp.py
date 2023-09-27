import Pyro4
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import skimage
from scipy.ndimage import affine_transform
from render import *

SHARED_DIR = "dex_shared_dir"

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

def sample_grasps(dim_file, num_samples, min_qual=0.5, max_qual=1.0):
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
	Returns
	-------
	List of lists containing grasps of the object with quality between min_qual and max_qual inclusive
	[[float: grasp_quality, float: grasp_center, float: grasp_depth, float: grasp_angle, float:
	   grasp_depth, float: gripper_width], ...]
	"""

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
		image_tensor: 1 x 32 x 32 x 1 tensor of depth image processed for grasp
	"""
	
	# construct pose tensor from grasp depth
	pose_tensor = np.zeros([1, 1])
	pose_tensor[0] = grasp[2]

	# process depth image wrt grasp
	# 1 - resize image tensor
	np_shape = np.asarray(d_im.shape).astype(np.float32)
	np_shape[0:2] *= (1/3)	
	output_shape = tuple(np_shape.astype(int))
	image_tensor = skimage.transform.resize(d_im, output_shape)

	# 2 - translate wrt to grasp angle: NOT WORKING RN
	dim_center_x = d_im.shape[0] / 2
	dim_center_y = d_im.shape[1] / 2
	translation = (1/3) * np.array([
		[1, 0, dim_center_x - grasp[0][1]],
		[0, 1, dim_center_y - grasp[0][0]]
	])	
	rotation = np.array([
		[np.cos(grasp[1]), -np.sin(grasp[1])],
		[np.sin(grasp[1]), np.cos(grasp[1])]])
	dim_translated = affine_transform(image_tensor, translation) 
	dim_rotated = affine_transform(dim_translated, rotation)

	# image_tensor = np.zeros([1, 32, 32, 1])
	print(grasp[0][1])
	print(grasp[0][0])

	return pose_tensor, dim_rotated 
	
if __name__ == "__main__":
	# renderer1 = Renderer()
	# mesh, image = renderer1.render_object("bar_clamp.obj", display=True, title="imported renderer")
	# d_im = renderer1.mesh_to_depth_im(mesh, display=False)

	Pyro4.config.COMMTIMEOUT = None
	server = Pyro4.Proxy("PYRO:Server@localhost:5000")
	print(server.test_extraction("depth_0.npy"))

	depth0 = np.load("dex_shared_dir/depth_0.npy")
	grasp = [(416, 286), -2.896613990462929, 0.607433762324266]	
	pose, image = extract_tensors(depth0, grasp)	
	# print("pose size:", pose.shape)
	# print("pose:", pose)	
	print("image size:", image.shape)
	print("image:", image)

"""	
# render barclamp object
barclamp_obj = PyTorchObject("bar_clamp.obj")
image = barclamp_obj.render_obj_file(display=False)
d_im = barclamp_obj.mesh_to_depth(display=False)
# barclamp_obj.save_nparr(d_im, "barclamp.npy")

# communication with server in dex3 docker container
Pyro4.config.COMMTIMEOUT = None

# uri = input("What is the Pytro uri of the server object? ").strip()
# server = Pyro4.Proxy("PYRO:Server@0.0.0.0:5000")
server = Pyro4.Proxy("PYRO:Server@localhost:5000")
print(server.add(4, 3))
# print(server.sample_grasps('bar_clamp.obj'))
# print(server.depth_im("barclamp.npy"))

# print(server.gqcnn_sample_grasps("depth_0.npy", 100))
# check, message = server.gqcnn_sample_grasps("barclamp.npy", 100)
# while not check:
#	check, message = server.gqcnn_sample_grasps("barclamp.npy", 100)
"""


