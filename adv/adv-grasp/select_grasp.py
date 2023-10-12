import Pyro4
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import skimage
import cv2
import PIL.Image as PImage
from render import *

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

def transform_im(d_im, cx, cy, translation, angle):
	"""
	Helper method for extract_tensors to rotate and translate a depth image based on a translation
	  and grasp angle using cv2.warpAffine method. Based on code from autolab_core.Image.transform 
	  method.
	Parameters
	----------
	d_im: np.ndarray
		Depth image to be rotated and translated
	cx: float
		X-coordinate of the center of the depth image 
	cy: float
		Y-coordinate of the center of the depth image 
	translation: np.ndarray
		2x1 vector that I don't really understand right now. 
	angle: float
		Grasp angle in randians used to rotate image to align with middle row of pixels
	Returns
	-------
	numpy.ndarray
		Depth image with affine transformation applied.	
	""" 
		
	theta = np.rad2deg(angle)

	# define matrix for translation 
	trans_mat = np.array(
		[[1, 0, translation[1, 0]], [0, 1, translation[0, 0]]], dtype=np.float32 
	)

	# define matrix for rotation 
	rot_mat = cv2.getRotationMatrix2D(
		(cx, cy), theta, 1
	)

	# convert to 3x3 matrices and combine transformations w/ matrix multiplication then revert to 2x3 
	trans_mat_aff = np.r_[trans_mat, [[0, 0, 1]]]
	rot_mat_aff = np.r_[rot_mat, [[0, 0, 1]]]
	full_mat = rot_mat_aff.dot(trans_mat_aff)
	full_mat = full_mat[:2, :]

	# apply transformation with cv2
	image = cv2.warpAffine(
		d_im,
		full_mat,
		(d_im.shape[1], d_im.shape[0]),
		flags=cv2.INTER_NEAREST
	)

	return image 

def crop_im(d_im, height, width):
	"""
	Helper method for extract_tensors to crop a depth image to specified height and width using
	  PIL. Based on code from autolab_core.Image.crop method. 
	Parameters
	----------
	d_im: np.ndarray
		Depth image to be cropped
	height: int
		Height for cropped image
	width: int
		Width for cropped image
	Returns
	-------
	np.ndarray
		Cropped depth image
	"""

	center_i = d_im.shape[0] / 2
	center_j = d_im.shape[1] / 2

	# crop using PIL
	start_row = int(np.floor(center_i - float(height) / 2))
	end_row = int(np.floor(center_i + float(height) / 2))
	start_col = int(np.floor(center_j - float(width) / 2))
	end_col = int(np.floor(center_j + float(width) / 2))

	im = PImage.fromarray(d_im)
	cropped_pil_im = im.crop(
		(
			start_col,
			start_row,
			end_col,
			end_row,
		)
	)
	crop_im = np.array(cropped_pil_im)
	
	return crop_im


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
	image_tensor = skimage.transform.resize(d_im.astype(np.float64), output_shape, order=1, anti_aliasing=False, mode="constant")

	# 2 - translate wrt to grasp angle and grasp center 
	dim_center_x = d_im.shape[1] // 2
	dim_center_y = d_im.shape[0] // 2
	# print("dim center:", dim_center_x, dim_center_y)	
	translation = (1/3) * np.array([
		[dim_center_y - grasp[0][1]],
		[dim_center_x - grasp[0][0]]
	])
	
	new_cx = image_tensor.shape[1] // 2
	new_cy = image_tensor.shape[0] // 2
	dim_rotated = transform_im(image_tensor.squeeze(), new_cx, new_cy, translation, grasp[1]) 

	im_cropped = crop_im(dim_rotated, 32, 32)

	return pose_tensor, im_cropped  
	
if __name__ == "__main__":
	# renderer1 = Renderer()
	# mesh, image = renderer1.render_object("bar_clamp.obj", display=True, title="imported renderer")
	# d_im = renderer1.mesh_to_depth_im(mesh, display=False)

	# Pyro4.config.COMMTIMEOUT = None
	# server = Pyro4.Proxy("PYRO:Server@localhost:5000")
	# print(server.test_extraction("depth_0.npy"))
	# print(server.gqcnn_sample_grasps("depth_0.npy", 100))

	depth0 = np.load("/home/hmitchell/pytorch3d/dex_shared_dir/depth_0.npy")
	grasp = [(416, 286), -2.896613990462929, 0.607433762324266]	

	pose, image = extract_tensors(depth0, grasp)	
	print("image tensor:", image)
	print("pose tensor:", pose)


	"""
	# testing by comparing output to saved output from gqcnn - SUCCESS 
	print("\nCOMPARISON")

	test = np.load("/home/hmitchell/pytorch3d/dex_shared_dir/cropped_test.npy", allow_pickle=True)
	print("tensor difference")
	tens_diff = test.squeeze() - image
	print(tens_diff)
	print(np.max(tens_diff))
	print(np.min(tens_diff))
	"""	



