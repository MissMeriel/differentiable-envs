import Pyro4
import os
import sys
import math
import logging
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

def sample_grasps(mesh, num_samples, camera, min_qual=0.002, max_qual=1.0):
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
	[[float: grasp_quality, float: grasp_center, float: grasp_axis, float: grasp_depth], ...]
	"""

	# test logging
	logger.info("sampling grasps")
	cands = []

	# adding renderer for grasp visualization
	renderer = Renderer()
	grasp_obj, _ = renderer.render_object("data/bar_clamp.obj", display=False)

	while len(cands) < num_samples:

		# randomly sample 10000 surface points for 5000 possible grasps
		samples_c0 = sample_points_from_meshes(mesh, 2000)[0]
		samples_c1 = sample_points_from_meshes(mesh, 2000)[0]
		norms = torch.linalg.norm((samples_c1 - samples_c0), dim=1)

		# mask to eliminate grasps that don't fit in the gripper
		mask = (norms > 0.0) & (norms <= 0.05)
		mask = mask.squeeze()
		norms = norms[mask]
		c0 = samples_c0[mask, :]
		c1 = samples_c1[mask, :]
	
		# computer grasp center and axis
		world_centers = (c0 + c1) / 2
		world_axes = (c1 - c0) / norms.unsqueeze(1)

		# get close_fingers result and quality from gqcnn
		logger.info("sending %d grasps to server", world_centers.shape[0])

		Pyro4.config.COMMTIMEOUT = None
		server = Pyro4.Proxy("PYRO:Server@localhost:5000")
		save_nparr(world_centers.detach().cpu().numpy(), "temp_centers.npy")
		save_nparr(world_axes.detach().cpu().numpy(), "temp_axes.npy")
		results = server.close_fingers("temp_centers.npy", "temp_axes.npy")
		logger.info("%d successful grasps returned", len(results))

		cands = cands + results

	# transform successful grasps to image space
	ret_grasps = []
	for i in range(len(cands)):
		g = cands[i]
		quality = g[1:]
		world_center = world_centers[g[0]]
		world_axis = world_axes[g[0]]
		world_points = torch.stack((world_center, world_axis))

		# save object to visualize grasp
		f_name = "vis_grasps/grasp_" + str(i) +".obj"
		renderer1.grasp_sphere(world_center, grasp_obj, f_name)

		# convert to camera space
		im_points = camera.transform_points(world_points)
		for i in range(im_points.shape[0]):		# fix depth value
			im_points[i][2] = 1/im_points[i][2]
		im_center = im_points[0]
		im_axis = im_points[1]
		depth = im_center[2].item()

		# calculate angle between im_axis and camera x-axis for Grasp2D
		x_axis = torch.tensor([1.0, 0.0, 0.0]).to(camera.device)
		dotp = torch.dot(im_axis, x_axis)
		axis_norm = torch.linalg.vector_norm(im_axis)
		logger.debug("image axis norm: %s", axis_norm.item())
		angle = torch.acos(dotp / axis_norm)
		logger.debug("angle in rad: %s", angle)

		im_g = [quality, (im_center[0].item(), im_center[1].item()), angle, depth]
		logger.info("image grasp: %s", im_g)
		ret_grasps.append(im_g)

		# create grasp object
		# angle = 0
		# grasp = [(im_center[0].item(), im_center[1].item()), angle, im_center[2].item()]  
		# print("grasp:", grasp)

	return ret_grasps 


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
	else:
		torch_dim = d_im

	# construct pose tensor from grasp depth
	pose_tensor = torch.zeros([1, 1])
	pose_tensor[0] = grasp[2]

	# process depth image wrt grasp (steps 1-3) 
	
	# 1 - resize image tensor
	out_shape = torch.tensor([torch_dim.shape], dtype=torch.float32)
	out_shape *= (1/3)		# using 1/3 based on gqcnn library - may need to change depending on input
	out_shape = tuple(out_shape.type(torch.int)[0][1:].numpy())
	logger.debug("out_shape: %s", out_shape)

	torch_transform = transforms.Resize(out_shape, antialias=False) 
	torch_image_tensor = torch_transform(torch_dim)
	logger.debug("torch_image_tensor shape: %s", torch_image_tensor.shape)


	# 2 - translate wrt to grasp angle and grasp center 
	theta = -1 * math.degrees(grasp[1])
	
	dim_cx = torch_dim.shape[2] // 2
	dim_cy = torch_dim.shape[1] // 2	
	
	translate = ((dim_cx - grasp[0][0]) / 3, (dim_cy - grasp[0][1]) / 3)

	cx = torch_image_tensor.shape[2] // 2
	cy = torch_image_tensor.shape[1] // 2

	# keep as two separate transformations so translation is performed before rotation
	torch_translated = transforms.functional.affine(
		torch_image_tensor,
		0,		# angle of rotation in degrees clockwise, between -180 and 180 inclusive
		translate,
		scale=1,	# no scale
		shear=0,	# no shear 
		interpolation=transforms.InterpolationMode.BILINEAR,
		center=(cx, cy)	
	)
	
	torch_translated = transforms.functional.affine(
		torch_translated,
		theta,
		translate=(0, 0),
		scale=1,
		shear=0,
		interpolation=transforms.InterpolationMode.BILINEAR,
		center=(cx, cy)
	)
	logger.debug("torch_translated shape: %s", torch_translated.shape)

	# 3 - crop image to size (32, 32)
	torch_cropped = transforms.functional.crop(torch_translated, cy-17, cx-17, 32, 32)
	image_tensor = torch_cropped.unsqueeze(0)

	return pose_tensor, image_tensor  
	
if __name__ == "__main__":
	# set up logging
	logger = logging.getLogger('select_grasp')
	logger.setLevel(logging.DEBUG)
	ch = logging.StreamHandler()
	ch.setLevel(logging.DEBUG)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	ch.setFormatter(formatter)
	logger.addHandler(ch)

	# Pyro4.config.COMMTIMEOUT = None
	# server = Pyro4.Proxy("PYRO:Server@localhost:5000")
	# print(server.test_extraction("depth_0.npy"))
	# print(server.gqcnn_sample_grasps("depth_0.npy", 100))
	# print(server.add(3,4))
	# print("meshpy principal dims:\n", server.close_fingers(0))

	renderer1 = Renderer()
	mesh, image = renderer1.render_object("data/bar_clamp.obj", display=False, title="imported renderer")
	d_im = renderer1.mesh_to_depth_im(mesh)
	d_im = d_im[:, :, np.newaxis]
	logger.debug("d_im shape: %s", d_im.shape)
	
	# bboxes = mesh.get_bounding_boxes()[0]
	# print("\nPyTorch bboxes:")
	# for i in range(3):
	#	box = tuple(bboxes[i])
	#	print(box[1] - box[0])

	# d_im = renderer1.mesh_to_depth_im(mesh, display=False)

	"""
	# testing tensor extraction and gqcnn prediction
	depth0 = np.load("/home/hmitchell/pytorch3d/dex_shared_dir/depth_0.npy")
	print("depth0 size:", depth0.shape)
	print("depth0:\n", depth0)
	grasp = [(416, 286), -2.896613990462929, 0.607433762324266]	
	pose, image = extract_tensors(depth0, grasp)	# tensor extraction
	model = KitModel("weights.npy")
	model.eval()
	run1 = Attack(model=model)
	print("prediction:", run1.run(pose, image))		# gqcnn prediction
	"""

	model = KitModel("weights.npy")
	model.eval()
	run1 = Attack(model=model)

	# [quality, (im_center[0].item(), im_center[1].item()), angle, depth]
	grasps = sample_grasps(mesh, 5, camera=renderer1.camera)
	for i in range(len(grasps)):
		g = grasps[i]
		qual = g[0]
		grasp = g[1:]
		pose, image = extract_tensors(d_im, grasp)
		logger.debug("pose shape: %s", pose.shape)
		logger.debug("image shape: %s", image.shape)
		prediction = run1.run(pose, image)
		t = "id: " + str(i) + " oracle:" + str(qual) + "\nprediction:" + str(prediction) + "\ncenter: " + str(grasp[0]) + "\nangle: " + str(grasp[1].item()) + "\ndepth: " + str(grasp[2])
		renderer1.display(image, title=t)


