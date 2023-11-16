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

class Grasp:

	def __init__(self, depth=None, im_center=None, im_angle=None, im_axis=None, world_center=None, world_axis=None, quality=None):
		"""
		Initialize a Grasp object
		Paramters
		---------
		depth: float
			Depth of grasp
		im_center: torch.tensor of size 3
			Grasp center in image coordinates
		im_angle: float
			Angle between grasp axis and camera x-axis; used for tensor extraction
		im_axis: torch.tensor of size 3
			Grasp axis in image coordinates
		world_center: torch.tensor of size 3
			Grasp center in world/mesh coordinates
		world_axis: torch.tensor of size 3
			Grasp axis in world/mesh coordinates
		quality: List of 2 oracle qualities of grasp
			quality[0]: Boolean quality from force_closure
			quality[1]: float quality value from robust_ferrari_canny
		Returns
		-------
		None
		"""

		if quality:
			self.fc_quality = quality[0]
			self.rfc_quality = quality[1]
		else:
			self.fc_quality = None
			self.rfc_quality = None

		self.depth = depth
		self.im_center = im_center
		self.im_axis = im_axis
		self.im_angle = im_angle
		self.world_center = world_center
		self.world_axis = world_axis

	def __str__(self):
		"""Returns a string with grasp information in image coordinates"""
		p_str = "quality: " + str(self.rfc_quality) + "\n\timage center: " + str(self.im_center) + "\n\timage angle: " + str(self.im_angle) + "\n\tdepth: " + str(self.depth)
		return p_str

	def title_str(self):
		"""Retruns a string like __str__, but without tabs"""
		p_str = "quality: " + str(self.rfc_quality) + "\nimage center: " + str(self.im_center) + "\nimage angle: " + str(self.im_angle) + "\ndepth: " + str(self.depth)
		return p_str

	def trans_world_to_im(self, camera):
		"""
		If world coordinate information is populated, calculate image coordinate information for grasp - in-place
		Parameters
		----------
		camera: pytorch3d.renderer.cameras.CameraBase
			Camera to convert grasp info to image space
		Returns
		-------
		None
		"""

		if (self.world_center) == None or (self.world_axis == None):
			logger.error("Grasp does not have world points to transform to image points")
			return None

		# convert to camera space
		world_points = -1 * torch.stack((self.world_center, self.world_axis))
		im_points = camera.transform_points(world_points)
		for i in range(im_points.shape[0]):		# fix depth value
			im_points[i][2] = 1/im_points[i][2]
		self.im_center = im_points[0]
		self.im_axis = im_points[1]
		self.depth = self.im_center[2].item()

		# calculate angle between im_axis and camera x-axis for im_angle
		x_axis = torch.tensor([-1.0, 0.0, 0.0]).to(camera.device)
		dotp = torch.dot(self.im_axis, x_axis)
		axis_norm = torch.linalg.vector_norm(self.im_axis)
		self.im_angle = torch.acos(dotp / axis_norm)

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

def sample_grasps(obj_f, num_samples, renderer, min_qual=0.002, max_qual=1.0, save_grasp=""):
	"""
	Samples grasps using dexnet oracle and returns grasps with qualities between min_qual and max_qual
	  inclusive. Contains RPC to oracle.py in running docker container. Saves new grasp object with an 
	  added sphere to visualize grasp if desired (default: does not save new grasp object)
	Parameters
	----------
	obj_f: String
		path to the .obj file of the mesh to be grasped
	num_samples: int
		Total number of grasps to evaluate (lower bound for number of grasps returned)
	renderer: Renderer object
		Renderer to use to render mesh object with camera to convert grasp info to image space	
	min_qual: float
		Float between 0.0 and 1.0, minimum quality for returned grasps, defaults to 0.002 according to robust ferrari canny method
		NOTE: Currently returns a grasp if force_closure metric returns 1 or robust ferrari canny quality is above min_qual
	max_qual: float
		Float between 0.0 and 1.0, maximum quality for returned grasps, defaults to 1.0
	save_grasp: String
		If empty string, no new grasp objects are saved
		If not empty, this is the path to the directory where to save new grasp objects that have a sphere added to visualize the grasp
	Returns
	-------
	List of Grasp objects with quality between min_qual and max_qual inclusive
	"""

	# test logging
	logger.info("sampling grasps")
	cands = []

	# renderer mesh
	mesh, _ = renderer.render_object(obj_f, display=False)

	while len(cands) < num_samples:

		# randomly sample surface points for possible grasps
		samples_c0 = sample_points_from_meshes(mesh, num_samples*1500)[0]
		samples_c1 = sample_points_from_meshes(mesh, num_samples*1500)[0]
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

		if save_grasp:
			# save object to visualize grasp
			logger.debug("saving new grasp visualization object")
			f_name = save_grasp + "/grasp_" + str(i) +".obj"
			renderer.grasp_sphere(world_center, mesh, f_name)

		grasp = Grasp(depth=world_center[:-1], world_center=world_center, world_axis=world_axis, quality=quality)
		grasp.trans_world_to_im(renderer.camera)

		logger.info("image grasp: %s", str(grasp))

		ret_grasps.append(grasp)

	return ret_grasps 


def extract_tensors(d_im, grasp):
	"""
	Use grasp information and depth image to get image and pose tensors in form of GQCNN input
	Parameters
	----------
	d_im: numpy.ndarray
		Numpy array depth image of object being grasped
	grasp: Grasp object
		Grasp to process depth image with respect to
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
	pose_tensor[0] = grasp.depth

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
	theta = -1 * math.degrees(grasp.im_angle)
	
	dim_cx = torch_dim.shape[2] // 2
	dim_cy = torch_dim.shape[1] // 2	
	
	translate = ((dim_cx - grasp.im_center[0]) / 3, (dim_cy - grasp.im_center[1]) / 3)

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

	# SET UP LOGGING
	logger = logging.getLogger('select_grasp')
	logger.setLevel(logging.INFO)
	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	ch.setFormatter(formatter)
	logger.addHandler(ch)

	renderer1 = Renderer()
	mesh, image = renderer1.render_object("data/bar_clamp.obj", display=False, title="imported renderer")
	d_im = renderer1.mesh_to_depth_im(mesh, display=True)
	d_im = d_im[:, :, np.newaxis]

	model = KitModel("weights.npy")
	model.eval()
	run1 = Attack(model=model)
	
	# TESTING TENSOR EXTRACTION AND GQCNN PREDICTION
	# depth0 = np.load("/home/hmitchell/pytorch3d/dex_shared_dir/depth_0.npy")
	# grasp = Grasp(0.607433762324266, (416, 286), -2.896613990462929)
	# pose, image = extract_tensors(depth0, grasp)	# tensor extraction
	# print("prediction:", run1.run(pose, image))		# gqcnn prediction
	# renderer1.display(image)
	

	# FIXED GRASP FOR TESTING
	# fixed_grasp = {
	# 	'force_closure_q': 1,
	# 	'robust_ferrari_canny_q': 0.00017405830492609492,
	# 	'world_center': torch.tensor([ 0.0157,  0.0167, -0.0599], device='cuda:0'), 
	# 	'world_axis': torch.tensor([ 0.6087, -0.0807, -0.7893], device='cuda:0'), 
	# 	'im_center': torch.tensor([333.6413, 185.6289,   0.5833], device='cuda:0'), 
	# 	'im_axis': torch.tensor([ 7.8896e+02, -3.6920e+02,  6.8074e-01], device='cuda:0'), 
	# 	'im_angle': torch.tensor(2.7039, device='cuda:0'),
	# 	'depth': 0.5792605876922607
	# }

	# DEBUGGING WORLD TO IMAGE COORD TRANSFORMATION
	# world_points = torch.stack((fixed_grasp["world_center"], fixed_grasp["world_axis"]))
	# world_points = torch.tensor([[-0.04, -0.04, -0.04], [-0.04, -0.04, -0.04]], device = renderer1.device)
	# image_grasp = world_to_im_grasp(world_points, 0, renderer1.camera)
	# im_center = image_grasp[1]
	# im_depth = image_grasp[2]
	# pose, image = extract_tensors(d_im, [im_center, 0, im_depth])
	# renderer1.display(image)

	# TESTING SAMPLE GRASPS METHOD AND VISUALIZING
	grasps = sample_grasps("data/bar_clamp.obj", 1, renderer=renderer1, save_grasp="vis_grasps")
	# VISUALIZE SAMPLED GRASPS
	for i in range(len(grasps)):
		grasp = grasps[i]
		qual = grasp.rfc_quality
		pose, image = extract_tensors(d_im, grasp)
		prediction = run1.run(pose, image)
		t = "id: " + str(i) + " prediction:" + str(prediction) + "\n" + grasp.title_str()
		renderer1.display(image, title=t)
