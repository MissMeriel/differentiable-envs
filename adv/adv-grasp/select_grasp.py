import Pyro4
import os
import sys
import math
import logging
import json
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

from pytorch3d.ops import sample_points_from_meshes

from render import *
from run_gqcnn import *

SHARED_DIR = "/home/hmitchell/pytorch3d/dex_shared_dir"

if torch.cuda.is_available():
	device = torch.device("cuda:0")
	torch.cuda.set_device(device)
else:
	print("cuda not available")
	device = torch.device("cpu")

class Grasp:

	def __init__(self, depth=None, im_center=None, im_angle=None, im_axis=None, world_center=None, world_axis=None, c0=None, c1=None, quality=None):
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
		c0: torch.tensor of size 3
			First contact point of grasp in world coordinates
		c1: torch.tensor of size 3
			Second contact point of grasp in world coordinates
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
		self.c0 = c0
		self.c1 = c1

	@classmethod
	def init_from_dict(cls, dict):
		"""Initialize grasp object from a dictionary"""
		return Grasp(depth=dict["depth"], im_center=dict["im_center"], im_angle=dict["im_angle"], im_axis=dict["im_axis"], world_center=dict["world_center"], world_axis=dict["world_axis"], c0=dict["c0"], c1=dict["c1"], quality=dict["quality"])

	@classmethod
	def read(cls, fname):
		"""Reads a JSON file fname with saved grasp information and initializes"""

		# read file
		with open(fname) as f:
			dictionary = json.load(f)

		# convert lists to tensors
		if dictionary["im_center"] and dictionary["im_axis"]:
			dictionary["im_center"] = torch.from_numpy(np.array(dictionary["im_center"])).to(device)
			dictionary["im_axis"] = torch.from_numpy(np.array(dictionary["im_axis"])).to(device)
		if dictionary["world_center"] and dictionary["world_axis"]:
			dictionary["world_center"] = torch.from_numpy(np.array(dictionary["world_center"])).to(device)
			dictionary["world_axis"] = torch.from_numpy(np.array(dictionary["world_axis"])).to(device)
		if dictionary["c0"] and dictionary["c1"]:
			dictionary["c0"] = torch.from_numpy(np.array(dictionary["c0"])).to(device)
			dictionary["c1"] = torch.from_numpy(np.array(dictionary["c1"])).to(device)

		return cls.init_from_dict(dictionary)

	def __str__(self):
		"""Returns a string with grasp information in image coordinates"""
		p_str = "quality: " + str(self.fc_quality) + " , " + str(self.rfc_quality) + "\n\timage center: " + str(self.im_center) + "\n\timage angle: " + str(self.im_angle) + "\n\tdepth: " + str(self.depth) + "\n\tworld center: " + str(self.world_center) + "\n\tworld axis: " + str(self.world_axis)
		return p_str

	def title_str(self):
		"""Retruns a string like __str__, but without tabs"""
		p_str = "quality: " + str(self.rfc_quality) + "\nimage center: " + str(self.im_center) + "\nimage angle: " + str(self.im_angle) + "\ndepth: " + str(self.depth)
		return p_str

	def save(self, fname):
		"""Saves a JSON file with grasp information in file fname"""

		# convert tensors to lists to save
		imc_list, imax_list, wc_list, was_list, c0_list, c1_list = None, None, None, None, None, None
		if (self.im_center is not None) and (self.im_axis is not None):
			imc_list = self.im_center.clone().detach().cpu().numpy().tolist()
			imax_list = self.im_axis.clone().detach().cpu().numpy().tolist()
		if (self.world_center is not None) and (self.world_axis is not None):
			wc_list = self.world_center.clone().detach().cpu().numpy().tolist()
			was_list = self.world_axis.clone().detach().cpu().numpy().tolist()
		if (self.c0 is not None) and (self.c1 is not None):
			c0_list = self.c0.clone().detach().cpu().numpy().tolist()
			c1_list = self.c1.clone().detach().cpu().numpy().tolist()

		grasp_data = {
			"depth": self.depth,
			"im_center": imc_list,
			"im_axis": imax_list,
			"im_angle": self.im_angle,
			"world_center": wc_list,
			"world_axis": was_list,
			"c0": c0_list,
			"c1": c1_list
		}

		if self.fc_quality and self.rfc_quality:
			grasp_data["quality"] = [self.fc_quality, self.rfc_quality]
		else:
			grasp_data["quality"] = None

		with open(fname, "w") as f:
			json.dump(grasp_data, f, indent=4)

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

		if (self.world_center == None) or (self.world_axis == None):
			logger.error("Grasp does not have world points to transform to image points")
			return None

		# convert grasp center (3D point) to camera space
		world_points = torch.stack((self.world_center, self.world_center))
		im_points = camera.transform_points(world_points)
		for i in range(im_points.shape[0]):		# fix depth value
			im_points[i][2] = 1/im_points[i][2]
		self.im_center = im_points[0]
		self.depth = self.im_center[2].item()

		# convert grasp axis (direction vector) to camera space
		R = camera.get_world_to_view_transform().get_matrix()[:, :3, :3]	# rotation matrix
		im_axis = torch.matmul(R, self.world_axis.unsqueeze(-1)).squeeze(-1).squeeze()	# apply only rotation, not translation
		self.im_axis = im_axis

		# calculate angle between im_axis and camera x-axis for im_angle
		x_axis = torch.tensor([-1.0, 0.0, 0.0]).to(camera.device)
		dotp = torch.dot(self.im_axis, x_axis)
		axis_norm = torch.linalg.vector_norm(self.im_axis)
		self.im_angle = torch.acos(dotp / axis_norm).item()

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
		#	results returns list of lists of form [[int: index, Bool: force closure quality, float: rfc quality, (array, array): contact points], ...]
		logger.info("%d successful grasps returned", len(results))

		cands = cands + results

	# transform successful grasps to image space
	ret_grasps = []
	for i in range(len(cands)):
		g = cands[i]
		quality = g[1:-1]
		contact0 = torch.tensor(g[-1][0]).to(renderer.device)
		contact1 = torch.tensor(g[-1][1]).to(renderer.device)
		world_center = world_centers[g[0]]
		world_axis = world_axes[g[0]]

		if save_grasp and i<num_samples:
			# save object to visualize grasp
			logger.debug("saving new grasp visualization object")
			f_name = save_grasp + "/grasp_" + str(i) +".obj"
			renderer.grasp_sphere((contact0, contact1), mesh, f_name)

		grasp = Grasp(depth=world_center[:-1], world_center=world_center, world_axis=world_axis, c0=contact0, c1=contact1, quality=quality)
		grasp.trans_world_to_im(renderer.camera)

		logger.info("image grasp: %s", str(grasp))
		print("\tcontact0:", grasp.c0)
		print("\tcontact1:", grasp.c1)

		ret_grasps.append(grasp)

	return ret_grasps[:num_samples]


def extract_tensors(d_im, grasp, logger):
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
		torch_dim = torch.tensor(d_im, dtype=torch.float32).permute(2, 0, 1).to(device)
	else:
		torch_dim = d_im

	# construct pose tensor from grasp depth
	pose_tensor = torch.zeros([1, 1])
	pose_tensor = pose_tensor.to(torch_dim.device)
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
	
	translate = ((grasp.im_center[0] - dim_cx) / 3, (grasp.im_center[1]- dim_cy) / 3)

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

def test_select_grasp(logger):
	model = KitModel("weights.npy")
	model.eval()
	run1 = Attack(model=model)

	# TESTING TENSOR EXTRACTION AND VISUALIZATION
	renderer1 = Renderer()
	depth0 = np.load("/home/hmitchell/pytorch3d/dex_shared_dir/depth_0.npy")
	grasp = Grasp(
		depth=0.607433762324266, 
		im_center=(416, 286), 
		im_angle=-2.896613990462929, 
	)
	pose, image = extract_tensors(depth0, grasp, logger)	# tensor extraction
	print("prediction:", run1.run(pose, image))		# gqcnn prediction
	renderer1.display(image)

	# TESTING GRASP SAMPLING
	mesh, image = renderer1.render_object("data/bar_clamp.obj", display=False, title="imported renderer")
	d_im = renderer1.mesh_to_depth_im(mesh, display=True)

	grasps = sample_grasps("data/bar_clamp.obj", 1, renderer=renderer1, save_grasp="vis_grasps")
	for i in range(len(grasps)):
		grasp = grasps[i]
		qual = grasp.rfc_quality
		pose, image = extract_tensors(d_im, grasp, logger)
		prediction = run1.run(pose, image)
		t = "id: " + str(i) + " prediction:" + str(prediction) + "\n" + grasp.title_str()
		renderer1.display(image, title=t)

	return "success"

def test_save_and_load_grasps(logger):

	renderer1 = Renderer()

	fg = {
		'fc_q': 1,
		'rfc_q': 0.00039880830039262474,
        # image center: tensor([344.3808, 239.4164,   0.5824], device='cuda:0')
        # image angle: 0.0
		'depth': 0.5824155807495117,
		'world_center': torch.tensor([ 2.7602e-02,  1.7584e-02, -9.2734e-05], device='cuda:0'),
		'world_axis': torch.tensor([-0.9385,  0.2661, -0.2201], device='cuda:0'),
		'c0': torch.tensor([0.0441, 0.0129, 0.0038], device='cuda:0'),
		'c1': torch.tensor([ 0.0112,  0.0222, -0.0039], device='cuda:0')
	}

	# DEBUGGING WORLD TO IMAGE AXIS TRANSFORMATION W fg2
	grasp = Grasp(
		quality=(fg['fc_q'], fg['rfc_q']), 
		depth=fg['depth'], 
		world_center=fg['world_center'], 
		world_axis=fg['world_axis'], 
		c0=fg['c0'], 
		c1=fg['c1'])

	print("before saving...")
	print(grasp)
	grasp.save("test-grasp.json")
	print("\nreading after saving...")
	g2 = Grasp.read("test-grasp.json")
	print(g2)
	
	grasp.trans_world_to_im(renderer1.camera)
	print("\nafter transforming before saving...")
	print(grasp)
	grasp.save('test-grasp2.json')
	print("\nreading after transforming and saving...")
	g2 = Grasp.read("test-grasp2.json")
	print(g2)

	return "success"
	
if __name__ == "__main__":

	# SET UP LOGGING
	logger = logging.getLogger('select_grasp')
	logger.setLevel(logging.INFO)
	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	ch.setFormatter(formatter)
	logger.addHandler(ch)

	# print(test_select_grasp(logger))
	print(test_save_and_load_grasps(logger))

	"""
	renderer1 = Renderer()
	mesh, image = renderer1.render_object("data/bar_clamp.obj", display=False, title="imported renderer")
	d_im = renderer1.mesh_to_depth_im(mesh, display=False)

	model = KitModel("weights.npy")
	model.eval()
	run1 = Attack(model=model)
	
	# FIXED GRASP FOR TESTING
	fixed_grasp = {
		'force_closure_q': 1,
		'robust_ferrari_canny_q': 0.00017405830492609492,
		'world_center': torch.tensor([ 0.0157,  0.0167, -0.0599], device='cuda:0'), 
		'world_axis': torch.tensor([ 0.6087, -0.0807, -0.7893], device='cuda:0'), 
		'im_center': torch.tensor([333.6413, 185.6289,   0.5833], device='cuda:0'), 
		'im_axis': torch.tensor([ 7.8896e+02, -3.6920e+02,  6.8074e-01], device='cuda:0'), 
		'im_angle': torch.tensor(2.7039, device='cuda:0'),
		'depth': 0.5792605876922607
	}

	fg2 = {
		'fc_q': 1,
		'rfc_q': 0.00039880830039262474,
        # image center: tensor([344.3808, 239.4164,   0.5824], device='cuda:0')
        # image angle: 0.0
		'depth': 0.5824155807495117,
		'world_center': torch.tensor([ 2.7602e-02,  1.7584e-02, -9.2734e-05], device='cuda:0'),
		'world_axis': torch.tensor([-0.9385,  0.2661, -0.2201], device='cuda:0'),
		'c0': torch.tensor([0.0441, 0.0129, 0.0038], device='cuda:0'),
		'c1': torch.tensor([ 0.0112,  0.0222, -0.0039], device='cuda:0')
	}

	# DEBUGGING WORLD TO IMAGE AXIS TRANSFORMATION W fg2
	grasp = Grasp(
		quality=(fg2['fc_q'], fg2['rfc_q']), 
		depth=fg2['depth'], 
		world_center=fg2['world_center'], 
		world_axis=fg2['world_axis'], 
		c0=fg2['c0'], 
		c1=fg2['c1'])

	grasp.trans_world_to_im(renderer1.camera)
	# renderer1.grasp_sphere((grasp.c0, grasp.c1), mesh, "vis_grasps/axis_test.obj")
	pose, image = extract_tensors(d_im, grasp, logger)
	renderer1.display(image, title="axis_test")
	model_out = model(pose, image)[0][0].item()
	print("model prediction:", model_out)
	"""

	"""
	# DEBUGGING WORLD TO IMAGE COORD TRANSFORMATION
	world_points = torch.stack((fixed_grasp["world_center"], fixed_grasp["world_axis"]))
	world_points = torch.tensor([[0.0, 0.0, 0.0], [-0.05, -0.05, -0.05]], device = renderer1.device)
	grasp = Grasp(depth=0, world_center=world_points[1], world_axis=world_points[0])
	image_grasp = grasp.trans_world_to_im(renderer1.camera)

	print("object bboxes:\n", mesh.get_bounding_boxes())
	print("\ninput points:\n", world_points)
	print("\npytorch camera project points:")
	print(renderer1.camera.transform_points(world_points))
	print("\noutput points:\n", grasp.im_center, "\n", grasp.im_axis, "\n")
	print(renderer1.camera.get_world_to_view_transform().get_matrix())

	pose, image = extract_tensors(d_im, grasp, logger)
	renderer1.display(image)

	# Find max and min vertices for each axis
	verts = mesh.verts_list()[0]
	max_index_x = torch.argmax(verts[:, 0])	# index of max value on x-axis
	max_vertex_x = verts[max_index_x] 		# vertex w max value on x-axis
	min_index_x = torch.argmin(verts[:, 0])	# index of min value on x-axis
	min_vertex_x = verts[min_index_x] 		# vertex w the min value on x-axis
	# same for y-axis
	max_index_y = torch.argmax(verts[:, 1])	# index of max value on x-axis
	max_vertex_y = verts[max_index_y]		# vertex w max value on x-axis
	min_index_y = torch.argmin(verts[:, 1])	# index of min value on x-axis
	min_vertex_y = verts[min_index_y]		# vertex w the min value on x-axis
	# same for z-axis
	max_index_z = torch.argmax(verts[:, 2])	# index of max value on x-axis
	max_vertex_z = verts[max_index_z]		# vertex w max value on x-axis
	min_index_z = torch.argmin(verts[:, 2])	# index of min value on x-axis
	min_vertex_z = verts[min_index_z]		# vertex w the min value on x-axis

	min_maxes = [min_vertex_x, max_vertex_x, min_vertex_y, max_vertex_y, min_vertex_z, max_vertex_z]
	labels = ["min_x", "max_x", "min_y", "max_y", "min_z", "max_z"]
	axis = torch.tensor([-1.0, 0.0, 0.0]).to(renderer1.camera.device)

	# transform points, and visualize points
	for label, mm in zip(labels, min_maxes):
		if label[-1] == "x":
			mm[1] = 0.0
			mm[2] = 0.0
		elif label[-1] == "y":
			mm[0] = 0.0
			mm[2] = 0.0
		elif label[-1] == "z":
			mm[0] = 0.0
			mm[1] = 0.0
		else:
			print("PROBLEM W LABEL!!")
			break

		grasp = Grasp(depth=mm[2].item(), world_center = mm, world_axis=axis)
		grasp.trans_world_to_im(renderer1.camera)
		grasp.im_angle = 0.0
		print("\n\n", label)
		print("original point:", mm)
		print("transformed point:", grasp.im_center)
		# cam_trans = renderer1.camera.transform_points(torch.stack((mm, mm)))
		# print("camera transform:", cam_trans[0])

		# view processed depth image
		_, image = extract_tensors(d_im, grasp, logger)
		renderer1.display(image, title=label)

		# # save new grasp object for visualization
		# fname = "vis_grasps/" + label + "2.obj"
		# renderer1.grasp_sphere(mm, mesh, fname)
		# print("saved new grasp object to ./" + fname)
	

	# world_points = torch.stack((min_vertex_x, max_vertex_x, min_vertex_y, max_vertex_y, min_vertex_z, max_vertex_z))
	# print("\nmin/max stacked:\n", world_points)
	# print("\nstacked -> transformed:\n", renderer1.camera.transform_points(world_points))
	# print("\nstacked -> transformed:\n", )
	"""

	"""
	# TESTING SAMPLE GRASPS METHOD AND VISUALIZING
	grasps = sample_grasps("data/bar_clamp.obj", 1, renderer=renderer1, save_grasp="")	#"vis_grasps")
	# VISUALIZE SAMPLED GRASPS
	for i in range(len(grasps)):
		grasp = grasps[i]
		qual = grasp.rfc_quality
		pose, image = extract_tensors(d_im, grasp, logger)
		prediction = run1.run(pose, image)
		t = "id: " + str(i) + " prediction:" + str(prediction) + "\n" + grasp.title_str()
		renderer1.display(image, title=t)
	"""
