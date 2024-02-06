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

	# SET UP LOGGING
	logger = logging.getLogger('select_grasp')
	logger.setLevel(logging.DEBUG)
	if not logger.handlers:
		ch = logging.StreamHandler()
		ch.setLevel(logging.DEBUG)
		formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		ch.setFormatter(formatter)
		logger.addHandler(ch)

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
			dictionary["world_center"] = torch.from_numpy(np.array(dictionary["world_center"])).to(device).float()
			dictionary["world_axis"] = torch.from_numpy(np.array(dictionary["world_axis"])).to(device).float()
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
		depth = self.depth
		if isinstance(self.depth, torch.Tensor):
			depth = self.depth.item()
		angle = self.im_angle
		if isinstance(self.im_angle, torch.Tensor):
			angle = self.im_angle.item()

		grasp_data = {
			"depth": depth,
			"im_center": imc_list,
			"im_axis": imax_list,
			"im_angle": angle,
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
			Grasp.logger.error("Grasp does not have world points to transform to image points")
			return None

		# convert grasp center (3D point) to 2D camera space
		world_points = torch.stack((self.world_center, self.world_center))
		im_points = camera.transform_points(world_points)
		for i in range(im_points.shape[0]):		# fix depth value
			im_points[i][2] = 1/im_points[i][2]
		self.im_center = im_points[0][:-1]
		self.depth = im_points[0][2]

		# convert grasp axis (direction vector) to 2D camera space
		R = camera.get_world_to_view_transform().get_matrix()[:, :3, :3]	# rotation matrix
		axis = torch.matmul(R, self.world_axis.unsqueeze(-1)).squeeze(-1).squeeze()	# apply only rotation, not translation
		axis[2] = 0
		axis = torch.nn.functional.normalize(axis, dim=-1)		# normalize axis without z-coord
		self.im_axis = axis[:-1]

		# convert normalized grasp axis back to world coordinates
		axis = camera.get_world_to_view_transform().inverse().transform_normals(axis.unsqueeze(0))
		self.world_axis = axis

		# calculate angle of im_axis
		self.im_angle = torch.atan2(self.im_axis[1], self.im_axis[0])

	@classmethod
	def sample_grasps(cls, obj_f, num_samples, renderer, oracle_method="dexnet", **kwargs): # min_qual=0.002, max_qual=1.0, save_grasp="", oracle_method="dexnet"):
		"""
		Samples grasps using oracle depending on oracle_method arg and returns grasps with qualities between min_qual and max_qual inclusive. 
		Saves new grasp object with an added sphere to visualize grasp if desired (default: does not save new grasp object).

		Parameters
		----------
		obj_f: String
			path to the .obj file of the mesh to be grasped
		num_samples: int
			Total number of grasps to evaluate (lower bound for number of grasps returned)
		renderer: Renderer object
			Renderer to use to render mesh object with camera to convert grasp info to image space
		
		* optional kwargs:
		min_qual: float
			Float between 0.0 and 1.0, minimum ferrari canny quality for returned grasps, defaults to 0.0005
				(0.002 according to dexnet robust ferrari canny method)
			NOTE: Currently returns a grasp if force_closure metric returns 1 or robust ferrari canny quality is above min_qual
		max_qual: float
			Float between 0.0 and 1.0, maximum ferrari canny quality for returned grasps, defaults to 1.0
		save_grasp: String
			If empty string, no new grasp objects are saved
			If not empty, this is the path to the directory where to save new grasp objects that have a sphere added to visualize the grasp
		oracle_method: String in ["dexnet", "pytorch"]
			"dexnet": use dexnet docker implementation of oracle
			"pytorch": use local implementation of oracle

		Returns
		-------
		List of Grasp objects with quality between min_qual and max_qual inclusive in descending order of rfc_quality
		"""

		if oracle_method not in ["dexnet", "pytorch"]:
			cls.logger.error("Grasp.sample_grasps argument `oracle_method` must be 'dexnet' or 'pytorch' -- defaulting to 'dexnet'.")
			oracle_method = "dexnet"

		if oracle_method == "dexnet":
			return cls.sample_grasps_dexnet(obj_f=obj_f, num_samples=num_samples, renderer=renderer, **kwargs)

		else:
			return cls.sample_grasps_pytorch(obj_f=obj_f, num_samples=num_samples, renderer=renderer, **kwargs)

	@classmethod
	def sample_grasps_pytorch(cls, obj_f, num_samples, renderer, **kwargs):
		"""
		Helper method for sample_grasps: samples grasps using local pytorch oracle implementation.

		Refer to `sample_grasps` method documentation for details on parameters and return values.
		"""

		cls.logger.error("Grasp.sample_grasps_pytorch not yet implemented.")
		return []

		# process kwargs
		keys = kwargs.keys()
		if "min_qual" not in keys:
			kwargs["min_qual"] = 0.002
		if "max_qual" not in keys:
			kwargs["max_qual"] = 1.0
		if "save_grasp" not in keys:
			kwargs["save_grasp"] = ""

	@classmethod
	def sample_grasps_dexnet(cls, obj_f, num_samples, renderer, **kwargs):
		"""
		Helper method for sample_grasps: samples grasps using dexnet oracle using RPC to oracle.py in running docker container.

		Refer to `sample_grasps` method documentation for details on parameters and return values.
		"""

		# process kwargs
		keys = kwargs.keys()
		if "min_qual" not in keys:
			kwargs["min_qual"] = 0.0005
		if "max_qual" not in keys:
			kwargs["max_qual"] = 1.0
		if "save_grasp" not in keys:
			kwargs["save_grasp"] = ""

		# test logging
		cls.logger.info("Sampling grasps...")
		cands = []

		# renderer mesh
		mesh, _ = renderer.render_object(obj_f, display=False)

		it = 0	# track iterations
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

			if it == 0:
				world_centers_batch = torch.clone(world_centers).unsqueeze(0)
				world_axes_batch = torch.clone(world_axes).unsqueeze(0)
			else:
				world_centers_add = torch.clone(world_centers)
				world_axes_add = torch.clone(world_axes)

				# match sizes for batches of world centers and axes
				diff = world_centers_batch.shape[1] - world_centers_add.shape[0]
				if diff > 0:
					world_centers_add = torch.cat((world_centers_add, torch.zeros(diff, 3, device=device)), 0)
					world_axes_add = torch.cat((world_axes_add, torch.zeros(diff, 3, device=device)), 0)
				elif diff < 0:
					world_centers_batch = torch.cat((world_centers_batch, torch.zeros(world_centers_batch.shape[0], -1*diff, 3, device=device)), 1)
					world_axes_batch = torch.cat((world_axes_batch, torch.zeros(world_axes_batch.shape[0], -1*diff, 3, device=device)), 1)

				world_centers_batch = torch.cat((world_centers_batch, world_centers_add.unsqueeze(0)), 0)
				world_axes_batch = torch.cat((world_axes_batch, world_axes_add.unsqueeze(0)), 0)

			# get close_fingers result and quality from gqcnn
			cls.logger.info("sending %d grasps to server", world_centers.shape[0])

			Pyro4.config.COMMTIMEOUT = None
			server = Pyro4.Proxy("PYRO:Server@localhost:5000")
			save_nparr(world_centers.detach().cpu().numpy(), "temp_centers.npy")
			save_nparr(world_axes.detach().cpu().numpy(), "temp_axes.npy")
			results = server.close_fingers("temp_centers.npy", "temp_axes.npy", kwargs["min_qual"], kwargs["max_qual"])
			#	results returns list of lists of form [[int: index, Bool: force closure quality, float: rfc quality, (array, array): contact points], ...]
			for res in results:
				res.append(it)	# track which batch/iteration grasp is from (for access to world centers/axes)
			cls.logger.info("%d successful grasps returned", len(results))

			cands = cands + results
			if len(results) > 0:
				it += 1

		# transform successful grasps to image space
		ret_grasps = []
		for i in range(len(cands)):
			g = cands[i]
			quality = g[1:-2]

			if quality[1] <= kwargs["max_qual"] and quality[1] >= kwargs["min_qual"]:
				contact0 = torch.tensor(g[-2][0]).to(renderer.device)
				contact1 = torch.tensor(g[-2][1]).to(renderer.device)
				world_center = world_centers_batch[g[-1]][g[0]]
				world_axis = world_axes_batch[g[-1]][g[0]]

				if kwargs["save_grasp"] and i<num_samples:
					# save object to visualize grasp
					cls.logger.debug("saving new grasp visualization object")
					f_name = kwargs["save_grasp"] + "/grasp_" + str(i) +".obj"
					renderer.grasp_sphere((contact0, contact1), mesh, f_name)

				grasp = Grasp(world_center=world_center, world_axis=world_axis, c0=contact0, c1=contact1, quality=quality)
				grasp.trans_world_to_im(renderer.camera)

				ret_grasps.append(grasp)

		# sort grasps in descending order of quality
		ret_grasps.sort(key=lambda x: x.rfc_quality, reverse=True)

		return ret_grasps[:num_samples]

	def extract_tensors(self, d_im):
		"""
		Use grasp information and depth image to get image and pose tensors in form of GQCNN input
		Parameters
		----------
		d_im: numpy.ndarray
			Numpy array depth image of object being grasped
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

		# check if grasp is 2D
		if (self.depth==None or self.im_center==None or self.im_angle==None):
			Grasp.logger.error("Grasp is not in 2D, must convert with camera intrinsics before tensor extraction.")
			return None, None

		# construct pose tensor from grasp depth
		pose_tensor = torch.zeros([1, 1])
		pose_tensor = pose_tensor.to(torch_dim.device)
		pose_tensor[0] = self.depth

		# process depth image wrt grasp (steps 1-3) 
		
		# 1 - resize image tensor
		out_shape = torch.tensor([torch_dim.shape], dtype=torch.float32)
		out_shape *= (1/3)		# using 1/3 based on gqcnn library - may need to change depending on input
		out_shape = tuple(out_shape.type(torch.int)[0][1:].numpy())

		torch_transform = transforms.Resize(out_shape, antialias=False) 
		torch_image_tensor = torch_transform(torch_dim)


		# 2 - translate wrt to grasp angle and grasp center 
		theta = math.degrees(self.im_angle)
		 
		dim_cx = torch_dim.shape[2] // 2
		dim_cy = torch_dim.shape[1] // 2	
		
		translate = ((self.im_center[0] - dim_cx) / 3, (self.im_center[1]- dim_cy) / 3)

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

		# 3 - crop image to size (32, 32)
		torch_cropped = transforms.functional.crop(torch_translated, cy-17, cx-17, 32, 32)
		image_tensor = torch_cropped.unsqueeze(0)

		return pose_tensor, image_tensor 

	def oracle_eval(self, obj_file, oracle_method="dexnet", robust=True):
		"""
		Get a final oracle evalution of a mesh object according to oracle_method

		Parameters
		----------
		obj_fil: String
			The path to the .obj file of the mesh to evaluate
		oracle_method: String
			Options: "dexnet" or "pytorch"; defaults to "dexnet"
			Indicates to use the dexnet oracle via remote call to docker, or local pytorch implementation
		robust: Boolean
			True: uses robust ferrari canny evaluation for oracle quality; default
			False: uses (non robust) ferrari canny evaluation for oracle quality

		Returns
		-------
		float: quality from ferarri canny evaluation

		"""
		if oracle_method == "dexnet":
			return self.oracle_eval_dexnet(obj_file, robust=robust)
		
		elif self.oracle_method == "pytorch":
			return self.oracle_eval_pytorch(obj_file)

		else:
			Grasp.logger.error("oracle evaluation method must be 'dexnet' or 'pytorch'.")

	def oracle_eval_dexnet(self, obj_file, robust=True):
		"""
		Get a final oracle evaluation of a mesh object via remote call to docker container.

		Refer to `oracle_eval` method documentation for details on parameters and return values.
		"""

		# check if object file is already saved in shared directory and copy there if not
		if not os.path.isfile(obj_file):
			Grasp.logger.error("Object for oracle evaluation does not exist.")
			return None

		obj_name = obj_file.split("/")[-1]
		obj_shared = SHARED_DIR + "/" + obj_name
		if not os.path.isfile(obj_shared):
			Grasp.logger.info("Saving object file to shared directory (%s) for oracle evaluation", SHARED_DIR)
			shutil.copyfile(obj_file, obj_shared)

		# save grasp info for oracle evaluation
		Pyro4.config.COMMTIMEOUT = None
		server = Pyro4.Proxy("PYRO:Server@localhost:5000")
		calc_axis = (self.c1 - self.c0) / torch.linalg.norm((self.c1 - self.c0))
		save_nparr(self.world_center.detach().cpu().numpy(), "temp_center.npy")
		save_nparr(calc_axis.detach().cpu().numpy(), "temp_axis.npy")
		results = server.final_eval("temp_center.npy", "temp_axis.npy", obj_name, robust=robust)

		return results

	def oracle_eval_pytorch(self, obj_file):
		"""
		Get a final oracle evaluation of a mesh object via local pytorch oracle implementation.

		Refer to `oracle_eval` method documentation for details on parameters and return values.
		"""

		Grasp.logger.error("Grasp.oracle_eval_pytorch not yet implemented.")
		return None

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

def test_select_grasp():
	Grasp.logger.info("Running test_select_grasp...")

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
	pose, image = grasp.extract_tensors(depth0)	# tensor extraction
	print("prediction:", run1.run(pose, image))		# gqcnn prediction
	renderer1.display(image)

	# TESTING GRASP SAMPLING
	mesh, image = renderer1.render_object("data/bar_clamp.obj", display=False, title="imported renderer")
	d_im = renderer1.mesh_to_depth_im(mesh, display=True)

	grasps = Grasp.sample_grasps("data/bar_clamp.obj", 1, renderer=renderer1, save_grasp="vis_grasps")
	for i in range(len(grasps)):
		grasp = grasps[i]
		qual = grasp.rfc_quality
		pose, image = grasp.extract_tensors(d_im)
		prediction = run1.run(pose, image)
		t = "id: " + str(i) + " prediction:" + str(prediction) + "\n" + grasp.title_str()
		renderer1.display(image, title=t)

	Grasp.logger.info("Finished running test_select_grasp.")

def test_save_and_load_grasps():
	Grasp.logger.info("Running test_save_and_load_grasps...")

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

	# test model inference and trans_world_to_im on saved grasps
	model = KitModel("weights.npy")
	model.eval()
	run1 = Attack(model=model)

	mesh, im = renderer1.render_object("data/bar_clamp.obj", display=False)
	d_im = renderer1.mesh_to_depth_im(mesh, display=False)

	g2.trans_world_to_im(renderer1.camera)
	pose, image = g2.extract_tensors(d_im)
	print("prediction:", run1.run(pose, image))		# gqcnn prediction

	Grasp.logger.info("Finished running test_save_and_load_grasps.")
	
def test_trans_world_to_im():
	Grasp.logger.info("Running test_trans_world_to_im...")

	r = Renderer()
	mesh, _ = r.render_object("data/new_barclamp.obj", display=False)
	dim = r.mesh_to_depth_im(mesh, display=False)

	model = KitModel("weights.npy")
	model.eval()
	run1 = Attack(model=model)

	grasp = Grasp.read("experiment-results/ex00/grasp.json")
	grasp.trans_world_to_im(camera=r.camera)
	r.grasp_sphere((grasp.c0, grasp.c1), mesh, "vis_grasps/test_trans_world_to_im.obj")
	pose, image = grasp.extract_tensors(dim)

	model_out = run1.run(pose, image)[0][0].item()
	r.display(image, title="processed dim\npred: "+str(model_out))

	Grasp.logger.info("Finished running test_trans_world_to_im.")

if __name__ == "__main__":

	# test_trans_world_to_im()
	# test_select_grasp()
	test_save_and_load_grasps()

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
	pose, image = grasp.extract_tensors(d_im)
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

	pose, image = grasp.extract_tensors(d_im)
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
		_, image = grasp.extract_tensors(d_im)
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
	grasps = Grasp.sample_grasps("data/bar_clamp.obj", 1, renderer=renderer1, save_grasp="")	#"vis_grasps")
	# VISUALIZE SAMPLED GRASPS
	for i in range(len(grasps)):
		grasp = grasps[i]
		qual = grasp.rfc_quality
		pose, image = grasp.extract_tensors(d_im)
		prediction = run1.run(pose, image)
		t = "id: " + str(i) + " prediction:" + str(prediction) + "\n" + grasp.title_str()
		renderer1.display(image, title=t)
	"""
