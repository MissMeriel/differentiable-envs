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

from quality_fork import *

SHARED_DIR = "/home/hmitchell/pytorch3d/dex_shared_dir"
EPS = 0.0000001

if torch.cuda.is_available():
	device = torch.device("cuda:0")
	torch.cuda.set_device(device)
else:
	print("cuda not available")
	device = torch.device("cpu")

class Grasp:

	# SET UP LOGGING
	logger = logging.getLogger('select_grasp')
	logger.setLevel(logging.INFO)
	if not logger.handlers:
		ch = logging.StreamHandler()
		ch.setLevel(logging.INFO)
		formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		ch.setFormatter(formatter)
		logger.addHandler(ch)

	def __init__(self, depth=None, im_center=None, im_angle=None, im_axis=None, world_center=None, world_axis=None, c0=None, c1=None, quality=None, prediction=None, oracle_method="dexnet", oracle_robust=True):
		"""
		Initialize a Grasp object
		Paramters
		---------
		depth: float or list or torch.tensor of size [batch, 1]
			Depth of grasp
		im_center: list or torch.tensor of size [batch, 2]				# or [batch_camera, batch_grasp, 2]
			Grasp center in image coordinates
		im_angle: float or list or torch.tensor of size [batch, 1]		# why not [batch_camera, batch_grasp, 1]?
			Angle between grasp axis and camera x-axis; used for tensor extraction
		im_axis: list or torch.tensor of size [batch, 2]				# or [batch_camera, batch_grasp, 1]
			Grasp axis in image coordinates
		world_center: list or torch.tensor of size [batch, 3]
			Grasp center in world/mesh coordinates
		world_axis: list or torch.tensor of size [batch, 3]				# or [batch_camera, batch_grasp, 3]
			Grasp axis in world/mesh coordinates
		c0: list or torch.tensor of size [batch, 3]
			First contact point of grasp in world coordinates
		c1: list or torch.tensor of size [batch, 3]
			Second contact point of grasp in world coordinates
		quality: float or list or torch.tensor of size [batch, 1]
			quality value from robust_ferrari_canny
		oracle_method: String
			Options: "dexnet" (default) or "pytorch"
			"dexnet": Oracle evaluation done via remote call to dex-net oracle on docker container
			"pytorch": Oracle evaluation done via local pytorch implementaiton
		oracle_robust: Boolean
			True: robust ferrari canny oracle evaluation (default)
			False: static ferrari canny oracle evaluation 

		Returns
		-------
		None
		"""

		self.im_center = im_center
		if isinstance(im_center, list):
			self.im_center = torch.from_numpy(np.array(im_center)).to(device).float()
		if self.im_center != None and self.im_center.dim() == 1:
			self.im_center = self.im_center.unsqueeze(0)

		self.im_axis = im_axis
		if isinstance(im_axis, list):
			self.im_axis = torch.from_numpy(np.array(im_axis)).to(device).float()
		if self.im_axis != None and self.im_axis.dim() == 1:
			self.im_axis = self.im_axis.unsqueeze(0)

		self.world_center = world_center
		if isinstance(world_center, list):
			self.world_center = torch.from_numpy(np.array(world_center)).to(device).float()
		if self.world_center != None and self.world_center.dim() == 1:
			self.world_center = self.world_center.unsqueeze(0)

		self.world_axis = world_axis
		if isinstance(world_axis, list):
			self.world_axis = torch.from_numpy(np.array(world_axis)).to(device).float()
		if self.world_axis != None and self.world_axis.dim() == 1:
			self.world_axis = self.world_axis.unsqueeze(0)

		self.c0 = c0
		if isinstance(c0, list):
			self.c0 = torch.from_numpy(np.array(c0)).to(device).float()
		if self.c0 != None and self.c0.dim() == 1:
			self.c0 = self.c0.unsqueeze(0)

		self.c1 = c1
		if isinstance(c1, list):
			self.c1 = torch.from_numpy(np.array(c1)).to(device).float()
		if self.c1 != None and self.c1.dim() == 1:
			self.c1 = self.c1.unsqueeze(0)

		self.depth = depth
		if isinstance(depth, float):
			self.depth = torch.tensor([depth]).to(device).float().unsqueeze()
		elif isinstance(depth, list):
			self.depth = torch.from_numpy(np.array(depth)).to(device).float()
		if self.depth != None and self.depth.dim() == 1:
			self.depth = self.depth.unsqueeze(0)

		self.im_angle = im_angle
		if isinstance(im_angle, float):
			self.im_angle = torch.tensor([im_angle]).to(device).float().unsqueeze()
		elif isinstance(im_angle, list):
			self.im_angle = torch.from_numpy(np.array(im_angle)).to(device).float()
		if self.im_angle != None and self.im_angle.dim() == 1:
			self.im_angle = self.im_angle.unsqueeze(0)

		self.quality = quality
		if isinstance(quality, float):
			self.quality = torch.tensor([quality]).to(device).float().unsqueeze()
		elif isinstance(quality, list):
			self.quality = torch.from_numpy(np.array(quality)).to(device).float()
		if self.im_axis != None and self.im_axis.dim() == 1:
			self.im_axis = self.im_axis.unsqueeze(0)

		self.prediction = prediction
		if isinstance(prediction, float):
			self.prediction = torch.tensor([prediction]).to(device).float().unsqueeze()
		elif isinstance(prediction, list):
			self.prediction = torch.from_numpy(np.array(quality)).to(device).float()
		if self.prediction != None and self.prediction.dim() == 1:
			self.prediction = self.prediction.unsqueeze(0)

		self.oracle_method = oracle_method
		if oracle_method not in ["dexnet", "pytorch"]:
			self.oracle_method = "dexnet"

		self.oracle_robust = oracle_robust
		if isinstance(oracle_robust, bool):
			self.oracle_robust = oracle_robust

	@classmethod
	def init_from_dict(cls, dict):
		"""Initialize grasp object from a dictionary"""
		init_keys = ["depth", "im_center", "im_angle", "im_axis", "world_center", "world_axis", "c0", "c1", "quality"]
		dict_keys = dict.keys()
		for key in init_keys:
			if key not in dict_keys:
				dict[key] = None
		if "oracle_method" not in dict_keys:
			dict["oracle_method"] = "dexnet"
		if "oracle_robust" not in dict_keys:
			dict["oracle_robust"] = True
		if "prediction" not in dict_keys:
			dict["prediction"] = None

		return Grasp(depth=dict["depth"], im_center=dict["im_center"], im_angle=dict["im_angle"], im_axis=dict["im_axis"], world_center=dict["world_center"], world_axis=dict["world_axis"], c0=dict["c0"], c1=dict["c1"], quality=dict["quality"], prediction=dict["prediction"], oracle_method=dict["oracle_method"], oracle_robust=dict["oracle_robust"])

	@classmethod
	def read(cls, fname):
		"""Reads a JSON file fname with saved grasp information and initializes"""

		# read file
		with open(fname) as f:
			dictionary = json.load(f)

		# convert lists to tensors
		if dictionary["im_center"] and dictionary["im_axis"]:
			dictionary["im_center"] = torch.from_numpy(np.array(dictionary["im_center"])).to(device).float()
			dictionary["im_axis"] = torch.from_numpy(np.array(dictionary["im_axis"])).to(device).float()

		if dictionary["depth"]:
			if np.array(dictionary["depth"]).shape == ():
				dictionary["depth"] = torch.from_numpy(np.array(dictionary["depth"])).to(device).unsqueeze(0).float()
			else:
				dictionary["depth"] = torch.from_numpy(np.array(dictionary["depth"])).to(device).float()
		
		if dictionary["world_center"] and dictionary["world_axis"]:
			dictionary["world_center"] = torch.from_numpy(np.array(dictionary["world_center"])).to(device).float()
			dictionary["world_axis"] = torch.from_numpy(np.array(dictionary["world_axis"])).to(device).float()
		
		if dictionary["c0"] and dictionary["c1"]:
			dictionary["c0"] = torch.from_numpy(np.array(dictionary["c0"])).to(device).float()
			dictionary["c1"] = torch.from_numpy(np.array(dictionary["c1"])).to(device).float()
		
		if isinstance(dictionary["quality"], list):
			# if not isinstance(dictionary["quality"][0], list):
			# 	dictionary["quality"] = torch.tensor([dictionary["quality"][1]]).to(device).float()
			# else:
			dictionary["quality"] = torch.from_numpy(np.array(dictionary["quality"])).to(device).float()

		return cls.init_from_dict(dictionary)

	@classmethod
	def read_batch(cls, fnames):

		dict_list = []
		batch_dict = {}

		for fname in fnames:

			# ensure path to json file
			if not os.path.isfile(fname):
				cls.logger.error("read_batch - %s grasp file file not found.", fname)
				continue
			if fname.split(".")[-1] != "json":
				cls.logger.error("read_batch - %s does not appear to be a json file, so cannot read grasp.", fname)
				continue
			
			with open(fname) as f:
				d = json.load(f)
			dict_list.append(d)
			
		length = len(dict_list[0])
		for d in dict_list:
			if len(d) != length:
				cls.logger.error("read_batch - batches of grasps must have the same attributes.")
				continue
			for k, v in d.items():
				batch_dict.setdefault(k, [])
				batch_dict[k].append(v)
				# if isinstance(v, list):
				# 	batch_dict[k].extend(v)
				# else:
				# 	batch_dict[k].append(v)

		return cls.init_from_dict(batch_dict)

	def __str__(self):
		"""Returns a string with grasp information in image coordinates"""
		p_str = "grasp:"
		
		if self.quality != None:
			p_str += "\n\tquality: " + str(self.quality.item())
		if self.im_center != None:
			p_str +=  "\n\timage center: " + str(self.im_center)
		if self.im_angle != None:
			p_str += "\n\timage angle: " + str(self.im_angle.item()) 
		if self.depth != None:
			p_str += "\n\tdepth: " + str(self.depth.item()) 	
		if self.world_center != None:
			p_str += "\n\tworld center: " + str(self.world_center) 
		if self.world_axis != None:
			p_str += "\n\tworld axis: " + str(self.world_axis)

		return p_str + "\n"

	def title_str(self):
		"""Retruns a string like __str__, but without tabs"""
		p_str = "quality: " + str(self.quality) + "\nimage center: " + str(self.im_center) + "\nimage angle: " + str(self.im_angle) + "\ndepth: " + str(self.depth)
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
		# if depth is not None:
		# 	depth = self.depth.clone().detach().cpu().numpy().tolist()
		depth = self.depth
		if isinstance(self.depth, torch.Tensor):
			depth = self.depth.clone().detach().cpu().numpy().tolist()
		angle = self.im_angle
		if isinstance(self.im_angle, torch.Tensor):
			angle = self.im_angle.clone().detach().cpu().numpy().tolist()
		quality = self.quality
		if isinstance(self.quality, torch.Tensor):
			quality = self.quality.clone().detach().cpu().numpy().tolist()
		prediction = self.prediction
		if isinstance(self.prediction, torch.Tensor):
			predicition = self.prediction.clone().detach().cpu().numpy().tolist()

		grasp_data = {
			"depth": depth,
			"im_center": imc_list,
			"im_axis": imax_list,
			"im_angle": angle,
			"world_center": wc_list,
			"world_axis": was_list,
			"c0": c0_list,
			"c1": c1_list, 
			"oracle_method": self.oracle_method,
			"oracle_robust": self.oracle_robust,
			"quality": quality,
			"prediction": prediction
		}

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
		im_points = camera.transform_points(self.world_center)
		for i in range(im_points.shape[0]):		# fix depth value
			im_points[i][2] = 1/im_points[i][2]
		self.im_center = im_points[..., :2]
		self.depth = im_points[..., [2]]

		# convert grasp axis (direction vector) to 2D camera space
		axis = camera.get_world_to_view_transform().transform_normals(self.world_axis)
		axis[..., 2] = 0

		axis = torch.nn.functional.normalize(axis, dim=-1)		# normalize axis without z-coord
		self.im_axis = axis[..., :2]

		# convert normalized grasp axis back to world coordinates
		self.world_axis = camera.get_world_to_view_transform().inverse().transform_normals(axis)

		# calculate angle of im_axis
		self.im_angle = torch.atan2(self.im_axis[...,1], self.im_axis[..., 0]).unsqueeze(-1)


	@classmethod
	def sample_grasps(cls, obj_f, num_samples, renderer, **kwargs): # min_qual=0.002, max_qual=1.0, save_grasp=""):
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
		sort: Boolean
			True: sort grasps in descending order of quality (default)
			False: return in order they were sampled in
		oracle_method: String in ["dexnet", "pytorch"]
			"dexnet": use docker dexnet oracle (default)
			"pytorch": use local pytorch oracle implementation
		oracle_robust: Boolean
			True: robust oracle evaluation (default)
			False: non-robust oracle evaluation

		Returns
		-------
		List of Grasp objects with quality between min_qual and max_qual inclusive in descending order of rfc_quality
		"""

		keys = kwargs.keys()
		if ("oracle_method" not in keys) or (kwargs["oracle_method"] != "pytorch"):
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
		if "sort" not in keys:
			kwargs["sort"] = True

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
		if "sort" not in keys:
			kwargs["sort"] = True
		if "oracle_robust" not in keys:
			kwargs["oracle_robust"] = True

		# logging
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
		
			# compute grasp center and axis
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
			results = server.close_fingers("temp_centers.npy", "temp_axes.npy", kwargs["min_qual"], kwargs["max_qual"], robust=kwargs["oracle_robust"])
			#	results returns list of lists of form [[int: index, float: ferrari canny quality, (array, array): contact points], ...]
			for res in results:
				res.append(it)	# track which batch/iteration grasp is from (for access to world centers/axes)
			cls.logger.info("%d successful grasps returned", len(results))

			cands = cands + results
			if len(results) > 0:
				it += 1

		# transform successful grasps to image space
		ret_grasps = []
		for i in range(len(cands)):
			if len(ret_grasps) >= num_samples:
				break

			g = cands[i]
			quality = g[2]

			# if quality <= kwargs["max_qual"] and quality >= kwargs["min_qual"]:	# server.py checks quality
			contact0 = torch.tensor(g[-2][0]).to(renderer.device)
			contact1 = torch.tensor(g[-2][1]).to(renderer.device)
			world_center = world_centers_batch[g[-1]][g[0]]
			world_axis = world_axes_batch[g[-1]][g[0]]

			grasp = Grasp(world_center=world_center, world_axis=world_axis, c0=contact0, c1=contact1, quality=quality, oracle_method="dexnet", oracle_robust=kwargs["oracle_robust"])
			grasp.trans_world_to_im(renderer.camera)

			if kwargs["save_grasp"]: 	# and i<num_samples:
				# save object to visualize grasp, grasp json, and approx. image of grasp
				if kwargs["save_grasp"][-1] == "/":
					kwargs["save_grasp"] = kwargs["save_grasp"][:-1]
				cls.logger.info("\nsaving new grasp visualization object")
				obj_name = kwargs["save_grasp"] + "/grasp_" + str(i) + ".obj"
				json_name = kwargs["save_grasp"] + "/grasp_" + str(i) + ".json"
				img_name = kwargs["save_grasp"] + "/grasp_" + str(i) + ".png"
				renderer.grasp_sphere((contact0, contact1), mesh, obj_name)
				grasp.save(json_name)
				renderer.draw_grasp(mesh, grasp.c0, grasp.c1, title=img_name, save=img_name)

			ret_grasps.append(grasp)

		if kwargs["sort"]:
			# sort grasps in descending order of quality
			ret_grasps.sort(key=lambda x: x.quality, reverse=True)

		return ret_grasps[:num_samples]

	def extract_tensors(self, d_im, debug=False):
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
		theta = -1 * math.degrees(self.im_angle[0])	# -1 because PyTorch transform goes clockwise and autolab_core goes counter-clockwise
		 
		dim_cx = torch_dim.shape[2] // 2
		dim_cy = torch_dim.shape[1] // 2
		
		translate = ((self.im_center[0][0] - dim_cx) / 3, (self.im_center[0][1]- dim_cy) / 3)

		# print("translate:", translate[0] / torch_image_tensor.shape[1], translate[1] / torch_image_tensor.shape[2])

		cx = torch_image_tensor.shape[2] // 2
		cy = torch_image_tensor.shape[1] // 2

		# keep as two separate transformations so translation is performed before rotation
		translated_only = transforms.functional.affine(
			torch_image_tensor,
			0,		# angle of rotation in degrees clockwise, between -180 and 180 inclusive
			translate,
			scale=1,	# no scale
			shear=0,	# no shear 
			interpolation=transforms.InterpolationMode.BILINEAR,
			center=(cx, cy)	
		)

		torch_rotated = transforms.functional.affine(
			translated_only,
			theta,
			translate=(0, 0),
			scale=1,
			shear=0,
			interpolation=transforms.InterpolationMode.BILINEAR,
			center=(cx, cy)
		)

		if debug:
			rotated_only = transforms.functional.affine(
				torch_image_tensor,
				theta,
				translate=(0,0),
				scale=1,
				shear=0,
				interpolation=transforms.InterpolationMode.BILINEAR,
				center=(cx, cy)
			)
			return pose_tensor, torch_image_tensor, translated_only, rotated_only, torch_rotated

		# 3 - crop image to size (32, 32)
		torch_cropped = transforms.functional.crop(torch_rotated, cy-17, cx-17, 32, 32)
		image_tensor = torch_cropped.unsqueeze(0)

		return pose_tensor, image_tensor 

	def extract_tensors_batch(self, dims, debug=False):

		r = Renderer()

		# check type of input_dim
		if isinstance(dims, np.ndarray):
			Grasp.error("extract_tensors_batch takes a tensor, not a numpy ndarray")
			return None, None
		
		# check if grasp is 2D
		if (self.depth==None or self.im_center==None or self.im_angle==None):
			Grasp.logger.error("Grasp is not in 2D, must convert with camera intrinsics before tensor extraction.")
			return None, None

		batch_size = dims.shape[0]	# dims.shape: [batch_size, 1, 480, 640]

		# construct pose tensor from grasp depth
		pose_tensor = self.depth

		# process depth image wrt grasp (steps 1-3) 
		# 1 - resize image tensors
		out_shape = torch.tensor([dims.squeeze(1).shape], dtype=torch.float32)
		out_shape *= (1/3)		# using 1/3 based on gqcnn library - may need to change depending on input
		out_shape = tuple(out_shape.type(torch.int)[0][1:].numpy())		# (160, 213)

		torch_transform = transforms.Resize(out_shape, antialias=False) 
		dims_resized = torch_transform(dims)		# shape: [batch_size, 1, 160, 213]

		# 2 - translation wrt to grasp angle and grasp center
		# 	translation matrix
		dim_cx = dims.shape[3] // 2	# 320
		dim_cy = dims.shape[2] // 2	# 240
		dim_cx_tens = torch.tensor([dim_cx]).expand(batch_size).to(device)
		dim_cy_tens = torch.tensor([dim_cy]).expand(batch_size).to(device)

		u = -1 * ((self.im_center[..., 0] - dim_cx_tens) / (dims_resized.shape[3])).float()
		v = -1 * ((self.im_center[..., 1] - dim_cy_tens) / (dims_resized.shape[2])).float()
		# 	should not be in pixels, but normalized based on image size, -1 to account for pytorch coordinates

		translate = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
		translate = translate.expand(batch_size, -1, -1).to(device).float()
		indices = torch.arange(batch_size)
		translate[indices, 0, 2] = u
		translate[indices, 1, 2] = v

		#	rotation matrix
		theta = self.im_angle.squeeze()	# no -1 for counter-clockwise, stay in radians
		cos = torch.cos(theta)
		sin = torch.sin(theta)

		rotation = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]).expand(batch_size, -1, -1).to(device).float()
		rotation[indices, 0, 0] = cos
		rotation[indices, 1, 1] = cos
		rotation[indices, 0, 1] = -1 * sin
		rotation[indices, 1, 0] = sin

		#	center and uncenter matrices
		uncenter = torch.tensor([[[1, 0, -0.5], [0, 1, 0.5], [0, 0, 1]]]).expand(batch_size, -1, -1).to(device).float()
		center = torch.tensor([[[1, 0, 0.5], [0, 1, -0.5], [0, 0, 1]]]).expand(batch_size, -1, -1).to(device).float()

		#	combine transformations and apply them
		if debug:
			trans_rot_mat = torch.matmul(translate, rotation)[:, :2, :]
			center_mat = center[:, :2, :]
			translate_mat = translate[:, :2, :]
			rotation_mat = rotation[:, :2, :]
			
			trans_grid = torch.nn.functional.affine_grid(translate_mat, dims_resized.shape)
			trans_only = torch.nn.functional.grid_sample(dims_resized, trans_grid)

			rot_grid = torch.nn.functional.affine_grid(rotation_mat, dims_resized.shape)
			rot_only = torch.nn.functional.grid_sample(dims_resized, rot_grid)

			trans_rot_grid = torch.nn.functional.affine_grid(trans_rot_mat, dims_resized.shape)
			trans_rot = torch.nn.functional.grid_sample(dims_resized, trans_rot_grid)

			center_grid = torch.nn.functional.affine_grid(center_mat, dims_resized.shape)
			centered = torch.nn.functional.grid_sample(dims_resized, center_grid)

			return pose_tensor, dims_resized, trans_only, rot_only, trans_rot, centered

		# affine_mat = torch.matmul(torch.matmul(translate, uncenter), torch.matmul(rotation, uncenter))
		affine_mat = torch.matmul(translate, rotation)
		affine_mat = affine_mat[:, :2, :]
		affine_grid = torch.nn.functional.affine_grid(affine_mat, dims_resized.shape)
		dims_transformed = torch.nn.functional.grid_sample(dims_resized, affine_grid)
		
		# print("\ncenter:\n", center[0])
		# print("\nuncenter:\n", uncenter[0])
		# print("\ntranslate:\n", translate[0])
		# print("\nrotation:\n", rotation[0])
		# print("\naffine mat:\n", affine_mat[0])
		# theta = theta[0].item()
		# print("\ncos:", math.cos(theta))
		# print("sin:", math.sin(theta))
		# u = u[0].item()
		# v = v[0].item()
		# print("u:", u)
		# print("v:", v)
		# u2 = u - math.cos(theta)/2 + math.sin(theta)/2 + 0.5
		# v2 = v - math.cos(theta)/2 - math.sin(theta)/2 + 0.5
		# print("u2:", u2)
		# print("v2:", v2)

		# r.display(dims_transformed[0].squeeze(0), title="translate & rotate only")

		# 3 - crop images to 32x32 pixels
		dims_transformed = dims_transformed[:, :, 64:96, 90:122]
		# print("dims_transformed:", dims_transformed.shape)

		return pose_tensor, dims_transformed

	def oracle_eval(self, obj_file, oracle_method=None, robust=None, renderer=None):
		"""
		Get a final oracle evalution of a mesh object according to oracle_method

		Parameters
		----------
		obj_fil: String
			The path to the .obj file of the mesh to evaluate
		oracle_method: String
			Options: "dexnet" or "pytorch"; defaults to self.oracle_method
			Indicates to use the dexnet oracle via remote call to docker, or local pytorch implementation
		robust: Boolean
			True: uses robust ferrari canny evaluation for oracle quality
			False: uses (non robust) ferrari canny evaluation for oracle quality
			defaults to self.oracle_robust

		Returns
		-------
		float: quality from ferarri canny evaluation

		"""
		check_method = (oracle_method in ["dexnet", "pytorch"])
		if oracle_method == "dexnet" or (not check_method and (self.oracle_method == "dexnet")):
			if isinstance(robust, bool):
				Grasp.logger.debug("Oracle eval - dexnet, robust %r", robust)
				return self.oracle_eval_dexnet(obj_file, robust=robust)
			else:
				Grasp.logger.debug("Oracle eval - dexnet, robust %r", self.oracle_robust)
				return self.oracle_eval_dexnet(obj_file, robust=self.oracle_robust)
		
		else:
			if not renderer:
				Grasp.logger.error("oracle_eval - pytorch oracle requires renderer argument")
			else:
				Grasp.logger.debug("Oracle eval - pytorch")
				return self.oracle_eval_pytorch(obj_file, renderer)

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

	def oracle_eval_pytorch(self, obj_file, renderer):
		"""
		Get a final oracle evaluation of a mesh object via local pytorch oracle implementation.

		Refer to `oracle_eval` method documentation for details on parameters and return values.
		"""

		if (self.world_center == None or self.world_axis == None):
			Grasp.logger.error("Grasp.oracle_eval_pytorch requires Grasp to have world_center and world_axis.")
			return None

		# convert to GraspTorch object
		width = torch.tensor([[0.05]], device=renderer.camera.device)
		center3D = self.world_center.unsqueeze(0)
		axis3D = self.world_axis.unsqueeze(0)
		gt = GraspTorch(center3D, axis3D=axis3D, width=width, camera_intr=renderer.camera)
		gt.make2D(updateCamera=False)

		# # check 2D information matches
		# print("depth:", g.depth.item() == gt.depth.squeeze(0).item())
		# print("angle diff:", g.im_angle.item() - gt.angle.item())
		# print("axis2D diff:", g.im_axis - gt.axis[0].float())
		# print("center2D:", g.im_center == gt.center[0].float())
		# print("axis3D diff:", g.world_axis - gt.axis3D[0].float())
		# print("center3D:", g.world_center == gt.center3D[0].float())

		# get ferrari canny quality
		config_dict = {
			"torque_scaling":1000,
			"soft_fingers":1,
			"friction_coef": 0.8, 
			"antipodality_pctile": 1.0 
		}
		mesh, _ = renderer.render_object(obj_file)

		com_qual_func = ComForceClosureParallelJawQualityFunction(config_dict)
		force_closure_qual = com_qual_func.quality(mesh, gt)

		com_qual_func = CannyFerrariQualityFunction(config_dict)
		return com_qual_func.quality(mesh, gt)

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

	# # TESTING TENSOR EXTRACTION AND VISUALIZATION
	renderer1 = Renderer()
	# depth0 = np.load("/home/hmitchell/pytorch3d/dex_shared_dir/depth_0.npy")
	# grasp = Grasp(
	# 	depth=0.607433762324266, 
	# 	im_center=(416, 286), 
	# 	im_angle=-2.896613990462929, 
	# )
	# pose, image = grasp.extract_tensors(depth0)	# tensor extraction
	# print("prediction:", run1.run(pose, image))		# gqcnn prediction
	# renderer1.display(image)

	# TESTING GRASP SAMPLING
	mesh, image = renderer1.render_object("data/new_barclamp.obj", display=True, title="imported renderer")
	d_im = renderer1.mesh_to_depth_im(mesh, display=True)

	grasps = Grasp.sample_grasps("data/new_barclamp.obj", 10, renderer=renderer1) #, save_grasp="example-grasps", sort=False)
	for i in range(len(grasps)):
		grasp = grasps[i]
		qual = grasp.quality
		pose, image = grasp.extract_tensors(d_im)
		prediction = run1.run(pose, image)
		print("\nprediction size:", prediction.shape, prediction)
		t = "id: " + str(i) + " prediction: " + str(prediction[0][0].item()) + "\noracle quality: " + str(qual.item()) #grasp.title_str()
		fname = "example-grasps/grasp_" + str(i) + "_pred.png"
		renderer1.display(image, title=t)	#, save=True, fname=fname)

	Grasp.logger.info("Finished running test_select_grasp.")

def test_save_and_load_grasps():
	Grasp.logger.info("Running test_save_and_load_grasps...")

	r = Renderer()
	fixed_grasp = {
		"quality": torch.tensor([0.00039880830039262474], device='cuda:0'),
		"depth": torch.tensor([0.5824155807495117], device='cuda:0'),
		'world_center': torch.tensor([ 2.7602e-02,  1.7584e-02, -9.2734e-05], device='cuda:0'),
		'world_axis': torch.tensor([-0.9385,  0.2661, -0.2201], device='cuda:0'),
		'c0': torch.tensor([0.0441, 0.0129, 0.0038], device='cuda:0'),
		'c1': torch.tensor([ 0.0112,  0.0222, -0.0039], device='cuda:0')
	}

	fixed_grasp2 = {
		"quality": 0.00039880830039262474,
		"depth": 0.5824155807495117,
		'world_center': torch.tensor([ 2.7602e-02,  1.7584e-02, -9.2734e-05], device='cuda:0'),
		'world_axis': torch.tensor([-0.9385,  0.2661, -0.2201], device='cuda:0'),
		'c0': torch.tensor([0.0441, 0.0129, 0.0038], device='cuda:0'),
		'c1': torch.tensor([ 0.0112,  0.0222, -0.0039], device='cuda:0')
	}

	fixed_grasp3 = {
		"quality": [0.00039880830039262474],
		"depth": torch.tensor([0.5824155807495117], device='cuda:0'),
		'world_center': [2.7602e-02,  1.7584e-02, -9.2734e-05],
		'world_axis': [-0.9385,  0.2661, -0.2201],
		'c0': [0.0441, 0.0129, 0.0038],
		'c1': [ 0.0112,  0.0222, -0.0039]
	}


	# Test init_from_dict with tensors vs init_from_dict with lists vs init_from_dict with floats vs read from json
	g = Grasp.init_from_dict(fixed_grasp)	# from tensors
	s1 = str(g)
	# print("\ns1:", s1)

	g2 = Grasp.init_from_dict(fixed_grasp2)	# from floats
	s2 = str(g2)
	# print("\ns2:", s2)

	g3 = Grasp.init_from_dict(fixed_grasp3)	# from lists
	s3 = str(g3)
	# print("\ns3:", s3)

	print("\nEqual when using init_from_dict with tensors vs floats vs lists?", s1 == s2 == s3)

	
	# Test with init_from_dict vs read from json
	# print("\ns1:", s1)
	g.save("test-grasp.json")
	g4 = Grasp.read("test-grasp.json")
	s4 = str(g4)
	# print("\ns4:", s4)
	print("\nEqual when using init_from_dict and read?", s1==s4, "\n")


	# Test trans_world_to_im from init_from_dict with tensors vs floats vs lists vs json
	g.trans_world_to_im(r.camera)
	s1 = str(g)
	g2.trans_world_to_im(r.camera)
	s2 = str(g2)
	g3.trans_world_to_im(r.camera)
	s3 = str(g3)
	g4.trans_world_to_im(r.camera)
	s4 = str(g4)
	print("\nEqual for trans_world_to_im from init_from_dict with tensors vs floats vs lists vs from read?", s1 == s2 == s3 == s4)
	

	# Test equality before and after saving after using trans_world_to_im
	# g.trans_world_to_im(r.camera)
	# s1 = str(g)
	# # print("\ns1:", s1)
	g.save('test-grasp.json')

	g5 = Grasp.read("test-grasp.json")
	s5 = str(g5)
	# print("\ns5:", s5)
	print("\nEqual after reading after trans_world_to_im?", s1==s5, "\n")


	# # Test dtypes and shapes of all attributes are equal
	# for grasp in [g, g4, g5]:
	# 	# print("im_angle:\t", grasp.im_angle.item())
		# print("\ngrasp")
		# print("\tquality:\t", type(grasp.quality), "\t", grasp.quality.dtype, "\t", grasp.quality)
		# print("\tdepth:\t\t", type(grasp.depth), "\t", grasp.depth.dtype, "\t", grasp.depth)
		# print("\tworld_center:\t", type(grasp.world_center), "\t", grasp.world_center.dtype, "\t", grasp.world_center)
		# print("\tworld_axis:\t", type(grasp.world_axis), "\t", grasp.world_axis.dtype, "\t", grasp.world_axis)
		# print("\tc0:\t\t", type(grasp.c0), "\t", grasp.c0.dtype, "\t", grasp.c0)
		# print("\tc1:\t\t", type(grasp.c1), "\t", grasp.c1.dtype, "\t", grasp.c1)
		# print("\tim_center:\t", type(grasp.im_center),"\t", grasp.im_center.dtype, "\t", grasp.im_center)
		# print("\tim_angle:\t", type(grasp.im_angle), "\t", grasp.im_angle.dtype, "\t", grasp.im_angle)
		# print("\tim_axis:\t", type(grasp.im_axis), "\t", grasp.im_axis.dtype, "\t", grasp.im_axis)


	# Test model inference and trans_world_to_im on saved grasps
	model = KitModel("weights.npy")
	model.eval()
	run1 = Attack(model=model)

	mesh, _ = r.render_object("data/new_barclamp.obj", display=False)
	d_im = r.mesh_to_depth_im(mesh, display=False)
	pose1, image1 = g.extract_tensors(d_im)
	pose2, image2 = g4.extract_tensors(d_im)
	pose3, image3 = g5.extract_tensors(d_im)

	pred1 = run1.run(pose1, image1)[0][0].item()	# gqcnn predictions
	pred2 = run1.run(pose2, image2)[0][0].item()
	pred3 = run1.run(pose3, image3)[0][0].item()
	print("\nprediction 1:", pred1)
	print("prediction 2:", pred2)
	print("prediction 3:", pred3)
	print("\nModel predictions the same on fixed and read grasps?", pred1 == pred2) # == pred3)


	# Test using trans_to_im again
	g.trans_world_to_im(r.camera)
	s1 = str(g)
	print("\nangle:", g.im_angle)
	print("\tim_axis:", g.im_axis)
	# print("\tworld_axis:", g.world_axis)
	print("\nangle:", g2.im_angle)
	print("\tim_axis:", g2.im_axis)
	# print("\tworld_axis:", g2.world_axis)
	print("\nEqual after using trans_to_im again?", s1 == s2)

	print("")
	Grasp.logger.info("Finished running test_save_and_load_grasps.")
	
def test_trans_world_to_im():
	Grasp.logger.info("Running test_trans_world_to_im...")

	r = Renderer()
	mesh, _ = r.render_object("data/bar_clamp.obj", display=False)
	dim = r.mesh_to_depth_im(mesh, display=False)

	grasp = Grasp.read("experiment-results/ex00/grasp.json")
	grasp.trans_world_to_im(camera=r.camera)

	pose, image = grasp.extract_tensors(dim)
	r.display(image, "processed depth image")
	r.draw_grasp(dim, grasp.c0, grasp.c1, "draw_grasp")

	r.grasp_sphere((grasp.c0, grasp.c1), mesh, "test_draw_grasp.obj", display=True)

	# # model evaluation
	# model = KitModel("weights.npy")
	# model.eval()
	# run1 = Attack(model=model)
	# pose, image = grasp.extract_tensors(dim)

	# model_out = run1.run(pose, image)[0][0].item()
	# r.display(image, title="processed dim\npred: "+str(model_out))

	Grasp.logger.info("Finished running test_trans_world_to_im.")

def test_oracle_selection():
	"""Test to ensure the correct oracle method is called based on grasp.oracle_method, grasp.oracle_robust, and argument methods"""

	Grasp.logger.info("Running test_oracle_selection...")

	grasp = Grasp.read("experiment-results/main/grasp-2/grasp.json")
	r = Renderer()

	print("\noracle_method:", grasp.oracle_method)
	print("oracle_robust:", grasp.oracle_robust)
	print("oracle eval:", grasp.oracle_eval("data/new_barclamp.obj"))

	print("\noracle_method:", grasp.oracle_method)
	print("oracle_robust:", False)
	print("oracle eval:", grasp.oracle_eval("data/new_barclamp.obj", robust=False))

	# print("\noracle_method: pytorch")
	# print("oracle_robust:", True)
	# print("oracle eval:", grasp.oracle_eval("data/new_barclamp.obj", oracle_method="pytorch", renderer=r))

	print("\noracle_method: blahblahblah")
	print("oracle_robust:", grasp.oracle_robust)
	print("oracle eval:", grasp.oracle_eval("data/new_barclamp.obj", oracle_method="blahblahlblah"))

	print("\noracle_method: blahblahblah")
	print("oracle_robust:", False)
	print("oracle eval:", grasp.oracle_eval("data/new_barclamp.obj", oracle_method="blahblahlblah", robust=False))

	print("\noracle_method:", grasp.oracle_method)
	print("oracle_robust: blahblah")
	print("oracle eval:", grasp.oracle_eval("data/new_barclamp.obj", robust="blahblah"), "\n")

	Grasp.logger.info("Finished running test_oracle_selection.")

def test_oracle_eval():
	Grasp.logger.info("Running test_oracle_eval...")

	r = Renderer()
	g = Grasp.read("example-grasps/grasp_0.json")
	g.trans_world_to_im(camera=r.camera)

	pytorch_qual = g.oracle_eval("data/new_barclamp.obj", oracle_method="pytorch", renderer=r)[0].item()
	dexnet_qual = g.oracle_eval("data/new_barclamp.obj", oracle_method="dexnet", robust=False)
	dexnet_qual_robust = g.oracle_eval("data/new_barclamp.obj", oracle_method="dexnet", robust=True)

	print("pytorch quality:", pytorch_qual)
	print("dexnet quality:", dexnet_qual)
	print("dexnet quality robust:", dexnet_qual_robust)
	print("diff:", pytorch_qual - dexnet_qual)

	# # iterate through dataset in data/data
	# data_dir = "data/data/data"
	# i = 0
	# obj = "data/new_barclamp.obj"
	# while True:
	# 	datafile = data_dir + str(i) + ".json"
	# 	if not os.path.isfile(datafile):
	# 		break

	# 	if i not in [0, 1]:
	# 		with open(datafile) as f:
	# 			grasp_dict = json.load(f)

	# 		init_dict = {
	# 			"world_center": grasp_dict["pytorch_w_center"],
	# 			"world_axis": grasp_dict["pytorch_w_axis"], 
	# 			"c0": grasp_dict["contact_points"][0],
	# 			"c1": grasp_dict["contact_points"][1]
	# 		}

	# 		g = Grasp.init_from_dict(init_dict)
	# 		g.trans_world_to_im(camera=r.camera)
	# 		pytorch_qual = g.oracle_eval(obj, oracle_method="pytorch", renderer=r)[0].item()
	# 		dexnet_qual = g.oracle_eval(obj, oracle_method="dexnet", robust=False)

	# 		print("\ngrasp " + str(i))
	# 		print("pytorch quality:", pytorch_qual)
	# 		print("dexnet quality:", dexnet_qual)
	# 		print("diff:", abs(pytorch_qual - dexnet_qual))

	# 	i += 1

	print("\n")
	Grasp.logger.info("Finished running test_oracle_eval.")

def test_batching():

	Grasp.logger.info("Running test_batching. Expecting 4 errors...")
	
	# try loading a batch of grasps from json files
	files = ["example-grasps/grasp_0.json", "example-grasps/grasp_1.json", "example-grasps/grasp_2.json", "example-grasps/grasp_3.json"]
	gb = Grasp.read_batch(files)
	assert gb.depth.shape == torch.Size([4,1])
	assert gb.im_center.shape == torch.Size([4,2])
	files2 = ["example-grasps/grasp_0.json", "example-grasps/grasp_1.png", "grasp_2.json"]
	gb2 = Grasp.read_batch(files2)
	assert gb2.depth.shape == torch.Size([1,1])
	assert gb2.im_center.shape == torch.Size([1,2])

	# try saving a batch of grasps and reading from one file
	gb.save("test-batch.json")
	gb_read = Grasp.read("test-batch.json")
	assert gb_read.depth.shape == torch.Size([4,1])
	assert gb_read.im_center.shape == torch.Size([4,2])

	# test trans_world_to_im in a batch
	r = Renderer()
	gb2_test = Grasp.read_batch(files2)
	gb2.trans_world_to_im(r.camera)
	assert torch.max(gb2_test.im_center - gb2.im_center).item() < EPS
	assert torch.max(gb2_test.im_axis - gb2.im_axis).item() < EPS
	assert torch.max(gb2_test.im_angle - gb2.im_angle).item() < EPS

	gb.trans_world_to_im(r.camera)
	g0 = Grasp.read("example-grasps/grasp_0.json")
	g1 = Grasp.read("example-grasps/grasp_1.json")
	g2 = Grasp.read("example-grasps/grasp_2.json")
	g3 = Grasp.read("example-grasps/grasp_3.json")
	check_grasps = [g0, g1, g2, g3]
	for i in range(4):
		message = "iteration " + str(i)
		assert torch.max(gb.im_angle[i] - check_grasps[i].im_angle).item() < EPS, message
		assert torch.max(gb.im_axis[i] - check_grasps[i].im_axis).item() < EPS, message
		assert torch.max(gb.im_center[i] - check_grasps[i].im_center).item() < EPS, message

	# test extract_tensors in a batch
	mesh, _ = r.render_object("data/new_barclamp.obj", display=False)
	dim = r.mesh_to_depth_im(mesh, display=False)
	dims = dim.repeat(4, 1, 1, 1)
	pose0, image0 = g0.extract_tensors(dim)
	print("\nBATCH PROCESSING")
	poses, images = gb.extract_tensors_batch(dims)

	print("\nimages:", images.shape)
	print("poses:", poses.shape)

	image1 = images[0]
	print("image0", image0.shape)
	print("image1:", image1.shape)
	diff = image1 - image0
	print("diff:", torch.max(diff))

	if torch.max(diff).item() > 0:
		r.display(image0, title="extract_tensors")
		r.display(image1, title="extract_tensors_batch")
		r.display(diff, title="diff")

	# compare model predictions
	model = KitModel("weights.npy")
	model.eval()
	run1 = Attack(model=model)
	print("extract_tensors:", run1.run(pose0, image0)[0][0].item())	#, g0.oracle_eval("data/new_barclamp.obj", oracle_method="dexnet", robust=True, renderer=r))
	print("extract_tensors_batch:", run1.run(poses, images)[0][0].item())

	Grasp.logger.info("Finished running test_batching.")

def test_processing_batching():

	Grasp.logger.info("Running test_processing_batching...")

	files = ["example-grasps/grasp_0.json", "example-grasps/grasp_1.json", "example-grasps/grasp_2.json", "example-grasps/grasp_3.json"]
	gb = Grasp.read_batch(files)
	g0 = Grasp.read("example-grasps/grasp_0.json")

	r = Renderer()
	gb.trans_world_to_im(r.camera)
	g0.trans_world_to_im(r.camera)

	mesh, _ = r.render_object("data/new_barclamp.obj", display=False)
	dim = r.mesh_to_depth_im(mesh, display=False)
	dims = dim.repeat(4, 1, 1, 1)
	pose, resized, trans, rot, trans_rot = g0.extract_tensors(dim, debug=True)
	pose_batch, resized_batch, trans_batch, rot_batch, trans_rot_batch, centered = gb.extract_tensors_batch(dims, debug=True)

	pose0 = pose_batch[0]
	pose_diff = torch.max(pose0 - pose)
	print("\npose_diff:", pose_diff.item(), "\n")

	resized0 = resized_batch[0]
	resized_diff = torch.max(resized0 - resized)
	print("resized_diff:", resized_diff.item(), "\n")

	trans0 = trans_batch[0]
	trans_diff = torch.max(trans0 - trans)
	print("trans_diff:", trans_diff.item(), "\n")

	rot0 = rot_batch[0]
	rot_diff = torch.max(rot0 - rot)
	print("rot_diff:", rot_diff.item(), "\n")

	r.display(centered[0], "center translation only")	# NOT WORKING AS EXPECTED
	r.display(trans0, "tensor_extraction translation only")
	r.display(trans, "tensor_extraction_batch translation only")
	r.display(trans-trans0, "diff translation only")

	test = torch.from_numpy(np.load("data/grasp_test2.npy")).permute(0,3,1,2)
	r.display(test, title="gqcnn grasp_to_tensors")

	save_dim = dim.unsqueeze(0).permute(0,2,3,1).cpu().detach().numpy()
	save_nparr(save_dim, "np_dim_test.npy")

	Grasp.logger.info("Finished running test_processing_batching.")

if __name__ == "__main__":

	# test_trans_world_to_im()
	# test_select_grasp()
	# test_save_and_load_grasps()
	# test_oracle_selection()
	# test_oracle_eval()
	# test_batching()
	test_processing_batching()

