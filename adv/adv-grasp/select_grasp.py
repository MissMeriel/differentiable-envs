import os
import math
import time
import logging
import json
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial._qhull import QhullError
from pytorch3d.ops import sample_points_from_meshes
from render import *
from run_gqcnn import *
from quality_fork import *

# imports only needed for dex-net oracle
import shutil
import Pyro4
from matplotlib.ticker import MaxNLocator

SHARED_DIR = "/home/hmitchell/pytorch3d/dex_shared_dir"
EPS = 0.00001

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

	def __init__(self, depth=None, im_center=None, im_angle=None, im_axis=None, world_center=None, world_axis=None, c0=None, c1=None, quality=None, prediction=None, oracle_method="pytorch", oracle_robust=None, objf=None):
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
		objf: String
			Path to .obj file associated with grasp

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
			self.depth = torch.tensor([depth]).to(device).float().unsqueeze(0)
		elif isinstance(depth, list):
			self.depth = torch.from_numpy(np.array(depth)).to(device).float()
		if self.depth != None and self.depth.dim() == 1:
			self.depth = self.depth.unsqueeze(0)

		self.im_angle = im_angle
		if isinstance(im_angle, float):
			self.im_angle = torch.tensor([im_angle]).to(device).float().unsqueeze(0)
		elif isinstance(im_angle, list):
			self.im_angle = torch.from_numpy(np.array(im_angle)).to(device).float()
		if self.im_angle != None and self.im_angle.dim() == 1:
			self.im_angle = self.im_angle.unsqueeze(0)

		self.quality = quality
		if isinstance(quality, float):
			self.quality = torch.tensor([quality]).to(device).float().unsqueeze(0)
		elif isinstance(quality, list):
			self.quality = torch.from_numpy(np.array(quality)).to(device).float()
		if self.quality != None and self.quality.dim() == 1:
			self.qualilty = self.quality.unsqueeze(0)

		self.prediction = prediction
		if isinstance(prediction, float):
			self.prediction = torch.tensor([prediction]).to(device).float().unsqueeze(0)
		elif isinstance(prediction, list):
			self.prediction = torch.from_numpy(np.array(prediction)).to(device).float()
		if self.prediction != None and self.prediction.dim() == 1:
			self.prediction = self.prediction.unsqueeze(0)

		self.oracle_method = oracle_method
		if oracle_method not in ["dexnet", "pytorch"]:
			self.oracle_method = "dexnet"

		self.oracle_robust = oracle_robust
		if isinstance(oracle_robust, bool):
			self.oracle_robust = oracle_robust

		self.objf = objf

	@classmethod
	def init_from_dict(cls, dict):
		"""Initialize grasp object from a dictionary"""
		init_keys = ["depth", "im_center", "im_angle", "im_axis", "world_center", "world_axis", "c0", "c1", "quality"]
		dict_keys = dict.keys()
		for key in init_keys:
			if key not in dict_keys:
				dict[key] = None
		if "oracle_method" not in dict_keys:
			dict["oracle_method"] = "pytorch"
		if "oracle_robust" not in dict_keys:
			dict["oracle_robust"] = None
		if "prediction" not in dict_keys:
			dict["prediction"] = None
		if "objf" not in dict_keys:
			dict["objf"] = None

		return Grasp(depth=dict["depth"], im_center=dict["im_center"], im_angle=dict["im_angle"], im_axis=dict["im_axis"], world_center=dict["world_center"], world_axis=dict["world_axis"], c0=dict["c0"], c1=dict["c1"], quality=dict["quality"], prediction=dict["prediction"], oracle_method=dict["oracle_method"], oracle_robust=dict["oracle_robust"], objf=dict["objf"])

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

		if "objf" in batch_dict.keys():
			obj_lst = batch_dict["objf"]
			obj = obj_lst.pop(0)
			for o in obj_lst:
				if o != obj:
					cls.logger.error("read_batch - all grasps must be for the same grasp object.")
					continue
			batch_dict["objf"] = o

		return cls.init_from_dict(batch_dict)

	def __iter__(self):
		"""Make Grasp class iterable"""
		self.index = 0
		self.length = self.num_grasps()

		return self

	def __next__(self):
		"""Make Grasp class iterable"""
		if self.index < self.length:
			self.index += 1
			return self[self.index-1]
		else:
			raise StopIteration

	def __getitem__(self, key):
		"""Make Grasp class sliceable"""
		if isinstance(key, slice):
			dep = self.depth[key] if self.depth is not None else None
			imc = self.im_center[key] if self.im_center is not None else None
			angle = self.im_angle[key] if self.im_angle is not None else None
			imax = self.im_axis[key] if self.im_axis is not None else None
			wc = self.world_center[key] if self.world_center is not None else None
			wax = self.world_axis[key] if self.world_axis is not None else None
			con0 = self.c0[key] if self.c0 is not None else None
			con1 = self.c1[key] if self.c1 is not None else None
			qual = self.quality[key] if self.quality is not None else None
			pred = self.prediction[key] if self.prediction is not None else None
			return Grasp(depth=dep, im_center=imc, im_angle=angle, im_axis=imax, world_center=wc, world_axis=wax, c0=con0, c1=con1, quality=qual, prediction=pred, oracle_method=self.oracle_method, oracle_robust=self.oracle_robust)
		elif isinstance(key, int):
			length = self.num_grasps()
			if key < 0:
				key += length
			if key >= length or key < 0:
				raise IndexError(f"The index {key} is out of range for accessing a Grasp object of length {length}")
			dep = self.depth[key] if self.depth is not None else None
			imc = self.im_center[key] if self.im_center is not None else None
			angle = self.im_angle[key] if self.im_angle is not None else None
			imax = self.im_axis[key] if self.im_axis is not None else None
			wc = self.world_center[key] if self.world_center is not None else None
			wax = self.world_axis[key] if self.world_axis is not None else None
			con0 = self.c0[key] if self.c0 is not None else None
			con1 = self.c1[key] if self.c1 is not None else None
			qual = self.quality[key] if self.quality is not None else None
			pred = self.prediction[key] if self.prediction is not None else None
			return Grasp(depth=dep, im_center=imc, im_angle=angle, im_axis=imax, world_center=wc, world_axis=wax, c0=con0, c1=con1, quality=qual, prediction=pred, oracle_method=self.oracle_method, oracle_robust=self.oracle_robust)
		else:
			raise TypeError("Invalid argument type for accessing a Grasp object.")

	def __hash__(self):
		return hash(self)

	def __eq__(self, obj):
		if type(self) != type(obj):
			return False
		if self.depth.shape != obj.depth.shape or not torch.min(torch.eq(self.depth, obj.depth)): return False
		if self.world_center is not None and obj.world_center is not None:
			if self.world_center.shape != obj.world_center.shape or torch.max(torch.sub(self.world_center, obj.world_center)) > EPS: return False
			if self.world_axis.shape != obj.world_axis.shape or torch.max(torch.sub(self.world_axis, obj.world_axis)) > EPS: return False
		if self.im_center is not None and obj.im_center is not None:
			if self.im_center.shape != obj.im_center.shape or torch.max(torch.sub(self.im_center, obj.im_center)) > EPS: return False
			if self.im_axis.shape != obj.im_axis.shape or torch.max(torch.sub(self.im_axis, obj.im_axis)) > EPS: return False
			if self.im_angle.shape != obj.im_angle.shape or torch.max(torch.sub(self.im_angle, obj.im_angle)) > EPS: return False
		if self.c0 is not None and obj.c0 is not None:
			if self.c0.shape != obj.c0.shape or torch.max(torch.sub(self.c0, obj.c0)) > EPS: return False
			if self.c1.shape != obj.c1.shape or torch.max(torch.sub(self.c1, obj.c1)) > EPS: return False
		if self.objf is not None and obj.objf is not None:
			if self.objf.split("/")[-1] != obj.objf.split("/")[-1]:
				if not ((self.objf.split("/")[-1] in ["new_barclamp.obj", "bar_clamp.obj"]) and (obj.objf.split("/")[-1] in ["new_barclamp.obj", "bar_clamp.obj"])): return False
		return True

	def __str__(self):
		"""Returns a string with grasp information in image coordinates"""
		p_str = "grasp:"
		
		if self.quality is not None:
			p_str += "\n\tquality: " + str(self.quality)
		if self.prediction is not None:
			p_str += "\n\tmodel prediction: " + str(self.prediction)
		if self.im_center is not None:
			p_str +=  "\n\timage center: " + str(self.im_center)
		if self.im_angle is not None:
			p_str += "\n\timage angle: " + str(self.im_angle)
		if self.im_axis is not None:
			p_str += "\n\timage angle: " + str(self.im_axis)
		if self.depth is not None:
			p_str += "\n\tdepth: " + str(self.depth)
		if self.world_center is not None:
			p_str += "\n\tworld center: " + str(self.world_center)
		if self.world_axis is not None:
			p_str += "\n\tworld axis: " + str(self.world_axis)
		if self.c0 is not None and self.c1 is not None:
			p_str += "\n\tcontact points: " + str(self.c0) + "\t" + str(self.c1)
		if self.oracle_method is not None:
			p_str += "\n\tworld axis: " + str(self.oracle_method)
		if self.objf is not None:
			p_str += "\n\tobj file: " + self.objf

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
			prediction = self.prediction.clone().detach().cpu().numpy().tolist()
		oracle_method, robust = self.oracle_method, self.oracle_robust
		if isinstance(oracle_method, torch.Tensor):
			oracle_method = oracle_method.clone().detach().cpu().numpy().tolist()
		if isinstance(robust, torch.Tensor):
			robust = robust.clone().detach().cpu().numpy().tolist()
		objf = self.objf if self.objf is not None else "unknown"

		grasp_data = {
			"depth": depth,
			"im_center": imc_list,
			"im_axis": imax_list,
			"im_angle": angle,
			"world_center": wc_list,
			"world_axis": was_list,
			"c0": c0_list,
			"c1": c1_list, 
			"oracle_method": oracle_method,
			"oracle_robust": robust,
			"quality": quality,
			"prediction": prediction,
			"objf": objf
		}

		with open(fname, "w") as f:
			json.dump(grasp_data, f, indent=4)

	def num_grasps(self):
		"""Return number of grasps in batch"""
		return self.world_center.shape[0]

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
	def sample_grasps(cls, obj_f, num_samples, renderer, **kwargs):
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
			Float between 0.0 and 1.0, minimum ferrari canny quality for returned grasps, defaults to 0.002
				(0.002 according to dexnet robust ferrari canny method)
		max_qual: float
			Float between 0.0 and 1.0, maximum ferrari canny quality for returned grasps, defaults to 1.0
		save_grasp: String
			If empty string, no new grasp objects are saved
			If not empty, this is the path to the directory where to save new grasp objects that have a sphere added to visualize the grasp
		sort: Boolean
			True: sort grasps in descending order of quality (default)
			False: return in order they were sampled in
		oracle_method: String in ["dexnet", "pytorch"]
			"dexnet": use docker dexnet oracle 
			"pytorch": use local pytorch oracle implementation (default)
		oracle_robust: Boolean
			True: robust oracle evaluation (default)
			False: non-robust oracle evaluation

		Returns
		-------
		List of Grasp objects with quality between min_qual and max_qual inclusive in descending order of rfc_quality
		"""

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

		if ("oracle_method" not in keys) or (kwargs["oracle_method"] != "dexnet"):
			return cls.sample_grasps_pytorch(obj_f=obj_f, num_samples=num_samples, renderer=renderer, **kwargs)
		else:
			return cls.sample_grasps_dexnet(obj_f=obj_f, num_samples=num_samples, renderer=renderer, **kwargs)

	@classmethod
	def sample_grasps_pytorch(cls, obj_f, num_samples, renderer, **kwargs):
		"""
		Helper method for sample_grasps: samples grasps using local pytorch oracle implementation.

		Refer to `sample_grasps` method documentation for details on parameters and return values.
		"""

		# logging
		cls.logger.info("Sampling grasps...")

		# render mesh
		mesh, _ = renderer.render_object(obj_f, display=False)

		# track grasps
		num_grasps, it, grasp_dict = 0, 0, {}
		grasp_dict["world_axis"] = []
		grasp_dict["world_center"] = []
		grasp_dict["quality"] = []
		grasp_dict["c0"] = []
		grasp_dict["c1"] = []
		grasp_dict["objf"] = obj_f
		grasp_dict["oracle_method"] = "pytorch"

		while (num_grasps < num_samples) and (it < 10):
			print(f"iteration {it}")
			# randomly sample surface points for possible grasps
			samples_c0 = sample_points_from_meshes(mesh, num_samples*200)[0]
			samples_c1 = sample_points_from_meshes(mesh, num_samples*200)[0]
			norms = torch.linalg.norm((samples_c1 - samples_c0), dim=1)

			# mask to eliminate grasps that don't fit in the gripper
			mask = (norms > 0.01) & (norms <= 0.05)
			mask = mask.squeeze()
			norms = norms[mask]
			c0 = samples_c0[mask, :]
			c1 = samples_c1[mask, :]
		
			# compute grasp center and axis, then check quality
			world_centers = (c0 + c1) / 2
			world_axes = (c1 - c0) / norms.unsqueeze(1)

			# update 3D axis based on 2D transformation
			camera = renderer.camera
			axis = camera.get_world_to_view_transform().transform_normals(world_axes)
			axis[..., 2] = 0
			axis = torch.nn.functional.normalize(axis, dim=-1)
			world_axes = camera.get_world_to_view_transform().inverse().transform_normals(axis)

			# temporary dictionary for candidate grasps
			temp_dict = {"world_axis": world_axes.detach().cpu().numpy().tolist(), "world_center": world_centers.detach().cpu().numpy().tolist(), "c0": c0.detach().cpu().numpy().tolist(), "c1": c1.detach().cpu().numpy().tolist()}
			temp_qual = torch.zeros(world_centers.shape[0], 1, dtype=torch.float64, device="cuda:0")

			cand_grasps = Grasp.init_from_dict(temp_dict)
			for i, g in enumerate(cand_grasps):
				if i % 100 == 0:
					cls.logger.info(f"Checking grasp {i} of {world_centers.shape[0]} in iteration {it}.")
				temp_qual[i] = g.oracle_eval(obj_f, oracle_method="pytorch", renderer=renderer)

			# check what grasps are in quality range
			mask = (temp_qual >= kwargs["min_qual"]) & (temp_qual <= kwargs["max_qual"])
			mask = mask.squeeze()
			temp_qual = temp_qual[mask]
			c0 = c0[mask, :]
			c1 = c1[mask, :]
			world_centers = world_centers[mask, :]
			world_axes = world_axes[mask, :]

			grasp_dict["world_axis"] = grasp_dict["world_axis"] + world_axes.detach().cpu().numpy().tolist()
			grasp_dict["world_center"] = grasp_dict["world_center"] + world_centers.detach().cpu().numpy().tolist()
			grasp_dict["quality"] = grasp_dict["quality"] + temp_qual.detach().cpu().numpy().tolist()
			grasp_dict["c0"] = grasp_dict["c0"] + c0.detach().cpu().numpy().tolist()
			grasp_dict["c1"] = grasp_dict["c1"] + c1.detach().cpu().numpy().tolist()
			num_grasps += torch.sum(mask).item()
			it += 1
			cls.logger.info(f"# successful grasps on iteration {it}: {torch.sum(mask)}")
  
		# sort in descending order of quality
		grasp = Grasp.init_from_dict(grasp_dict)
		if kwargs["sort"]:
			sort_indices = torch.argsort(grasp.quality, 0, descending=True).squeeze()
			grasp.world_center = torch.index_select(grasp.world_center, 0, sort_indices)
			grasp.world_axis = torch.index_select(grasp.world_axis, 0, sort_indices)
			grasp.c0 = torch.index_select(grasp.c0, 0, sort_indices)
			grasp.c1 = torch.index_select(grasp.c1, 0, sort_indices)
			grasp.quality = torch.index_select(grasp.quality, 0, sort_indices)

		if grasp.num_grasps() > num_samples:
			return grasp[:num_samples]
		else:
			return grasp

	@classmethod
	def sample_grasps_dexnet(cls, obj_f, num_samples, renderer, **kwargs):
		"""
		Helper method for sample_grasps: samples grasps using dexnet oracle using RPC to oracle.py in running docker container.

		Refer to `sample_grasps` method documentation for details on parameters and return values.
		"""

		# process kwargs
		if kwargs["save_grasp"][-1] == "/":
			kwargs["save_grasp"] = kwargs["save_grasp"][:-1]

		# logging
		cls.logger.info("Sampling grasps...")
		cands = []

		# renderer mesh
		mesh, _ = renderer.render_object(obj_f, display=False)

		it = 0	# track iterations
		while len(cands) < num_samples:

			# randomly sample surface points for possible grasps
			samples_c0 = sample_points_from_meshes(mesh, num_samples*1000)[0]
			samples_c1 = sample_points_from_meshes(mesh, num_samples*1000)[0]
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
			print("sample_grasps_dexnet robust:", kwargs["oracle_robust"])
			results = server.close_fingers("temp_centers.npy", "temp_axes.npy", kwargs["min_qual"], kwargs["max_qual"], robust=kwargs["oracle_robust"])
			#	results returns list of lists of form [[int: index, float: ferrari canny quality, (array, array): contact points], ...]
			for res in results:
				res.append(it)	# track which batch/iteration grasp is from (for access to world centers/axes)
			cls.logger.info("%d successful grasps returned", len(results))

			cands = cands + results
			if len(results) > 0:
				it += 1

		# transform successful grasps to image space and convert to Grasp object

		grasp_dict = {"world_center": [], "world_axis": [], "c0": [], "c1": [], "quality": []}
		cls.logger.debug("sample_grasps_dexnet - number of grasps: %d", len(cands))
		for i in range(len(cands)+1):

			# return final grasp batch
			cls.logger.debug("sample_grasps_dexnet - Line 510: %s", str(len(grasp_dict["c0"]) >= num_samples))
			if len(grasp_dict["c0"]) >= num_samples:
				for key in ["c0", "c1", "quality"]:
					grasp_dict[key] = torch.tensor(grasp_dict[key]).to(renderer.device).float()
					# print("key:", key, "\tshape:", grasp_dict[key].shape)

				grasp = cls.init_from_dict(grasp_dict)

				if kwargs["sort"]:
					sort_indices = torch.argsort(grasp.quality, 0, descending=True).squeeze()
					grasp.world_center = torch.index_select(grasp.world_center, 0, sort_indices)
					grasp.world_axis = torch.index_select(grasp.world_axis, 0, sort_indices)
					grasp.c0 = torch.index_select(grasp.c0, 0, sort_indices)
					grasp.c1 = torch.index_select(grasp.c1, 0, sort_indices)
					grasp.quality = torch.index_select(grasp.quality, 0, sort_indices)

				grasp.trans_world_to_im(renderer.camera)

				if kwargs["save_grasp"]:
					# save object to visualize grasp, grasp json, and approx. image of grasp
					pathname = kwargs["save_grasp"]
					if not os.path.isdir(pathname):
						os.mkdir(pathname)
					j, json_batch_name = 0, pathname + "/grasp_batch.json"
					while os.path.exists(json_batch_name):
						json_batch_name = pathname + "/grasp_batch_" + str(i) + ".json"
						j += 1
					grasp.save(json_batch_name)

					for j, g in enumerate(grasp):
						cls.logger.info("saving new grasp visualization object")
						pathname = pathname + "/grasp_" + str(j) + "/"
						if not os.path.exists(pathname):
							os.mkdir(pathname)
						img_name = pathname + "grasp_image.png"
						obj_name = pathname + "grasp_vis.obj"
						renderer.grasp_sphere((g.c0, g.c1), mesh, display=False, save=obj_name)
						renderer.draw_grasp(mesh, g.c0, g.c1, save=img_name, display=False)

				return grasp

			g = cands[i]
			quality = g[2]

			wc, wa = grasp_dict["world_center"], grasp_dict["world_axis"]
			if i == 0:
				grasp_dict["world_center"] = world_centers_batch[g[-1]][g[0]].unsqueeze(0)
				grasp_dict["world_axis"] = world_axes_batch[g[-1]][g[0]].unsqueeze(0)
			else:
				grasp_dict["world_center"] = torch.cat((wc, world_centers_batch[g[-1]][g[0]].unsqueeze(0)), 0)
				grasp_dict["world_axis"] = torch.cat((wa, world_axes_batch[g[-1]][g[0]].unsqueeze(0)), 0)
			grasp_dict["c0"] += [g[-2][0]]
			grasp_dict["c1"] += [g[-2][1]]
			grasp_dict["quality"] += [quality]

		cls.logger.error("Error - sample_grasps with batching isn't working.")

	def random_grasps(self, num_samples=64, sigma=1e-4, camera=None):
		"""
		Populate a singular grasp (or first grasp in batch) with similar grasps from random variations 
		Parameters
		----------
		num_samples: float
			Number of total grasps in result
		sigma: float
			Standard deviation of outputs normal distribution
		Return
		------
		None
		"""

		if self.num_grasps() > 1:
			self.c0 = self.c0[0].unsqueeze(0)
			self.c1 = self.c1[0].unsqueeze(0)

		# vary world centers
		original_center = self.world_center[0]
		expanded_center = self.world_center[0].unsqueeze(0).expand(num_samples, -1)
		noise_center = torch.normal(mean=torch.zeros_like(expanded_center), std=sigma)
		world_center = expanded_center + noise_center
		world_center[0] = original_center
		self.world_center = world_center

		# vary world axes
		original_axis = self.world_axis[0]
		expanded_axis = self.world_axis[0].unsqueeze(0).expand(num_samples, -1)
		noise_axis = torch.normal(mean=torch.zeros_like(expanded_axis), std=sigma)
		world_axis = expanded_axis + noise_axis
		world_axis[0] = original_axis
		# re-normalize world axes
		self.world_axis = world_axis / torch.norm(world_axis, dim=-1, keepdim=True)

		# update other grasp info
		self.quality = None
		self.prediction = None
		
		# update 2D grasp info
		if camera:
			self.trans_world_to_im(camera=camera)

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
		theta = -1 * math.degrees(self.im_angle[0])	# -1 because PyTorch transform goes clockwise and autolab_core goes counter-clockwise
		 
		dim_cx = torch_dim.shape[2] // 2
		dim_cy = torch_dim.shape[1] // 2
		
		translate = ((self.im_center[0][0] - dim_cx) / 3, (self.im_center[0][1]- dim_cy) / 3)

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

		# torch_rotated2 = transforms.functional.affine(
		# 	translated_only,
		# 	theta,
		# 	translate=(0, 0),
		# 	scale=1,
		# 	shear=0, 
		# 	interpolation=transforms.InterpolationMode.BILINEAR,
		# 	center=(torch_image_tensor.shape[2] / 2, torch_image_tensor.shape[1] / 2)
		# )

		# # for debugging - rotation only, no translation
		# rotated_only = transforms.functional.affine(
		# 	torch_image_tensor,
		# 	theta,
		# 	translate=(0,0),
		# 	scale=1,
		# 	shear=0,
		# 	interpolation=transforms.InterpolationMode.BILINEAR,
		# 	center=(cx, cy)
		# )

		# 3 - crop image to size (32, 32)
		torch_cropped = transforms.functional.crop(torch_rotated, cy-17, cx-17, 32, 32)
		image_tensor = torch_cropped.unsqueeze(0)

		return pose_tensor, image_tensor
		# return pose_tensor, torch_image_tensor, translated_only, rotated_only, torch_rotated, image_tensor, torch_rotated2

	def extract_tensors_batch(self, dims):

		r = Renderer()

		# check type of input_dim
		if isinstance(dims, np.ndarray):
			Grasp.error("extract_tensors_batch takes a tensor, not a numpy ndarray")
			return None, None
		
		# check if grasp is 2D
		if (self.depth==None or self.im_center==None or self.im_angle==None):
			Grasp.logger.error("Grasp is not in 2D, must convert with camera intrinsics before tensor extraction.")
			return None, None

		# dims.shape: [batch_size, 1, 480, 640] = [batch_size, channels, H, W]
		batch_size = self.num_grasps()
		if dims.dim() != 4:
			dims = dims.repeat(batch_size, 1, 1, 1)
		if dims.shape[0] != batch_size:
			dims = dims[0].repeat(batch_size, 1, 1, 1)

		# construct pose tensor from grasp depth
		pose_tensor = self.depth

		# process depth image wrt grasp (steps 1-3) 
		# 1 - resize image tensors
		out_shape = torch.tensor([dims.squeeze(1).shape], dtype=torch.float32)
		out_shape *= (1/3)		# using 1/3 based on gqcnn library - may need to change depending on input
		out_shape = tuple(out_shape.type(torch.int)[0][1:].numpy())		# (160, 213)

		torch_transform = transforms.Resize(out_shape, antialias=False) 
		dims_resized = torch_transform(dims)		# shape: [batch_size, 1, 160, 213] = [batch_size, channels, H, W]

		# 2 - translation wrt to grasp angle and grasp center
		# 	translation matrix
		dim_cx = dims.shape[3] // 2	# 320
		dim_cy = dims.shape[2] // 2	# 240
		dim_cx_tens = torch.tensor([dim_cx]).expand(batch_size).to(device)
		dim_cy_tens = torch.tensor([dim_cy]).expand(batch_size).to(device)

		u = (2 * ((dim_cx_tens - self.im_center[..., 0])/3) / (dims_resized.shape[3])).float()	# not in pixels, but normalized on (image size * 2)
		v = (2 * ((dim_cy_tens - self.im_center[..., 1])/3) / (dims_resized.shape[2])).float()

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

		#	apply transformations
		translate_mat = translate[:, :2, :]
		rotation_mat = rotation[:, :2, :]
		
		# 	translation only
		trans_grid = torch.nn.functional.affine_grid(translate_mat, dims_resized.shape)
		trans_only = torch.nn.functional.grid_sample(dims_resized, trans_grid)

		# 	rotation only
		rot_grid = torch.nn.functional.affine_grid(rotation_mat, dims_resized.shape)
		rot_only = torch.nn.functional.grid_sample(dims_resized, rot_grid)

		# 	translation then rotation (applied separately)
		trans_then_rot = torch.nn.functional.grid_sample(trans_only, rot_grid)

		# 3 - crop images to 32x32 pixels
		top = dims_resized.shape[2] // 2 - 17	# 63
		left = dims_resized.shape[3] // 2 - 17	# 89
		dims_transformed = trans_then_rot[:, :, top:top+32, left:left+32]	# [:, :, 63:95, 89:121]

		return pose_tensor, dims_transformed

	def oracle_eval(self, obj_file, oracle_method=None, robust=None, renderer=None, grad=True):
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
			Used only for dex-net oracle method
			True: uses robust ferrari canny evaluation for oracle quality
			False: uses (non robust) ferrari canny evaluation for oracle quality
			defaults to self.oracle_robust
		renderer:
		grad: Boolean
			Used only for pytorch oracle method
			True: keep gradient information for output tensor
			False: no gradient information for output tensor

		Returns
		-------
		float: quality from ferarri canny evaluation

		"""
		if oracle_method == "dexnet" or (self.oracle_method == "dexnet"):
			if isinstance(robust, bool):
				return self.oracle_eval_dexnet(obj_file, robust=robust)
			else:
				return self.oracle_eval_dexnet(obj_file, robust=self.oracle_robust)
		
		else:
			if not renderer:
				Grasp.logger.error("oracle_eval - pytorch oracle requires renderer argument")
			else:
				return self.oracle_eval_pytorch(obj_file, renderer, grad=grad)

	def oracle_eval_dexnet(self, obj_file, robust=True):
		"""
		Get a final oracle evaluation of a mesh object via remote call to docker container.

		Refer to `oracle_eval` method documentation for details on parameters and return values.
		"""

		print("oracle_eval_dexnet robust:", robust)

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
		# calc_axis = (self.c1 - self.c0) / torch.linalg.norm((self.c1 - self.c0))	# ensure grasp axis normalized
		# calc_axes = (self.c1 - self.c0) / torch.linalg.norm((self.c1 - self.c0), dim=1).unsqueeze(1)
		calc_axes = self.world_axis / torch.norm(self.world_axis, dim=-1, keepdim=True)
		save_nparr(self.world_center.detach().cpu().numpy(), "temp_center.npy")
		save_nparr(calc_axes.detach().cpu().numpy(), "temp_axis.npy")
		# print("\nobj_name:", type(obj_name), obj_name, "\n\n")
		results = server.final_evals("temp_center.npy", "temp_axis.npy", obj_name, robust=robust)

		# update quality info
		self.quality = torch.from_numpy(np.array(results)).unsqueeze(1).to(self.world_axis.device).float()

		return self.quality

	def oracle_eval_pytorch(self, obj, renderer, grad=True):
		"""
		Get a final oracle evaluation of a mesh object via local pytorch oracle implementation.

		Refer to `oracle_eval` method documentation for details on parameters and return values.
		"""

		if (self.world_center == None or self.world_axis == None):
			Grasp.logger.error("Grasp.oracle_eval_pytorch requires Grasp to have world_center and world_axis.")
			return None

		config_dict = {
			"torque_scaling":1000,
			"soft_fingers":1,
			"friction_coef": 0.8,
			"antipodality_pctile": 1.0 
    	}

		if isinstance(obj, Meshes): mesh = obj
		else: mesh, _ = renderer.render_object(obj, display=False)

		if self.num_grasps() > 1:
			self.quality = torch.zeros_like(self.depth)
			for i, g in enumerate(self):
				if g.c0 is not None and g.c1 is not None: contact_points = torch.stack((g.c0, g.c1), 0)
				else: contact_points = None
				
				width = torch.tensor([[0.05]], device=device)
				grasp_torch = GraspTorch(center=g.world_center, axis3D=g.world_axis, width=width, camera_intr=renderer.rasterizer.cameras, contact_points=contact_points, friction_coef=config_dict["friction_coef"], torque_scaling=config_dict["torque_scaling"])
				
				try:
					com_qual_func = CannyFerrariQualityFunction(config_dict)
					quality = com_qual_func.quality(mesh, grasp_torch).float().item()
				except QhullError:
					quality =  0.0

				self.quality[i] = quality

		else:

			if self.c0 is not None and self.c1 is not None: contact_points = torch.stack((self.c0, self.c1), 0)
			else: contact_points = None
			
			width = torch.tensor([[0.05]], device=device)
			grasp_torch = GraspTorch(center=self.world_center, axis3D=self.world_axis, width=width, camera_intr=renderer.rasterizer.cameras, contact_points=contact_points, friction_coef=config_dict["friction_coef"], torque_scaling=config_dict["torque_scaling"])
			
			try:
				com_qual_func = CannyFerrariQualityFunction(config_dict)
				self.quality = com_qual_func.quality(mesh, grasp_torch).float().to(self.world_center.device)
			except QhullError:
				self.quality = torch.tensor([0.0]).to(device)

		if grad:
			self.quality.requires_grad_(True)
		else:
			self.quality.requires_grad_(False)

		return self.quality


	def vis_grasp_dataset(self, obj_file, directory, renderer):
		"""Visualize a dataset of grasps in the given directory"""

		if not os.path.isdir(directory): os.mkdir(directory)

		if directory[-1] == "/": directory = directory[:-1]
		batch_fname, img_fname, dimg_fname = directory + "/grasp-batch.json", directory + "/grasp-object.png", directory + "/depth-im.png"
		pdimg_fname, gimg_fname, gsphimg_fname = directory + "/processed-depth-im.png", directory + "/grasp-vis.png", directory + "/grasp-sphere.png"

		# save top level info
		self.save(batch_fname)
		mesh, im = renderer.render_object(obj_file, display=False)
		vol_mesh, vol_bounding, vol_hull = get_volumes(mesh)
		depth_im = renderer.mesh_to_depth_im(mesh, display=False)
		renderer.display(im, title=obj_file.split("/")[-1], save=img_fname)
		renderer.display(depth_im, title=obj_file.split("/")[-1], save=dimg_fname)

		# save grasp specific information
		if self.num_grasps() == 1:
			qual_title = ""
			if self.quality is not None: qual_title += f"\nQuality: {self.quality.item()}"
			if self.prediction is not None: qual_title += f"\nPrediction: {self.prediction.item()}"
			qual_title += f"\n volumes: mesh {vol_mesh}, bb {vol_bounding}, hull {vol_hull}"
			# processed depth image
			_, p_dim = self.extract_tensors(depth_im)
			title = "Depth image w.r.t. grasp" + qual_title
			renderer.display(p_dim, title=title, save=pdimg_fname)

			# grasp sphere rendering
			title = "Grasp visualization" + qual_title
			contacts = [self.c0, self.c1] if (self.c0 is not None and self.c1 is not None) else None
			if contacts is not None: 
				gsphere_mesh = renderer.grasp_sphere(center=contacts, obj=mesh, display=False)
			else:
				gsphere_mesh = renderer.grasp_sphere(center=self.world_center, obj=mesh, display=False)
			renderer.display(gsphere_mesh, title=title, save=gsphimg_fname)

			# if applicable: draw grasp/grasp vis
			if contacts is not None:
				renderer.draw_grasp(mesh, contacts[0], contacts[1], title=title, save=gimg_fname, display=False)

			return None
		
		for i, g in enumerate(self):
			# nested directory for each grasp
			local_dir = directory + f"/grasp_{i}/"
			if not os.path.isdir(local_dir): os.mkdir(local_dir)
			pdimg_fname, gimg_fname, gsphimg_fname = local_dir + pdimg_fname.split("/")[-1], local_dir + gimg_fname.split("/")[-1], local_dir + gsphimg_fname.split("/")[-1]
			
			qual_title = ""
			if g.quality is not None: qual_title += f"\nQuality: {g.quality.item()}"
			if g.prediction is not None: qual_title += f"\nPrediction: {g.prediction.item()}"

			# processed depth image
			_, p_dim = g.extract_tensors(depth_im)
			title = f"Depth image w.r.t. grasp {i}" + qual_title
			renderer.display(p_dim, title=title, save=pdimg_fname)

			# grasp sphere rendering
			title = f"Grasp visualization for grasp {i}" + qual_title
			contacts = [g.c0, g.c1] if (g.c0 is not None and g.c1 is not None) else None
			if contacts is not None: 
				gsphere_mesh = renderer.grasp_sphere(center=contacts, grasp_obj=mesh, display=False)
			else:
				gsphere_mesh = renderer.grasp_sphere(center=g.world_center, grasp_obj=mesh, display=False)
			renderer.display(gsphere_mesh, title=title, save=gsphimg_fname)

			# if applicable: draw grasp/grasp vis
			if contacts is not None:
				renderer.draw_grasp(mesh, contacts[0], contacts[1], title=title, save=gimg_fname, display=False)

def get_volumes(mesh):
	vol_bounding = compute_mesh_bounding_volume(mesh)
	vol_mesh = compute_mesh_volume(mesh)
	vol_hull = compute_mesh_hull_volume(mesh)
	return (vol_mesh, vol_bounding, vol_hull)

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

	renderer1 = Renderer()
	model = KitModel("weights.npy")
	model.eval()
	run1 = Attack(model=model, renderer=renderer1)

	# TESTING GRASP SAMPLING
	mesh, image = renderer1.render_object("data/new_barclamp.obj", display=True, title="imported renderer")
	d_im = renderer1.mesh_to_depth_im(mesh, display=True)
	dims = d_im.repeat(3, 1, 1, 1)

	grasp = Grasp.sample_grasps("data/new_barclamp.obj", 3, renderer=renderer1) #, save_grasp = "example-grasps/batch-test")

	print("im_center shape:", grasp.im_center.shape)
	print("im_axis shape:", grasp.im_axis.shape)
	print("angle shape:", grasp.im_angle.shape, "\n\n")
	print("world_center shape:", grasp.world_center.shape)
	print("world_axis shape:", grasp.world_axis.shape)
	print("c0 shape:", grasp.c0.shape)
	print("c1 shape:", grasp.c1.shape)
	print("quality:", grasp.quality.shape)
	print(grasp.quality, "\n")

	poses, images = grasp.extract_tensors_batch(dims)
	pred = run1.run(poses, images)
	print("prediction:", pred.shape, "\n", pred)
	
	# for i in range(len(grasps)):
	# 	grasp = grasps[i]
	# 	qual = grasp.quality
	# 	pose, image = grasp.extract_tensors(d_im)
	# 	prediction = run1.run(pose, image)
	# 	print("\nprediction size:", prediction.shape, prediction)
	# 	t = "id: " + str(i) + " prediction: " + str(prediction[0][0].item()) + "\noracle quality: " + str(qual.item()) #grasp.title_str()
	# 	fname = "example-grasps/grasp_" + str(i) + "_pred.png"
	# 	renderer1.display(image, title=t)	#, save=True, fname=fname)

	Grasp.logger.info("Finished running test_select_grasp.")

def test_select_grasp_pytorch():
	Grasp.logger.info("Testing grasp sampling with pytorch - test_select_grasp_pytorch")

	fname, num = "temp_0.json", 1
	while os.path.isfile(fname):
		fname = "temp_" + str(num) + ".json"
		num += 1

	r = Renderer()
	g = Grasp.sample_grasps("data/new_barclamp.obj", 5, renderer=r)
	assert g.num_grasps() == 5
	qual = g.quality
	g.save(fname)

	g = Grasp.read(fname)
	os.remove(fname)
	assert torch.max(torch.sub(qual, g.quality)).item() == 0

	qual2 = g.oracle_eval("data/new_barclamp.obj", renderer=r)
	assert torch.max(torch.sub(qual, qual2)).item() == 0

	g.trans_world_to_im(camera=r.camera)
	qual3 = g.oracle_eval("data/new_barclamp.obj", renderer=r)
	assert torch.max(torch.sub(qual, qual3)).item() == 0

	Grasp.logger.info("Finished testing grasp sampling with pytorch - test_select_grasp_pytorch")

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

	g2 = Grasp.init_from_dict(fixed_grasp2)	# from floats

	g3 = Grasp.init_from_dict(fixed_grasp3)	# from lists

	print("Equal when using init_from_dict with tensors vs floats vs lists?", g == g2 == g3)
	assert g == g2
	assert g2 == g3
	
	# Test with init_from_dict vs read from json
	g.save("test-grasp.json")
	g4 = Grasp.read("test-grasp.json")
	print("Equal when using init_from_dict and read?", g == g4)
	assert g == g4


	# Test trans_world_to_im from init_from_dict with tensors vs floats vs lists vs json
	g.trans_world_to_im(r.camera)
	g2.trans_world_to_im(r.camera)
	g3.trans_world_to_im(r.camera)
	g4.trans_world_to_im(r.camera)
	print("Equal for trans_world_to_im from init_from_dict with tensors vs floats vs lists vs from read?", g == g2 == g3 == g4)
	assert g == g2 == g3 == g4
	

	# Test equality before and after saving after using trans_world_to_im
	g.save('test-grasp.json')

	g5 = Grasp.read("test-grasp.json")
	print("Equal after reading after trans_world_to_im?", g==g5, "\n")
	assert g == g5

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
	print("\nModel predictions the same on fixed and read grasps?", pred1 == pred2 == pred3)
	assert pred1 == pred2 == pred3


	# Test using trans_to_im again
	g.trans_world_to_im(r.camera)
	print("Equal after using trans_to_im again?", g==g2)
	assert g == g2

	Grasp.logger.info("Finished running test_save_and_load_grasps.")
	
def test_trans_world_to_im():
	Grasp.logger.info("Running test_trans_world_to_im...")
	display_images = []
	display_titles = []

	r = Renderer()
	mesh, _ = r.render_object("data/bar_clamp.obj", display=False)
	dim = r.mesh_to_depth_im(mesh, display=False)

	grasp = Grasp.read("experiment-results/ex00/grasp.json")
	grasp.trans_world_to_im(camera=r.camera)

	pose, image = grasp.extract_tensors(dim)

	draw_grasp_im = r.draw_grasp(dim, grasp.c0, grasp.c1, "draw_grasp", display=False)
	display_images.append(draw_grasp_im)
	display_titles.append("draw_grasp")

	# r.grasp_sphere((grasp.c0, grasp.c1), mesh, "test_draw_grasp.obj", display=True)
	mesh = r.grasp_sphere((grasp.c0, grasp.c1), mesh, display=False)
	display_images.append(mesh)
	display_titles.append("grasp_sphere")


	# model evaluation
	model = KitModel("weights.npy")
	model.eval()
	run1 = Attack(model=model)
	pose, image = grasp.extract_tensors(dim)

	model_out = run1.run(pose, image)[0][0].item()
	# r.display(image, title="processed dim\npred: "+str(model_out))
	display_images.append(image)
	display_titles.append("processed depth image\npred: " + str(model_out))

	r.display(images=display_images, title=display_titles, shape=(1,3))

	Grasp.logger.info("Finished running test_trans_world_to_im.")

def test_batching():

	Grasp.logger.info("Running test_batching. Expecting 4 errors...")
	
	# try loading a batch of grasps from json files
	files = ["example-grasps/grasp_0.json", "example-grasps/grasp_1.json", "example-grasps/grasp_2.json", "example-grasps/grasp_3.json"]
	gb = Grasp.read_batch(files)
	assert gb.depth.shape == torch.Size([4,1])
	assert gb.im_center.shape == torch.Size([4,2])
	assert gb.num_grasps() == 4
	files2 = ["example-grasps/grasp_0.json", "example-grasps/grasp_1.png", "grasp_2.json"]
	gb2 = Grasp.read_batch(files2)
	assert gb2.depth.shape == torch.Size([1,1])
	assert gb2.im_center.shape == torch.Size([1,2])
	assert gb2.num_grasps() == 1

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
	assert g0.num_grasps() == 1
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
	# dims = dim.repeat(4, 1, 1, 1)
	pose0, image0 = g0.extract_tensors(dim)
	poses, images = gb.extract_tensors_batch(dim)

	assert torch.max(pose0 - poses[0]) == 0

	image1 = images[0]
	pose1 = poses[0]
	
	diff = image1 - image0
	Grasp.logger.debug("processed image diff: %f", torch.max(diff))

	if torch.max(diff).item() > 0:
		r.display([image0, image1, diff], title="extract_tensors vs batched")
		# r.display(image1, title="extract_tensors_batch")
		# r.display(diff, title="diff")

	# compare model predictions
	model = KitModel("weights.npy")
	model.eval()
	run1 = Attack(model=model)
	pred0, pred1 = run1.run(pose0, image0)[0][0].item(), run1.run(poses, images)[0][0].item()
	Grasp.logger.debug("extract_tensors: %f", pred0)
	Grasp.logger.debug("extract_tensors_batch: %f", pred1)
	Grasp.logger.debug("prediction diff: %f", pred1 - pred0)

	Grasp.logger.info("Finished running test_batching.")

def test_slicing():
	Grasp.logger.info("Running test_slicing...")
	
	# try loading a batch of grasps from json files
	files = ["example-grasps/grasp_0.json", "example-grasps/grasp_1.json", "example-grasps/grasp_2.json", "example-grasps/grasp_3.json"]
	gb = Grasp.read_batch(files)
	assert gb.depth.shape == torch.Size([4,1])
	assert gb.im_center.shape == torch.Size([4,2])
	assert gb.num_grasps() == 4
	g0 = Grasp.read("example-grasps/grasp_0.json")
	assert g0.num_grasps() == 1
	files2 = ["example-grasps/grasp_0.json", "example-grasps/grasp_1.json"]
	gb2 = Grasp.read_batch(files2)
	assert gb2.num_grasps() == 2

	print("testing __eq__")
	assert gb != 4
	assert gb == gb
	assert g0 == g0
	assert gb2 == gb2
	g0_dup = Grasp.read("example-grasps/grasp_0.json")
	assert g0 == g0_dup
	g0_dup.depth[0][0] = 45.4
	assert g0 != g0_dup

	g_slice0 = gb[0]
	assert g_slice0 == g0
	gb_slice = gb[0:2]
	assert gb_slice == gb2

	for g in gb:
		assert g.num_grasps() == 1

	for i, g in enumerate(gb):
		assert g.num_grasps() == 1

	Grasp.logger.info("Finished running test_slicing.")

def test_oracle_check(iteration):
	Grasp.logger.info(f"Running oracle volatility check {iteration}...")

	grasp_files = ["example-grasps/grasp_0.json", "example-grasps/grasp_1.json", "example-grasps/grasp_2.json", "example-grasps/grasp_3.json"]
	files = [str_ for str_ in grasp_files for _ in range(16)]
	assert len(files) == 64

	gb = Grasp.read_batch(files)
	start1 = time.time()
	oracle_quals = gb.oracle_eval("data/new_barclamp.obj", oracle_method="dexnet")
	end1 = time.time()
	oracle_quals_nr = gb.oracle_eval("data/new_barclamp.obj", oracle_method="dexnet", robust=False)
	end2 = time.time()
	oracle_quals = oracle_quals.squeeze().detach().cpu().numpy()
	oracle_quals_nr = oracle_quals_nr.squeeze().detach().cpu().numpy()
	assert len(oracle_quals) == len(files)

	robust_time = end1 - start1
	nonrobust_time = end2 - end1
	Grasp.logger.info(f"Time for robust evaluation: {robust_time}")
	Grasp.logger.info(f"Time for non-robust evaluation: {nonrobust_time}")

	x_values = [0, 1, 2, 3]
	total_points = [oracle_quals[0:16], oracle_quals[16:32], oracle_quals[32:48], oracle_quals[48:]]
	assert len(total_points) == 4
	nr_points = [oracle_quals_nr[0:16], oracle_quals_nr[16:32], oracle_quals_nr[32:48], oracle_quals_nr[48:]]

	plt.figure(figsize=(9, 9))
	for i, color1_set in enumerate(total_points):
		if i == 0:
			plt.scatter([x_values[i]]*len(color1_set), color1_set, color='red', label='robust')
		else:
			plt.scatter([x_values[i]]*len(color1_set), color1_set, color='red')

	for i, color2_set in enumerate(nr_points):
		if i == 0:
			plt.scatter([x_values[i]]*len(color2_set), color2_set, color='blue', label='non-robust')
		else:
			plt.scatter([x_values[i]]*len(color2_set), color2_set, color='blue')

	plt.xlabel('Grasp')
	plt.ylabel('Oracle Quality')
	title_str = f"Consistency of oracle evaluations\nbatch_size: 64\nnum_samples: 25\nrobust time (s): {robust_time}\nnon-robust time (s): {nonrobust_time}"
	plt.title(title_str)
	plt.legend()
	plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

	plt.show()
	# filesave = "ex" + str(10+iteration) + ".png"
	# Grasp.logger.info(f"SAVING FIGURE {iteration}")
	# plt.savefig(filesave)

def test_random_grasps():
	Grasp.logger.info("Testing random_grasps...")
	r = Renderer()

	# randomize grasp from file
	g_orig = Grasp.read("example-grasps/grasp_0.json")
	g = Grasp.read("example-grasps/grasp_0.json")
	g.random_grasps(num_samples=10, camera=r.camera)
	assert g.num_grasps() == 10
	g.oracle_eval("data/new_barclamp.obj", renderer=r)
	print("\nrand qualities:", g.quality.shape, "\n", g.quality)

	files = ["example-grasps/grasp_0.json", "example-grasps/grasp_1.json"]
	gb = Grasp.read_batch(files)
	gb.random_grasps(num_samples=5, camera=r.camera)
	assert gb.num_grasps() == 5
	assert gb.c0.shape[0] == 1
	assert gb.c1.shape[0] == 1
	res = gb.oracle_eval("data/new_barclamp.obj", renderer=r)
	print("\n\nrand batch qualities:", gb.quality.shape, "\n", gb.quality)

	Grasp.logger.info("Finished testing random_grasps.")

def test_pytorch_oracle():
	Grasp.logger.info("Running test_pytorch_oracle() to test PyTorch oracle.")

	r = Renderer()
	mesh, _ = r.render_object("data/new_barclamp.obj", display=False)
	config_dict = {
		"torque_scaling":1000,
		"soft_fingers":1,
		"friction_coef": 0.8,
		"antipodality_pctile": 1.0 
	}

	Grasp.logger.info("Run Ferarri Canny quality function from quality_fork.py")
	_, device = pytorch_setup()

	center2d = torch.tensor([[344.3809509277344, 239.4164276123047]],device=device)
	angle = torch.tensor([[0.3525843322277069 + math.pi]],device=device)
	depth = torch.tensor([[0.5824159979820251]],device=device)
	width = torch.tensor([[0.05]],device=device)

	grasp1 = GraspTorch(center2d, angle, depth, width, r.camera, friction_coef=config_dict["friction_coef"], torque_scaling=config_dict["torque_scaling"])
	
	com_qual_func = CannyFerrariQualityFunction(config_dict)
	print("canny ferrari:", com_qual_func.quality(mesh, grasp1))

	Grasp.logger.info("Comparison of select_grasp.Grasp and quality_fork.GraspTorch")
	# GraspTorch object
	center3D = torch.tensor([[ 0.027602000162005424, 0.017583999782800674, -9.273400064557791e-05]], device=device)
	axis3D   = torch.tensor([[-0.9384999871253967, 0.2660999894142151, -0.22010000050067902]], device=device)
	graspt = GraspTorch(center3D, axis3D=axis3D, width=width, camera_intr=r.camera,friction_coef=config_dict["friction_coef"], torque_scaling=config_dict["torque_scaling"]) 
	graspt.make2D(updateCamera=False)
	# Grasp object
	grasp = Grasp(world_center=center3D, world_axis=axis3D)
	grasp.trans_world_to_im(camera=r.camera)
	# comparisons
	assert torch.max(torch.sub(graspt.center, grasp.im_center)).item() == 0
	assert torch.max(torch.sub(graspt.angle, grasp.im_angle)).item() < EPS
	assert torch.max(torch.sub(graspt.depth, grasp.depth)).item() == 0
	assert torch.max(torch.sub(graspt.axis, grasp.im_axis)).item() < EPS
	assert torch.max(torch.sub(graspt.axis3D, grasp.world_axis)).item() == 0
	# quality comparisons
	graspt_qual, grasp_qual = com_qual_func.quality(mesh, graspt), grasp.oracle_eval("data/new_barclamp.obj", oracle_method="pytorch", renderer=r)
	assert torch.max(torch.sub(graspt_qual.to(grasp_qual.device), grasp_qual)).item() < EPS

	Grasp.logger.info("PyTorch oracle evaluation on batched grasps.")
	# files = ["example-grasps/grasp_" + str(i) + ".json" for i in range(5)]
	# grasps = Grasp.read_batch(files)
	grasps = Grasp.read("grasp-dataset/grasp-batch.json")
	grasps.trans_world_to_im(camera=r.camera)
	batch_qual = grasps.oracle_eval("data/new_barclamp.obj", oracle_method="pytorch", renderer=r)
	for i, grasp in enumerate(grasps):
		qual = grasp.oracle_eval("data/new_barclamp.obj", oracle_method="pytorch", renderer=r)
		assert torch.sub(qual, batch_qual[i]).item() < EPS

if __name__ == "__main__":

	# test_trans_world_to_im()
	# test_select_grasp()
	# test_save_and_load_grasps()
	# test_batching()
	# test_slicing()
	# test_oracle_check(1)
	# test_random_grasps()
	# generate_grasp_dataset("grasp-dataset")
	# vis_rand_grasps()
	# test_pytorch_oracle()
	# test_select_grasp_pytorch()
 
	# visualize grasp-batch.json
	r = Renderer()
	g = Grasp.sample_grasps("data/new_barclamp.obj", 10, r, min_qual=0.002, max_qual=0.005)
	print(g.quality)
	# g.trans_world_to_im(camera=r.camera)
	# g.vis_grasp_dataset(obj_file="data/new_barclamp.obj", directory="grasp-dataset2", renderer=r)
