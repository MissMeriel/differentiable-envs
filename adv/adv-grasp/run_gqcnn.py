import os
import json
import logging
import numpy as np
import torch
import pytorch3d
from tqdm import tqdm
from pytorch3d.io import save_obj
from pytorch3d.loss import mesh_edge_loss, mesh_normal_consistency, mesh_laplacian_smoothing
from enum import Enum


from pytorch3d_ext import Renderer, mesh_properties
from gqcnn_pytorch import KitModel
import matplotlib.pyplot as plt
from grasp import GraspTorch
import select_grasp as sg
import quality_fork as quality
import math
from quality_fork import (
    GQCNNQualityFunction, 
    RobustCannyFerrariQualityFunction,
    RobustMinWeightQualityFunction,
    minWeightQualityFunction,
    CannyFerrariQualityFunction,
    pytorch_setup
)
import imageio

SHARED_DIR = "/home/hmitchell/pytorch3d/dex_shared_dir"

class AttackMethod(Enum):
	RANDOM_FUZZ = 1
	GQCNN_DOWN = 2
	GQCNN_CF_DIFF_NO_CF_GRAD = 3
	GQCNN_CF_DIFF = 4
	GQCNN_DOWN_CF_UP = 5
	GQCNN_UP_CF_DOWN = 6
	GQCNN_MINWEIGHT_DIFF = 7
	GQCNN_DOWN_MINWEIGHT_UP = 8
	GQCNN_UP_MINWEIGHT_DOWN = 9
	NO_GQCNN_GRAD = 10
	SELF_COLLISION_UP = 11
	GQCNN_UP = 12
	CF_UP = 13
	CF_DOWN = 14
	MW_UP = 15
	MW_DOWN = 16
	

class Attack: 

	# SET UP LOGGING
	logger = logging.getLogger('run_gqcnn')
	logger.setLevel(logging.INFO)
	if not logger.handlers:
		ch = logging.StreamHandler()
		ch.setLevel(logging.INFO)
		formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		ch.setFormatter(formatter)
		logger.addHandler(ch)

	def __init__(self, num_plots=0, steps_per_plot=0, model=None, renderer=None, oracle_method="pytorch"):
		self.num_plots = num_plots
		self.steps_per_plot = steps_per_plot
		self.num_steps = num_plots * steps_per_plot

		self.model = model
		if model == None:
			model = KitModel("weights.npy")
			model.eval()
			self.model = model

		self.renderer = renderer
		if renderer == None:
			self.renderer = Renderer()

		self.oracle_method = oracle_method
		if oracle_method not in ["dexnet", "pytorch"]:
			Attack.logger.debug("Attack oracle evaluation method must be 'dexnet' or 'pytorch' -- defaulting to 'pytorch'.")
			self.oracle_method = "pytorch"

		# self.loss_weights = {
		# 	"edge": 5,
		# 	"normal": 0.4,
		# 	"smooth": 5
		# }

		self.loss_alpha = None
		self.losses = None
		self.track_qual = None
		self.attack_method = None

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

		# check for bad output from Grasp.extract_tensors
		if pose_tensor==None or image_tensor==None:
			Attack.logger.error("Pose tensor and/or image tensor input is bad.")
			return None

		return self.model(pose_tensor, image_tensor)

	@staticmethod
	def scale_oracle(x):
		"""Scale the oracle quality to be more like the model prediction"""
		return 1 / (1 + torch.exp(-2500 * (x - torch.tensor(0.001))))

	@staticmethod
	def plot_losses(losses, dir):
		fig = plt.figure(figsize=(7, 5))
		ax = fig.gca()
		ax2 = ax.twinx()
		ax2.plot(losses['dist'], label="dist loss",color='tab:blue')
		index = [0] + (np.nonzero(np.diff(losses['index']))[0]+1).tolist() + [len(losses['index'])-1]
		ax.vlines(index,ymin=0,ymax=1,color='tab:green')
		ax.plot(losses['prediction'], label="prediction loss",color='tab:red')
		ax.set_xlabel("Iteration", fontsize="16")
		ax.set_ylabel("Loss", fontsize="16")
		ax2.set_ylabel("Dist", fontsize="16")
		ax.set_title("Loss vs iterations", fontsize="18")
		plt.savefig(dir+"losses.png")
		plt.close()

	@staticmethod
	def plot_qual(losses, dir):
		fig = plt.figure(figsize=(7, 5))
		ax = fig.gca()
		for k, l in losses.items():
			ax.plot(l, label=k)
		ax.axhline(y=0.5, color='r', linestyle='--', label='threshold')
		ax.legend(fontsize="16")
		ax.set_xlabel("Iteration", fontsize="16")
		ax.set_ylabel("Quality (Scaled)", fontsize="16")
		ax.set_title("Oracle and GQCNN Quality Over Time", fontsize="18")
		#ax.set_ylim(0, 1.1)
		plt.savefig(dir+"qual.png")
		plt.close()

	def calc_loss(self, adv_mesh, grasp, method, index):
		"""
		Calculates loss for the adversarial attack to maximize difference between oracle and model prediction

		Parameters
		----------
		adv_mesh:
		grasp:
		index:
		Returns
		-------
		Tensor: calculated loss
		float: "is close" boolean comparing two qualities
		Tensor: component of loss from self collision distance
		Tensor: component of loss from adv attack
		"""
		if  isinstance(method, AttackMethod):
			method = [method]
		method = set(method)

		adv_mesh_clone = adv_mesh.clone()
		dim = self.renderer.mesh_to_depth_im(adv_mesh_clone, display=False)
		pose, image = quality.GQCNNQualityFunction.extract_tensors_batch(grasp=grasp,d_ims=dim)
		out = self.run(pose, image)
		cur_pred = out[:,1:].to(adv_mesh.device)
		eps_check = False

		dist_loss = self.eval_self_collision_dist(grasp, adv_mesh_clone)
		oracle_pred = self.oracle_eval(grasp, adv_mesh_clone, renderer=self.renderer)

		# initialize to 0, so we can optionally add to it
		adv_loss = torch.zeros([len(adv_mesh),grasp.num_grasps()],device=adv_mesh.device)
		minweight_pred = self.oracle_eval(grasp, adv_mesh_clone, renderer=self.renderer,mode='minweight')

		loss_dict = {}

		if AttackMethod.SELF_COLLISION_UP in method:
			loss_dict[AttackMethod.SELF_COLLISION_UP] = dist_loss
		
		if AttackMethod.CF_UP in method:
			loss_dict[AttackMethod.CF_UP] = -oracle_pred
		
		if AttackMethod.CF_DOWN in method:
			loss_dict[AttackMethod.CF_DOWN] = oracle_pred

		if AttackMethod.MW_UP in method:
			loss_dict[AttackMethod.MW_UP] =  -minweight_pred
		
		if AttackMethod.MW_DOWN in method:
			loss_dict[AttackMethod.MW_DOWN] =  minweight_pred

		if AttackMethod.NO_GQCNN_GRAD in method:
			cur_pred= cur_pred.detach()

		# NO ORACLE
		if AttackMethod.GQCNN_DOWN in method:
			loss_dict[AttackMethod.GQCNN_DOWN] = cur_pred
		
		if AttackMethod.GQCNN_UP in method:
			loss_dict[AttackMethod.GQCNN_UP] = -cur_pred

		# NO ORACLE GRADIENT - check current model prediction
		elif AttackMethod.GQCNN_CF_DIFF_NO_CF_GRAD in method:
			other_methods = method - {AttackMethod.GQCNN_CF_DIFF_NO_CF_GRAD}
			if len(other_methods) > 0:
				tqdm.write(f'NO_ORACLE_GRAD may have unknown effects alongside {AttackMethod.GQCNN_CF_DIFF_NO_CF_GRAD.name}')
			oracle_pred = oracle_pred.detach()
			abs_diff = torch.abs(torch.sub(cur_pred, oracle_pred))
			adv_loss = torch.sub(1.0, abs_diff)
			if self.epsilon is not None and abs_diff >= self.epsilon:
				eps_check = True
	
		# ORACLE GRADIENT
		if len(method & { AttackMethod.GQCNN_CF_DIFF,  AttackMethod.GQCNN_UP_CF_DOWN,  AttackMethod.GQCNN_DOWN_CF_UP}) > 0:

			# since there is a threshold in robust canny-ferrari, we need
			# a two different modes to get the gradient we want
			GQCNN_DOWN_CF_UP = oracle_pred -0.5 - cur_pred
			GQCNN_UP_CF_DOWN = cur_pred - oracle_pred +0.5
			if AttackMethod.GQCNN_CF_DIFF in method and oracle_pred > cur_pred:
				loss_dict[AttackMethod.GQCNN_CF_DIFF]= -GQCNN_DOWN_CF_UP
			else :
				loss_dict[AttackMethod.GQCNN_CF_DIFF]= -GQCNN_UP_CF_DOWN
			loss_dict[AttackMethod.GQCNN_DOWN_CF_UP] = -GQCNN_DOWN_CF_UP
			loss_dict[AttackMethod.GQCNN_UP_CF_DOWN] = -GQCNN_UP_CF_DOWN

			if self.epsilon is not None and abs_diff >= self.epsilon:
				eps_check = True


		self.losses["prediction"].append(adv_loss.item())


		self.losses["dist"].append((dist_loss/10000).item())
		self.losses["index"].append(index)
		self.track_qual["gqcnn prediction"].append(cur_pred.item())
		self.track_qual["oracle quality"].append(oracle_pred.item())
		self.track_qual["minWeight quality"].append(minweight_pred.item())
		self.track_qual["dist loss"].append(dist_loss.item())
		self.track_qual["did hit object"].append(torch.any(grasp.contact_mask.flatten()).item())
		#self.track_qual["index"].append(index)

		return loss_dict, eps_check

	def perturb(self, mesh, param, grasp:GraspTorch, method, index):
		"""
		Perturb the mesh for the adversarial attack

		Parameters
		----------
		mesh: pytorch3d.structures.Meshes
			Mesh to be perturbed
		param: torch.tensor
		index: outer loop index for storing with loss to track successful steps
		grasp:
		Returns
		-------
		float: loss, pytorch.3d.structures.Meshes: adv_mesh
			loss: loss calculated by calc_loss
			adv_mesh: perturbed mesh structure
		"""
  
		# NO ORACLE / NO ORACLE GRADIENT / ORACLE GRADIENT - perturb vertices
		if param.shape[-1] == 4:
			# update grasp as 2D
			angle = grasp.im_angle + param[...,2]
			depth = grasp.depth + param[...,3]
			im_center = grasp.im_center + param[...,:2]
			grasp = GraspTorch(angle=angle, depth=depth,im_center=im_center, friction_coef=grasp.friction_coef,
					  num_cone_faces=grasp.num_cone_faces, torque_scaling=grasp.torque_scaling, camera_intr=grasp.camera_intr)
			adv_mesh = mesh
			updated_obj = grasp
		elif param.shape[-1] == 6:
			# update grasp as 3D
			axis3D = torch.nn.functional.normalize(grasp.world_axis + param[...,:3],dim=-1)
			center3D = grasp.world_center + param[...,3:]
			grasp = GraspTorch(world_axis=axis3D, world_center=center3D, friction_coef=grasp.friction_coef,
					  num_cone_faces=grasp.num_cone_faces, torque_scaling=grasp.torque_scaling, camera_intr=grasp.camera_intr)
			adv_mesh = mesh
			updated_obj = grasp
		elif param.shape[-1] == 3:
			# update mesh vertices
			adv_mesh = mesh.offset_verts(param)
			updated_obj = adv_mesh

		grasp.make2D(camera_intr=self.renderer.camera)
		grasp = grasp.apply_to_mesh(adv_mesh, is_watertight=self.mesh_properties.is_watertight, 
                                    is_inverted=self.mesh_properties.is_inverted, use_dexnet_normal=False)

		
		loss, eps_check = self.calc_loss(adv_mesh, grasp, method, index)

		return loss, updated_obj, eps_check 

	def attack_setup(self, dir, logfile, grasp, method):
		"""Set up attack by saving info and resetting losses"""

		# save attack info
		if not os.path.exists(dir):
			os.makedirs(dir)
		grasp.save(dir+"grasp.json")
		if isinstance(method, AttackMethod):
			method_string = method.name
		else:
			method_string = str([m.name for m in method])
		data = {
			"learning_rate": self.learning_rate,
			"momentum": self.momentum,
			"optimizer": "SGD",
			"loss weight (alpha)": self.loss_alpha, 
			"method": method_string
		}
		with open(logfile, "w") as f:
			json.dump(data, f, indent=4)

		# reset loss tracking to plot at the end
		self.losses = {
			"prediction": [],
			"dist": [],
			"index": []
			# "edge": [],
			# "normal": [],
			# "smoothing": []
		}

		self.track_qual = {
			"gqcnn prediction": [],
			"oracle quality": [],
			"minWeight quality": [],
			"dist loss": [],
			"did hit object": []
			#"index": []
		}

	def attack_random_fuzz(self, mesh, grasp, dir, lr):
		"""Run an attack on the model using random mesh perturbations with step size of lr"""
		Attack.logger.info("Running random perturbation attack!")

		param = torch.zeros(mesh.verts_packed().shape, device=mesh.device, requires_grad=True)
		adv_mesh = mesh.clone()

		# ADVERSARIAL LOOP
		for i in range(1, self.num_steps):

			# perturb
			v = torch.normal(0.0, 1.0, param.shape).to(param.device)
			v = torch.nn.functional.normalize(v)
			random_step = self.learning_rate * v
			adv_mesh = mesh.offset_verts(random_step)

			if i % self.steps_per_plot == 0:
				# snapshot
				mesh2 = adv_mesh.clone()
				self.snapshot(mesh=mesh2, grasp=grasp, dir=dir, iteration=i, orig_pdim=self.orig_pdim, logfile=self.logfile, method=AttackMethod.RANDOM_FUZZ)

		# plot qualities over time
		Attack.plot_qual(self.track_qual, dir)

		# save final object
		final_mesh = mesh.offset_verts(param)
		self.snapshot(mesh=final_mesh, grasp=grasp, dir=dir, iteration=i+1, orig_pdim=self.orig_pdim, logfile=self.logfile, method=AttackMethod.RANDOM_FUZZ, save_mesh=True)

		return final_mesh

	def attack(self, mesh, grasp, dir, lr, momentum=None, loss_alpha=None, method=AttackMethod.GQCNN_CF_DIFF, epsilon=None):
		"""
		Run an attack on the model for number of steps specified in self.num_steps

		Parameters
		----------
		mesh: pytorch3d.structures.Meshes
			Mesh to perturb for adversarial attack
		grasp: Grasp 
		dir: String
			Directory to which to save results
		lr: int
			Learning rate passed to optimizer or step size
		momentum: float
			Momentum passed to optimizer
		loss_alpha: float
		method: MethodAttack
			Type of attack to run
		epsilon: float
			When the difference between the model prediction and oracle evaluation reaches or surpasses epsilon, exit adversarial loop.
			Only applicable for AttackMethods ORACLE_GRAD and NO_ORACLE_GRAD. 
		Returns
		-------
		pytorch3d.structures.Meshes: adv_mesh
			Final adversarial mesh
		"""
		if isinstance(method, AttackMethod):
			Attack.logger.info(f"Running attack with method {method.name}!")
		else:
			Attack.logger.info(f"Running attack with methods {[m.name for m in method]}!")

		#self.mesh_properties = torch.jit.script(mesh_properties(mesh)) Not currently faster
		self.mesh_properties = mesh_properties(mesh)
		# set attack parameters
		self.learning_rate = lr
		self.momentum = momentum
		self.loss_alpha = loss_alpha
		if epsilon is not None and ((method == AttackMethod.GQCNN_CF_DIFF_NO_CF_GRAD) or (method == AttackMethod.GQCNN_CF_DIFF)):
			self.epsilon = epsilon
		else:
			self.epsilon = None

		# set up attack: save info + reset losses
		grasp.c0, grasp.c1 = None, None 	# don't use contact information bc perturbations will change this
		if dir[-1] != "/": dir = dir+"/"
		logfile = dir+"logfile.txt"
		self.attack_setup(dir=dir, logfile=logfile, grasp=grasp, method=method)
		dim = self.renderer.mesh_to_depth_im(mesh, display=False)
		_, orig_pdim = quality.GQCNNQualityFunction.extract_tensors_batch(grasp=grasp,d_ims=dim)
		self.orig_pdim = orig_pdim
		self.logfile = logfile

		# snapshot
		self.snapshot(mesh=mesh, grasp=grasp, dir=dir, iteration=0, orig_pdim=orig_pdim, logfile=logfile, method=method)

		if method == AttackMethod.RANDOM_FUZZ:
			return self.attack_random_fuzz(mesh, grasp, dir, lr)
		
		optimize_mesh = True
		if optimize_mesh:
			param = torch.zeros(mesh.verts_packed().shape, device=mesh.device, requires_grad=True)
		else:
			# TODO, options for size 4
			# Make this work, currently no gradient computed
			param = torch.zeros(list(grasp.world_center[...,0].shape) + [6], device=mesh.device, requires_grad=True)

		# option to use steps of fixed size, inspired by fgsm (https://pytorch.org/tutorials/beginner/fgsm_tutorial.html)
		# otherwise, will use SGD and scale by gradient magnitude
		use_fixed_step = True

		if not use_fixed_step:
			optimizer = torch.optim.SGD([param], lr=lr, momentum=momentum)

		perturbed_by_param = mesh.clone()

		# initialize stack of known feasible meshes, for reverting during optimization
		param_safe=[param.detach().clone()]
		
		if  hasattr(grasp,'reference_dist'):
			delattr(grasp,'reference_dist')

		collision_loss_scale =  1e4
		collision_loss_sat = 1
		backoff = 2
		prune_likely_collision_gradients = True

		self_collision_index = None
		if AttackMethod.SELF_COLLISION_UP in method:
			self_collision_index = method.index(AttackMethod.SELF_COLLISION_UP)
			other_indices = [i for i in range(len(method)) if i != self_collision_index]
		else:
			other_indices = range(len(method))
		even_weight = 1/len(other_indices)

		# ratio between start and end when doing exponential backoff. 2 will take average, 10 will reduce step by 90%
		
		loss_last = 0
		# ADVERSARIAL LOOP
		# torch.autograd.set_detect_anomaly(True)
		resets = 0
		best_loss = float('inf')
		self.qual_measures_raw = []
		self.loss_mag = []
		self.update_scale = []
		self.optim_status = []
		attack_failed = None
		for i in tqdm(range(1, self.num_steps),desc='attack outer iterations',leave=False):
			attempts = 0 # used to decide whether to revert to previous mesh in param_safe
			if attack_failed is not None:
				break
			if not use_fixed_step:
				optimizer.zero_grad()
			while True:
				fail = False
				param_updated = False
				self.param_grad_list = []
				if len(self.loss_mag) > 0:
					# drop gradient information for last iteration, can release graph
					self.loss_mag[-1] = self.loss_mag[-1].detach()
				self.loss_mag.append(torch.zeros((1,len(method)), device=mesh.device))
				self.update_scale.append(torch.zeros((1,len(method)), device=mesh.device))
				self.optim_status.append(np.array([[i, len(param_safe), attempts, resets]]))
				loss_dict, perturbed_by_param, eps_check = self.perturb(mesh, param, grasp, method, i)

				for ind, method_type in enumerate(method):
					self.loss_mag[-1][0,ind] = loss_dict[method_type]
					
					if torch.isfinite(self.loss_mag[-1][0,ind]).item():
						param_back = None # clear local cache variable
						try:
							self.loss_mag[-1][0,ind].backward(retain_graph=True)
							if param.grad is not None:
								if torch.all(torch.isfinite(param.grad).flatten()).item():
									self.param_grad_list.append(param.grad.detach().clone())
									param.grad.zero_()
								else:
									tqdm.write(f'found inf grad value {method_type} ind {ind} val {self.loss_mag[-1]}')
									fail = True or fail
									break
							else:
								tqdm.write(f'found None grad value {method_type} ind {ind} val {self.loss_mag[-1]}')
								self.param_grad_list.append(torch.zeros_like(param))

						except Exception as e:
							tqdm.write(method_type)
							tqdm.write(e)
							fail=True or fail
							break
					else:
						tqdm.write(method_type)
						tqdm.write('loss mag not finite')
						fail=True or fail
						break
				# check that both loss AND its derivative are finite, otherwise triggering backoff
				if not fail:
					param_back = None # clear local cache variable
					try:
						
						self.update_scale[-1][0,other_indices] = even_weight
						param = param.detach()
						
						if self_collision_index is not None:
							self.update_scale[-1][0,self_collision_index] = torch.minimum(self.loss_mag[-1][0,self_collision_index].detach() / collision_loss_scale, 
														  torch.ones_like(self.loss_mag[-1][0,self_collision_index])*collision_loss_sat)
							# 
							# 
							self.update_scale[-1] = torch.nn.functional.normalize(self.update_scale[-1]) 
							grad_other = torch.zeros_like(param)
							update_scale = torch.zeros_like(param[0][0])
							for grad_ind in other_indices:
								grad_other += self.param_grad_list[grad_ind]
								grad_other += torch.nn.functional.normalize(self.param_grad_list[grad_ind].flatten(),dim=0).reshape(param.shape)
								update_scale += self.update_scale[-1][0,grad_ind]
							if prune_likely_collision_gradients:
								collision_dirs = torch.linalg.vecdot(grad_other,self.param_grad_list[self_collision_index]) < 0
								grad_other[collision_dirs]  = 0
							grad_other = torch.nn.functional.normalize(grad_other.flatten(),dim=0).reshape(grad_other.shape)
							param -=  lr * grad_other.detach() * update_scale.detach()
							if self.loss_mag[-1][0,self_collision_index] != 0:
								param -=  lr * torch.nn.functional.normalize(self.param_grad_list[self_collision_index].detach().flatten(),dim=0).reshape(grad_other.shape) * self.update_scale[-1][0,self_collision_index].detach()
						else:
							self.update_scale[-1] = torch.nn.functional.normalize(self.update_scale[-1]) 
							for method_index in range(len(method)):
								param -= lr * torch.nn.functional.normalize(self.param_grad_list[method_index].detach().flatten(),dim=0).reshape(param.shape) * self.update_scale[-1][0,method_index].detach() 
							
						tqdm.write(f'i_loop {i} o_steps {len(param_safe)} methods: {[m.name for m in method]} loss: {self.loss_mag[-1].numpy(force=True)} update scale: {(self.update_scale[-1]).numpy(force=True)}')
						param = param.detach()
						param.requires_grad_(requires_grad=True)
						param_updated = True

					except Exception as e:
						tqdm.write(method_type)
						tqdm.write(e)
						fail=True
						pass
					else:
						# and torch.any((param != param_back).flatten()).item()
						if not fail and torch.all(torch.isfinite(param).flatten()).item():
							loss_last = self.loss_mag[-1][0,ind].item()
							param_safe.append(param.detach().clone())
							if resets > 0:
								resets -= 0.5
							# no infs, we can compute loss
							break
						

				tqdm.write(f'reducing param diff by {backoff}, try {attempts} reset counter {resets}')
				if attempts > 3 or not param_updated or not torch.all(torch.isfinite(param).flatten()).item():
					if resets > 3:
						tqdm.write(f'ending after reset {resets} times in a row, unlikely to recover')
						if i <= 3:
							attack_failed = f'{i} ending after reset {resets} times in a row, unlikely to recover'
						break
					if param_updated or resets < 1:
						resets += 1
					attempts = 0
					for _ in range(math.floor(resets)):
						param = param_safe.pop(-1)
					tqdm.write(f'resetting with {len(param_safe)+1}')
				attempts += 1
				# apply backoff ratio
				if len(param_safe) == 0:
					tqdm.write('attempting to backoff, but param empty')
					attack_failed = f'{i} attempting to backoff, but param empty'
					break
				param = ((param + (param_safe[-1]*(backoff-1)))/backoff).detach()
				param.requires_grad_(requires_grad=True)
				if not use_fixed_step:
					optimizer = torch.optim.SGD([param], lr=lr, momentum=momentum)
					optimizer.zero_grad()


			curr_loss = torch.sum(self.loss_mag[-1][0,other_indices]).item()
			self.qual_measures_raw.append(torch.zeros((1,5), device=mesh.device))
			self.qual_measures_raw[-1][0,0],self.qual_measures_raw[-1][0,1],self.qual_measures_raw[-1][0,2],self.qual_measures_raw[-1][0,3],self.qual_measures_raw[-1][0,4] = self.track_qual["gqcnn prediction"][-1],self.track_qual["minWeight quality"][-1], self.track_qual["oracle quality"][-1], self.track_qual["dist loss"][-1], self.track_qual["did hit object"][-1]


			if (i % self.steps_per_plot == 0) or (i+1 == self.num_steps):
				# snapshot
				mesh2 = perturbed_by_param.clone()
				self.snapshot(mesh=mesh2, grasp=grasp, dir=dir, iteration=i, orig_pdim=orig_pdim, logfile=logfile, method=method)
			
			if curr_loss < best_loss:
				best_loss = curr_loss
				mesh2 = perturbed_by_param.clone()
				self.snapshot(mesh=mesh2, grasp=grasp, dir=dir, iteration=i, orig_pdim=orig_pdim, logfile=logfile, method=method,name='best')
			if eps_check:
				# model prediction and oracle evaluation are sufficiently different, break loop
				if i % self.steps_per_plot != 0:
					mesh2 = perturbed_by_param.clone()
					self.snapshot(mesh=mesh2, grasp=grasp, dir=dir, iteration=i, orig_pdim=orig_pdim, logfile=logfile, method=method)
				break

		# plot losses/qualities over time
		Attack.plot_losses(self.losses, dir)
		Attack.plot_qual(self.track_qual, dir)

		# save final object
		final_mesh = mesh.offset_verts(param)
		self.snapshot(mesh=final_mesh, grasp=grasp, dir=dir, iteration=i+1, orig_pdim=orig_pdim, logfile=logfile, method=method, save_mesh=True)
		delattr(self, 'param_grad_list')
		return final_mesh, attack_failed

	def snapshot(self, mesh, grasp, dir, iteration, orig_pdim, logfile, method, save_mesh=False, name=None):
		"""Save information at current point in attack."""
		# TODO: Implement for batch of grasps in attack

		# store other numpy tensors in the grasp
		other_dict = {}
		if name is None and hasattr(self, 'param_grad_list') and len(self.param_grad_list) > 0 and len(self.qual_measures_raw) > 0:			
			other_dict['param_grad'] = torch.cat([grad.unsqueeze(0) for grad in self.param_grad_list],dim=0).numpy(force=True)
			other_dict['qual_measures_raw'] = torch.cat(self.qual_measures_raw, dim=0).numpy(force=True)
			other_dict['loss_mag'] = torch.cat(self.loss_mag, dim=0).numpy(force=True)
			other_dict['gradient_update_scale'] = torch.cat(self.update_scale, dim=0).numpy(force=True)
			other_dict['optim_status'] = np.concatenate(self.optim_status, axis=0)


			self.qual_measures_raw = []
			self.loss_mag = []
			self.update_scale = []
			self.optim_status = []

		if name is None:
			name = f'it-{iteration}'

		# different mesh volumes for considering change
		vol_mesh, vol_bounding, vol_hull = mesh_properties.get_volumes(mesh)
		mesh_file = dir + f"{name}.obj"
		grasp_file = dir + f"{name}-grasp.obj"
		quality_files = (dir + f"{name}-cf.obj", dir + f"{name}-mw.obj")
		mat_file = dir + f"{name}-grasp.mat"
		grasp = grasp.apply_to_mesh(mesh, is_watertight=self.mesh_properties.is_watertight, 
                                    is_inverted=self.mesh_properties.is_inverted, use_dexnet_normal=False)


		self.write_grasp_pytorch(grasp=grasp, obj=mesh, renderer=self.renderer, path=grasp_file, quality_path=quality_files)
		save_obj(mesh_file, verts=mesh.verts_list()[0], faces=mesh.faces_list()[0])
		image = self.renderer.render_mesh(mesh, display=False)

		dim = self.renderer.mesh_to_depth_im(mesh, display=False)
		pose, processed_dim = quality.GQCNNQualityFunction.extract_tensors_batch(grasp=grasp,d_ims=dim)
		other_dict['model_pred'] = self.run(pose, processed_dim)[:,1:].detach().item()
		other_dict['minWeight'] = self.oracle_eval(grasp=grasp,obj_file=mesh, renderer=self.renderer, grad=True, write_path=mat_file,mode='minweight').detach().item()
		oracle_qual = self.oracle_eval(grasp=grasp,obj_file=mesh, renderer=self.renderer, grad=True, write_path=mat_file,other_dict=other_dict).detach().item()

		if orig_pdim is not None: depth_diff = orig_pdim - processed_dim

		# RANDOM FUZZ ONLY
		if method == AttackMethod.RANDOM_FUZZ:
			self.track_qual["gqcnn prediction"].append(other_dict['model_pred'])
			self.track_qual["minWeight quality"].append(other_dict['minWeight'])
			self.track_qual["oracle quality"].append(oracle_qual)
			self.track_qual["dist loss"].append(0)
			self.track_qual["did hit object"].append(torch.any(grasp.contact_mask.flatten()).item())

		title = f"Iteration {iteration}: oracle quality {oracle_qual:.4f}, mw prediction {other_dict['minWeight']:.4f}, gqcnn prediction {other_dict['model_pred']:.4f}\n volumes: mesh {vol_mesh}, bb {vol_bounding}, hull {vol_hull}"
		fname = dir + name + ".png"
		image = image.squeeze(0)
		image = image[140:340, 270:370, :]
		self.renderer.display(images=[image, processed_dim, depth_diff], shape=(1,3), title=title, save=fname)
		
		# store renders for creating a .gif animation
		if hasattr(self, 'renders_list'):
			self.renders_list += [imageio.imread(fname)]
		else:
			self.renders_list = [imageio.imread(fname)]
		tqdm.write(f"save: {fname}")

		if save_mesh:
			imageio.mimsave(dir + f"it.gif", self.renders_list,fps=1,loop=10000)
			self.renders_list = []


		# if not save_mesh:
		# 	os.remove(mesh_file)

		# add info to logfile
		data = {
			"iteration": iteration,
			"model prediction": other_dict['model_pred'],
			"min weight raw": other_dict['minWeight'] ,
			"oracle quality raw": oracle_qual,
		}
		with open(logfile, "a") as f:
			json.dump(data, f, indent=4)
			

	def oracle_eval(self, grasp, obj_file, oracle_method=None, robust=True, renderer=None, grad=True, mode='actual',write_path=None, other_dict={}):
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
		if oracle_method == "dexnet" or (hasattr(grasp, 'oracle_method') and (grasp.oracle_method == "dexnet")):
			if isinstance(robust, bool):
				return oracle_eval_dexnet(obj_file, robust=robust)
			else:
				return oracle_eval_dexnet(obj_file, robust=grasp.oracle_robust)
		
		else:
			if not renderer:
				GraspTorch.logger.error("oracle_eval - pytorch oracle requires renderer argument")
			else:
				return self.oracle_eval_pytorch(grasp, obj_file, renderer, grad=grad, robust=robust, mode=mode, write_path=write_path, other_dict=other_dict)

	def oracle_eval_dexnet(self, obj_file, robust=True):
		"""
		Get a final oracle evaluation of a mesh object via remote call to docker container.

		Refer to `oracle_eval` method documentation for details on parameters and return values.
		"""

		tqdm.write("oracle_eval_dexnet robust:", robust)

		# check if object file is already saved in shared directory and copy there if not
		if not os.path.isfile(obj_file):
			GraspTorch.logger.error("Object for oracle evaluation does not exist.")
			return None

		obj_name = obj_file.split("/")[-1]
		obj_shared = SHARED_DIR + "/" + obj_name
		if not os.path.isfile(obj_shared):
			GraspTorch.logger.info("Saving object file to shared directory (%s) for oracle evaluation", SHARED_DIR)
			shutil.copyfile(obj_file, obj_shared)

		# save grasp info for oracle evaluation
		Pyro4.config.COMMTIMEOUT = None
		server = Pyro4.Proxy("PYRO:Server@localhost:5000")
		# calc_axis = (self.c1 - self.c0) / torch.linalg.norm((self.c1 - self.c0))	# ensure grasp axis normalized
		# calc_axes = (self.c1 - self.c0) / torch.linalg.norm((self.c1 - self.c0), dim=1).unsqueeze(1)
		calc_axes = self.world_axis / torch.norm(self.world_axis, dim=-1, keepdim=True)
		save_nparr(self.world_center.detach().cpu().numpy(), "temp_center.npy")
		save_nparr(calc_axes.detach().cpu().numpy(), "temp_axis.npy")
		# tqdm.write("\nobj_name:", type(obj_name), obj_name, "\n\n")
		results = server.final_evals("temp_center.npy", "temp_axis.npy", obj_name, robust=robust)

		# update quality info
		self.quality = torch.from_numpy(np.array(results)).unsqueeze(1).to(self.world_axis.device).float()

		return self.quality

	def write_grasp_pytorch(self, grasp, obj, renderer, path, quality_path=None, other_dict={}):
		"""
			Save a colored mesh of the grasp points for the current object using pytorch version of dexnet library
		"""
		if (grasp.world_center == None or grasp.world_axis == None):
			GraspTorch.logger.error("Grasp.oracle_eval_pytorch requires Grasp to have world_center and world_axis.")
			return None

		config_dict = {
			"torque_scaling":1000,
			"soft_fingers":1,
			"friction_coef": 0.8,
			"antipodality_pctile": 1.0 
		}

		if isinstance(obj, pytorch3d.structures.Meshes): mesh = obj
		else: mesh, _ = renderer.render_object(obj, display=False)

		if grasp.num_grasps() == 1:
			grasp.quality = torch.zeros_like(grasp.depth)
			# for i, g in enumerate(self):
			g = grasp[0]
			if g.contact_points is not None: contact_points = g.contact_points
			else: contact_points = None
			
			width = torch.tensor([[0.05]], device=mesh.device)
			# grasp_torch = GraspTorch(world_center=g.world_center, world_axis=g.world_axis, width=width, camera_intr=renderer.rasterizer.cameras, friction_coef=config_dict["friction_coef"], torque_scaling=config_dict["torque_scaling"])

			# grasp_torch = grasp_torch.apply_to_mesh(mesh,is_watertight=grasp.mesh_properties.is_watertight, is_inverted=grasp.mesh_properties.is_inverted)

			g.write_obj(path, include_coordinate=True, include_line_o_action=True)

			if quality_path is not None:

				com_qual_func = CannyFerrariQualityFunction(config_dict)
				com_qual_func.quality(mesh, grasp)
				com_qual_func.write_obj(quality_path[0])

				com_qual_func = minWeightQualityFunction(config_dict)
				com_qual_func.quality(mesh, grasp)
				com_qual_func.write_obj(quality_path[1])


	def oracle_eval_pytorch(self, grasp, obj, renderer, grad=True, robust=True, mode='actual',write_path=None, other_dict={}):
		"""
		Get a final oracle evaluation of a mesh object via local pytorch oracle implementation.

		Refer to `oracle_eval` method documentation for details on parameters and return values.
		"""
		if mode == 'minweight':
			if robust:
				qual_class = RobustMinWeightQualityFunction
			else:
				qual_class = minWeightQualityFunction
		else:
			if robust:
				qual_class = RobustCannyFerrariQualityFunction
			else:
				qual_class = CannyFerrariQualityFunction



		if (grasp.world_center == None or grasp.world_axis == None):
			GraspTorch.logger.error("Grasp.oracle_eval_pytorch requires Grasp to have world_center and world_axis.")
			return None

		config_dict = {
			"torque_scaling":1000,
			"soft_fingers":1,
			"friction_coef": 0.8,
			"antipodality_pctile": 1.0 
		}

		if isinstance(obj, pytorch3d.structures.Meshes): mesh = obj
		else: mesh, _ = renderer.render_object(obj, display=False)

		if grasp.num_grasps() > 1:
			grasp.quality = torch.zeros_like(grasp.world_axis[...,0:1])
			for i, g in tqdm(enumerate(grasp),desc='grasps'):
				
				try:
					com_qual_func = qual_class(config_dict)
					quality = com_qual_func.quality(mesh, g,
										is_watertight=self.mesh_properties.is_watertight, is_inverted=self.mesh_properties.is_inverted).float().item()

				except QhullError:
					quality =  0.0

				g.quality[i] = quality

		else:

			try:
				com_qual_func = qual_class(config_dict)
				grasp.quality = com_qual_func.quality(mesh, grasp, 
										is_watertight=self.mesh_properties.is_watertight, is_inverted=self.mesh_properties.is_inverted).float().to(grasp.world_center.device)
			except QhullError:
				grasp.quality = torch.tensor([0.0]).to(grasp.device)

		if grad:
			grasp.quality.requires_grad_(True)
		else:
			grasp.quality.requires_grad_(False)

		if write_path is not None:
			com_qual_func.savemat(write_path, other_items=other_dict)

		return grasp.quality

	def eval_self_collision_dist(self,grasp, obj):
		# computes a self collision loss, either comparing minimum distance directly OR computing the distance
		# energy, https://www.cs.cmu.edu/~kmcrane/Projects/RepulsiveShells/index.html
		
		# configuration. TODO, test and maybe expose
		use_energy = False
		use_all_in_min = True
		# only process full vector if needed
		verts = obj.verts_packed()
		verts.requires_grad_(True)
		all_dist, coords = self.mesh_properties(verts)
		all_dist = mesh_properties.make_invalid_dist_inf(all_dist, coords)

		min_dist = torch.min(all_dist)
		
		# # DEBUG TODO REMOVE
		# self.last_all_dist = all_dist.numpy(force=True)
		# self.last_coords = coords.numpy(force=True)
		# store a reference distance if it doesn't exist, should be first time we reach here per grasp
		if not hasattr(grasp,'reference_dist'):
			if use_energy:
				grasp.reference_dist = torch.sum(torch.log(all_dist[torch.isfinite(all_dist)])).detach().item()
			else:
				grasp.reference_dist = min_dist.item()

		if use_energy:
			# repulsive shell formula
			dist_loss = torch.square(grasp.reference_dist - torch.sum(torch.log(all_dist[torch.isfinite(all_dist)])))
		else:
			if use_all_in_min:
				# allows for us to have a distance threshold, above which we ignore in gradient
				relevant_dist = all_dist[torch.logical_and(torch.isfinite(all_dist), all_dist < 4*grasp.reference_dist)]
	#				clamp_dist = torch.clamp(all_dist[torch.isfinite(all_dist)], max=2*self.reference_dist)
				# use reference, if valid in barier. Use min/10 if we're closer than reference 
				if min_dist < grasp.reference_dist/10:
					tqdm.write('inside outer boundary region')
					# dist_loss = -torch.sum(torch.log(relevant_dist - min_dist/10 ))
					# dist_loss = torch.sum(1/(relevant_dist - min_dist/10 ))
					dist_loss = torch.sum(torch.square(torch.log(relevant_dist - min_dist/10 )))
				else:
					# dist_loss = -torch.sum(1/(relevant_dist - self.reference_dist/10 ))
					dist_loss = torch.sum(torch.square(torch.log(relevant_dist - grasp.reference_dist/10)))
			
			else:
				# if we only care about closest pair to collision, then no sum needed
				clamp_dist = torch.clamp(min_dist, max=4*grasp.reference_dist)
				if min_dist < grasp.reference_dist/10:
					tqdm.write('inside outer boundary region')
					dist_loss = torch.sum(torch.square(torch.log(clamp_dist - min_dist/10 )))
				else:
					dist_loss = torch.sum(torch.square(torch.log(clamp_dist - grasp.reference_dist/10)))
		dist_loss.requires_grad_(True)

		return dist_loss


def test_run():
	"""Test prediction of gqcnn_pytorch model"""

	Attack.logger.info("Running test_run...")

	if torch.cuda.is_available():
		device = torch.device("cuda:0")
		torch.cuda.set_device(device)
	else:
		tqdm.write("cuda not available")
		device = torch.device("cpu")

	depth0 = np.load("/home/hmitchell/pytorch3d/dex_shared_dir/depth_0.npy")
	grasp = GraspTorch(depth=0.607433762324266, im_center=(416, 286), im_angle=-2.896613990462929, device=device)

	# load input tensors from gqcnn library for prediction
	pose0 = torch.from_numpy(np.load("data/pose_tensor1_raw.npy")).float().to(device)
	image0 = torch.from_numpy(np.load("data/image_tensor1_raw.npy")).float().permute(0,3,1,2).to(device)

	# tensors from pytorch extraction
	pose1, image1 = quality.GQCNNQualityFunction.extract_tensors(grasp=grasp,d_im=depth0)

	# tensors from pytorch extraction & pytorch depth image
	renderer = Renderer(device=device)
	mesh, _ = renderer.render_object("data/bar_clamp.obj", display=False)
	dim = renderer.mesh_to_depth_im(mesh, display=False)
	pose2, image2 = quality.GQCNNQualityFunction.extract_tensors(grasp=grasp,d_im=dim)

	# testing with new barclamp object
	mesh2, _ = renderer.render_object("data/new_barclamp.obj", display=False)
	dim2 = renderer.mesh_to_depth_im(mesh2, display=False)
	pose3, image3 = quality.GQCNNQualityFunction.extract_tensors(grasp=grasp,d_im=dim2)

	# instantiate GQCNN PyTorch model
	model = KitModel("weights.npy",device=device)
	model.eval()

	# instantiate Attack class and run prediction
	run1 = Attack(model=model)
	tqdm.write(run1.run(pose0, image0)[0][1].item())
	tqdm.write(run1.run(pose1, image1)[0][1].item())
	tqdm.write(run1.run(pose2, image2)[0][1].item())	# original barclamp object
	tqdm.write(run1.run(pose3, image3)[0][1].item())	# new barclamp object

	# test model with varying batch sizes
	pose4 = torch.cat([pose1, pose2, pose3], 0)
	image4 = torch.cat([image1, image2, image3], 0)
	tqdm.write("\n")
	tqdm.write(pose4.shape, image4.shape)
	tqdm.write(run1.run(pose4, image4))

	pose5 = torch.cat([pose1, pose2, pose2, pose3, pose1, pose3, pose3, pose1, pose2])
	image5 = torch.cat([image1, image2, image2, image3, image1, image3, image3, image1, image2])
	tqdm.write(pose5.shape, image5.shape)
	tqdm.write(run1.run(pose5, image5))

	pose6 = torch.cat([pose4, pose5], dim=0)
	image6 = torch.cat([image4, image5], dim=0)
	tqdm.write(pose6.shape, image6.shape)
	tqdm.write(run1.run(pose6, image6))
	
	Attack.logger.info("Finished test_run.")

def test_attack():
	"""Test attack on gqcnn_pytorch model"""

	Attack.logger.info("Running test_attack...")

	# RENDER MESH AND DEPTH IMAGE
	renderer = Renderer()
	mesh, _ = renderer.render_object("data/new_barclamp.obj", display=False)
	dim = renderer.mesh_to_depth_im(mesh, display=False)

	# FIXED GRASP TO ATTACK
	grasp = GraspTorch.read("grasp-batch.json")[0]
	tqdm.write("oracle quality:", grasp.quality.item())

	# SET UP ATTACK
	model = KitModel("weights.npy")
	model.eval()
	run1 = Attack(num_plots=10, steps_per_plot=10, model=model, renderer=renderer, oracle_method=grasp.oracle_method)

	# RUN INITIAL MODEL PREDICTION
	pose, image = quality.GQCNNQualityFunction.extract_tensors_batch(grasp=grasp,d_ims=dim)
	pred = run1.run(pose, image)
	tqdm.write("initial model prediction:", pred[1].item())

	Attack.logger.info("ATTACK")
	adv_mesh, final_pic = run1.attack(mesh, grasp, "test-attack", lr=1e-5, momentum=0.0)
	renderer.display(final_pic, title="final_grasp", save="test-attack/final-grasp.png")

	Attack.logger.info("Finished running test_attack.")

def test_run2():
	Attack.logger.info("Running test_run2...")

	# set up
	gqcnn_model = KitModel("weights.npy")
	gqcnn_model.eval()
	r = Renderer()
	model = Attack(model=gqcnn_model, renderer=r)
	mesh, _ = r.render_object("data/new_barclamp.obj", display=False)
	depth_im = r.mesh_to_depth_im(mesh, display=False)

	# run model on batch of grasps
	grasp = GraspTorch.read("grasp-batch.json")
	grasp.trans_world_to_im(camera=r.camera)
	poses, images = quality.GQCNNQualityFunction.extract_tensors_batch(grasp=grasp,d_ims=depth_im)
	out = model.run(poses, images)
	out = out[:,1:]
	tqdm.write(f"prediction: {out.shape} {out.device}\n{out}")

	grasp.prediction = out
	grasp.save("grasp-batch.json")

	# run model on individual grasps
	for i, g in tqdm(enumerate(grasp),desc='grasps'):
		pose, image = g.extract_tensors(depth_im)
		pred = model.run(pose, image)[:, 1:]
		tqdm.write(f"\nGrasp {i}: \n\tModel prediction: {pred.item()}\n\tOriginal prediction: {out[i].item()}\n\tOracle quality: {g.quality.item()}")
		if not out[i].item() == pred.item():
			tqdm.write(f"\tPrediction diff: {torch.sub(pred, out[i]).item()}")

	Attack.logger.info("Done running test_run2.")

if __name__ == "__main__":
	# test_run2()

	r = Renderer()
	mesh, _ = r.render_object("data/new_barclamp.obj", display=False)
	g = GraspTorch.read("grasp-batch.json")
	grasps = [g[0], g[1], g[3], g[4], g[5], g[6]]

	# d = "exp-results/random-fuzz/"
	# grasp_dirs = {d+"grasp0/": g[0], d+"grasp1/": g[1], d+"grasp3/": g[2], d+"grasp4/": g[3], d+"grasp5/": g[4], d+"grasp6/": g[5]}

	# oracle_grad_lr = [(1e-6, 0.0), (1e-6, 0.99), (1e-5, 0.0), (1e-5, 0.9), (1e-5, 0.99), (1e-4, 0.0), (1e-4, 0.9)]	# learning rate and momentum combinations
	# oracle_grad_alpha = [0.1, 0.3, 0.5, 0.7]
 
	# learning rate and momentum combinations
	# random_fuzz_lr = {"lr0": 1e-6, "lr1":1e-5, "lr3": 1e-4}

	model = KitModel("weights.npy")
	model.eval()
	run1 = Attack(num_plots=10, steps_per_plot=10, model=model, renderer=r, oracle_method="pytorch")

	# for d in grasp_dirs.keys():
	# 	for ext in random_fuzz_lr.keys():
	# 		save = d + ext + "/"
	# 		lr = random_fuzz_lr[ext]
	# 		grasp = grasp_dirs[d]
	# 		tqdm.write(f"saving to: {save} with learning rate {lr}")
	# 		run1.attack(mesh=mesh, grasp=grasp, dir=save, lr=lr, momentum=None, loss_alpha=None, method="random-fuzz")

	# tqdm.write("ATTACK SET 1\n")
	graspObj = grasps[1]
	dir = "test/no-oracle/"
	run1.attack(mesh=mesh, grasp=graspObj, dir=dir, lr=1e-5, momentum=0.9, loss_alpha=None, method="no-oracle")
	# run1.attack(mesh=mesh, grasp=grasp, dir=d+"lr0-weight1/", lr=1e-6, momentum=0.0, loss_alpha=0.3, method="oracle-grad")
	# run1.attack(mesh=mesh, grasp=grasp, dir=d+"lr0-weight2/", lr=1e-6, momentum=0.0, loss_alpha=0.5, method="oracle-grad")
	# run1.attack(mesh=mesh, grasp=grasp, dir=d+"lr0-weight3/", lr=1e-6, momentum=0.0, loss_alpha=0.7, method="oracle-grad")
