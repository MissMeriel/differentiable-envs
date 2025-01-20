import torch
import torch.optim as optim

from torch.masked import masked_tensor
from torchvision import transforms
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import math
import json
from pytorch3d.io import load_obj, save_obj
from pytorch3d import _C
import pytorch3d.transforms as tf
from scipy.spatial import ConvexHull
from scipy.io import savemat
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (
        look_at_view_transform,
        PerspectiveCameras,
        MeshRenderer,
        MeshRasterizer,
        SoftPhongShader,
        RasterizationSettings,
        PointLights,
        TexturesVertex
)
from qpth.qp import QPFunction

# used to save grasp visualization
import trimesh
from grasp import GraspTorch
from pytorch3d_ext import mesh_properties as mp
import pytorch3d_ext as p3d_ex
from gqcnn_pytorch import KitModel
#from ll4ma_opt.problems.problem import Problem
#from ll4ma_opt.problems import SteinWrapper
#from ll4ma_opt.solvers import GradientDescent,BFGSMethod


class GraspQualityFunction():    #ABC):
    """Abstract grasp quality class."""

    def __init__(self):
        # Set up logger - can't because it's from autolab_core.
        # self._logger = Logger.get_logger(self.__class__.__name__)
        self._logger = 0

    def __call__(self, state, actions, params=None):
        """Evaluates grasp quality for a set of actions given a state."""
        return self.quality(state, actions, params)
    
    def apply_grasp(self,state, actions:GraspTorch, is_watertight=True, is_inverted=False):
        if actions.applied_to_object is False :
            with record_function("solveForIntersection"):
                actions = actions.apply_to_mesh(state, is_watertight=is_watertight, is_inverted=is_inverted)
        self.Mesh = state
        self.Grasps = actions
        if torch.any(actions.contact_mask):
            self.G = actions.grasp_matrix
            return True
        else:
            return False

    def write_obj(self, path):
        if self.Grasps is not None and hasattr(self, 'quality_cache') and hasattr(self, 'equation_cache'):
            colors = [[0, 255, 255],[0, 0, 0]]
            sphere_radius = self.Grasps.finger_radius
            vectors = self.equation_cache[...,:-1] 
            #vectors = self.equation_cache[...,:-1] / self.equation_cache[...,(-1,)]
            vectors = vectors.reshape((-1,6)).numpy(force=True)
            mesh_list = []
            com_np = self.Grasps.object_com.numpy(force=True).reshape(1,3)
            force_dir_np = vectors[:,:3]
            torque_axis_np =vectors[:,3:]

            force_points_np = np.concatenate((com_np, com_np+force_dir_np*sphere_radius*10),axis=0)
            mesh_list.append(trimesh.creation.cylinder(segment=force_points_np, radius=sphere_radius/10))
            mesh_list[-1].visual.face_colors = colors[0]

            force_points_np = np.concatenate((com_np-torque_axis_np*sphere_radius*5, com_np+torque_axis_np*sphere_radius*5),axis=0)
            mesh_list.append(trimesh.creation.cylinder(segment=force_points_np, radius=sphere_radius/10))
            mesh_list[-1].visual.face_colors = colors[1]

            merged_mesh = trimesh.util.concatenate(mesh_list)

            merged_mesh.export(path)

            
    def savemat(self, path, other_items=None):
        # TODO move to grasp object and call there, passing quality as a dict
        # collects a dictionary of interesting internal state, then saves it
        dict_to_save = {}
        if self.Grasps is not None:
            dict_to_save['axis3D'] = self.Grasps.world_axis.numpy(force=True)
            dict_to_save['center3D'] = self.Grasps.world_center.numpy(force=True)
            dict_to_save['contact_points']= self.Grasps.contact_points.numpy(force=True)
            dict_to_save['contact_normals']= self.Grasps.contact_normals.numpy(force=True)
            dict_to_save['face_normals']= self.Grasps.face_normals.numpy(force=True)
            dict_to_save['object_com'] = self.Grasps.object_com.numpy(force=True)
            dict_to_save['faces_index'] =self.Grasps.faces_index.numpy(force=True)
            dict_to_save['grasp_matrix'] = self.Grasps.grasp_matrix.numpy(force=True)
            ep0, ep1 = self.Grasps.endpoints3D
            dict_to_save['endpoints3D'] =torch.cat((ep0.unsqueeze(0),ep1.unsqueeze(0)),dim=0).numpy(force=True)

        if other_items is not None:
            dict_to_save = dict_to_save | other_items
        if hasattr(self, 'quality_cache'):
            dict_to_save['quality'] = self.quality_cache.numpy(force=True)
        if hasattr(self, 'equation_cache'):
            dict_to_save['equation_cache'] = self.equation_cache.numpy(force=True)

        savemat(path, dict_to_save)

        return

    #@abstractmethod
    def quality(self, state, actions, params=None):
        """Evaluates grasp quality for a set of actions given a state.
        Parameters
        ----------
        state : :obj:`object`
            State of the world e.g. image.
        actions : :obj:`list`
            List of actions to evaluate e.g. parallel-jaw or suction grasps.
        params : :obj:`dict`
            Optional parameters for the evaluation.
        Returns
        -------
        :obj:`numpy.ndarray`
            Vector containing the real-valued grasp quality
            for each candidate.
        """
        pass

class ParallelJawQualityFunction(GraspQualityFunction):
    """Abstract wrapper class for parallel jaw quality functions ()."""

    def __init__(self, config):
        GraspQualityFunction.__init__(self)
        # Read Shared parameters.
        self._friction_coef = config["friction_coef"]
        self._max_friction_cone_angle = np.arctan(self._friction_coef)


    def friction_cone_angle(self, action):
        """Compute the angle between the axis and the boundaries of the
        friction cone."""
        if action.contact_points is None or action.contact_normals is None:
            invalid_friction_ang_msg = ("Cannot compute friction cone angle"
                                        " without precomputed contact points"
                                        " and normals.")
            raise ValueError(invalid_friction_ang_msg)
        dot_prod = torch.sum(torch.mul(action.contact_normals, action.ray_directions),-1) 
        # not sure if necessary, should already be bounded -1 to 1. 
        dot_prod = torch.minimum(torch.maximum(dot_prod, torch.tensor(-1.0)), torch.tensor(1.0))
        angle = torch.arccos(dot_prod)
        max_angle = torch.max(angle,0).values

        return max_angle

    
    def force_closure(self, action):
        """Determine if the (2 contact) grasp is in force closure."""
        return (self.friction_cone_angle(action) <
                self._max_friction_cone_angle)


class GQCNNQualityFunction(ParallelJawQualityFunction):
    def __init__(self, weights_path, renderer):
        self.model = KitModel(weight_file=weights_path, device=renderer.device)
        self.model.eval()
        self.renderer = renderer
    
    
    def quality(self, state, actions):
        d_im = self.renderer.mesh_to_depth_im(state, display=False)
        poses, images = self.extract_tensors_batch(actions, d_im)
        return self.model(poses, images)
        

    @staticmethod
    def extract_tensors(grasp, d_im):
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
            torch_dim = torch.tensor(d_im, dtype=torch.float32).permute(2, 0, 1).to(grasp.device)
        else:
            torch_dim = d_im

        # check if grasp is 2D
        if (grasp.depth==None or grasp.im_center==None or grasp.im_angle==None):
            GraspTorch.logger.error("Grasp is not in 2D, must convert with camera intrinsics before tensor extraction.")
            return None, None

        # construct pose tensor from grasp depth
        pose_tensor = torch.zeros([1, 1])
        pose_tensor = pose_tensor.to(torch_dim.device)
        pose_tensor[0] = grasp.depth.float()

        # process depth image wrt grasp (steps 1-3) 
        
        # 1 - resize image tensor
        out_shape = torch.tensor([torch_dim.shape], dtype=torch.float32)
        out_shape *= (1/3)		# using 1/3 based on gqcnn library - may need to change depending on input
        out_shape = tuple(out_shape.type(torch.int)[0][1:].numpy())

        torch_transform = transforms.Resize(out_shape, antialias=False) 
        torch_image_tensor = torch_transform(torch_dim)

        # 2 - translate wrt to grasp angle and grasp center 
        theta = -1 * math.degrees(grasp.im_angle[0])	# -1 because PyTorch transform goes clockwise and autolab_core goes counter-clockwise
         
        dim_cx = torch_dim.shape[2] // 2
        dim_cy = torch_dim.shape[1] // 2
        
        translate = ((grasp.im_center[...,0] - dim_cx) / 3, (grasp.im_center[...,1]- dim_cy) / 3)

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
        # torch_scaled = transforms.functional.affine(
        #     torch_rotated,
        #     0,
        #     translate=(0, 0),
        #     scale=scale,
        #     shear=0,
        #     interpolation=transforms.InterpolationMode.BILINEAR,
        #     center=(cx, cy)
        # )
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
    @staticmethod
    def extract_tensors_batch(grasp, d_ims):

        # r = Renderer()

        # check type of input_dim
        if isinstance(d_ims, np.ndarray):
            GraspTorch.error("extract_tensors_batch takes a tensor, not a numpy ndarray")
            return None, None
        
        # check if grasp is 2D
        if (grasp.depth==None or grasp.im_center==None or grasp.im_angle==None):
            GraspTorch.logger.error("Grasp is not in 2D, must convert with camera intrinsics before tensor extraction.")
            return None, None

        # dims.shape: [batch_size, 1, 480, 640] = [batch_size, channels, H, W]
        batch_size = grasp.num_grasps()
        if d_ims.dim() != 4:
            d_ims = d_ims.repeat(batch_size, 1, 1, 1)
        if d_ims.shape[0] != batch_size:
            d_ims = d_ims[0].repeat(batch_size, 1, 1, 1)

        # construct pose tensor from grasp depth
        pose_tensor = grasp.depth.float()

        # process depth image wrt grasp (steps 1-3) 
        # 1 - resize image tensors
        out_shape = torch.tensor([d_ims.squeeze(1).shape], dtype=torch.float32)
        out_shape *= (1/3)		# using 1/3 based on gqcnn library - may need to change depending on input
        out_shape = tuple(out_shape.type(torch.int)[0][1:].numpy())		# (160, 213)

        torch_transform = transforms.Resize(out_shape, antialias=False) 
        dims_resized = torch_transform(d_ims)		# shape: [batch_size, 1, 160, 213] = [batch_size, channels, H, W]

        # 2 - translation wrt to grasp angle and grasp center
        # 	translation matrix
        dim_cx = d_ims.shape[3] // 2	# 320
        dim_cy = d_ims.shape[2] // 2	# 240
        dim_cx_tens = torch.tensor([dim_cx]).expand(batch_size).to(d_ims.device)
        dim_cy_tens = torch.tensor([dim_cy]).expand(batch_size).to(d_ims.device)

        u = (2 * ((dim_cx_tens - grasp.im_center[..., 0])/3) / (dims_resized.shape[3])).float()	# not in pixels, but normalized on (image size * 2)
        v = (2 * ((dim_cy_tens - grasp.im_center[..., 1])/3) / (dims_resized.shape[2])).float()

        translate = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
        translate = translate.expand(batch_size, -1, -1).to(d_ims.device).float()
        indices = torch.arange(batch_size)
        translate[indices, 0, 2] = u
        translate[indices, 1, 2] = v

        #	rotation matrix
        theta = grasp.im_angle.squeeze()	# no -1 for counter-clockwise, stay in radians
        cos = torch.cos(theta).float()
        sin = torch.sin(theta).float()

        rotation = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]).expand(batch_size, -1, -1).to(d_ims.device).float()
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

class minWeightQualityFunction(ParallelJawQualityFunction):
    """Computes the minimum external wrench required to disrupt a grasp"""
    def __init__(self, config, min_quality=0.004):
        self.min_quality = min_quality
        ParallelJawQualityFunction.__init__(self, config)


    def quality(self, state, actions, is_watertight=True, is_inverted=False):
        """Given a batch of parallel-jaw grasps, compute the minimum magnitude external
        wrench required to disrupt the grasp.

        Parameters
        ----------
        state : :obj:`Pytorch3D Meshes object `
            A Meshes object of size 1 containing a watertight mesh
        action: :obj:`Grasp`
            A suction grasp in image space that encapsulates center and axis
        params: dict
            Stores params used in computing quality.

        Returns
        -------
        :obj:`numpy.ndarray`
            Array of the quality for each grasp.
        """
        if not self.apply_grasp(state,actions,is_watertight,is_inverted):
            # no intersection found, just return 0 quality
            return torch.zeros_like(actions.world_axis[...,0])
        
        closest = torch.zeros(self.Grasps.contact_mask.shape, device=state.device, dtype=torch.float64)
        closest_eq = torch.zeros(list(self.Grasps.contact_mask.shape)+[7], device=state.device, dtype=torch.float64)
        if torch.any(self.Grasps.contact_mask):
            with record_function("minweight"):
                closest[self.Grasps.contact_mask], closest_eq[self.Grasps.contact_mask] = self.find_min_weight(self.G)
        self.quality_cache = closest
        self.equation_cache = closest_eq
        return closest / self.min_quality
    
    @staticmethod
    def find_min_weight(G):       
        original_shape = G.shape
        G_unwrapped = G.view((original_shape[0]*original_shape[1], -1, 6)).transpose(0,1)

        # https://arxiv.org/pdf/2302.13687
        #
        # maximize l
        # s.t. G * alpha = 0
        #      sum(alpha) = 1
        #      alpha - l >=0 (all alpha)
        #
        # minimize epsilon @ [alpha; l] @ eye(7) @ [alpha; l] - l
        # s.t.   [G_unwrapped,zeros] @ [alpha; l] = 0
        #        [ones(6), 0] @ [alpha; l] = 1
        #        [eye, 1] >= 0
        # 

        quadratic_term = 1e-8 * torch.eye(G_unwrapped.shape[1]+1,device=G.device,dtype=G.dtype)
        linear_term = torch.nn.functional.pad(torch.zeros((G_unwrapped.shape[1],1),device=G.device,dtype=G.dtype),(0,0,0,1),value=-1).squeeze(1)
        # [G_unwrapped,zeros] @ [alpha; l] = 0
        # [ones(6), 0] @ [alpha; l] = 1
        #
        # 
        # [batch, 6, n_forces] @ [n_forces,1] = zeros(6)
        # [batch, 1, n_forces] @ [n_forces,1] = ones(1)
        equality_A_first = torch.nn.functional.pad(G_unwrapped.transpose(2,1), (0,0,0,1,0,0),value=1)
        equality_A = torch.nn.functional.pad(equality_A_first, (1,0,0,0,0,0)) # pad 1 zero to last dim
        equality_b = torch.tensor((0,0,0,0,0,0,1),device=G.device,dtype=G.dtype).unsqueeze(0).expand((equality_A.shape[0],-1))

        #        [eye, 1] >= 0
        inequality_G = torch.concatenate((-torch.eye(G_unwrapped.shape[1],device=G.device,dtype=G.dtype), 
                                          torch.ones((G_unwrapped.shape[1],1),device=G.device,dtype=G.dtype)),dim=1)
        inequality_h =  torch.zeros((G_unwrapped.shape[1]),device=G.device)

        qp = QPFunction(check_Q_spd=False)

        alpha_weights = qp(quadratic_term, linear_term, # optimization
                                            inequality_G, inequality_h, #inequality constraints
                                            equality_A,equality_b) # equality constraints)
        # test feasibility?

        # find minimum value
        min_out = torch.min(alpha_weights[...,:-1],dim=-1)
        min_alpha = min_out.values
        indices = min_out.indices
        equations_unnorm = torch.nn.functional.pad(torch.gather(input=G_unwrapped,index=indices.unsqueeze(-1).unsqueeze(-1).expand([-1,-1,6]),dim=-2).squeeze(-2),[0,1],value=-1)
        equations = equations_unnorm / torch.linalg.vector_norm(equations_unnorm[...,:-1],dim=-1).unsqueeze(-1)
        return min_alpha, equations

class CannyFerrariQualityFunction(ParallelJawQualityFunction):
    """Computes the minimum external wrench required to disrupt a grasp"""
    def __init__(self, config,min_quality=0.002):
        self.min_quality = min_quality
        ParallelJawQualityFunction.__init__(self, config)
    
    def savemat(self, path, other_items=None):
        # for this quality, we can store the convex hull as well
        if other_items is None:
            other_items = {}
        original_shape = self.G.shape
        G_unwrapped = self.G.view((original_shape[0]*original_shape[1], -1, 6))
        simplices_list = np.zeros((G_unwrapped.shape[1],), dtype=object)
        equations_list = np.zeros((G_unwrapped.shape[1],), dtype=object)
        for batch_idx in range(G_unwrapped.shape[1]):
            miniG = G_unwrapped[:,batch_idx,:]
            simplices,equations = qHullTorch.apply(miniG.cpu())
            simplices_list[batch_idx] = simplices.numpy(force=True)
            equations_list[batch_idx] = equations.numpy(force=True)
        other_items['hull_simplices'] = simplices_list
        other_items['hull_equations'] = equations_list
        super().savemat(path, other_items=other_items)

    def quality(self, state, actions, is_watertight=True, is_inverted=False):
        """Given a batch of parallel-jaw grasps, compute the minimum magnitude external
        wrench required to disrupt the grasp.

        Parameters
        ----------
        state : :obj:`Pytorch3D Meshes object `
            A Meshes object of size 1 containing a watertight mesh
        action: :obj:`Grasp`
            A suction grasp in image space that encapsulates center and axis
        params: dict
            Stores params used in computing quality.

        Returns
        -------
        :obj:`numpy.ndarray`
            Array of the quality for each grasp.
        """
        
        if not self.apply_grasp(state,actions,is_watertight,is_inverted):
            # no intersection found, just return 0 quality
            return torch.zeros_like(actions.world_axis[...,0])
        closest = torch.zeros(self.Grasps.contact_mask.shape, device=state.device, dtype=torch.float64)
        closest_eq = torch.zeros(list(self.Grasps.contact_mask.shape)+[7], device=state.device, dtype=torch.float64)
        with record_function("minHull"):
            closest[self.Grasps.contact_mask], closest_eq[self.Grasps.contact_mask] = CannyFerrariQualityFunction.find_min_dist_to_hull(self.G)
        
        self.quality_cache = closest
        self.equation_cache = closest_eq
        return closest / self.min_quality
        
    
    @staticmethod
    def qp_wrap(facets):
        ### TODO Re-enable for in-feasible grasps. Need to make sure sign is negative
        # square facet matrix
        Gsquared_full  = torch.linalg.matmul(facets,torch.permute(facets, [0,2,1]))
        wrench_regularizer=1e-10
        # find all facets where the inputs are less than the regularizer, so we can ignore the qp results
        non_zero_wrench_facets = torch.logical_not(torch.all(torch.all(torch.abs(Gsquared_full) < wrench_regularizer,dim=-1),dim=-1))
        dist_full = torch.zeros_like(Gsquared_full[...,0:1,0:1])
        x_full = torch.zeros_like(Gsquared_full[...,0])
        P = None
        if torch.any(non_zero_wrench_facets):
            Gsquared = Gsquared_full[non_zero_wrench_facets]

            regulizer_mat = (wrench_regularizer * torch.eye(Gsquared.shape[1], device = facets.device, dtype=facets.dtype))
            n_dim = Gsquared.shape[1]
            n_batch = Gsquared.shape[0]
            P = 2 * (Gsquared + regulizer_mat).transpose(1,2)
            q = torch.zeros((n_batch,n_dim), device = facets.device, dtype=facets.dtype)
            G = -torch.eye(n_dim, device = facets.device, dtype=facets.dtype).unsqueeze(0).expand((n_batch,n_dim,n_dim))
            h = torch.zeros((n_batch,n_dim), device = facets.device, dtype=facets.dtype)
            A = torch.ones((n_batch,1,n_dim), device = facets.device, dtype=facets.dtype)
            b = torch.ones((n_batch,1),device = facets.device, dtype=facets.dtype)

            x = QPFunction(check_Q_spd=False)(P, q, G, h, A , b)
            dist = torch.sqrt(torch.matmul(x.unsqueeze(1), torch.matmul(P, x.unsqueeze(2)))/2)
        

            dist_full[non_zero_wrench_facets] = dist
            x_full[non_zero_wrench_facets] = x

        return dist_full, x_full, P
   
    @staticmethod
    def compute_hyperplane_above(facets):
        hull_pad = torch.nn.functional.pad(facets,(0,1,0,1),value=1)
        hull_pad[...,-1,-1] = 0
        b_vec = torch.tensor([0,0,0,0,0,0,1],dtype=facets.dtype,device=facets.device).expand((hull_pad.shape[0],7))
        normals = torch.nn.functional.normalize(torch.linalg.solve_ex(hull_pad, b_vec)[0][:,0:6])
        dists = torch.abs(torch.linalg.vecdot(torch.mean(facets,dim=1),normals))
        if torch.any(dists > 0.1):
            print(torch.max(dists))
        return dists
    
    @staticmethod
    def find_min_dist_to_hull(G):       
        original_shape = G.shape
        G_unwrapped = G.view((original_shape[0]*original_shape[1], -1, 6))
        is_feasible_minimal = torch.zeros((G_unwrapped.shape[1]),dtype=torch.bool,device=G.device)
        is_feasible_multi = torch.zeros((G_unwrapped.shape[1]),dtype=torch.bool,device=G.device)
        closest = torch.zeros(is_feasible_minimal.shape, dtype=G.dtype, device=G.device)
        closest_eq = torch.zeros(list(is_feasible_minimal.shape)+[7], dtype=G.dtype, device=G.device)
        facets_infeasible_local = []
        count_facets_per_grasp = []
        facets_feasible_multi_local = []
        facets_feasible_minimal_local = []
        with record_function("ConvexHull-Loop"):
            for batch_idx in range(G_unwrapped.shape[1]):
                miniG = G_unwrapped[:,batch_idx,:]
                [simplices,equations] = qHullTorch.apply(miniG)
                areas = torch.abs(torch.linalg.det(miniG[simplices,:]))
                finite_size = areas > 1e-10
                #simplices_finite = simplices[finite_size]
                dist_from_origin = equations[...,-1]#equations[finite_size,-1]
                is_above_origin = dist_from_origin<0
                
                if torch.all(is_above_origin) and torch.numel(is_above_origin) > 0:

                    # if all negative, closest point is max (min of abs)
                    min_dist, min_dex = torch.max(dist_from_origin,0)
                    facet = torch.nn.functional.pad(miniG[torch.unique(simplices[min_dist == equations[:,-1]].flatten())],[0,1],value=1)
                    if facet.shape[-2] > 6:
                        is_feasible_multi[batch_idx] = 1
                        facets_feasible_multi_local.append(facet)
                    else:
                        is_feasible_minimal[batch_idx] = 1
                        facets_feasible_minimal_local.append(facet)
                    closest_eq[batch_idx] = equations[min_dex]
                else:
                    facet = miniG[simplices[torch.logical_not(is_above_origin),:],:]
                    facets_infeasible_local.append(facet)
                    count_facets_per_grasp.append(facet.shape[0])
        
                
        if len(facets_feasible_multi_local) > 0:
            # facets = torch.cat(facets_feasible_local,dim=0)   
            # closest_feasible = CannyFerrariQualityFunction.compute_hyperplane_above(facets)
            # closest[is_feasible] = closest_feasible
            
            #facets = torch.nested.nested_tensor(facets_feasible_local,requires_grad=True, layout=torch.jagged).to_padded_tensor(padding=0)
            facets = torch.nn.utils.rnn.pad_sequence(facets_feasible_multi_local,batch_first=True)
            closest_feasible = hullTorch.computePlaneLeastSquare(facets)
            closest_feasible = torch.abs(closest_feasible[:,-1])
            closest[is_feasible_multi] = closest_feasible

        if len(facets_feasible_minimal_local) > 0:
            facets = torch.nn.utils.rnn.pad_sequence(facets_feasible_minimal_local,batch_first=True)
            closest_feasible = hullTorch.computePlaneCross(facets)
            closest_feasible = torch.abs(closest_feasible[:,-1])
            closest[is_feasible_minimal] = closest_feasible

        allow_neg_fc = 0
        # TODO re-introduce, consider if we want to still drop zero size facets
        if allow_neg_fc and len(facets_infeasible_local) > 0:
            facets = torch.cat(facets_infeasible_local,dim=0)
            ### reassemble simplices into batch may be too many and need to serialize
            ## maybe we only (re)compute the important one in pytorch, drop others for memory
            with record_function("qp_wrap"):
                dist,x,P = CannyFerrariQualityFunction.qp_wrap(facets)


            start_ind = 0
            closest_infeasible = torch.zeros(len(count_facets_per_grasp),1, dtype=torch.float64)
            for index,length in enumerate(count_facets_per_grasp):
                end_ind = start_ind + length
                dist_local = dist[start_ind:end_ind]
                minReturn = torch.min(dist_local[torch.logical_not(torch.logical_or(torch.isnan(dist_local),dist_local<0))],dim=0)
                closest_infeasible[index] = minReturn.values
                start_ind = end_ind
        else:
            closest_infeasible = 0
        
        
        closest[torch.logical_not(torch.logical_or(is_feasible_multi,is_feasible_minimal))] = closest_infeasible
        return closest, closest_eq

class RobustCannyFerrariQualityFunction(CannyFerrariQualityFunction):
    """Measures the probability that grasps near a reference grasp will have high Canny Ferrari quality."""
    def __init__(self, config,min_quality=0.002):
        CannyFerrariQualityFunction.__init__(self, config,min_quality=min_quality)

    def quality(self, state, actions, is_watertight=False, is_inverted=False):
        """Given a parallel-jaw grasp, the probability that grasps near it 
        reference grasp will have high Canny Ferrari quality.

        Parameters
        ----------
        state : :obj:`Pytorch3D Meshes object `
            A Meshes object of size 1 containing a watertight mesh
        action: :obj:`Grasp`
            A suction grasp in image space that encapsulates center and axis
        params: dict
            Stores params used in computing quality.

        Returns
        -------
        :obj:`numpy.ndarray`
            Array of the quality for each grasp.
        """
        # 
        noised_grasps = actions.generateNoisyGrasps(25)
        noised_grasps = noised_grasps.apply_to_mesh(state, ignore_backface_check=False, is_watertight=is_watertight, is_inverted=is_inverted)
        noised_tensor = super().quality(state, noised_grasps, is_watertight=is_watertight, is_inverted=is_inverted)
        # expected quality
        qual_tensor = torch.mean(noised_tensor)

        return qual_tensor

class RobustMinWeightQualityFunction(minWeightQualityFunction):
    """Measures the probability that grasps near a reference grasp will have high Canny Ferrari quality."""
    def __init__(self, config,min_quality=0.002):
        minWeightQualityFunction.__init__(self, config,min_quality=min_quality)

    def quality(self, state, actions, is_watertight=False, is_inverted=False):
        """Given a parallel-jaw grasp, the probability that grasps near it 
        reference grasp will have high Canny Ferrari quality.

        Parameters
        ----------
        state : :obj:`Pytorch3D Meshes object `
            A Meshes object of size 1 containing a watertight mesh
        action: :obj:`Grasp`
            A suction grasp in image space that encapsulates center and axis
        params: dict
            Stores params used in computing quality.

        Returns
        -------
        :obj:`numpy.ndarray`
            Array of the quality for each grasp.
        """
        # 
        noised_grasps = actions.generateNoisyGrasps(25)
        noised_grasps = noised_grasps.apply_to_mesh(state, ignore_backface_check=False, is_watertight=is_watertight, is_inverted=is_inverted)
        noised_tensor = super().quality(state, noised_grasps, is_watertight=is_watertight, is_inverted=is_inverted)
        # expected quality
        qual_tensor = torch.mean(noised_tensor)

        return qual_tensor

class hullTorch(torch.autograd.Function):
    # attempt to keep convex hull computation on gpu. 
    # complicated algorithm, still missing a lot
    @staticmethod
    def forward(_, miniG):
        miniG_homogeneous = torch.nn.functional.pad(miniG,[0,1],value=1)

        # initial simplex and planes
        initial_hull, initial_neighbors, points_used = hullTorch.find_initial_set(miniG_homogeneous)
        
        planes = hullTorch.computePlanes(miniG_homogeneous,initial_hull)
        
        unused_points = miniG_homogeneous[...,torch.logical_not(points_used),:]
        dists = hullTorch.comparePoints(unused_points, planes)
        #while(unused_points.shape[-2] > 0):
        # loop over points
        # TODO, filter points to only be non-hull
        
        # TODO choose point to add to hull (largest positive?)
        # TODO add point: use comparison to find ridges
        
        # remove point from unused points
        # 
        # # can we avoid a copy? overwrite by index? 
        # simplices_removed =
        # simplices_added =
        # TODO update hull, update neighbors
        #
        # TODO, maintain hull to point comparisons 
        #    update planes 
        #  dists_new = hullTorch.comparePoints(unused_points, planes_new)
        return torch.tensor(initial_hull,dtype=torch.long)

    @staticmethod
    def find_initial_set(points):
        extrema = torch.unique(torch.cat((torch.max(points[...,:-1], dim=-1).indices, torch.min(points, dim=-1).indices),dim=-1),dim=-1)
        non_extrema = hullTorch.setdiff_flat(extrema, torch.arange(points.shape[-2],device=points.device))

        non_extrama_combinations = torch.combinations(non_extrema,r=(points.shape[-1]-extrema.shape[-1]))
        simplices = torch.cat((extrema.expand((non_extrama_combinations.shape[0],-1)), non_extrama_combinations),dim=-1)

        det_A, _ = hullTorch.hull_det(points, simplices)

        simplex_initial_unique = simplices[torch.max(det_A,dim=-1).indices,:].unsqueeze(-2)
        simplex_initial_stride = list(simplex_initial_unique.stride())
        simplex_initial_stride[-2] = 1
        simplex_initial_stride[-1] = 1

        simplices_hull = torch.as_strided(simplex_initial_unique,[7,6],stride=simplex_initial_stride)

        range_for_neighbors = torch.arange(points.shape[-1],device=points.device).unsqueeze(-2)
        neighbors_shape = list(range_for_neighbors.shape)
        neighbors_shape[-2] = 7
        neighbors = torch.as_strided(range_for_neighbors.repeat(neighbors_shape),[7,6],stride=simplex_initial_stride,storage_offset=1)

        points_used = torch.zeros(points[...,0].shape,device=points.device,dtype=torch.bool)
        points_used.scatter_(-1, simplex_initial_unique.squeeze(-2),1)

        return simplices_hull, neighbors, points_used
        # https://github.com/qhull/qhull/blob/c7bee59d068a69f427b1273e71cdc5bc455a5bdd/src/libqhull/poly2.c#L2347


    @staticmethod
    def setdiff_flat(ints1, ints2):
        # https://stackoverflow.com/a/62407582
        combined = torch.cat((ints1, ints2),dim=-1)
        uniques, counts = combined.unique(return_counts=True,dim=-1)
        difference = uniques[counts == 1]
        # intersection = uniques[counts > 1]
        return difference

    @staticmethod
    def hull_det(points, hull):
        hull_points = p3d_ex.multi_gather_tris(points, hull.contiguous()) # simplices, verts, dims
        
        # # if splitting normal and reference point
        # hull_vecs = torch.nn.functional.pad(hull_points[...,1:,:] - hull_points[...,0:1,:],[0,0,0,1])
        # hull_vecs[...,-1,-1] = 1
        if hull_points.shape[-1] == (hull_points.shape[-2]+1):
            A_matrices = torch.nn.functional.pad(hull_points,[0,0,0,1])
            A_matrices[...,-1,-1] = 1
        else:
            A_matrices = hull_points
        
        # # I feel like this is close to a real thing, but can't find evidence of it
        # hull_points = torch.nn.functional.pad(hull_points,[0,1]).unsqueeze(0).transpose(-2,-1)
        # data_points = torch.nn.functional.pad(points.unsqueeze(-1).unsqueeze(0),[0,0,0,1],value=1)
        # shape = list(torch.broadcast_shapes(hull_points.shape[:-2],data_points.shape[:-2])) + [-1,-1]
        # A_matrices = torch.cat((hull_points.expand(shape),data_points.expand(shape)),dim=-1)
        
        det_A = torch.linalg.det(A_matrices)
        return det_A, A_matrices

    @staticmethod
    def comparePoints(points, planes):
        distances = torch.linalg.vecdot(planes.unsqueeze(-2), points.unsqueeze(-3))

        return distances

    @staticmethod
    def computePlaneCross(points):
        # generalized cross product to create planes
        # https://madoshakalaka.github.io/2019/03/02/generalized-cross-product-for-high-dimensions.html

        points_reshape = points.unsqueeze(-3).expand(list(points.shape[:-2])+[7,6,7])
        identity_shape = list(points_reshape.shape)[:-3]+[7,1,7]
        identity = torch.eye(7,device=points.device,dtype=points.dtype).reshape([1,7,1,7]).expand(identity_shape)
        points_with_eye = torch.cat((points_reshape,identity),dim=-2)
        planes = torch.linalg.det(points_with_eye)
        planes = planes / torch.linalg.vector_norm(planes[...,:-1],dim=-1,keepdim=True)
        # planes[...,-1] = torch.mean(torch.linalg.vecdot(points[...,:-1],planes[...,:-1]),dim=-1)

        return planes

    @staticmethod
    def computePlaneLeastSquare(points):
        planes = torch.linalg.svd(points).Vh[...,-1,:]
        planes = planes / torch.linalg.vector_norm(planes[...,:-1],dim=-1,keepdim=True)
        # planes[...,-1] = torch.mean(torch.linalg.vecdot(points[...,:-1],planes[...,:-1]),dim=-1)

        return planes


class qHullTorch(torch.autograd.Function):
    @staticmethod
    def forward(_, miniG):
        # both fingers hit a backface, so convex hull will throw an error
        if torch.all(miniG == 0):
            return (torch.tensor([range(6)],dtype=torch.long,device=miniG.device),
                    torch.zeros((1,7),dtype=torch.long,device=miniG.device))
        # I don't remember why I was afraid of this, hasn't happened in memory
        if torch.any(torch.isnan(miniG)):
            breakpoint()
        miniGnumpy = miniG.numpy(force=True)
        try:
            hull = ConvexHull(miniGnumpy) # ,qhull_options='QJ'
        except: 
            try:
                hull = ConvexHull(miniGnumpy,qhull_options='Qs') # ,qhull_options='QJ' 'Qs' # search all initial
            except:
                hull = ConvexHull(miniGnumpy,qhull_options='QJ') # ,qhull_options='QJ' 'Qs' # search all initial
        return ( torch.tensor(hull.simplices,dtype=torch.long,device=miniG.device),
                torch.tensor(hull.equations,dtype=miniG.dtype,device=miniG.device) )
    @staticmethod
    def backward(_, grad_output):
        return None
    
class ComForceClosureParallelJawQualityFunction(ParallelJawQualityFunction):
    """Measures the distance to the estimated center of mass for antipodal
    parallel-jaw grasps."""
    def __init__(self, config):
        self._antipodality_pctile = config["antipodality_pctile"]
        ParallelJawQualityFunction.__init__(self, config)
    def quality(self, state, actions, params=None):
        """Given a parallel-jaw grasp, compute the distance to the center of
        mass of the grasped object.

        Parameters
        ----------
        state : :obj:`Pytorch3D Meshes object `
            A Meshes object of size 1 containing a watertight mesh
        action: :obj:`Grasp`
            A suction grasp in image space that encapsulates center and axis
        params: dict
            Stores params used in computing quality.

        Returns
        -------
        :obj:`numpy.ndarray`
            Array of the quality for each grasp.
        """

        with record_function("solveForIntersection"):
            actions, faces_index, contactFound = actions.solveForIntersection(state)
       
        # Compute antipodality.
        antipodality_q = ParallelJawQualityFunction.friction_cone_angle(self, actions)

        # Can rank grasps, instead of compute absolute score. Only makes sense if seeding many grasps
        # antipodality_thresh = abs(
        #     np.percentile(antipodality_q, 100 - self._antipodality_pctile))
        
        max_q = torch.norm(torch.diff(state.get_bounding_boxes(),1,2))
        quality = torch.ones(list(actions.axis3D.shape)[:-1],device=actions.axis3D.device, dtype=actions.axis3D.dtype) * max_q
        
        in_force_closure = ParallelJawQualityFunction.force_closure(self, actions)
        
        dist = torch.norm(actions.center3D - actions.object_com,dim=-1)
        quality[in_force_closure] = dist[in_force_closure]
        # some kind of damped sigmoid, not sure where it comes from
        e_inverse = torch.exp(torch.tensor(-1))
        quality = (torch.exp(-quality / max_q) - e_inverse) / (1 - e_inverse)

        return quality


# Our code
def pytorch_setup():
        # set PyTorch device, use cuda if available
        if torch.cuda.is_available():
                device = torch.device("cuda:0")
                torch.cuda.set_device(device)
        else:
                print("cuda not available")
                device = torch.device("cpu")

        lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])
        dist = torch.linspace(0.5,0.7,4).reshape(1,1,-1).expand(6,6,4).reshape(-1,1)
        elev = torch.linspace(80,100,6).reshape(1,-1,1).expand(6,6,4).reshape(-1,1)
        azim = torch.linspace(-5,5,6).reshape(-1,1,1).expand(6,6,4).reshape(-1,1)

    # camera with info from gqcnn primesense
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)        # camera located above object, pointing down
        fl = torch.tensor([[525.0]],requires_grad=True)
        pp = torch.tensor([[319.5, 239.5]],requires_grad=True)
        im_size = torch.tensor([[480, 640]])

        camera = PerspectiveCameras(focal_length=fl, principal_point=pp, in_ndc=False, image_size=im_size, device=device, R=R, T=T)

        raster_settings = RasterizationSettings(
                 image_size=(480, 640),  # image size (H, W) in pixels
                 blur_radius=0.0,
                 faces_per_pixel=1
        )

        rasterizer = MeshRasterizer(
                 cameras = camera,
                 raster_settings = raster_settings
        )

        renderer = MeshRenderer(
                 rasterizer = rasterizer,
                 shader = SoftPhongShader(
                          device = device,
                          cameras = camera,
                          lights = lights
                 )
        )
    
        return renderer, device    

def test_wine():



    renderer, device = pytorch_setup()
    with record_function("load_obj"):
        verts, faces_idx, _ = load_obj("adv/adv-grasp/data/Wineglass_800_tex.obj")
    faces = faces_idx.verts_idx
    verts_rgb = torch.ones_like(verts)[None]
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)],
        textures=textures
    )
    config_dict = {
        "torque_scaling":1000,
        "soft_fingers":1,
        "friction_coef": 0.8, # TODO use 0.8 in practice
        "antipodality_pctile": 1.0 
    }

    com_qual_func = CannyFerrariQualityFunction(config_dict, min_quality=1)

    eye = torch.tensor([[0.0, 0.6, 0.0]])	# 
    up = torch.tensor([[0.0, 0.0, 1.0]])
    at = torch.tensor([[0.0, 0.0, 0.0]])
    R, T = look_at_view_transform(eye=eye, up=up, at=at)	# camera located above object, pointing down

    # camera intrinsics
    fl = torch.tensor([[525.0]])
    pp = torch.tensor([[319.5, 239.5]]) 
    im_size = torch.tensor([[480, 640]])

    camera = PerspectiveCameras(focal_length=fl, principal_point=pp, in_ndc=False, image_size=im_size, device=device, R=R, T=T)[0]


    center3D = torch.tensor([[0.0, -0.00, -0.034]], device=device,requires_grad=True)
    axis3D   = torch.tensor([[1.0,0.0,0.0]], device=device,requires_grad=True)
    graspObj = GraspTorch(world_center=center3D, world_axis=axis3D, width=0.05,
                        friction_coef=config_dict["friction_coef"], 
                        torque_scaling=config_dict["torque_scaling"],camera_intr=camera)
    graspObj = graspObj.apply_to_mesh(mesh,is_watertight=False)
    graspObj.write_obj('wine_grasp.obj',include_coordinate=True,include_line_o_action=True)
    original_quality = com_qual_func.quality(mesh, graspObj)   
    print("original (w/backface): ", original_quality)

    graspObj = graspObj.apply_to_mesh(mesh, ignore_backface_check=True)
    original_quality = com_qual_func.quality(mesh, graspObj)   
    print("original (ignore backface): ", original_quality)
    axis3D.retain_grad() 
    center3Dyz = center3D[:,1:].clone()
    center3Dyz.retain_grad() 
    
    optimizer = optim.Rprop([axis3D, center3Dyz], lr=0.0001)
    #optimizer = optim.SGD([center3D], lr=0.25, momentum=0.0)
    print('original-grasp', axis3D.squeeze().numpy(force=True), center3D.squeeze().numpy(force=True))
    for i in range(20):
        optimizer.zero_grad()
        center3D=torch.cat((torch.zeros_like(center3D[:,0:1]),(center3Dyz)),dim=-1)
        graspObj = GraspTorch(world_center=center3D, world_axis=axis3D, width=0.05,
                        friction_coef=config_dict["friction_coef"], torque_scaling=config_dict["torque_scaling"])
        noised_grasps = graspObj.generateNoisyGrasps(25)
        noised_grasps = noised_grasps.apply_to_mesh(mesh, ignore_backface_check=True)
        graspObj.write_obj(f'optim-grasp{i}.obj',include_coordinate=True,include_line_o_action=True)
        noised_tensor = com_qual_func.quality(mesh, noised_grasps)
        #com_qual_func.savemat(f'robust_sgd_iterates{i}.mat')
        qual_tensor = torch.sum(torch.nn.functional.relu(-(noised_tensor - 0.004)))
        # com_qual_func.savemat(f'quality_out{i}.mat')
        qual_tensor.backward()
        print('iteration: ', i)
        print('raw cf: ',noised_tensor.squeeze().numpy(force=True))
        print('count success: ',np.sum(noised_tensor.squeeze().numpy(force=True) > 0.004))
        print('loss score:',qual_tensor.squeeze().numpy(force=True))
        print('grasp-update', axis3D.grad.squeeze().numpy(force=True), center3Dyz.grad.squeeze().numpy(force=True))
        optimizer.step()
        print('new-grasp', axis3D.squeeze().numpy(force=True), center3D.squeeze().numpy(force=True))
        print()

    # test slicing
    a = noised_grasps[0:5]
    b = a[torch.tensor((0,0,1,0,1),dtype=torch.bool,device=device)]
    c = b[0]
def test_quality():
    # load PyTorch3D mesh from .obj file
    renderer, device = pytorch_setup()
    with record_function("load_obj"):
        verts, faces_idx, _ = load_obj("adv/adv-grasp/data/new_barclamp.obj")
    faces = faces_idx.verts_idx
    verts_rgb = torch.ones_like(verts)[None]
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)],
        textures=textures
    )
    unconnectivity = mp(mesh,forceNormalDist=False)
    # with record_function("distfunction"):
        # print(self_collision(mesh, unconnectivity,forceNormalDist=False))
    with record_function("distfunction_n"):
        print(mp.self_collision(unconnectivity,mesh))
    config_dict = {
        "torque_scaling":1000,
        "soft_fingers":1,
        "friction_coef": 0.8, # TODO use 0.8 in practice
        "antipodality_pctile": 1.0 
    }
    print("mesh vol:", mp.compute_mesh_volume(mesh).numpy(force=True))
    print("mesh bb vol:", mp.compute_mesh_bounding_volume(mesh).numpy(force=True))
    print("mesh hull vol:", mp.compute_mesh_hull_volume(mesh).numpy(force=True))
    # Test intersection finding
    test_grasps_compute = []
    test_grasps_set = []
    dicts = []
    for i in range(0,12):
        f = open('adv/adv-grasp/data/data/data'+str(i)+'.json')
        dicts.append(json.load(f))
        center3D = torch.tensor([dicts[-1]['pytorch_w_center']],device=device,requires_grad=True)
        axis3D = torch.tensor([dicts[-1]['pytorch_w_axis']],device=device,requires_grad=True)
        axis3D.retain_grad()        

        test_grasps_compute.append(GraspTorch(world_center=center3D, world_axis=axis3D, width=0.05,
                                               friction_coef=config_dict["friction_coef"], torque_scaling=config_dict["torque_scaling"]))
        test_grasps_compute[-1] = test_grasps_compute[-1].apply_to_mesh(mesh)
        print("contact points:", i)
        print(test_grasps_compute[-1].contact_points.squeeze().numpy(force=True))
        print(np.array(dicts[-1]['contact_points']))
        print("contact normals:", i)
        print(test_grasps_compute[-1].contact_normals.squeeze().numpy(force=True))
        print(-torch.nn.functional.normalize(torch.tensor(dicts[-1]['normals_1'],device=device).transpose(0,1).double(),dim=-1).numpy(force=True)) # .json has inward normal
        test_grasps_set.append(GraspTorch(world_center=center3D, world_axis=axis3D, width=0.05,
                                          friction_coef=config_dict["friction_coef"], torque_scaling=config_dict["torque_scaling"]))
        test_grasps_set[-1].contact_points = torch.tensor(dicts[-1]['contact_points'],device=device).unsqueeze(1).double()
        test_grasps_set[-1].contact_normals = -torch.nn.functional.normalize(torch.tensor(dicts[-1]['normals_1'],device=device).transpose(0,1).unsqueeze(1).double(),dim=-1)
        test_grasps_set[-1].applied_to_object=True
        test_grasps_set[-1].object_com=test_grasps_compute[-1].object_com.clone()
        test_grasps_set[-1].mesh = mesh

    # Test intermediate values, forces/torques
    tests = [[3e-3, 1e-1, test_grasps_set], [1, 1, test_grasps_compute]]
    for test in tests:
        for i in range(len(dicts)):
            n_force =test[2][i].normal_force_magnitude.unsqueeze(-1)

            cone = torch.mul(test[2][i].friction_cone, n_force)
            torques = torch.mul(test[2][i].friction_torques, n_force)

            # print("n_force:",list(n_force.squeeze().numpy(force=True)),(dicts[i]['n_0'],dicts[i]['n_1']),i)
            # print("cone:", i)
            # print(cone.transpose(0,1).numpy(force=True).reshape(16,3))
            # print(np.array(dicts[i]['forces_1']).T)
            # np.testing.assert_allclose(cone.transpose(0,1).numpy(force=True).reshape(16,3), 
                            # np.array(dicts[i]['forces_1']).T,
                            # atol=test[0],rtol=test[1])
        
            # print("torqes:", i)
            # print(torques.transpose(0,1).numpy(force=True).reshape(16,3))
            # print(np.array(dicts[i]['torques_1']).T)
            # np.testing.assert_allclose(torques.transpose(0,1).numpy(force=True).reshape(16,3),
            #                 np.array(dicts[i]['torques_1']).T,
            #                 atol=test[0],rtol=test[1])
            
            com_qual_func = CannyFerrariQualityFunction(config_dict)
            G = test[2][i].grasp_matrix
            G_unwrapped = G.reshape((G.shape[0]*G.shape[1], -1, 6))
            #G = G[list(range(8))+list(range(10,18))+list(range(8,10))+list(range(18,20)),:]
            # print("G:", i)
            # print(G_unwrapped.transpose(0,1).numpy(force=True).reshape(-1,6)[16:,:])
            # print(np.array(dicts[i]['G']).T[16:,:])
            # np.testing.assert_allclose(G_unwrapped.transpose(0,1).numpy(force=True).reshape(-1,6)[16:,:],
            #                 np.array(dicts[i]['G']).T[16:,:],
            #                 atol=test[0],rtol=test[1])
    # TODO, test grasp matrix, need to account for order of soft finger torsion terms
    # Test dists
    # for i in range(len(dicts)):
    #     com_qual_func = CannyFerrariQualityFunction(config_dict)
    #     G = com_qual_func.compute_grasp_matrix(mesh, test_grasps_set[i])
    #     G_unwrapped = G.reshape((G.shape[0]*G.shape[1], -1, 6))
    #     dists,_ = minHull.distWrap(G_unwrapped)
    #     np.testing.assert_allclose(dists.reshape((-1)).numpy(force=True),
    #             np.array(dicts[i]['dists']).T,
    #             atol=0.04,rtol=test[1])

    # Test final grasp quality
    tests = [[1e-3, 1e-5, test_grasps_set], [1e-2, 1e-3, test_grasps_compute]]
    print("atol", tests[0][0], "rtol", tests[0][1],"atol", tests[1][0], "rtol", tests[1][1])
    print("torch (w/o col), torch (w/ col), dexnet")
    # for i in range(len(dicts)):            
    #         com_qual_func = CannyFerrariQualityFunction(config_dict)
    #         # print("set")
    #         torch_quality_no_col = com_qual_func.quality(mesh, tests[0][2][i]).numpy(force=True)[0]
    #         np.testing.assert_allclose(torch_quality_no_col,
    #                 np.array(dicts[i]['ferrari_canny_fc08']).T,
    #                 atol=tests[0][0],rtol=tests[0][1])
    #         # print("compute")
    #         torch_quality_col = com_qual_func.quality(mesh, tests[1][2][i]).numpy(force=True)[0]
    #         np.testing.assert_allclose(torch_quality_col,
    #                 np.array(dicts[i]['ferrari_canny_fc08']).T,
    #                 atol=tests[1][0],rtol=tests[1][1])
    #         print(torch_quality_no_col[0],",", torch_quality_col[0],",",
    #         dicts[i]['ferrari_canny_fc08'],";")

    # #torch.autograd.set_detect_anomaly(True)
    center3D = torch.tensor([dicts[2]['pytorch_w_center']],device=device,requires_grad=True)
    axis3D = torch.tensor([dicts[2]['pytorch_w_axis']],device=device,requires_grad=True)
    axis3D.retain_grad() 
    center3D.retain_grad()
    graspObj = GraspTorch(world_center=center3D, world_axis=axis3D, width=0.05,
                           friction_coef=config_dict["friction_coef"], torque_scaling=config_dict["torque_scaling"]).apply_to_mesh(mesh)
    print('before', axis3D.grad)
    qual_tensor = com_qual_func.quality(mesh, graspObj)
    print('quality', qual_tensor)
    qual_tensor.backward(inputs=axis3D)
    print('after', axis3D.grad)
    print('value', axis3D)

    noised_grasps = graspObj.generateNoisyGrasps(25)
    noised_tensor = com_qual_func.quality(mesh, noised_grasps)
    torch.set_printoptions(precision=8)
    print(qual_tensor)
    print(noised_tensor)
    torch.set_printoptions(precision=4)

    # center3D = torch.tensor([dicts[2]['pytorch_w_center']],device=device,requires_grad=True)
    # axis3D = torch.tensor([dicts[2]['pytorch_w_axis']],device=device,requires_grad=True)
    # axis3D.retain_grad() 
    # center3D.retain_grad()
    # optimizer = optim.Rprop([axis3D, center3D], lr=0.00001)
    # #optimizer = optim.SGD([center3D], lr=0.25, momentum=0.0)
    # print('original-grasp', axis3D.squeeze().numpy(force=True), center3D.squeeze().numpy(force=True))
    # for i in range(20):
    #     optimizer.zero_grad()
    #     graspObj = GraspTorch(center3D, axis3D=axis3D, width=0.05,
    #                     friction_coef=config_dict["friction_coef"], torque_scaling=config_dict["torque_scaling"]).apply_to_mesh(mesh)
    #     noised_grasps = graspObj.generateNoisyGrasps(25)
    #     noised_tensor = com_qual_func.quality(mesh, noised_grasps)   
    #     com_qual_func.savemat(f'robust_sgd_iterates{i}.mat')
    #     qual_tensor = torch.sum(torch.nn.functional.relu(-(noised_tensor - 0.002)))
    #     # com_qual_func.savemat(f'quality_out{i}.mat')
    #     qual_tensor.backward()
    #     print('iteration: ', i)
    #     print('raw cf: ',noised_tensor.squeeze().numpy(force=True))
    #     print('count success: ',np.sum(noised_tensor.squeeze().numpy(force=True) > 0.002))
    #     print('loss score:',qual_tensor.squeeze().numpy(force=True))
    #     print('grasp-update', axis3D.grad.squeeze().numpy(force=True), center3D.grad.squeeze().numpy(force=True))
    #     optimizer.step()
    #     print('new-grasp', axis3D.squeeze().numpy(force=True), center3D.squeeze().numpy(force=True))
    #     print()

    # center3D = torch.tensor([dicts[2]['pytorch_w_center']],device=device,requires_grad=True)
    # axis3D = torch.tensor([dicts[2]['pytorch_w_axis']],device=device,requires_grad=True)
    # axis3D.retain_grad() 
    # center3D.retain_grad()
    # optimizer = optim.SGD([axis3D, center3D], lr=0.001, momentum=0.0)
    # for i in range(10):
    #     optimizer.zero_grad()
    #     graspObj = GraspTorch(center3D, axis3D=axis3D, width=0.05,
    #                     friction_coef=config_dict["friction_coef"], torque_scaling=config_dict["torque_scaling"]).apply_to_mesh(mesh)
   
    #     qual_tensor = -com_qual_func.quality(mesh, graspObj)
    #     # com_qual_func.savemat(f'quality_out{i}.mat')
    #     qual_tensor.backward()
    #     print('quality:',-qual_tensor.squeeze().numpy(force=True))
    #     print('grad', axis3D.grad.squeeze().numpy(force=True), center3D.grad.squeeze().numpy(force=True))
    #     print('value', axis3D.squeeze().numpy(force=True), center3D.squeeze().numpy(force=True))
    #     optimizer.step()
    

    center2d = torch.tensor([[344.3809509277344, 239.4164276123047]],device=device)
    angle = torch.tensor([[0.3525843322277069 + math.pi]],device=device)
    depth = torch.tensor([[0.5824159979820251]],device=device)
    width = torch.tensor([[0.05]],device=device)

    grasp1 = GraspTorch(im_center=center2d, im_angle=angle, depth=depth, width=width, camera_intr=renderer.rasterizer.cameras,friction_coef=config_dict["friction_coef"], torque_scaling=config_dict["torque_scaling"]) 
        

    center3D = torch.tensor([[ 0.027602000162005424, 0.017583999782800674, -9.273400064557791e-05]], device=device)
    axis3D   = torch.tensor([[-0.9384999871253967, 0.2660999894142151, -0.22010000050067902]], device=device)

    grasp2 = GraspTorch(world_center=center3D, world_axis=axis3D, width=width, camera_intr=renderer.rasterizer.cameras,friction_coef=config_dict["friction_coef"], torque_scaling=config_dict["torque_scaling"]) 
    grasp2.make2D(updateCamera=False)
    
    gq_qual = GQCNNQualityFunction(renderer=p3d_ex.Renderer(device=device),weights_path='adv/adv-grasp/weights.npy')
    gqcnn_qual = gq_qual.quality(state=mesh, actions=grasp2[0])
    print(f'gqcnn qualit {gqcnn_qual}')

    center3D = torch.tensor([[-0.03714486211538315, -0.029467197135090828, 0.01168159581720829]], device=device)
    axis3D   = torch.tensor([[-0.974246621131897, -0.19650164246559143, -0.11059238761663437]], device=device)

    grasp3 = GraspTorch(world_center=center3D, world_axis=axis3D, width=width, camera_intr=renderer.rasterizer.cameras,friction_coef=config_dict["friction_coef"], torque_scaling=config_dict["torque_scaling"]) 
    # Call ComForceClosureParallelJawQualityFunction init with parameters from gqcnn (from gqcnn/cfg/examples/replication/dex-net_2.1.yaml 

    # with record_function("FastAntipodalityFunction"):
    #     com_qual_func = ComForceClosureParallelJawQualityFunction(config_dict)

    # # Call quality with the Grasp2D and mesh
    #     com_qual_func.quality(mesh, grasp3)

    with record_function("CannyFerrari"):
        com_qual_func = CannyFerrariQualityFunction(config_dict)
        com_qual_func.quality(mesh, grasp3)
    com_qual_func.write_obj('CannyFerrari_test.obj')
    com_qual_func.savemat('CannyFerrari_test.mat')
    with record_function("minWeight"):
        com_qual_func = minWeightQualityFunction(config_dict)
        com_qual_func.quality(mesh, grasp3)
    com_qual_func.write_obj('minWeight_test.obj')

    with record_function("robustMinWeight"):
        com_qual_func = RobustMinWeightQualityFunction(config_dict)
        com_qual_func.quality(mesh, grasp3)


# def test_stein():

#     renderer, device = pytorch_setup()
#     with record_function("load_obj"):
#         verts, faces_idx, _ = load_obj("adv/adv-grasp/data/new_barclamp.obj")
#     faces = faces_idx.verts_idx
#     verts_rgb = torch.ones_like(verts)[None]
#     textures = TexturesVertex(verts_features=verts_rgb.to(device))

#     mesh = Meshes(
#         verts=[verts.to(device)],
#         faces=[faces.to(device)],
#         textures=textures
#     )
#     config_dict = {
#         "torque_scaling":1000,
#         "soft_fingers":1,
#         "friction_coef": 0.8, # TODO use 0.8 in practice
#         "antipodality_pctile": 1.0 
#     }
#     i=0
#     f = open('datadatadata'+str(i)+'.json')
#     grasp_dict = json.load(f)

#     center3D = torch.tensor([grasp_dict['pytorch_w_center']],device=device,requires_grad=True)
#     axis3D = torch.tensor([grasp_dict['pytorch_w_axis']],device=device,requires_grad=True)
#     axis3D.retain_grad() 
#     center3D.retain_grad()

#     com_qual_func = CannyFerrariQualityFunction(config_dict)
    
#     GT = lambda center,axis:GraspTorch(center, axis3D=axis, width=0.05,friction_coef=config_dict["friction_coef"], torque_scaling=config_dict["torque_scaling"]).apply_to_mesh(mesh)
#     graspObj = GraspTorch(center3D, axis3D=axis3D, width=0.05,friction_coef=config_dict["friction_coef"], torque_scaling=config_dict["torque_scaling"]).apply_to_mesh(mesh)

#     grasp = GT(center3D,axis3D)
#     qual = com_qual_func.quality(mesh, grasp)
#     qual.backward(inputs=[axis3D, center3D])
#     print('quality', qual)
#     print('center', center3D, center3D.grad)
#     print('axis3D', axis3D, axis3D.grad)

#     prob = graspQualityOpt(com_qual_func,GT,mesh)
#     num_particles = 10
#     stein_problem = SteinWrapper(prob,num_particles,repulsive_weight=1e-3)


#     solver = BFGSMethod(stein_problem, alpha=0.01, rho=0.2,min_alpha=1e-5)
#     initial_solution = torch.cat((axis3D, center3D),dim=1).T.numpy(force=True)
#     particles = np.random.normal(initial_solution,1e-4,(num_particles,6,1))
#     stacked_particles = particles.reshape((-1,1))
#     result = solver.optimize(stacked_particles,max_iterations=10)
#     result.display()
#     new_particles = result.iterates.reshape((result.iterates.shape[0],-1,6,1))
#     tensor_iterates = prob.make_tensor(new_particles)
#     for iterate_index in range(new_particles.shape[0]):
#         for particle_index in range(new_particles.shape[1]):
#             axis = tensor_iterates[iterate_index,particle_index, :3].T
#             center = tensor_iterates[iterate_index,particle_index, 3:].T
#             grasp = GT(center,axis)
#             com_qual_func.quality(mesh, grasp)
#             com_qual_func.savemat(f'stein_iterates{iterate_index}_{particle_index}.mat')

# class graspQualityOpt(Problem):
#     def __init__(self, qualityObj=None, grasp=None, mesh=None):
#         self.grasp = grasp
#         self.device = mesh.device
#         self.qualityObj = qualityObj
#         self.mesh = mesh
#         super().__init__()
#         self.make_tensor = self.test
#     def test(self, x):
#         return torch.DoubleTensor(x).to(self.device)
#     def size(self):
#         return 6
#     def tensor_batch_cost(self, x):
#         axis = x[:,:3,0]
#         center = x[:,3:,0]
#         return -self.qualityObj.quality(self.mesh,self.grasp(center,axis)).squeeze(-1)
#     def tensor_cost(self, x):
#         axis = x[:3].T
#         center = x[3:].T
#         return -self.qualityObj.quality(self.mesh,self.grasp(center,axis))

def test_dist():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    with record_function("load_obj"):
        verts, faces_idx, _ = load_obj("adv/adv-grasp/data/new_barclamp.obj")
    faces = faces_idx.verts_idx
    verts_rgb = torch.ones_like(verts)[None]
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)],
        textures=textures
    )
    with record_function("test_compilation"):
        unconnectivity = torch.jit.script(mp(mesh,forceNormalDist=False))
    verts = mesh.verts_packed()
    verts.requires_grad_(True)
    distances,barys = unconnectivity( verts )
    print(distances)
    distances,barys = unconnectivity( verts )
    print(distances)

    with record_function("test_compiled"):
        for i in range(20):
            distances,barys = unconnectivity( verts )
            print(distances)

    with record_function("test_construction"):
        unconnectivity = mp(mesh,forceNormalDist=False)
    verts = mesh.verts_packed()
    verts.requires_grad_(True)
    with record_function("test_regular"):
        for i in range(20):
            distances,barys = unconnectivity( verts )
            print(distances)

if __name__ == "__main__":
    #minHull.apply(torch.tensor(dict['G']).transpose(0,1).reshape((20,1,1,6)))
    np.set_printoptions(edgeitems=30, linewidth=100)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False, with_stack=False) as prof:
        with record_function("test_quality"):
            #model(inputs)
            with torch.enable_grad():        
                # test_stein()
                # test_quality()
                # test_wine()
                test_dist()
                
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))
    prof.export_chrome_trace("trace-solve.json")