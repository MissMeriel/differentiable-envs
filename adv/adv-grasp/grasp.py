import torch
import torch.optim as optim
import os
from torch.masked import masked_tensor
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import math
import json
import itertools
import logging
from pytorch3d.io import load_obj, save_obj
from pytorch3d import _C
import pytorch3d.transforms as tf
from scipy.spatial import ConvexHull
from scipy.io import savemat
from pytorch3d.structures import Meshes, Pointclouds
import pytorch3d_ext as p3d_ex
from pytorch3d.renderer import (
        look_at_view_transform,
)
# used by block print
import sys
# used to save grasp visualization
import trimesh

class GraspTorch(object):

    logger = logging.getLogger('select_grasp')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    """Parallel-jaw grasp in image space.

    Attributes
    ----------
    center : :obj:`autolab_core.Point`
        Point in image space.
    angle : float
        Grasp axis angle with the camera x-axis.
    depth : float
        Depth of the grasp center in 3D space.
    width : float
        Distance between the jaws in meters.
    camera_intr : :obj:`autolab_core.CameraIntrinsics`
        Frame of reference for camera that the grasp corresponds to.
    contact_points : list of :obj:`numpy.ndarray`
        Pair of contact points in image space.
    contact_normals : list of :obj:`numpy.ndarray`
        Pair of contact normals in image space.
    """
    def __init__(self,
                 world_center=None,
                 im_center=None,
                 im_angle=0.0,
                 depth=1.0,
                 width=0.05,
                 camera_intr=None,
                 contact_points=None,
                 c0=None,
                 c1=None,
                 contact_normals=None,
                 world_axis=None,
                 quality=None,
                 num_cone_faces=8, 
                 friction_coef=0.8,
                 torque_scaling=None,
                 im_axis=None, 
                 prediction=None, 
                 oracle_method="pytorch", 
                 oracle_robust=None, 
                 objf=None,):
        self.width = width
        self.camera_intr = camera_intr
        self.num_cone_faces = num_cone_faces
        self.friction_coef = friction_coef
        self.applied_to_object = False
        self.quality = quality
        self.torque_scaling = torque_scaling
        self.finger_radius=0.005

        if world_center is not None or world_axis is not None: # 3D grasp
            if world_center is not None and world_axis is not None:
                if world_center.shape[-1] != 3:
                    raise ValueError("world_center should have three coordinates")
                if world_axis.shape[-1] != 3:
                    raise ValueError("world_axis should have three coordinates")
                self.world_center = world_center.double()
                self.world_axis = world_axis.double()
            else:
                raise ValueError("If providing a 3D grasp, must include both world_axis and world_center")
        # self.axis3D = torch.nn.functional.normalize(self.axis3D,dim=-1)

        if im_center is not None : # 2D grasp
            if im_center.shape[-1] != 2:
                raise ValueError("im_center should have two coordinates")
            self.im_center = im_center.double()
            
            self.im_angle = im_angle.double()
            if len(self.im_angle.shape) == 0:
                self.im_angle = self.im_angle.unsqueeze(-1)
            self.depth = depth.double()
            if len(self.depth.shape) == 0:
                self.depth = self.depth.unsqueeze(-1)
            if im_axis is not None:
                self.im_axis = im_axis
            else:
                self.im_axis = torch.cat((torch.cos(self.im_angle.unsqueeze(-1)), torch.sin(self.im_angle).unsqueeze(-1)), dim=-1) # TODO, do we need to check if last dim is 1?

        if im_center is not None and world_center is None:
            center_in_camera = torch.cat((self.im_center, self.depth), dim=-1)
            self.world_center = camera_intr.unproject_points(center_in_camera.float(), world_coordinates=True).double()
            axis_in_camera = torch.cat((self.im_axis, torch.zeros(list(self.im_axis.shape)[:-1]+[1], device=self.im_axis.device, dtype=self.im_axis.dtype)),dim=-1)
            self.world_axis =camera_intr.get_world_to_view_transform().inverse().transform_normals(axis_in_camera.float()).double()
        
        if contact_points is not None:
            self.contact_points = contact_points.double()
        elif c0 is not None and c1 is not None:
            self.contact_points = torch.cat((c0.unsqueeze(-2), c1.unsqueeze(-2)),dim=-2)
            
        if contact_normals is not None:
            self.contact_normals = contact_normals.double()

    @classmethod
    def read(cls, fname, device=None):
        """Reads a JSON file fname with saved grasp information and initializes"""
        if device is None:
            if torch.cuda.is_available():
                device = torch.device(f"cuda:0")
                torch.cuda.set_device(device)
            else:
                print("cuda not available")
                device = torch.device("cpu")
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
        
        if "c0" in dictionary:
            if dictionary["c0"] and dictionary["c1"]:
                dictionary["c0"] = torch.from_numpy(np.array(dictionary["c0"])).to(device).float()
                dictionary["c1"] = torch.from_numpy(np.array(dictionary["c1"])).to(device).float()

        if "contact_points" in dictionary and dictionary["contact_points"] is not None:
            dictionary["contact_points"] = torch.from_numpy(np.array(dictionary["contact_points"])).to(device).float()
        
        if isinstance(dictionary["quality"], list):
            # if not isinstance(dictionary["quality"][0], list):
            # 	dictionary["quality"] = torch.tensor([dictionary["quality"][1]]).to(device).float()
            # else:
            dictionary["quality"] = torch.from_numpy(np.array(dictionary["quality"])).to(device).float()

        return cls.init_from_dict(dictionary,device=device)

    @classmethod
    def read_batch(cls, fnames, device=None):

        if device is None:
            if torch.cuda.is_available():
                device = torch.device(f"cuda:0")
                torch.cuda.set_device(device)
            else:
                print("cuda not available")
                device = torch.device("cpu")

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
                if isinstance(v, list) and len(v)==1:
                    v = v[0]
                batch_dict[k].append(v)

        if "objf" in batch_dict.keys():
            obj_lst = batch_dict["objf"]
            obj = obj_lst.pop(0)
            for o in obj_lst:
                if o != obj:
                    cls.logger.error("read_batch - all grasps must be for the same grasp object.")
                    continue
            batch_dict["objf"] = o

        return cls.init_from_dict(batch_dict,device=device)

    @classmethod
    def init_from_dict(cls, dict, device=None):
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

        for key, value in dict.items():
            if isinstance(value, list) or isinstance(value, np.ndarray):
                try:
                    dict[key] = torch.tensor(value, device=device)
                except:
                    print(f'failed to convert {key} storing None')
                    dict[key] = None

        return GraspTorch(depth=dict["depth"], im_center=dict["im_center"], im_angle=dict["im_angle"], im_axis=dict["im_axis"], world_center=dict["world_center"], world_axis=dict["world_axis"], c0=dict["c0"], c1=dict["c1"], quality=dict["quality"], prediction=dict["prediction"], oracle_method=dict["oracle_method"], oracle_robust=dict["oracle_robust"], objf=dict["objf"])


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

    def __hash__(self):
        return hash(self)

    def __eq__(self, obj, EPS_world = 1e-5, EPS_image = 1e-3):
        
        if type(self) != type(obj):
            return False
        if self.world_center.shape != obj.world_center.shape or torch.max(torch.abs(torch.sub(self.world_center, obj.world_center))) > EPS_world: return False
        if self.world_axis.shape != obj.world_axis.shape or torch.max(torch.abs(torch.sub(self.world_axis, obj.world_axis))) > EPS_world: return False

        if hasattr(self,'depth') and hasattr(obj,'depth'):
            if self.depth.shape != obj.depth.shape or not torch.min(torch.eq(self.depth, obj.depth)): return False
        if hasattr(self,'im_center') and hasattr(obj,'im_center') is not None:
            if self.im_center.shape != obj.im_center.shape or torch.max(torch.abs(torch.sub(self.im_center, obj.im_center))) > EPS_image: return False
            if self.im_axis.shape != obj.im_axis.shape or torch.max(torch.abs(torch.sub(self.im_axis, obj.im_axis))) > EPS_world: return False
            if self.im_angle.shape != obj.im_angle.shape or torch.max(torch.abs(torch.sub(self.im_angle, obj.im_angle))) > EPS_world: return False
        if hasattr(self,'contact_points') and hasattr(obj,'contact_points'):
            if self.contact_points.shape != obj.contact_points.shape or torch.max(torch.abs(torch.sub(self.contact_points, obj.contact_points)).flatten()) > EPS_world: return False
        if hasattr(self,'objf') and hasattr(obj,'objf'):
            if self.objf.split("/")[-1] != obj.objf.split("/")[-1]:
                if not ((self.objf.split("/")[-1] in ["new_barclamp.obj", "bar_clamp.obj"]) and (obj.objf.split("/")[-1] in ["new_barclamp.obj", "bar_clamp.obj"])): return False
        return True

    def __str__(self):
        """Returns a string with grasp information in image coordinates"""
        p_str = "grasp:"
        
        if self.quality is not None:
            p_str += "\n\tquality: " + str(self.quality)
        if hasattr(self,'prediction') and self.prediction is not None:
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
        imc_list, imax_list, wc_list, was_list, contact_points_list, depth, angle, quality, prediction, oracle_method, robust, objf = None, None, None, None, None, None, None, None, None, None, None, None
        if hasattr(self,'im_center') and hasattr(self,'im_axis'):
            imc_list = self.im_center.clone().detach().cpu().numpy().tolist()
            imax_list = self.im_axis.clone().detach().cpu().numpy().tolist()
        if (self.world_center is not None) and (self.world_axis is not None):
            wc_list = self.world_center.clone().detach().cpu().numpy().tolist()
            was_list = self.world_axis.clone().detach().cpu().numpy().tolist()
        if hasattr(self,'contact_points'):
            contact_points_list = self.contact_points.clone().detach().cpu().numpy().tolist()
        # if depth is not None:
        # 	depth = self.depth.clone().detach().cpu().numpy().tolist()
        if hasattr(self,'depth'):
            depth = self.depth
            if isinstance(self.depth, torch.Tensor):
                depth = self.depth.clone().detach().cpu().numpy().tolist()
        if hasattr(self,'im_angle'):
            angle = self.im_angle
            if isinstance(self.im_angle, torch.Tensor):
                angle = self.im_angle.clone().detach().cpu().numpy().tolist()
        if hasattr(self,'quality'):
            quality = self.quality
            if isinstance(self.quality, torch.Tensor):
                quality = self.quality.clone().detach().cpu().numpy().tolist()
        if hasattr(self,'prediction'):
            prediction = self.prediction
            if isinstance(self.prediction, torch.Tensor):
                prediction = self.prediction.clone().detach().cpu().numpy().tolist()
        if hasattr(self, 'oracle_method'):
            oracle_method, robust = self.oracle_method, self.oracle_robust
            if isinstance(oracle_method, torch.Tensor):
                oracle_method = oracle_method.clone().detach().cpu().numpy().tolist()
            if isinstance(robust, torch.Tensor):
                robust = robust.clone().detach().cpu().numpy().tolist()
        
        objf = self.objf if hasattr(self,'objf') else "unknown"

        grasp_data = {
            "depth": depth,
            "im_center": imc_list,
            "im_axis": imax_list,
            "im_angle": angle,
            "world_center": wc_list,
            "world_axis": was_list,
            "contact_points": contact_points_list,
            "oracle_method": oracle_method,
            "oracle_robust": robust,
            "quality": quality,
            "prediction": prediction,
            "objf": objf
        }

        with open(fname, "w") as f:
            json.dump(grasp_data, f, indent=4)

    def __getitem__(self,key):

        width = self.width
        camera_intr=self.camera_intr
        num_cone_faces=self.num_cone_faces
        friction_coef=self.friction_coef
        torque_scaling=self.torque_scaling
    


        if isinstance(key, np.ndarray) or torch.is_tensor(key) or isinstance(key, slice) or isinstance(key, int):

            if isinstance(key, int):
                world_center = self.world_center.reshape((-1,3))[key]
                axis3D = self.world_axis.reshape((-1,3))[key]
                if hasattr(self, 'contact_points'):
                    if torch.numel(self.contact_points) == 0:
                        contact_points = self.contact_points
                    else:
                        contact_points = self.contact_points.reshape((-1,2,3))[key,:]
                else:
                    contact_points = None
                if hasattr(self, 'contact_normals'):
                    if torch.numel(self.contact_normals) == 0:
                        contact_normals = self.contact_normals
                    else:
                        contact_normals = self.contact_normals.reshape((-1,2,3))[key,:]
                else:
                    contact_normals = None       
                if hasattr(self,'im_center'):
                    im_center = self.im_center.reshape((-1,2))[key,:]

                    if torch.is_tensor(self.im_angle) and torch.numel(self.im_angle) > 1:
                        im_angle = self.im_angle.flatten()[key]
                    else:
                        im_angle = self.im_angle

                    if torch.is_tensor(self.depth) and torch.numel(self.depth) > 1 :
                        depth = self.im_angle.flatten()[key]
                    else:
                        depth = self.depth
                else:
                    im_angle, depth, im_center = None, None, None
            else:
                if torch.numel(self.world_center[...,0]) > 1:
                    world_center = self.world_center[key]
                else:
                    world_center = self.world_center
            
                if torch.is_tensor(self.world_axis) and torch.numel(self.world_axis[...,0]) > 1 :
                    axis3D = self.world_axis[key]
                else:
                    axis3D = self.world_axis
                
                if hasattr(self, 'contact_points'):
                    contact_points = self.contact_points[:,key]
                else:
                    contact_points = None

                if hasattr(self, 'contact_normals'):
                    contact_normals = self.contact_normals[:,key]
                else:
                    contact_normals = None       

                if hasattr(self,'im_center'):
                    if torch.numel(self.im_center[...,0]) > 1:
                        im_center = self.im_center[key]
                    else:
                        im_center = self.im_center


                    if torch.is_tensor(self.im_angle) and torch.numel(self.im_angle) > 1:
                        im_angle = self.im_angle[key]
                    else:
                        im_angle = self.im_angle

                    if torch.is_tensor(self.depth) and torch.numel(self.depth) > 1 :
                        depth = self.depth[key]
                    else:
                        depth = self.depth
                else:
                    im_angle, depth, im_center = None, None, None

            sliced = GraspTorch(world_center=world_center,
                 width=width,
                 camera_intr=camera_intr,
                 contact_points=contact_points,
                 contact_normals=contact_normals,
                 world_axis=axis3D,
                 num_cone_faces=num_cone_faces, 
                 friction_coef=friction_coef,
                 torque_scaling=torque_scaling,
                 im_angle=im_angle,
                 im_center=im_center,
                 depth=depth)


            if hasattr(self, 'mesh'):
                sliced.mesh = self.mesh
            if hasattr(self, 'object_com'):
                sliced.object_com = self.object_com
            if hasattr(self, 'contact_mask'):

                if torch.numel(self.contact_mask) <= 1:
                    # indexing will fail, but scalar can just direct assign
                    contact_mask = self.contact_mask
                else:
                    contact_mask = torch.zeros_like(self.contact_mask)
                    contact_inds = torch.nonzero(self.contact_mask.flatten()).squeeze(-1)
                    contact_mask[contact_inds[key]] = 1
                sliced.contact_mask = contact_mask

            sliced.applied_to_object = self.applied_to_object
            return sliced
    
    def num_grasps(self):
        return len(self)

    def __len__(self,):
        return torch.numel(self.world_axis[...,0])

    def make2D(self, updateCamera=False,camera_intr=None):
        if camera_intr==None:
            camera_intr = self.camera_intr
        else:
            self.camera_intr = camera_intr
        if camera_intr==None:
            # TODO error
            return None
        if len(self.world_axis.shape) == 1:
            world_axis = self.world_axis.reshape((1,3))
            world_center = self.world_center.reshape((1,3))
        else:
            world_axis = self.world_axis
            world_center = self.world_center

        if updateCamera: # in order to keep specified axis, we need to update the camera
            
            world_axis = camera_intr.get_world_to_view_transform().transform_normals(world_axis.float()).double() # in world
            
            cameraDir = torch.tensor([0.,0.,1.],device=self.world_axis.device,dtype=world_axis.dtype,requires_grad=True) # camera Z
            cameraDir = cameraDir.reshape([1]*(len(world_axis.shape)-1)+[3])
            cameraDir = cameraDir.expand(world_axis.shape)
            rotVec = torch.cross(cameraDir, world_axis, dim=-1) # vector orthogonal to both
            rotVecNorm = torch.linalg.vector_norm(rotVec,dim=-1) # includes angle information
            rotAngle = torch.asin( rotVecNorm ) - math.pi/2 # compare current angle to 90 deg, assume cameraDir, axis3D are unit
            rodVec = torch.unsqueeze(rotAngle/rotVecNorm,-1) * rotVec # axis scaled by angle

            rotToGraspAxis = tf.Rotate(tf.so3_exp_map(rodVec.reshape(-1,3)))
            transformOrig = camera_intr.get_world_to_view_transform()
            transformFull = transformOrig.compose(rotToGraspAxis)
            R = transformFull.get_matrix()[...,:3,:3]
            T = transformFull.get_matrix()[..., 3,:3]
            camera_intr.get_world_to_view_transform(R=R, T=T) # acts as setter for camera_intr
            self.camera_intr = camera_intr

        else: # to keep specified camera, we need to update the axis
            world_axis = camera_intr.get_world_to_view_transform().transform_normals(world_axis.float()).double()
            world_axis[...,2] = 0
            world_axis = torch.nn.functional.normalize(world_axis,dim=-1)
            world_axis = camera_intr.get_world_to_view_transform().inverse().transform_normals(world_axis.float()).double()
            axis_shape = self.world_axis.shape

            if len(camera_intr) > 1:
                axis_shape = [len(self.camera_intr)] + list(axis_shape)
            self.world_axis = world_axis.reshape(axis_shape)
            
        self.im_center = camera_intr.get_full_projection_transform().transform_points(world_center.float())[..., :2].double()
        self.depth = camera_intr.get_world_to_view_transform().transform_points(world_center.float())[..., [2]].double()
        self.im_axis = torch.nn.functional.normalize(camera_intr.get_world_to_view_transform().transform_normals(world_axis.float()).double(),dim=-1)[..., :2]
        self.im_angle = torch.atan2(self.im_axis[..., 1],self.im_axis[..., 0])

        return camera_intr

    def write_obj(self, path, include_coordinate=False, include_line_o_action=False):
        sphere_list = [trimesh.transformations.translation_matrix(self.world_center.reshape((3,)).numpy(force=True))]
        sphere_radius = self.finger_radius
        sphere_colors = [[255, 0, 255],[255, 0, 0],[0, 0, 255]]
        mesh_list = []
        if hasattr(self,'contact_points') and self.contact_points is not None and torch.numel(self.contact_points) > 0:
            contact_np = self.contact_points.reshape((2,3)).numpy(force=True)
            for i in [0,1]:
                sphere_list.append(trimesh.transformations.translation_matrix(contact_np[i,:]))
                if hasattr(self,'contact_normals') and self.contact_normals is not None:
                    try:
                        normal_dir_np = self.contact_normals.reshape((2,3)).numpy(force=True)[i,:]
                        normal_points_np = np.concatenate((contact_np[(i,),:], contact_np[(i,),:]+normal_dir_np*sphere_radius*2),axis=0)
                        mesh_list.append(trimesh.creation.cylinder(segment=normal_points_np, radius=sphere_radius/10))
                        mesh_list[-1].visual.face_colors = sphere_colors[i+1]
                    except:
                        print(f'failed to save normals {self.contact_normals}')
        if hasattr(self,'camera_intr') and self.camera_intr is not None:
            transform_np = self.camera_intr.get_world_to_view_transform().inverse().get_matrix().reshape((4,4)).transpose(0,1).numpy(force=True)
            trimesh_cam = trimesh.scene.Camera(focal=self.camera_intr.focal_length.reshape(2).numpy(force=True),
                                               resolution=self.camera_intr.image_size.reshape(2).numpy(force=True))
            cam_list = trimesh.creation.camera_marker(trimesh_cam, origin_size=sphere_radius*2)
            cam_mesh = trimesh.util.concatenate(cam_list)
            cam_mesh = cam_mesh.apply_transform(transform_np)
            mesh_list.append(cam_mesh)

        
        for i in range(len(sphere_list)):
            mesh_list.append(trimesh.creation.uv_sphere(transform=sphere_list[i],radius=sphere_radius))
            mesh_list[-1].visual.face_colors = sphere_colors[i]      
        if include_coordinate:
            mesh_list.append(trimesh.creation.axis(origin_size=sphere_radius/10))
        if include_line_o_action:
            p1,p2 = self.endpoints3D
            endpoints_np = np.concatenate((p1.reshape(1,3).numpy(force=True),p2.reshape(1,3).numpy(force=True)))
            mesh_list.append(trimesh.creation.cylinder(segment=endpoints_np, radius=sphere_radius/20))
            mesh_list[-1].visual.face_colors = [255, 0, 255]

        merged_mesh = trimesh.util.concatenate(mesh_list)

        merged_mesh.export(path)

    @staticmethod
    def normal_diagonal_3D(reference, var_triple, sampleCount, meanZero=False):
        output_size = [sampleCount] + list(reference.shape)
        #reference = torch.unsqueeze(reference,0)
        reference = reference.expand(output_size)
        output_samples = torch.zeros_like(reference)
        for i in range(3):
            output_samples[...,i] = torch.normal(output_samples[...,i], var_triple[i] ** 2, generator=torch.cuda.manual_seed(i))
        return output_samples
    
    def generateNoisyGrasps(self, sampleCount=1):
                #          center,
                #  angle=0.0,
                #  depth=1.0,
                #  width=0.05,
                #  camera_intr=None,
                #  contact_points=None,
                #  contact_normals=None,
                #  axis3D=None,
                #  num_cone_faces=8, 
                #  friction_coef=0.5,
                #  torque_scaling=None
        sigma_grasp_trans_x= math.sqrt(0.005 ** 2 + 0.01 ** 2)
        sigma_grasp_trans_y= math.sqrt(0.005 ** 2 + 0.01 ** 2)
        sigma_grasp_trans_z= math.sqrt(0.005 ** 2 + 0.01 ** 2)
        sigma_grasp_rot_x= math.sqrt(0.001 ** 2 + 0.01 ** 2)
        sigma_grasp_rot_y= math.sqrt(0.001 ** 2 + 0.01 ** 2)
        sigma_grasp_rot_z= math.sqrt(0.001 ** 2 + 0.01 ** 2)
        R_sample_sigma = torch.eye(3,device=self.world_axis.device,dtype=self.world_axis.dtype) # 3x3

        t_var = (sigma_grasp_trans_x,sigma_grasp_trans_y,sigma_grasp_trans_z)
        r_var = (sigma_grasp_rot_x, sigma_grasp_rot_y, sigma_grasp_rot_z)

        # R_sample_sigma is treated as (1 grasp dim times) x 3 x 3
        center_in_noise_frame = torch.squeeze(torch.matmul(torch.transpose(R_sample_sigma,-1,-2),torch.unsqueeze(self.world_center,-1)))
        center_noised_in_noise_frame = self.normal_diagonal_3D(center_in_noise_frame, t_var, sampleCount) + center_in_noise_frame
        # R_sample_sigma is treated as 1 x (1 grasp dim times) x 3 x 3
        center3D = torch.squeeze(R_sample_sigma.matmul(torch.unsqueeze(center_noised_in_noise_frame,-1)),-1)

        axis_in_noise_frame = torch.matmul(torch.transpose(R_sample_sigma,-1,-2), torch.unsqueeze(self.world_axis,-1))
        randRotVel = self.normal_diagonal_3D(self.world_axis, r_var, sampleCount)
        randRotVel = torch.reshape(randRotVel, (-1,3))
        randRelRotMat = tf.so3_exp_map(randRotVel)
        randRelRotMat = torch.reshape(randRelRotMat, list(center3D.shape)+[3])
        # R_sample_sigma is treated as 1 x (1 grasp dim times) x 3 x 3
        axis3D = torch.squeeze(torch.matmul(R_sample_sigma, torch.matmul(randRelRotMat, axis_in_noise_frame)),-1)
        return GraspTorch(center3D, world_axis=axis3D, 
                          friction_coef=self.friction_coef, num_cone_faces=self.num_cone_faces,
                          torque_scaling=self.torque_scaling, width=self.width,
                          camera_intr=self.camera_intr)
    
    @property
    def ray_directions(self):
        axisBatched = torch.unsqueeze(self.world_axis,0)
        return torch.cat([axisBatched, -axisBatched], 0)
    
    @property
    def tform_to_camera(self):
        """Returns a pytorch3d transform to go from world to camera (pixel) coordinates"""
        return self.camera_intr.get_full_projection_transform()
    
    @property
    def endpoints(self):
        """Returns the grasp endpoints."""
        p1 = self.im_center - (self.width_px / 2) * self.im_axis
        p2 = self.im_center + (self.width_px / 2) * self.im_axis
        return p1, p2
    
    @property
    def endpoints3D(self):
        """Returns the grasp endpoints."""
        p1 = self.world_center - (self.width / 2) * self.world_axis
        p2 = self.world_center + (self.width / 2) * self.world_axis
        return p1, p2


    @property
    def width_px(self):
        """Returns the width in pixels."""
        if self.camera_intr is None:
            missing_camera_intr_msg = ("Must specify camera intrinsics to"
                                       " compute gripper width in 3D space.")
            raise ValueError(missing_camera_intr_msg)
        # Form the jaw locations in 3D space at the given depth.
        p1 =torch.cat((torch.zeros([self.depth.shape[0],2],device=self.depth.device,dtype=self.depth.dtype), self.depth), dim=-1)
        p2 =torch.cat((self.depth, torch.zeros([self.depth.shape[0],1],device=self.depth.device,dtype=self.depth.dtype), self.depth), dim=1)

        # Project into pixel space.
        u1 = self.camera_intr.transform_points(p1.float())
        u2 = self.camera_intr.transform_points(p2.float())
        return torch.norm(u1 - u2,dim=-1)
    
    @property
    def friction_torques(self):
        """
        Get the torques that can be applied by a set of force vectors at the contact point.

        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            the forces applied at the contact

        Returns
        -------
        success : bool
            whether or not computation was successful
        torques : 3xN :obj:`numpy.ndarray`
            the torques that can be applied by given forces at the contact
        """
        # TODO error
        if self.friction_cone is None:
            return None   
        
        

        n_force = self.normal_force_magnitude.unsqueeze(-1).unsqueeze(0)
        forces = torch.mul(self.friction_cone, n_force)

        momentArm = self.contact_points - self.object_com
        momentArm = momentArm.expand([forces.shape[0]]+list(momentArm.shape))
        torques = torch.linalg.cross(momentArm, forces, dim=-1)
        if torch.any(torch.isnan(torques)):
            breakpoint()
        return torques
    
    @property
    def normal_force_magnitude(self):
        """ Returns the component of the force that the contact would apply along the normal direction.

        Returns
        -------
        float
            magnitude of force along object surface normal
        """
        if self.applied_to_object is False:
            return None   
        in_direction_norm = torch.nn.functional.normalize(self.ray_directions,dim=-1)

        in_normal = -self.contact_normals

        normal_force_mag = torch.sum(torch.mul(in_normal, in_direction_norm),-1)

        return torch.nn.functional.relu(normal_force_mag)
        
    @property
    def grasp_matrix(self):
        """ Computes the grasp map between contact forces and wrenchs on the object in its reference frame.

        Returns
        -------
        G : 6xM :obj:`numpy.ndarray`
            grasp map
        """
        if self.applied_to_object is False:
            return None  
        bounding_box = self.mesh.get_bounding_boxes()
        bounding_lengths = torch.diff(bounding_box, dim=-1 )
        if self.torque_scaling == None:
            median_length = torch.median(bounding_lengths)
            torque_scaling = torch.pow(median_length, -1)
        else:
            torque_scaling = self.torque_scaling

        n_force = self.normal_force_magnitude.unsqueeze(-1)
        normals = torch.mul(-self.contact_normals , n_force)
        soft_fingers = True
        finger_radius=self.finger_radius

        n_force = n_force.unsqueeze(0)
        forces = torch.mul(self.friction_cone, n_force)
        torques = torch.mul(self.friction_torques, n_force)

        G = torch.cat([forces, torques*torque_scaling], dim=-1)
        if soft_fingers:
            torsion = np.pi * finger_radius**2 * self.friction_coef * normals * torque_scaling
            G_torsion = torch.zeros(torsion.shape, device=torsion.device, dtype=torsion.dtype)
            G_torsion = torch.cat((G_torsion, torsion),-1)
            G_torsion = G_torsion.unsqueeze(0)
            G = torch.cat((G, G_torsion, -G_torsion), 0)

        return G

    @property
    def friction_cone(self):
        """ Computes the friction cone and normal for all contact points.

        Parameters
        ----------
        num_cone_faces : int
            number of cone faces to use in discretization
        friction_coef : float 
            coefficient of friction at contact point
        
        Returns
        -------
        success : bool
            False when cone can't be computed
        cone_support : :obj:`numpy.ndarray`
            array where each column is a vector on the boundary of the cone
        normal : normalized 3x1 :obj:`numpy.ndarray`
            outward facing surface normal
        """

        if self.applied_to_object is False:
            return None

        def cross_unit(vec1, vec2):
            vec3 =  torch.linalg.cross(vec1, vec2, dim=-1)
            return torch.nn.functional.normalize(vec3,dim=-1)
        
        normal_in = -self.contact_normals
        # get unit vectors orthogonal to normal. Defaults to aligning to x axis, falls back on y axis if normal is exactly parallel to x
        ref = torch.eye(2, m=3,device=self.contact_normals.device, dtype=self.contact_normals.dtype).expand(list(self.contact_normals.shape[:-1]) + [2,3])
        yvec = cross_unit(normal_in.unsqueeze(-2), ref) # may be degenerate, so perform twice
        yvec_error = abs(torch.linalg.vector_norm(yvec,dim=-1) - 1)
        yvec = torch.gather(input=yvec,index=torch.argmin(yvec_error,dim=-1).unsqueeze(-1).unsqueeze(-1).expand(list(yvec.shape[:-2])+[1,3]),dim=-2).squeeze(-2)
 
        xvec = cross_unit(yvec, normal_in)
        yvec = cross_unit(normal_in, xvec)
        # TODO check if contact would slip https://github.com/BerkeleyAutomation/dex-net/blob/cccf93319095374b0eefc24b8b6cd40bc23966d2/src/dexnet/grasping/contacts.py#L251


        def reshape_local(vec, num_faces):
            vec = vec.unsqueeze(0)
            target_shape = list(vec.shape)
            target_shape[0] = num_faces
            return vec.expand(target_shape)
            
        yvec_expanded = reshape_local(yvec, self.num_cone_faces)
        xvec_expanded = reshape_local(xvec, self.num_cone_faces)

        sampleAngles = torch.linspace(0, 2 * math.pi, self.num_cone_faces+1, device=self.contact_normals.device)
        sampleAngles = sampleAngles[:-1]
        sampleAngles = torch.reshape(sampleAngles, [self.num_cone_faces] + [1] * (len(xvec_expanded.shape)-1))

        tan_vec = torch.mul(xvec_expanded, torch.cos(sampleAngles)) + torch.mul(yvec_expanded, torch.sin(sampleAngles))
        friction_cone = -self.contact_normals + self.friction_coef * tan_vec
        return friction_cone

        
    def apply_to_mesh(self, mesh, contact_points=None, ignore_backface_check=False, is_watertight=True, is_inverted=False, use_dexnet_normal=False, use_cramer=False): 
        """Compute where grasp contacts a mesh, state is meshes pytorch3D object"""
            # for grasp in grasp 
        # for each ray (2)
        # for each triangle (vectorize for parallel)
        #  compute intersection

            # find intersection
            # 1. Inside triangle
            # 2. In direction for ray
            # 3. Closest to start point (gripper closes until contact)
            # that intersection is contact_point
            # angle between ray and normal is contact_normal
        #https://en.m.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
        opposite_dir_rays = self.ray_directions
        ray_o = self.world_center - (opposite_dir_rays * self.width / 2)
        ray_d = opposite_dir_rays # [axis3d, -axis3d]
        # ray is n, 3
        
        target_shape = ray_o.shape
        # moller_trumbore assumes flat list of rays (2d Nx3)
        ray_o_flat = torch.flatten(ray_o, end_dim=-2)
        ray_d_flat = torch.flatten(ray_d, end_dim=-2)
        if contact_points is None:
            if use_cramer:
                with record_function("moller-cramer"):
                    u, v, t = p3d_ex.moller_trumbore(ray_o_flat, ray_d_flat, mesh)
            else:
                with record_function("moller-solve"):
                    u, v, t  = p3d_ex.moller_tumbore_solve(ray_o_flat,ray_d_flat,mesh)
                    t = t.clone()
            u = torch.unflatten(u, 0, target_shape[:-1])
            v = torch.unflatten(v, 0, target_shape[:-1])
            t = torch.unflatten(t, 0, target_shape[:-1])
            # correct dir, not too far, actually hits triangle
            inside1 = ((t >= 0.0) * (t < self.width) * (u >= 0.0) * (v >= 0.0) * ((u + v) <= 1.0)).bool()  # (n_rays, n_faces)
            t[torch.logical_not(inside1)] = float('Inf')
            # (n_rays, n_faces)
            min_out = torch.min(t, -1,keepdim=True)

            contact_points = ray_o + ray_d * min_out.values
            faces_index = min_out.indices
            grasps_in_contact = torch.all(torch.all(torch.logical_not(torch.isinf(min_out.values)),dim=-1),dim=0)
            if len(contact_points.shape) == 2:
                if not grasps_in_contact:
                    contact_points = torch.zeros((0,3),device=contact_points.device)
            else:
                contact_points = contact_points[:,grasps_in_contact]
        else:
            grasps_in_contact = torch.ones(contact_points[...,-1,-1].shape,device=contact_points.device, dtype=torch.bool)
        
        
        #verts = mesh.verts_packed()[mesh.faces_packed()[faces_index,:]]
        # experimental, weighted vertex normals
        #vertex_normals = mesh.verts_normals_packed()[mesh.faces_packed()[faces_index,:]]
        #u_vals = torch.gather(u, 2, faces_index).unsqueeze(3).unsqueeze(4)
        #v_vals = torch.gather(v, 2, faces_index).unsqueeze(3).unsqueeze(4)
        #w_vals = 1 - u_vals - v_vals
        #weights = torch.cat((w_vals,u_vals,v_vals),-2)
        #minReturn=torch.min(weights,dim=-2)
        #normsVert = torch.sum(torch.multiply(vertex_normals,weights),dim=-2).squeeze(-2)
        # verts[[0,1],:,:,minReturn.indices.squeeze(),:] = torch.mean(verts, dim=-2,keepdim=False)
        # vertex_normals[[0,1],:,:,minReturn.indices.squeeze(),:] = mesh.faces_normals_packed()[faces_index,:]
        if torch.any(grasps_in_contact):
            if use_dexnet_normal:
                (idxs_face, masks, sphereDirs) = GraspTorch.sphereSamples(contact_points, mesh)
                normsEstimated = self.svdSpherePointsNormal(contact_points, sphereDirs, masks, ray_d[:,grasps_in_contact])
            else:
                normsEstimated = GraspTorch.avgSphereArcNormal(mesh, contact_points)


            # optional correction if allowing mesh backfaces (some vertex order reversed)
            if(ignore_backface_check or not is_watertight):
                normsEstimated = normsEstimated * -torch.sign(torch.sum(normsEstimated * ray_d[:,grasps_in_contact], dim=-1,keepdim=True))
            # optional correction if we know that all vertices have order reversed
            elif(is_inverted):
                normsEstimated = -normsEstimated

            if is_watertight:
                if len(contact_points.shape) == 2:
                    contact_is_outside = torch.all(torch.sum(normsEstimated * ray_d, dim=-1) < 0 ,dim=0)
                else:
                    contact_is_outside = torch.all(torch.sum(normsEstimated * ray_d[:,grasps_in_contact], dim=-1) < 0 ,dim=0)# dot product of normal and finger should be opposite, less than 0
                contact_is_outside_full = torch.zeros_like(grasps_in_contact)
                contact_is_outside_full[grasps_in_contact] = contact_is_outside
                grasps_in_contact = torch.logical_and(grasps_in_contact,  contact_is_outside_full)
                if len(contact_points.shape) == 2:
                    if not contact_is_outside:
                        contact_points = torch.zeros((0,3),device=contact_points.device)
                        normsEstimated = torch.zeros((0,3),device=contact_points.device)
                else:
                    contact_points = contact_points[:,contact_is_outside]
                    normsEstimated = normsEstimated[:,contact_is_outside]
                if torch.numel(normsEstimated) > 0 and torch.any( torch.linalg.vector_norm(normsEstimated) < 0.95):
                    print('found bad normal')

        grasp_with_contact = self[grasps_in_contact]
        # normsSphereAvg = self.avgSpherePoints(state, idxs_face, masks, self.contact_points)
        #torch.mean(verts, dim=-2,keepdim=True)
        #vertex_normals.scatter_( state.faces_normals_packed()[faces_index,:])
        # TODO, update to use built in https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/interp_face_attrs.html
        #self.contact_normals = (torch.sum(torch.multiply(vertex_normals,weights),dim=-2).squeeze(-2) + torch.squeeze(state.faces_normals_packed()[faces_index,:],-2))/2
        normsFace = torch.squeeze(mesh.faces_normals_packed()[faces_index,:],-2)
        grasp_with_contact.contact_mask = grasps_in_contact
        grasp_with_contact.face_normals = normsFace
        grasp_with_contact.applied_to_object = True
        grasp_with_contact.contact_points = contact_points
        if torch.any(grasps_in_contact):
            grasp_with_contact.contact_normals = normsEstimated
        else:
            grasp_with_contact.contact_normals = contact_points
        grasp_with_contact.mesh = mesh
        grasp_with_contact.object_com = p3d_ex.mesh_properties.compute_mesh_COM(mesh,is_watertight=is_watertight)
        grasp_with_contact.faces_index = faces_index

        return grasp_with_contact

    @property
    def count_misses(self):
        if hasattr(self, 'contact_mask'):
            return torch.count_nonzero(torch.logical_not(self.contact_mask))
        else:
            return float('nan')

    @staticmethod
    def dexnetRadius(mesh, gridDist=1.5):
        # matches min scaling from:
        # https://github.com/BerkeleyAutomation/dex-net/blob/cccf93319095374b0eefc24b8b6cd40bc23966d2/src/dexnetdatabase/mesh_processor.py#L281
        sdf_dim = 100
        sdf_padding = 5

        maxDim = torch.max(torch.diff(mesh.get_bounding_boxes(),dim=-1))
        scaling = (maxDim / (sdf_dim - sdf_padding * 2)) # box to meters
        sphereRadius = scaling * gridDist
        return sphereRadius, scaling

    @staticmethod
    def sphereSamples(surface_point, mesh):
        # rejection ish samples around surface_point that are on mesh
        point = surface_point.unsqueeze(-2) # add dim for samples

        steps = 3 # positive int, probably odd
        steps_cubed = steps**3
        obj_target_scale = 0.040
        sphereRadius,scaling = GraspTorch.dexnetRadius(mesh)
        step_ends = (steps-1)/2
        step_tensor = torch.linspace(start=-step_ends,end=step_ends,steps=steps,device=mesh.device,dtype=surface_point.dtype)
        sphereDirsTuple = torch.meshgrid(step_tensor,step_tensor,step_tensor,indexing='ij')
        sphereDirs = torch.cat((sphereDirsTuple[0].reshape(steps_cubed,1),sphereDirsTuple[1].reshape(steps_cubed,1),sphereDirsTuple[2].reshape(steps_cubed,1)),1)
        sphereDirs = torch.nn.functional.normalize(sphereDirs.double(),dim=-1) * sphereRadius
        sphereDirs = sphereDirs.reshape([1]*(len(point.shape)-2)+ [steps_cubed, 3])
        pointSphere = sphereDirs + point
        origShape = pointSphere.shape
        pointSphereCloud = Pointclouds([pointSphere.reshape(-1,3)])

        (idxs_face, dists, face_edge_shared) = GraspTorch.checkSamplesAreOnMesh(pointSphereCloud, mesh)

        dists = dists.reshape([-1, steps_cubed, 1])
        face_edge_shared = face_edge_shared.reshape([-1, steps_cubed, 1])
        idxs_face = idxs_face.reshape([-1, steps_cubed, 1])
        minDist = (scaling * np.sqrt(2) / 2)**2  # square meters 
        masks = torch.split(torch.logical_and(dists < minDist,face_edge_shared), dim=0, split_size_or_sections=1)
        if torch.count_nonzero(masks[1]) < 1:
            print('bad match')
        return idxs_face, masks, sphereDirs

    @staticmethod
    def checkSamplesAreOnMesh(samples, mesh):
        # check if sample points are near surface of mesh
        verts_packed = mesh.verts_packed()
        faces_packed = mesh.faces_packed()
        tris = verts_packed[faces_packed]
        edges_packed = mesh.edges_packed()
        segms = verts_packed[edges_packed]

        dists_face, idxs_face = _C.point_face_dist_forward(samples.points_packed().float(), 
                                                samples.cloud_to_packed_first_idx(), 
                                                tris.float(), 
                                                mesh.mesh_to_faces_packed_first_idx(), 
                                                samples.num_points_per_cloud().max().item(),
                                                5e-6)
        dists_edge, idxs_edge = _C.point_edge_dist_forward(samples.points_packed().float(), 
                                                samples.cloud_to_packed_first_idx(), 
                                                segms.float(), 
                                                mesh.mesh_to_edges_packed_first_idx(), 
                                                samples.num_points_per_cloud().max().item(),
                                                )
        dists = dists_edge
        edges_to_check = mesh.faces_packed_to_edges_packed()[idxs_face,:]
        face_edge_shared = torch.any(edges_to_check == idxs_edge.unsqueeze(1), dim=1)
        dists[face_edge_shared] = dists_face[face_edge_shared]
        
        return idxs_face, dists, face_edge_shared

    @staticmethod
    def avgSphereArcNormal(mesh, surface_point, drop_opposite_normals=False):
        # Assumes that nearby triangles have same winding. If winding is inconsistent, mean will be weird
        radius,_ = GraspTorch.dexnetRadius(mesh)
        verts_packed = mesh.verts_packed()
        faces_packed = mesh.faces_packed()
        tris = verts_packed[faces_packed]

        # find reference frame for each face
        face_normals_unsqeeze = mesh.faces_normals_packed().unsqueeze(-1)
        edge_dir = torch.nn.functional.normalize(tris[:,1,:]-tris[:,0,:],dim=-1).unsqueeze(-1)
        face_rot_ms = torch.cat((edge_dir, torch.cross(face_normals_unsqeeze,edge_dir),face_normals_unsqeeze),-1)
        
        # rotate each face to that frame
        tris_rot = torch.matmul(tris.unsqueeze(-2), face_rot_ms.unsqueeze(-3)).squeeze(-2)
        # rotate contacts to that frame
        sp_shape = list(surface_point.shape[:-1]) + [1] * len(face_rot_ms.shape[:-1]) + [3]
        rm_shape = [1] * len(surface_point.shape[:-1]) + [face_rot_ms.shape[0]] + [3,3]
        surface_point_rot = torch.matmul(surface_point.reshape(sp_shape), face_rot_ms.double().reshape(rm_shape)).squeeze(-2)
        tris_2D = tris_rot[...,:-1]
        sphere_plane_dist = surface_point_rot[...,-1] - tris_rot[...,0,-1]
        intersects_plane = torch.abs(sphere_plane_dist) < radius
        sphere_center_2D = surface_point_rot[...,:-1]
        # get radius of projection of sphere to triangle plane, 0 out all non-intersections
        radius_2D = torch.zeros_like(sphere_plane_dist)
        radius_compare = radius**2 - sphere_plane_dist**2
        radius_compare_positive = radius_compare > 0
        radius_2D[radius_compare_positive] = torch.sqrt( radius_compare[radius_compare_positive] )
        radius_2D_inv = torch.zeros_like(radius_2D)
        radius_2D_inv[intersects_plane] = 1 / radius_2D[intersects_plane]
        # triangle centered at projection with radius 0 
        tris_2D_unit = (tris_2D - sphere_center_2D.unsqueeze(-2)) * radius_2D_inv.unsqueeze(-1).unsqueeze(-1)
    #    # masking to reduce computation, che 
    #    tris_2D_unit = tris_2D_unit[intersects_plane]

        # # can filter on mesh_on_contact_proj all less than face_radius

        edge_dir_2D = tris_2D_unit[...,(1,2,0),:] - tris_2D_unit
        # # mesh_on_contact_proj and edge_dir_2D define line segment
        # # solve for intersection with unit circle
        # https://mathworld.wolfram.com/Circle-LineIntersection.html
        edge_length_2D_sq = torch.sum(edge_dir_2D**2,dim=-1)
        # # x1y2 - x2y1
        edge_det_2D = tris_2D_unit[...,0] * tris_2D_unit[...,(1,2,0),1] - tris_2D_unit[...,1] * tris_2D_unit[...,(1,2,0),0]
        edge_disc_2D = edge_length_2D_sq - edge_det_2D ** 2
        edge_length_2D_sq_finite = edge_length_2D_sq!=0
        projection_contact_on_edge_2D = torch.zeros_like(edge_dir_2D[...,(1,0)])
        projection_contact_on_edge_2D[edge_length_2D_sq_finite] = edge_dir_2D[...,(1,0)][edge_length_2D_sq_finite] * torch.cat((edge_det_2D[edge_length_2D_sq_finite].unsqueeze(-1),-edge_det_2D[edge_length_2D_sq_finite].unsqueeze(-1)),dim=-1) / edge_length_2D_sq[edge_length_2D_sq_finite].unsqueeze(-1)
        projection_contact_on_edge_2D_norm = torch.sum((projection_contact_on_edge_2D - tris_2D_unit) * edge_dir_2D,dim=-1) / edge_length_2D_sq
        edge_norm_2d = torch.cat((edge_dir_2D[...,(1)].unsqueeze(-1),-edge_dir_2D[...,(0)].unsqueeze(-1)),dim=-1)
        contact_is_above_line = torch.sum(-tris_2D_unit * edge_norm_2d, dim = -1) < 0
        contact_is_above = torch.all( contact_is_above_line, dim=-1)
        proj_falls_in_seg = torch.all(
            torch.logical_and(
                projection_contact_on_edge_2D_norm > 0,
                projection_contact_on_edge_2D_norm < 1,
                ),dim=-1)
        contains_projected_sphere = torch.logical_and(contact_is_above,
            torch.all(torch.linalg.vector_norm(projection_contact_on_edge_2D,dim=-1) > 1,dim=-1))
        contact_fully_contained = torch.logical_and(proj_falls_in_seg, contains_projected_sphere)
        intersects_line = edge_disc_2D > 0
        projection_contact_on_edge_2D[torch.isnan(projection_contact_on_edge_2D)] = 0
        # filter out known bad lines
        #edge_dir_2D = edge_dir_2D[intersects_line]
        #edge_det_2D = edge_det_2D[intersects_line]
        #edge_length_2D_sq = edge_length_2D_sq[intersects_line]
        #edge_disc_2D = edge_disc_2D[intersects_line]
        #tris_2D_unit = tris_2D_unit[intersects_line]
        #projection_contact_on_edge_2D = projection_contact_on_edge_2D[intersects_line]
        # can filter on disc, if neg, no intersection
        edge_length_2D_sq = edge_length_2D_sq.unsqueeze(-1)
        edge_length_2D_sq_non_zero = torch.abs(edge_length_2D_sq) != 0 # torch.finfo(torch.float32).eps
        x_scale = (torch.sign(edge_dir_2D[...,1]) * edge_dir_2D[...,0]).unsqueeze(-1)
        y_scale = abs(edge_dir_2D[...,1]).unsqueeze(-1)
        offset_to_tri_sphere_inter_2D_num = torch.zeros_like(edge_disc_2D.unsqueeze(-1).expand(list(edge_disc_2D.shape)+[2]))
        offset_to_tri_sphere_inter_2D_num[intersects_line] = torch.cat((x_scale[intersects_line],y_scale[intersects_line]),dim=-1) * torch.sqrt(edge_disc_2D[intersects_line]).unsqueeze(-1)
        offset_to_tri_sphere_inter_2D = torch.zeros_like(offset_to_tri_sphere_inter_2D_num)
        offset_to_tri_sphere_inter_2D[edge_length_2D_sq_non_zero.squeeze(-1)] = offset_to_tri_sphere_inter_2D_num[edge_length_2D_sq_non_zero.squeeze(-1)] / edge_length_2D_sq[edge_length_2D_sq_non_zero].unsqueeze(-1)
        offset_to_tri_sphere_inter_2D=offset_to_tri_sphere_inter_2D.unsqueeze(-2)
        intersection_sphere_edge_2D = projection_contact_on_edge_2D.unsqueeze(-2) + torch.cat((offset_to_tri_sphere_inter_2D,-offset_to_tri_sphere_inter_2D),dim=-2)
        intersection_sphere_edge_2D_norm = torch.sum((intersection_sphere_edge_2D - tris_2D_unit.unsqueeze(-2)) * edge_dir_2D.unsqueeze(-2),dim=-1)[edge_length_2D_sq_non_zero.squeeze(-1)] / edge_length_2D_sq[edge_length_2D_sq_non_zero].unsqueeze(-1)
        intersects_segment = torch.logical_and(intersection_sphere_edge_2D_norm > 0,intersection_sphere_edge_2D_norm < 1)
        
        intersection_sphere_edge_2D_finite = torch.logical_and(torch.all(torch.logical_not(torch.isnan(intersection_sphere_edge_2D)),dim=-1),intersection_sphere_edge_2D[...,0]!=0)
        
        intersects_tri = torch.any(torch.any(intersects_segment,dim=-1),dim=-1)
        intersection_angles = torch.zeros_like(intersection_sphere_edge_2D[...,1])
        intersection_angles[intersection_sphere_edge_2D_finite] = torch.atan2(intersection_sphere_edge_2D[intersection_sphere_edge_2D_finite][:,1],intersection_sphere_edge_2D[intersection_sphere_edge_2D_finite][:,0])
        #

        ## sort (or mask, seems equivalent?) intersections on angle, filling missing with pi, append the -pi,pi
        
        #intersection_angles[torch.logical_not(intersects_segment)] = float('nan')
        intersection_angles_shape = list(intersection_angles.shape[:-2]) + [-1]
        intersection_angles = torch.reshape(intersection_angles, intersection_angles_shape)
        pad_size = list(intersection_angles.shape[:-1]) + [1]
        #start_pad = -math.pi * torch.ones(pad_size, dtype=intersection_angles.dtype, device=intersection_angles.device)
        #end_pad = math.pi * torch.ones(pad_size, dtype=intersection_angles.dtype, device=intersection_angles.device)
        #intersection_angles = torch.cat((start_pad, intersection_angles, end_pad),dim=-1)
        intersection_angles = torch.sort(intersection_angles, dim=-1).values
        
        ## for each pair of interesections, check if point at intermediate angle is inside triangle
        
        dif_angles = torch.cat((torch.diff(intersection_angles, dim=-1), torch.full_like(intersection_angles[...,0:1],torch.nan)),dim=-1)
        dif_wrap = intersection_angles[...,0:1] - intersection_angles
        dif_wrap[torch.isnan(dif_wrap)] = 0
        dif_wrap = torch.min(dif_wrap,keepdim=True, dim=-1)
        dif_angles = dif_angles.scatter_(-1, dif_wrap.indices, dif_wrap.values + 2 * math.pi )
        intersection_angles[intersection_angles.isnan()] = 0

        dif_angles_nan=torch.isnan(dif_angles)
        dif_angles_finite = torch.logical_and(torch.logical_not(dif_angles_nan), dif_angles!=0)
        dif_angles_clean = dif_angles[dif_angles_finite]
        dif_angles[dif_angles_nan] = 0

        mid_angles = intersection_angles + dif_angles/2

        mid_points = torch.cat((torch.cos(mid_angles).unsqueeze(-1),torch.sin(mid_angles).unsqueeze(-1)),dim=-1)
        mid_points[torch.isnan(mid_points)] = 0
        dif_angles_half = torch.zeros_like(dif_angles)
        dif_angles_half = dif_angles_clean/2

        arc_com_2D_unit_scale = torch.zeros_like(dif_angles)
        arc_com_2D_unit_scale[dif_angles_finite] = (torch.sin(dif_angles_half)/(dif_angles_half))
        arc_com_2D_unit = arc_com_2D_unit_scale.unsqueeze(-1) * mid_points
        arc_com_2D = arc_com_2D_unit * radius_2D.unsqueeze(-1).unsqueeze(-1) + sphere_center_2D.unsqueeze(-2)

        mid_points = mid_points.unsqueeze(-2)
        edge_dir_2D = edge_dir_2D.unsqueeze(-3)
        tris_2D_unit = tris_2D_unit.unsqueeze(-3)
        # ## if inside of triangle, add the angle between them to count for that triangle/circle combo
        # ## check if midpoints is inside triangle by checking if it is on same side of all edges
        # ## https://stackoverflow.com/a/3461533
        mid_point_in_tri = torch.all(edge_dir_2D[...,0] * (mid_points[...,1]-tris_2D_unit[...,1]) - edge_dir_2D[...,1] * (mid_points[...,0]-tris_2D_unit[...,0]) > 0,dim=-1)   


        sin_dif_over_dif = torch.sin(dif_angles_clean)/dif_angles_clean
    
        moment_of_inertia_1 = torch.zeros_like(dif_angles)
        moment_of_inertia_2 = torch.zeros_like(dif_angles)
        # only compute where dif angle exists
        moment_of_inertia_1[dif_angles_finite] = 0.5 * (1 + sin_dif_over_dif - 2 * torch.square(torch.sin(dif_angles_clean))/torch.square(dif_angles_clean))
        moment_of_inertia_2[dif_angles_finite] = 0.5 * (1 - sin_dif_over_dif)
        moment_of_inertia_3 = moment_of_inertia_1+moment_of_inertia_2
        # match size back up

        # rotation matrix used a couple places
        face_rot_shape = [1] * len(surface_point.shape[:-1]) + [face_rot_ms.shape[0]] + [1] + [3] * 2
        # rotate stuff back to 3D
        # tris_rot = torch.matmul(tris.unsqueeze(-2), face_rot_ms.unsqueeze(-3)).squeeze(-2)
        fac_rot = face_rot_ms.double().reshape(face_rot_shape)
        fac_rot_inverse = fac_rot.transpose(-2,-1)

        # get arc center of mass into mesh coordinates
        dim_3 =  torch.zeros_like(arc_com_2D[...,0:1])
        tri_shape = len(surface_point.shape[:-1]) * [1] + [-1] + 2 * [1]
        center_local_3D = torch.cat((arc_com_2D,dim_3+tris_rot[...,0,2:3].reshape(tri_shape)),-1).unsqueeze(-2)
        arc_com_3D = torch.matmul(center_local_3D, fac_rot_inverse).squeeze(-2)
        arc_com_3D[torch.logical_not(mid_point_in_tri)] = 0

        # get overall center of mass from weighted average
        arc_mass = radius_2D.unsqueeze(-1) * dif_angles
        arc_mass[torch.logical_not(mid_point_in_tri)] = 0


        # solution 1: smoother weighted average normal
        arc_mass_per_face = torch.sum(arc_mass,dim=-1,keepdim=True)
        
        face_normals_reshape = mesh.faces_normals_packed().reshape(rm_shape[:-1]).expand(torch.broadcast_shapes(arc_mass_per_face.size(),rm_shape[:-1]))
        # optional check for normals that are opposite the one at the collision point, and therefore don't contribute to overall normal
        if drop_opposite_normals:
            collision_faces_index = torch.argmax(torch.all(contact_is_above_line,dim=-1,keepdim=True).to(dtype=torch.uint8),dim=-2,keepdim=True)
            collision_faces_size = list(collision_faces_index.shape)[:-1] + [3]
            collision_faces_index = collision_faces_index.expand(collision_faces_size)
            collision_faces = torch.gather(face_normals_reshape,-2, collision_faces_index)
            dots = torch.linalg.vecdot(collision_faces, face_normals_reshape)
            arc_mass_per_face[dots < 0] = 0
            
        normalContribution = face_normals_reshape * arc_mass_per_face
        surface_normals_avg = torch.nn.functional.normalize(torch.sum(normalContribution,dim=-2), dim=-1)

        # # solution 2: more dexnet like, SVD on intersections. Does weird stuff at corners
        # # moment of inertia -> Cov -> SVD
        # arc_com_total = torch.sum(arc_mass.unsqueeze(-1) * arc_com_3D,dim=(-3,-2)) / torch.sum(arc_mass,dim=(-1,-2)).unsqueeze(-1) 
        # 
        # moment_of_inertia_vec = torch.cat((moment_of_inertia_1.unsqueeze(-1),moment_of_inertia_2.unsqueeze(-1),moment_of_inertia_3.unsqueeze(-1)),dim=-1)
        # moment_of_inertia_local = torch.diag_embed(moment_of_inertia_vec)
        # moment_of_inertia_local[torch.logical_not(mid_point_in_tri)] = 0
        # moment_of_inertia_local = moment_of_inertia_local * (radius_2D*radius_2D).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # # transform to shared CoM 
        # # https://en.m.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor_of_rotation
        # arc_displacement = (arc_com_total.unsqueeze(-2).unsqueeze(-2) - arc_com_3D)
        # moment_of_inertia_local_aligned = torch.matmul(torch.matmul(fac_rot, moment_of_inertia_local),  fac_rot_inverse)
        # # outter product
        # m_o_i_2 = torch.matmul(arc_displacement.unsqueeze(-1), arc_displacement.unsqueeze(-2))
        # # inner product
        # m_o_i_1 = torch.matmul(arc_d12 = moment_of_inertia_local_aligned + m_o_i_1 - 1
        # moment_of_inertia_global[torch.logical_not(mid_point_in_tri)] = 0

        # moment_of_inertia_total = torch.sum(moment_of_inertia_global,dim=(-4,-3))
        # moment_of_inertia_total_diag = torch.diagonal(moment_of_inertia_total,dim1=-1,dim2=-2)
        # moment_of_inertia_total_trace = torch.sum(moment_of_inertia_total_diag,keepdim=True,dim=-1)
        # moment_of_inertia_total_trace_mat = torch.diag_embed(moment_of_inertia_total_trace.expand(list(moment_of_inertia_total_trace.shape)[:-1]+[3]))
        # cov = moment_of_inertia_total_trace_mat/2-moment_of_inertia_total
        # eig_return = torch.linalg.eigh(cov)
        # surface_normals = torch.real(eig_return.eigenvectors[...,0])

        # savemat('segments.mat',{'segments':segments_3D[linearized_segments_mask].numpy(force=True),'eigvec':torch.real(eig_return[1]).numpy(force=True),'eigval':torch.real(eig_return[0]).numpy(force=True)})
        if torch.any(torch.linalg.vector_norm(surface_normals_avg,dim=-1)<1e-5):
            raise Exception("Non-unit normal found, check contact on mesh")
        #return surface_normals * -torch.sign(torch.sum(surface_normals * in_rays, dim=-1,keepdim=True))
        return surface_normals_avg

    # @staticmethod
    # def avgSpherePointsNormal(self, mesh, idxs_face, masks, surface_point):
    #     normals = torch.zeros_like(surface_point).reshape((len(masks),3))
    #     for maskInd in range(len(masks)):
    #         normals[maskInd,:] = torch.mean(mesh.faces_normals_packed()[idxs_face.squeeze()[maskInd, masks[maskInd].squeeze()],:], dim=0)
    #     normals = normals.reshape(surface_point.shape)
    #     return normals

    @staticmethod
    def svdSpherePointsNormal(surface_point, sphereDirs, masks, in_rays):

        normals = torch.zeros_like(surface_point).reshape((len(masks),3))
        for maskInd in range(len(masks)):
            (U, S, V) = torch.pca_lowrank(sphereDirs.squeeze()[masks[maskInd].squeeze(),:],center=True)
            normals[maskInd,:] = V[:, -1]
        normals = normals.reshape(surface_point.shape)
        return normals * -torch.sign(torch.sum(normals * in_rays, dim=-1,keepdim=True))


    @property
    def feature_vec(self):
        """Returns the feature vector for the grasp.

        `v = [p1, p2, depth]` where `p1` and `p2` are the jaw locations in
        image space.
        """
        p1, p2 = self.endpoints
        return np.r_[p1, p2, self.depth]