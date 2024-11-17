import torch
import torch.optim as optim
import os
from torch.masked import masked_tensor
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import math
import json
import itertools
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
from qpth.qp import QPFunction, QPSolvers
# used by block print
import sys
# used to save grasp visualization
import trimesh
#from ll4ma_opt.problems.problem import Problem
#from ll4ma_opt.problems import SteinWrapper
#from ll4ma_opt.solvers import GradientDescent,BFGSMethod

# Grasp2D class copied from: https://github.com/BerkeleyAutomation/gqcnn/blob/master/gqcnn/grasping/grasp.py
class GraspTorch(object):
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
                 center,
                 angle=0.0,
                 depth=1.0,
                 width=0.05,
                 camera_intr=None,
                 contact_points=None,
                 contact_normals=None,
                 axis3D=None,
                 num_cone_faces=8, 
                 friction_coef=0.5,
                 torque_scaling=None):
        self.width = width
        self.camera_intr = camera_intr
        self.num_cone_faces = num_cone_faces
        self.friction_coef = friction_coef
        self.applied_to_object = False
        self.torque_scaling = torque_scaling
        self.finger_radius=0.005
        if(center.shape[-1] == 3): # 3D grasp
            self.center3D = center.double()
            self.axis3D = axis3D.double()
           # self.axis3D = torch.nn.functional.normalize(self.axis3D,dim=-1)


        elif(center.shape[-1] == 2): # 2D grasp:
            self.center = center.double()
            self.angle = angle.double()
            self.depth = depth.double()
            self.axis = torch.cat((torch.cos(self.angle), torch.sin(self.angle)), dim=-1) # TODO, do we need to check if last dim is 1?

            center_in_camera = torch.cat((self.center, self.depth), dim=-1)
            self.center3D = camera_intr.unproject_points(center_in_camera.float(), world_coordinates=True).double()
            axis_in_camera = torch.cat((self.axis, torch.zeros(list(self.axis.shape)[:-1]+[1], device=self.axis.device, dtype=self.axis.dtype)),dim=-1)
            self.axis3D =camera_intr.get_world_to_view_transform().inverse().transform_normals(axis_in_camera.float()).double()

        else:
            # TODO error
            self = None
        
        if contact_points is not None:
            self.contact_points = contact_points.double()
        if contact_normals is not None:
            self.contact_normals = contact_normals.double()

    def __getitem__(self,key):
        if isinstance(key, np.ndarray) or torch.is_tensor(key) or isinstance(key, slice):


            if torch.numel(self.center3D[...,0]) > 1:
                center = self.center3D[key]
            else:
                center = self.center3D
            


            if torch.is_tensor(self.axis3D) and torch.numel(self.axis3D[...,0]) > 1 :
                axis3D = self.axis3D[key]
            else:
                axis3D = self.axis3D
            
            width = self.width
            
            camera_intr=self.camera_intr
            num_cone_faces=self.num_cone_faces
            friction_coef=self.friction_coef
            torque_scaling=self.torque_scaling
            
            if hasattr(self, 'contact_points'):
                contact_points = self.contact_points[:,key]
            else:
                contact_points = None

            if hasattr(self, 'contact_normals'):
                contact_normals = self.contact_normals[:,key]
            else:
                contact_normals = None       

            sliced = GraspTorch(center,
                 width=width,
                 camera_intr=camera_intr,
                 contact_points=contact_points,
                 contact_normals=contact_normals,
                 axis3D=axis3D,
                 num_cone_faces=num_cone_faces, 
                 friction_coef=friction_coef,
                 torque_scaling=torque_scaling)
            
            # if grasp was 2D, add the 2D info back in
            if hasattr(self,'center'):
                if torch.numel(self.center[...,0]) > 1:
                    sliced.center = self.center[key]
                else:
                    sliced.center = self.center


                if torch.is_tensor(self.angle) and torch.numel(self.angle[...,0]) > 1:
                    sliced.angle = self.angle[key]
                else:
                    sliced.angle = self.angle

                if torch.is_tensor(self.depth) and torch.numel(self.depth[...,0]) > 1 :
                    sliced.depth = self.depth[key]
                else:
                    sliced.depth = self.depth

            if hasattr(self, 'mesh'):
                sliced.mesh = self.mesh
            if hasattr(self, 'object_com'):
                sliced.object_com = self.object_com
            if hasattr(self, 'contact_mask'):
                contact_mask = torch.zeros_like(self.contact_mask)
                contact_inds = torch.nonzero(self.contact_mask.flatten()).squeeze(-1)
                contact_mask[contact_inds[key]] = 1
                sliced.contact_mask = contact_mask

            sliced.applied_to_object = self.applied_to_object
            return sliced
        
    def __len__(self,):
        return self.axis3D[...,0].size

    def make2D(self, updateCamera=False,camera_intr=None):
        if camera_intr==None:
            camera_intr = self.camera_intr
        if camera_intr==None:
            # TODO error
            return None

        if updateCamera: # in order to keep specified axis, we need to update the camera
            
            axis3D = camera_intr.get_world_to_view_transform().transform_normals(self.axis3D.float()).double() # in world
            
            cameraDir = torch.tensor([0.,0.,1.],device=self.axis3D.device,dtype=axis3D.dtype,requires_grad=True) # camera Z
            cameraDir = cameraDir.reshape([1]*(len(axis3D.shape)-1)+[3])
            cameraDir = cameraDir.expand(axis3D.shape)
            rotVec = torch.cross(cameraDir, axis3D, dim=-1) # vector orthogonal to both
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
            axis3D = camera_intr.get_world_to_view_transform().transform_normals(self.axis3D.float()).double()
            axis3D[...,2] = 0
            axis3D = torch.nn.functional.normalize(axis3D,dim=-1)
            axis3D = camera_intr.get_world_to_view_transform().inverse().transform_normals(axis3D.float()).double()
            self.axis3D = axis3D
            
        self.center = camera_intr.get_full_projection_transform().transform_points(self.center3D.float())[..., :2].double()
        self.depth = camera_intr.get_world_to_view_transform().transform_points(self.center3D.float())[..., [2]].double()
        self.axis = torch.nn.functional.normalize(camera_intr.get_world_to_view_transform().transform_normals(self.axis3D.float()).double(),dim=-1)[..., :2]
        self.angle = torch.atan2(self.axis[..., 1],self.axis[..., 0])

        return camera_intr

    def write_obj(self, path, include_coordinate=False, include_line_o_action=False):
        sphere_list = [trimesh.transformations.translation_matrix(self.center3D.reshape((3,)).numpy(force=True))]
        sphere_radius = self.finger_radius
        sphere_colors = [[255, 0, 255],[255, 0, 0],[0, 0, 255]]
        mesh_list = []
        if hasattr(self,'contact_points') and self.contact_points is not None:
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
        R_sample_sigma = torch.eye(3,device=self.axis3D.device,dtype=self.axis3D.dtype) # 3x3

        t_var = (sigma_grasp_trans_x,sigma_grasp_trans_y,sigma_grasp_trans_z)
        r_var = (sigma_grasp_rot_x, sigma_grasp_rot_y, sigma_grasp_rot_z)

        # R_sample_sigma is treated as (1 grasp dim times) x 3 x 3
        center_in_noise_frame = torch.squeeze(torch.matmul(torch.transpose(R_sample_sigma,-1,-2),torch.unsqueeze(self.center3D,-1)))
        center_noised_in_noise_frame = self.normal_diagonal_3D(center_in_noise_frame, t_var, sampleCount) + center_in_noise_frame
        # R_sample_sigma is treated as 1 x (1 grasp dim times) x 3 x 3
        center3D = torch.squeeze(R_sample_sigma.matmul(torch.unsqueeze(center_noised_in_noise_frame,-1)),-1)

        axis_in_noise_frame = torch.matmul(torch.transpose(R_sample_sigma,-1,-2), torch.unsqueeze(self.axis3D,-1))
        randRotVel = self.normal_diagonal_3D(self.axis3D, r_var, sampleCount)
        randRotVel = torch.reshape(randRotVel, (-1,3))
        randRelRotMat = tf.so3_exp_map(randRotVel)
        randRelRotMat = torch.reshape(randRelRotMat, list(center3D.shape)+[3])
        # R_sample_sigma is treated as 1 x (1 grasp dim times) x 3 x 3
        axis3D = torch.squeeze(torch.matmul(R_sample_sigma, torch.matmul(randRelRotMat, axis_in_noise_frame)),-1)
        return GraspTorch(center3D, axis3D=axis3D, 
                          friction_coef=self.friction_coef, num_cone_faces=self.num_cone_faces,
                          torque_scaling=self.torque_scaling, width=self.width,
                          camera_intr=self.camera_intr)
    
    @property
    def ray_directions(self):
        axisBatched = torch.unsqueeze(self.axis3D,0)
        return torch.cat([axisBatched, -axisBatched], 0)
    
    @property
    def tform_to_camera(self):
        """Returns a pytorch3d transform to go from world to camera (pixel) coordinates"""
        return self.camera_intr.get_full_projection_transform()
    
    @property
    def endpoints(self):
        """Returns the grasp endpoints."""
        p1 = self.center - (self.width_px / 2) * self.axis
        p2 = self.center + (self.width_px / 2) * self.axis
        return p1, p2
    
    @property
    def endpoints3D(self):
        """Returns the grasp endpoints."""
        p1 = self.center3D - (self.width / 2) * self.axis3D
        p2 = self.center3D + (self.width / 2) * self.axis3D
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
        # get unit vectors orthogonal to normal
        ref = torch.eye(1, m=3,device=self.contact_normals.device, dtype=self.contact_normals.dtype).expand(self.contact_normals.shape)
        yvec = cross_unit(normal_in, ref) # may be degenerate, so perform twice
 
        xvec = cross_unit(yvec, normal_in)
        yvec = cross_unit(normal_in, xvec)
        # TODO check if contact would slip https://github.com/BerkeleyAutomation/dex-net/blob/cccf93319095374b0eefc24b8b6cd40bc23966d2/src/dexnet/grasping/contacts.py#L251


        def reshape_local(vec, num_faces):
            vec = vec.unsqueeze(0)
            target_shape = list(vec.shape)
            target_shape[0] = num_faces
            return vec.expand(target_shape)
            
        yvec = reshape_local(yvec, self.num_cone_faces)
        xvec = reshape_local(xvec, self.num_cone_faces)

        sampleAngles = torch.linspace(0, 2 * math.pi, self.num_cone_faces+1, device=self.contact_normals.device)
        sampleAngles = sampleAngles[:-1]
        sampleAngles = torch.reshape(sampleAngles, [self.num_cone_faces] + [1] * (len(xvec.shape)-1))

        tan_vec = torch.mul(xvec, torch.cos(sampleAngles)) + torch.mul(yvec, torch.sin(sampleAngles))
        friction_cone = -self.contact_normals + self.friction_coef * tan_vec
        return friction_cone

        
    def apply_to_mesh(self, mesh, contact_points=None, ignore_backface_check=False, is_watertight=True, is_inverted=False, use_dexnet_normal=False): 
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
        ray_o = self.center3D - (opposite_dir_rays * self.width / 2)
        ray_d = opposite_dir_rays # [axis3d, -axis3d]
        # ray is n, 3
        
        target_shape = ray_o.shape
        # moller_trumbore assumes flat list of rays (2d Nx3)
        ray_o_flat = torch.flatten(ray_o, end_dim=-2)
        ray_d_flat = torch.flatten(ray_d, end_dim=-2)
        if contact_points is None:
            mesh_unwrapped = multi_gather_tris(mesh.verts_packed(), mesh.faces_packed())

            u, v, t = moller_trumbore(ray_o_flat, ray_d_flat, mesh_unwrapped.double())

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
        if use_dexnet_normal:
            (idxs_face, masks, sphereDirs) = GraspTorch.sphereSamples(contact_points, mesh)
            normsEstimated = self.svdSpherePointsNormal(contact_points, sphereDirs, masks, ray_d)
        else:
            normsEstimated = GraspTorch.avgSphereArcNormal(mesh, contact_points)

        # optional correction if allowing mesh backfaces (some vertex order reversed)
        if(ignore_backface_check or not is_watertight):
            normsEstimated = normsEstimated * -torch.sign(torch.sum(normsEstimated * ray_d, dim=-1,keepdim=True))
        # optional correction if we know that all vertices have order reversed
        elif(is_inverted):
            normsEstimated = -normsEstimated

        if is_watertight:
            contact_is_outside = torch.all(torch.sum(normsEstimated * ray_d, dim=-1) < 0 ,dim=0)# dot product of normal and finger should be opposite, less than 0
            grasps_in_contact = torch.logical_and(grasps_in_contact,  contact_is_outside)
            contact_points = contact_points[:,contact_is_outside]
            normsEstimated = normsEstimated[:,contact_is_outside]
        

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
        grasp_with_contact.contact_normals = normsEstimated
        grasp_with_contact.mesh = mesh
        grasp_with_contact.object_com = compute_mesh_COM(mesh,is_watertight=is_watertight)
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
        # m_o_i_1 = torch.matmul(arc_displacement.unsqueeze(-2), arc_displacement.unsqueeze(-1)).squeeze(-1)
        # m_o_i_1 = torch.diag_embed(m_o_i_1.expand(list(m_o_i_1.shape)[:-1]+[3]))
        # moment_of_inertia_global = moment_of_inertia_local_aligned + m_o_i_1 - m_o_i_2
        # moment_of_inertia_global[torch.logical_not(mid_point_in_tri)] = 0

        # moment_of_inertia_total = torch.sum(moment_of_inertia_global,dim=(-4,-3))
        # moment_of_inertia_total_diag = torch.diagonal(moment_of_inertia_total,dim1=-1,dim2=-2)
        # moment_of_inertia_total_trace = torch.sum(moment_of_inertia_total_diag,keepdim=True,dim=-1)
        # moment_of_inertia_total_trace_mat = torch.diag_embed(moment_of_inertia_total_trace.expand(list(moment_of_inertia_total_trace.shape)[:-1]+[3]))
        # cov = moment_of_inertia_total_trace_mat/2-moment_of_inertia_total
        # eig_return = torch.linalg.eigh(cov)
        # surface_normals = torch.real(eig_return.eigenvectors[...,0])

        # savemat('segments.mat',{'segments':segments_3D[linearized_segments_mask].numpy(force=True),'eigvec':torch.real(eig_return[1]).numpy(force=True),'eigval':torch.real(eig_return[0]).numpy(force=True)})

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


class GraspQualityFunction():    #ABC):
    """Abstract grasp quality class."""

    def __init__(self):
        # Set up logger - can't because it's from autolab_core.
        # self._logger = Logger.get_logger(self.__class__.__name__)
        self._logger = 0

    def __call__(self, state, actions, params=None):
        """Evaluates grasp quality for a set of actions given a state."""
        return self.quality(state, actions, params)
    
    def savemat(self, path, other_items=None):
        # TODO move to grasp object and call there, passing quality as a dict
        # collects a dictionary of interesting internal state, then saves it
        dict_to_save = {}
        if self.Grasps is not None:
            dict_to_save['axis3D'] = self.Grasps.axis3D.numpy(force=True)
            dict_to_save['center3D'] = self.Grasps.center3D.numpy(force=True)
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

def compute_mesh_tri_areas(mesh):
    """Compute the area of each mesh triangle, using mean as 4th point. Used in CoM/Volume for non-watertight"""
    
    verts_tri = multi_gather_tris(mesh.verts_packed(), mesh.faces_packed())
    com_per_triangle = torch.mean(verts_tri, 1) 

    verts = mesh.verts_packed()
    verts.requires_grad_(True)
    # compute all edge vectors
    edges_tri = multi_gather_tris(verts, mesh.faces_packed()) # n_faces, edge, xyz
    area_per_triangle = torch.linalg.norm(torch.linalg.cross(edges_tri[:,0,:],edges_tri[:,1,:]),dim=-1,keepdim=True)/2

    return area_per_triangle, com_per_triangle

    

def compute_mesh_tetra_volumes(mesh):
    """Compute the volume of each mesh tetra, using mean as 4th point. Used in CoM/Volume for watertight"""
    #https://forums.cgsociety.org/t/how-to-calculate-center-of-mass-for-triangular-mesh/1309966
    mesh_unwrapped = multi_gather_tris(mesh.verts_packed(), mesh.faces_packed())
    # B, F, 3, 3
    # assume Faces, verts, coords
    totalCoords = torch.sum(mesh_unwrapped, 1) # used in both mean and CoM, so split out

    meanVert = torch.sum(totalCoords,0) / (totalCoords.shape[0] * totalCoords.shape[1])

    totalCoords = totalCoords + meanVert
    com_per_triangle = totalCoords / 4

    # add dims and expand "average vertex" to match mesh. Will be used to go from triangles to
    # tetrahedrons
    meanVert_expand = torch.reshape(meanVert, [1, 1, 3]).expand(mesh_unwrapped.shape[0],1,3)

    mesh_tetra = torch.cat([mesh_unwrapped, meanVert_expand], 1)
    mesh_tetra = torch.cat([mesh_tetra, torch.ones([mesh_tetra.shape[0],4,1],device=mesh_unwrapped.device,dtype=mesh_unwrapped.dtype)], -1)
    vol_per_triangle = torch.reshape(torch.linalg.det(mesh_tetra),(mesh_tetra.shape[0],1))
    # det([[x1,y1,z1,1],[x2,y2,z2,1],[x3,y3,z3,1],[x4,y4,z4,1]]) / 6 
    # technically a scaled volume, since we dropped the division by 6
    # does det on last 2 dims, considers at least first 1 to be batch dim
    return vol_per_triangle, com_per_triangle

def compute_mesh_bounding_volume(mesh):
    bb = mesh.get_bounding_boxes()
    ranges = torch.diff(bb,dim=-1).squeeze(-1)[0]
    return torch.prod(ranges,dim=-1)

def compute_mesh_volume(mesh, ignore_backface_check=False):
    """Compute the volume for a mesh, assume uniform density."""
    vol_per_triangle,_ = compute_mesh_tetra_volumes(mesh)
    if ignore_backface_check:
        vol_per_triangle = torch.abs(vol_per_triangle)
    return torch.sum(vol_per_triangle)/6

def compute_mesh_hull_volume(mesh):
    verts = mesh.verts_packed()
    verts_np = verts.numpy(force=True)
    hull = ConvexHull(verts_np).simplices
    # get unique from hull, update hull to use unique indices
    indices_kept, hull_new = np.unique(hull.reshape(-1),return_inverse=True)
    hull_new = np.reshape(hull_new, hull.shape)
    faces = torch.tensor(hull_new, dtype=torch.long, device=mesh.device)
    verts = verts[indices_kept,:]
    mesh_hull = Meshes([verts],[faces])
    # sub sample verts with unique indices
    return compute_mesh_volume(mesh_hull,ignore_backface_check=True)
    # construct mesh


def compute_mesh_COM(mesh, is_watertight=True): 
    """Compute the center of mass for a mesh, assume uniform density."""
    if is_watertight:
        vol_per_triangle, com_per_triangle = compute_mesh_tetra_volumes(mesh)
    else:
        vol_per_triangle, com_per_triangle = compute_mesh_tri_areas(mesh)

    com = torch.sum(com_per_triangle * vol_per_triangle,dim=0) / torch.sum(vol_per_triangle)
    if not torch.any(torch.isfinite(com.flatten())):
        raise Exception("com computation diverged")
        breakpoint()
    return com

class mesh_properties:
    def __init__(self, mesh):
        """Find properties of a mesh, including topology, watertightness, and handedness
           topology is all pairs of non-neighbor triangles, edges, and vertices in a mesh"""

        
        self.compute_connectivity(mesh)
        # initially set using a manifold check during connectivity. This also checks for self collision, but that is very expensive for large meshes
        # self.is_watertight = self.is_watertight and self_collision_min(mesh.detach(), self, forceNormalDist=0) > 0


        if self.is_watertight:
            volume = compute_mesh_volume(mesh, ignore_backface_check=False)
            bounding_volume = compute_mesh_bounding_volume(mesh)
            if abs(volume) < bounding_volume * 1e-8 or abs(volume) > bounding_volume:
                self.is_watertight = False
                self.is_inverted = False
            else:
                self.is_inverted = volume.item() < 0
        else:
            self.is_inverted = False
        
    def compute_connectivity(self, mesh):
        # find mapping between vertices and all faces they're part of
        faces = mesh.faces_packed()
        inverseFaceMap = [[] for _ in range(mesh._V)]
        for rowInd in range(faces.shape[0]):
            for ind in range(3):
                inverseFaceMap[faces[rowInd,ind]].append((rowInd, ind))
            
        # find mapping between edges and all faces they're part of
        edges_per_face = mesh.faces_packed_to_edges_packed()
        inverseFaceEdgeMap = [[] for _ in range(mesh.num_edges_per_mesh())]
        for rowInd in range(edges_per_face.shape[0]):
            for ind in range(3):
                inverseFaceEdgeMap[edges_per_face[rowInd,ind]].append((rowInd,ind))

        # For all edges, mark which faces they come from AND what index they are in that face
        connectivity_edge_ind = torch.full((mesh._F,mesh._F), -1, dtype=torch.int64, device=mesh.device)
        for vert_idx in range(len(inverseFaceEdgeMap)):
            for pair in itertools.combinations(inverseFaceEdgeMap[vert_idx],2):
                if pair[0][0] != pair[1][0]:
                    connectivity_edge_ind[pair[0][0],pair[1][0]] = pair[0][1]
                    connectivity_edge_ind[pair[1][0],pair[0][0]] = pair[1][1]
            
        # For all verts, mark which non-edge faces they come from AND what index they are in that face
        connectivity_vert_ind = torch.full((mesh._F,mesh._F), -1, dtype=torch.int64, device=mesh.device)
        for vert_idx in range(len(inverseFaceMap)):
            for pair in itertools.combinations(inverseFaceMap[vert_idx],2):
                if pair[0][0] != pair[1][0] and connectivity_edge_ind[pair[0][0],pair[1][0]] == -1:
                    connectivity_vert_ind[pair[0][0],pair[1][0]] = pair[0][1]
                    connectivity_vert_ind[pair[1][0],pair[0][0]] = pair[1][1]
        
        connectivity_edge_bool = connectivity_edge_ind > -1
        connectivity_vert_bool = connectivity_vert_ind > -1

        # use mask to get all unconnected triangles (symmetric, so only triu)
        tri_u_ind = torch.triu_indices(mesh._F,mesh._F, device=mesh.device)
        connectivity_bool = torch.logical_or(connectivity_edge_bool, torch.logical_or(connectivity_vert_bool,torch.eye(mesh._F,dtype=bool,device=mesh.device)))
        unconnectivity_flat_mask = torch.logical_not(connectivity_bool[tri_u_ind[0],tri_u_ind[1]])
        self.tri_unconnectivity = tri_u_ind[:,unconnectivity_flat_mask]

        # convert to tensor of the form (face_index, unconnected_edge_index) and (face_index, unconnected_vert_index)
        # first vertex of edge matches index, edges_packed()[edges_per_face[:,0]] == faces[:,0]
        # So opposite edge is always (shared vertex + 1) mod 3 and opposite vert is (shared edge + 2) mod 3  
        edge_faces_indices = torch.nonzero(connectivity_edge_bool, as_tuple=True)
        vert_indices_in_face = connectivity_edge_ind[edge_faces_indices]
        vert_indices = faces[edge_faces_indices[0], vert_indices_in_face]
        self.vert_unconnectivity = torch.cat((edge_faces_indices[1].unsqueeze(0), vert_indices.unsqueeze(0)),dim=0)

        vert_faces_indices = torch.nonzero(connectivity_vert_bool, as_tuple=True)
        edge_indices_in_face = connectivity_vert_ind[vert_faces_indices]
        edge_indices = edges_per_face[vert_faces_indices[0], edge_indices_in_face]
        self.edge_unconnectivity = torch.cat((vert_faces_indices[1].unsqueeze(0), edge_indices.unsqueeze(0)),dim=0)

        self.is_watertight = all([len(edge_faces) % 2==0 for edge_faces in inverseFaceEdgeMap])
            

def self_collision(mesh, unconnectivity, forceNormalDist=True):
    # function to compute the distance between all specified triangles pairs in a mesh.
    # can return nan or 0 if triangles are in collision. Also returns barycentric coordinates
    # that resulted in those distance
    #
    # unconnectivity is a mesh_unnconnectivity object, the includes lists of indices to compare, including
    # triangle to triangle, triangle to edge, and triangle to vertex comparisons
    #
    # forceNormalDist is whether we want the euclidian distance between triangles (False) or the distance in the normal
    #   direction of each triangle (true). Since distance along the normal depends on which triangle use use for the normal, 
    #   this is not symmetric for triangle to triangle comparisons. Therefore, (nearly) twice as many distances will be returned 
    #   in the (True) case. Triangle-edge and triangle-vertex are not affected and are never constrained to normal

    # useful for debugging
    # torch.autograd.set_detect_anomaly(True)
 
    # since forceNormalDist is not symmetric, we need to duplicate the list of pairs, with the order swapped
    if forceNormalDist:
        tri_unconnectivity = torch.cat((unconnectivity.tri_unconnectivity,torch.flip(unconnectivity.tri_unconnectivity,[0])),dim=1)
    else:
        tri_unconnectivity = unconnectivity.tri_unconnectivity
    edge_unconnectivity = unconnectivity.edge_unconnectivity
    vert_unconnectivity = unconnectivity.vert_unconnectivity

    quadratic_term_tri, linear_term_tri, const_term_tri, quadratic_term_edg, linear_term_edg, const_term_edg, quadratic_term_vtx, linear_term_vtx, const_term_vtx, equality_A_tri, equality_b_tri = __assemble_quad_terms(mesh, tri_unconnectivity, edge_unconnectivity, vert_unconnectivity, forceNormalDist)
    # build inequality constraints: solution falls in triangle (or along edge)

    # modified barycentric, 0,0,0 is center and 2/3, -1/3, -1/3  is corner0 
    bary_G_tri = torch.cat((-torch.eye(4,device=mesh.device),
                        torch.tensor([[1,1,0,0]],device=mesh.device),
                        torch.tensor([[0,0,1,1]],device=mesh.device)), dim=0)
    bary_h_tri = torch.tensor([1/3,1/3,1/3,1/3,1/3,1/3],device=mesh.device)

    bary_G_edg = torch.cat((-torch.eye(3,device=mesh.device),
                        torch.tensor([[1,1,0]],device=mesh.device),
                        torch.tensor([[0,0,1]],device=mesh.device)), dim=0)
    # center for edge is at [0.5, 0], so slight tweak
    bary_h_edg = torch.tensor([1/3,1/3,1/2,1/3,1/2],device=mesh.device)

    bary_G_vtx = torch.cat((-torch.eye(2,device=mesh.device),
                        torch.tensor([[1,1]],device=mesh.device)), dim=0)
    bary_h_vtx = torch.tensor([1/3,1/3,1/3],device=mesh.device)

    equality_A_edg =  torch.tensor([],device=mesh.device) 
    equality_b_edg =  torch.tensor([],device=mesh.device) 

    equality_A_vtx =  torch.tensor([],device=mesh.device) 
    equality_b_vtx =  torch.tensor([],device=mesh.device) 

    # solve qp with our custom wrapper. It returns distance, in addition to solution
    # it also, in the forceNormalDist case, replaces distance with inf when solution does not exist
    dist_all_tri, bary_coords_tri = __qp_bary_feasible(quadratic_term_tri, linear_term_tri, const_term_tri, equality_A_tri, equality_b_tri, bary_G_tri, bary_h_tri)

    dist_all_edg, bary_coords_edg = __qp_bary_feasible(quadratic_term_edg, linear_term_edg, const_term_edg, equality_A_edg, equality_b_edg, bary_G_edg, bary_h_edg)
    
    dist_all_vtx, bary_coords_vtx = __qp_bary_feasible(quadratic_term_vtx, linear_term_vtx, const_term_vtx, equality_A_vtx, equality_b_vtx, bary_G_vtx, bary_h_vtx)

    # for ease of use outside of this function, we go back to traditional bary-centric coordinates before returning
    # modified barycentric, 0,0,0 is center and 2/3, -1/3, -1/3  is corner0 
    # traiditional barycentric 1/3,1/3,1/3 is center and 0,0,0 is corner0

    bary_coords_tri_corrected = bary_coords_tri + 1/3

    # center for edge is at [0.5, 0], so slight tweak
    # correct first two by 1/3, 3rd by 1/2, add zero column
    bary_coords_edg_corrected = torch.cat((bary_coords_edg + torch.tensor([[1/3,1/3,1/2]],device=bary_coords_edg.device), torch.zeros_like(bary_coords_edg[:,0:1])),dim=1)

    # correct first two by 1/3, add 2 zero columns
    bary_coords_vtx_corrected =  torch.cat((bary_coords_vtx + 1/3, torch.zeros_like(bary_coords_vtx)),dim=1)

    # cat all before returning
    dist_all = torch.cat((dist_all_tri,dist_all_edg,dist_all_vtx),dim=0)
    bary_coords_corrected = torch.cat((bary_coords_tri_corrected, bary_coords_edg_corrected, bary_coords_vtx_corrected),dim=0)

    return dist_all, bary_coords_corrected

def __assemble_quad_terms(mesh, tri_unconnectivity, edge_unconnectivity, vert_unconnectivity, forceNormalDist):   
    verts = mesh.verts_packed()
    verts.requires_grad_(True)
    # compute all edge vectors
    # NOTE minor repeated calcs with shared edges
    
    tris = multi_gather_tris(verts, mesh.faces_packed()) # n_faces, corner, xyz

    E1 = tris[:, 1] - tris[:, 0]  # vector of edge 1 on triangle (n_faces, 3)
    E2 = tris[:, 2] - tris[:, 0]  # vector of edge 2 on triangle (n_faces, 3)

    edges = multi_gather_tris(verts, mesh.edges_packed()) # n_edges, end, xyz

    E_edge = edges[:, 1] - edges[:, 0]  # vector of edge 1 on triangle (n_faces, 3)

    # reference vectors between comparisons, since origins are centered in our modified barycentric

    # triangle center to triangle center, when no vertices or edges are shared
    w_diff_vec_tri = torch.mean(tris[tri_unconnectivity[1]],dim=1) - torch.mean(tris[tri_unconnectivity[0]],dim=1)  # n_tri_unconnect, xyz

    # triangle center to (non-neighbor) edge center, when vertex is shared
    w_diff_vec_edg = torch.mean(edges[edge_unconnectivity[1]],dim=1) - torch.mean(tris[edge_unconnectivity[0]],dim=1) # n_edg_unconnect, xyz

    # triangle center to (non-neighbor) vertex, when edge is shared
    w_diff_vec_vtx = verts[vert_unconnectivity[1]] - torch.mean(tris[vert_unconnectivity[0]],dim=1) # n_edg_unconnect, xyz

    # expand edge vector directions to full size (per face or edge they appear in) for linear and quadratic terms
    TR1_E1_full_tri = E1[tri_unconnectivity[0]]
    TR1_E2_full_tri = E2[tri_unconnectivity[0]]
    TR2_E1_full_tri = E1[tri_unconnectivity[1]]
    TR2_E2_full_tri = E2[tri_unconnectivity[1]]

    TR1_E1_full_edg = E1[edge_unconnectivity[0]]
    TR1_E2_full_edg = E2[edge_unconnectivity[0]]

    TR2_E_full_edg = E_edge[edge_unconnectivity[1]]

    TR1_E1_full_vtx = E1[vert_unconnectivity[0]]
    TR1_E2_full_vtx = E2[vert_unconnectivity[0]]

    # linear terms, 

    #  concatenate all relevant edge vectors

    # tri
    triangle_vecs_tri = torch.cat(
                    (-TR1_E1_full_tri.unsqueeze(1),
                        -TR1_E2_full_tri.unsqueeze(1),
                        TR2_E1_full_tri.unsqueeze(1),
                        TR2_E2_full_tri.unsqueeze(1)),
                        dim=1)
    
    # edge
    triangle_vecs_edg = torch.cat(
                    (-TR1_E1_full_edg.unsqueeze(1),
                        -TR1_E2_full_edg.unsqueeze(1),
                        TR2_E_full_edg.unsqueeze(1)),
                        dim=1)
    
    # vert
    triangle_vecs_vtx = torch.cat(
                    (-TR1_E1_full_vtx.unsqueeze(1),
                     -TR1_E2_full_vtx.unsqueeze(1)),
                        dim=1)
    
    # compare triangle edge directions to vector between objects to compare
    linear_term_tri = 2 * triangle_vecs_tri @ w_diff_vec_tri.unsqueeze(2)

    linear_term_edg = 2 * triangle_vecs_edg @ w_diff_vec_edg.unsqueeze(2)

    linear_term_vtx = 2 * triangle_vecs_vtx @ w_diff_vec_vtx.unsqueeze(2)


    # const terms, not used in quadratic program, only to correct distance afterwards.
    # only depend on distance between "centers"

    const_term_tri = torch.linalg.vecdot(w_diff_vec_tri,w_diff_vec_tri)

    const_term_edg = torch.linalg.vecdot(w_diff_vec_edg,w_diff_vec_edg)

    const_term_vtx = torch.linalg.vecdot(w_diff_vec_vtx,w_diff_vec_vtx)


    # quadratic terms

    # edge - edge products

    # within triangle terms, compute for each triangle then expand

    # within "first" edges
    E1_E1 = torch.linalg.vecdot(E1,E1)
    TR1_TR1_E1_E1_full_tri = E1_E1[tri_unconnectivity[0]].unsqueeze(1)
    TR2_TR2_E1_E1_full_tri = E1_E1[tri_unconnectivity[1]].unsqueeze(1)

    TR1_TR1_E1_E1_full_edg = E1_E1[edge_unconnectivity[0]].unsqueeze(1)

    TR1_TR1_E1_E1_full_vtx = E1_E1[vert_unconnectivity[0]].unsqueeze(1)

    # within "second" edges
    E2_E2 = torch.linalg.vecdot(E2,E2)
    TR1_TR1_E2_E2_full_tri = E2_E2[tri_unconnectivity[0]].unsqueeze(1)
    TR2_TR2_E2_E2_full_tri = E2_E2[tri_unconnectivity[1]].unsqueeze(1)

    TR1_TR1_E2_E2_full_edg = E2_E2[edge_unconnectivity[0]].unsqueeze(1)

    TR1_TR1_E2_E2_full_vtx = E2_E2[vert_unconnectivity[0]].unsqueeze(1)

    # between "first" edges and "second edges"
    E1_E2 = torch.linalg.vecdot(E1,E2)
    TR1_TR1_E1_E2_full_tri = E1_E2[tri_unconnectivity[0]].unsqueeze(1)
    TR2_TR2_E1_E2_full_tri = E1_E2[tri_unconnectivity[1]].unsqueeze(1)

    TR1_TR1_E1_E2_full_edg = E1_E2[edge_unconnectivity[0]].unsqueeze(1)

    TR1_TR1_E1_E2_full_vtx = E1_E2[vert_unconnectivity[0]].unsqueeze(1)

    # single edge terms, can be 1st, 2nd, or even 3rd edge
    E_E = torch.linalg.vecdot(E_edge,E_edge)
    TR2_TR2_E_E_full_edg = E_E[edge_unconnectivity[1]].unsqueeze(1)

    # between triangle terms, use expanded version, repeated calcs when forceNormalDist is true
    # TODO, detect forceNormalDist and share computation
    TR1_TR2_E1_E1_full_tri = -torch.linalg.vecdot(TR1_E1_full_tri,TR2_E1_full_tri).unsqueeze(1)
    TR1_TR2_E1_E2_full_tri = -torch.linalg.vecdot(TR1_E1_full_tri,TR2_E2_full_tri).unsqueeze(1)
    TR1_TR2_E2_E1_full_tri = -torch.linalg.vecdot(TR1_E2_full_tri,TR2_E1_full_tri).unsqueeze(1)
    TR1_TR2_E2_E2_full_tri = -torch.linalg.vecdot(TR1_E2_full_tri,TR2_E2_full_tri).unsqueeze(1)

    quadratic_term_tri = 2 * torch.cat((TR1_TR1_E1_E1_full_tri, TR1_TR1_E1_E2_full_tri, TR1_TR2_E1_E1_full_tri, TR1_TR2_E1_E2_full_tri,
                    TR1_TR1_E1_E2_full_tri, TR1_TR1_E2_E2_full_tri, TR1_TR2_E2_E1_full_tri, TR1_TR2_E2_E2_full_tri,
                    TR1_TR2_E1_E1_full_tri, TR1_TR2_E2_E1_full_tri, TR2_TR2_E1_E1_full_tri, TR2_TR2_E1_E2_full_tri,
                    TR1_TR2_E1_E2_full_tri, TR1_TR2_E2_E2_full_tri, TR2_TR2_E1_E2_full_tri, TR2_TR2_E2_E2_full_tri),dim=1).reshape([-1,4,4])

    # triangle-edge terms, use expanded version
    TR1_TR2_E1_E_full_edg = -torch.linalg.vecdot(TR1_E1_full_edg,TR2_E_full_edg).unsqueeze(1)
    TR1_TR2_E2_E_full_edg = -torch.linalg.vecdot(TR1_E2_full_edg,TR2_E_full_edg).unsqueeze(1)

    quadratic_term_edg = 2 * torch.cat(
            (TR1_TR1_E1_E1_full_edg, TR1_TR1_E1_E2_full_edg, TR1_TR2_E1_E_full_edg,
                TR1_TR1_E1_E2_full_edg, TR1_TR1_E2_E2_full_edg, TR1_TR2_E2_E_full_edg, 
                TR1_TR2_E1_E_full_edg, TR1_TR2_E2_E_full_edg, TR2_TR2_E_E_full_edg, ),dim=1).reshape([-1,3,3])
    
    # triangle-vertex terms, use expanded version. Note that vertex does not actually appear!
    quadratic_term_vtx = 2 * torch.cat(
            (TR1_TR1_E1_E1_full_vtx, TR1_TR1_E1_E2_full_vtx,
                TR1_TR1_E1_E2_full_vtx, TR1_TR1_E2_E2_full_vtx, ),dim=1).reshape([-1,2,2])
    
        # build equality constraints, if needed
    if forceNormalDist:
        # need additional constraint that solution is in normal direction of 1st triangle
        # encoded as an equality constraint in the optimization
        normal_vecs_tri = mesh.faces_normals_packed()[tri_unconnectivity[0]]

        TR_1_N_tri = mesh.faces_normals_packed()[tri_unconnectivity[0]]

        # compute projection of edge vectors to containing triangle normal.
        # perform only once per triangle and expand later
        TR_1_E1_N = torch.linalg.vecdot( mesh.faces_normals_packed() , E1)
        TR_1_E2_N = torch.linalg.vecdot( mesh.faces_normals_packed() , E2)

        # compute projection of second triangle edge vectors onto first triangle normal
        TR_2_E1_N_tri = torch.linalg.vecdot( TR_1_N_tri , TR2_E1_full_tri)
        TR_2_E2_N_tri = torch.linalg.vecdot( TR_1_N_tri , TR2_E2_full_tri)

        # use projection of edge onto normal to compute rejection of normal from vector
        # perform only once per triangle, expand later
        vec_reject_TR1_E1 = (E1 - TR_1_E1_N.unsqueeze(1) * mesh.faces_normals_packed())
        vec_reject_TR1_E2 = (E2 - TR_1_E2_N.unsqueeze(1) * mesh.faces_normals_packed())

        equality_A_tri = torch.cat(
                      (vec_reject_TR1_E1[tri_unconnectivity[0]].unsqueeze(-1),
                       vec_reject_TR1_E2[tri_unconnectivity[0]].unsqueeze(-1),
                       ((TR_2_E1_N_tri.unsqueeze(1) * TR_1_N_tri) - TR2_E1_full_tri).unsqueeze(-1) ,
                       ((TR_2_E2_N_tri.unsqueeze(1) * TR_1_N_tri) - TR2_E2_full_tri).unsqueeze(-1)),dim=-1)
        
        # compute rejection of vector normal from the vector between triangles
        equality_b_tri = w_diff_vec_tri - (torch.linalg.vecdot(w_diff_vec_tri,normal_vecs_tri).unsqueeze(1) * normal_vecs_tri)
        


        # Don't ever do normal constraint for neighbors, since they can pass through each others sides, not just faces
        #
        # normal_vecs_edg = mesh.faces_normals_packed()[edge_unconnectivity[0]]
        # normal_vecs_vtx = mesh.faces_normals_packed()[vert_unconnectivity[0]]
        # TR_1_N_edg = mesh.faces_normals_packed()[edge_unconnectivity[0]]
        # TR_2_E_N_edg = torch.linalg.vecdot( TR_1_N_edg , TR2_E_full_edg)
        #
        # equality_A_edg = torch.cat(
        #               (vec_reject_TR1_E1[edge_unconnectivity[0]].unsqueeze(-1),
        #                vec_reject_TR1_E2[edge_unconnectivity[0]].unsqueeze(-1),
        #                ((TR_2_E_N_edg.unsqueeze(1) * TR_1_N_edg) - TR2_E_full_edg).unsqueeze(-1)),dim=-1)
        # equality_A_vtx = torch.cat(
        #               (vec_reject_TR1_E1[vert_unconnectivity[0]].unsqueeze(-1),
        #                vec_reject_TR1_E2[vert_unconnectivity[0]].unsqueeze(-1)),dim=-1)
        #
        # equality_b_edg = -(torch.linalg.vecdot(w_diff_vec_edg,normal_vecs_edg).unsqueeze(1) * normal_vecs_edg - w_diff_vec_edg)
        # equality_b_vtx = -(torch.linalg.vecdot(w_diff_vec_vtx,normal_vecs_vtx).unsqueeze(1) * normal_vecs_vtx - w_diff_vec_vtx)

    else:
        # no equality constraint needed, so empty tensor
        equality_A_tri = torch.tensor([],device=mesh.device) 
        equality_b_tri = torch.tensor([],device=mesh.device)

    
    return quadratic_term_tri, linear_term_tri, const_term_tri, quadratic_term_edg, linear_term_edg, const_term_edg, quadratic_term_vtx, linear_term_vtx, const_term_vtx, equality_A_tri, equality_b_tri


def __applyQuad(bary_coords, quadratic_term, linear_term, const_term):
    # computes distance for a batch of quadratic programs and their solutions
    # needed because QPFunction only returns the coordinates of the closest points on the triangles
    bc1 = bary_coords.unsqueeze(1)
    bc2 = bary_coords.unsqueeze(2)

    # for numeric reasons, this can be negative when it should be 0
    dist_raw = (torch.matmul(bc1, torch.matmul(quadratic_term, bc2))/2).squeeze(2).squeeze(1) + torch.linalg.vecdot(linear_term.squeeze(2), bary_coords) + const_term
    # replace negative values with 0, indicating a collision has taken place
    dist_relu = torch.nn.functional.relu(dist_raw)
    return dist_relu
    

def __qp_bary_feasible(quadratic_term, linear_term, const_term, equality_A, equality_b, inequality_G, inequality_h):
    # optimization is not possible if quadratic term is not semi-positive-definite (spd)
    # this means the parabola must have single minimum (non flat and facing up)

    # there is probably a better way to enforce this, but I force it to be SPD by repeadtedly adding a scaled identiy matrix.
    # scaled identity matrix is equivalent to L2 regularization on the triangle coordinates. ie, the optimization
    # is solving for a scaled objective that is distance + l2norm(coordinates). Intuitively, when there are 
    # infinit solutions, like when two triangles are at least partially parallel, this will cause us to find
    # the coordinates that are "smallest" along the parallel part. "Smallest" means closest to some origin. If
    # useRefPointBias is True, this origin is at one of the triangle corners. If it is false, this origin
    # is at the triangle center. 
    #
    # Solving for the identity matrix that will make our matrix spd seemed complicated, so for now
    # we compute a lower bound, the size of the smallest (largest negative) eigenvalue, compared to a small
    # positive epsilon (spd_target). We double the size of this target every time we fail to create an spd
    # matrix, then try again, up to max_spd_iter
    quadratic_term_diagonal = torch.diagonal(quadratic_term,dim1=-1,dim2=-2)
    quadratic_term_orig = quadratic_term_diagonal.clone()
    quadratic_term_reg = quadratic_term_diagonal.clone()
    
    quadratic_term_diagonal
    spd_min_eig=1e-6
    spd_target = spd_min_eig
    max_spd_iter = 4
    
    while max_spd_iter > 0:
        max_spd_iter-=1
        quadratic_term_diagonal.copy_(quadratic_term_reg)
        eigs = torch.real(torch.linalg.eigvals(quadratic_term.detach()))
        not_spd = torch.any(eigs < spd_min_eig,dim=1)

        # print(f'found {torch.where(not_spd)[0].shape[0]} non spd quadratic terms')
        if not torch.any(not_spd):
            break
        spd_target = spd_target * 2
        dist_regularizer = spd_target  - torch.min(eigs[not_spd],dim=1)[0]
        quadratic_term_reg[not_spd] += dist_regularizer.unsqueeze(-1)
        # regulizer_mat = (dist_regularizer.unsqueeze(1).unsqueeze(2) * torch.eye(quadratic_term.shape[-1], device = quadratic_term.device).unsqueeze(0))

        # quadratic_term_reg[not_spd] = quadratic_term_reg[not_spd] + regulizer_mat

    # solve all, including infeasible
    qp = QPFunction(check_Q_spd=False)
    __blockPrint()
    bary_coords = qp(quadratic_term, linear_term.squeeze(2), # optimization
                                                inequality_G, inequality_h, #inequality constraints
                                                equality_A,equality_b) # equality constraints)
    __enablePrint()
    
    # we can check that our inequality constraint was met (Gz <= h), which indicates bary coordinates fall outside triangle
    invalid_bary = torch.any(torch.matmul(bary_coords, inequality_G.transpose(0,1)) > inequality_h.unsqueeze(0),dim=1)

    

    # if any invalid solutions are found, we need to repeat the computation without them
    # pytorch fails to compute any derivatives if an invalid distance exists
    if torch.any(invalid_bary):
        valid_bary = torch.logical_not(invalid_bary)
        
        if equality_A.size(0) == 0:
            equality_A_filt = equality_A
            equality_b_filt = equality_b
        else:
            equality_A_filt = equality_A[valid_bary]
            equality_b_filt = equality_b[valid_bary]
        __blockPrint()
        bary_coords_feasible = qp(quadratic_term[valid_bary], linear_term[valid_bary].squeeze(2), # optimization
                                                inequality_G, inequality_h, #inequality constraints
                                                equality_A_filt,equality_b_filt) # equality constraints)
        __enablePrint()
        bary_coords = float('Inf') * torch.ones_like(bary_coords)
        bary_coords[valid_bary] = bary_coords_feasible
        quadratic_term_diagonal.copy_(quadratic_term_orig)
        dist_feasible = __applyQuad(bary_coords_feasible, quadratic_term[valid_bary], linear_term[valid_bary], const_term[valid_bary])
        dist_all = float('Inf') * torch.ones_like(const_term)
        dist_all[valid_bary] = dist_feasible
    else:
        quadratic_term_diagonal.copy_(quadratic_term_orig)
        dist_all = __applyQuad(bary_coords, quadratic_term, linear_term, const_term)



    return dist_all, bary_coords

# Disable
def __blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def __enablePrint():
    sys.stdout = sys.__stdout__

def self_collision_min(mesh, unconnectivity, forceNormalDist=1):
    # wrapper on self_collision that returns the minimum valid distance
    dists_raw, bary_coords_raw = self_collision(mesh=mesh, unconnectivity=unconnectivity, forceNormalDist=forceNormalDist)
    invalid_bary = torch.logical_or(torch.all(bary_coords_raw < 0,dim=1),
                                   torch.all(bary_coords_raw[:,[0,2]] + bary_coords_raw[:,[1,3]] > 1,dim=1))
    dists_filtered = dists_raw
    dists_filtered[invalid_bary] = float('Inf')
    return torch.min(dists_filtered)


class minWeightQualityFunction(ParallelJawQualityFunction):
    """Computes the minimum external wrench required to disrupt a grasp"""
    def __init__(self, config, min_quality=0.004):
        self.min_quality = 0.004
        ParallelJawQualityFunction.__init__(self, config)


    def quality(self, state, actions, is_watertight=True, is_inverted=False, mode=None):
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
        if actions.applied_to_object is False :
            with record_function("solveForIntersection"):
                actions = actions.apply_to_mesh(state, is_watertight=is_watertight, is_inverted=is_inverted)
        if actions.applied_to_object is False :
            # no intersection found, just return 0 quality
            return torch.zeros_like(actions.axis3D[...,0])
        self.Mesh = state
        self.Grasps = actions
        self.G = actions.grasp_matrix
        with record_function("minHull"):
            closest = self.find_min_weight(self.G)
        self.quality_cache = closest
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
        min_alpha = alpha_weights[:,-1]
        
        return min_alpha

class CannyFerrariQualityFunction(ParallelJawQualityFunction):
    """Computes the minimum external wrench required to disrupt a grasp"""
    def __init__(self, config,min_quality=0.002):
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
        if actions.applied_to_object is False :
            with record_function("solveForIntersection"):
                actions = actions.apply_to_mesh(state, is_watertight=is_watertight, is_inverted=is_inverted)
        if actions.applied_to_object is False :
            # no (or bad) intersection found, just return nan quality
            return torch.full_like(actions.axis3D[...,0], float('nan'))
        self.Mesh = state
        self.Grasps = actions
        self.G = actions.grasp_matrix
        with record_function("minHull"):
            closest = CannyFerrariQualityFunction.find_min_dist_to_hull(self.G)
        self.quality_cache = closest
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
        is_feasible = torch.zeros((G_unwrapped.shape[1]),dtype=torch.bool,device=G.device)
        facets_infeasible_local = []
        count_facets_per_grasp = []
        facets_feasible_local = []
        with record_function("ConvexHull-Loop"):
            for batch_idx in range(G_unwrapped.shape[1]):
                miniG = G_unwrapped[:,batch_idx,:]
                [simplices,equations] = qHullTorch.apply(miniG)
                areas = torch.abs(torch.linalg.det(miniG[simplices,:]))
                finite_size = areas > 1e-8
                simplices = simplices[finite_size]
                dist_from_origin = equations[finite_size,-1]
                is_above_origin = dist_from_origin<0
                
                if torch.all(is_above_origin):
                    is_feasible[batch_idx] = 1
                    # if all negative, closest point is max (min of abs)
                    min_dex = torch.argmax(dist_from_origin)
                    
                    facet = miniG[simplices[(min_dex),:],:].unsqueeze(0)
                    facets_feasible_local.append(facet)
                else:
                    facet = miniG[simplices[torch.logical_not(is_above_origin),:],:]
                    facets_infeasible_local.append(facet)
                    count_facets_per_grasp.append(facet.shape[0])
        
        if len(facets_feasible_local) > 0:
            facets = torch.cat(facets_feasible_local,dim=0)
            closest_feasible = CannyFerrariQualityFunction.compute_hyperplane_above(facets)
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
        closest = torch.zeros(is_feasible.shape, dtype=G.dtype, device=G.device)
        closest[is_feasible] = closest_feasible
        closest[torch.logical_not(is_feasible)] = closest_infeasible
        return closest

    dtype=torch.float64

class RobustCannyFerrariQualityFunction(CannyFerrariQualityFunction):
    """Measures the probability that grasps near a reference grasp will have high Canny Ferrari quality."""
    def __init__(self, config,min_quality=0.002):
        CannyFerrariQualityFunction.__init__(self, config,min_quality=min_quality)

    def quality(self, state, actions, mode='actual', is_watertight=True, is_inverted=False):
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
        mode: string from 'quality_increase', 'quality_decrease', 'actual'
            Determines whether we get the "answer" or just something that gives a
            a gradient for changing the answer in a particular direction

        Returns
        -------
        :obj:`numpy.ndarray`
            Array of the quality for each grasp.
        """
        # 
        noised_grasps = actions.generateNoisyGrasps(25).apply_to_mesh(state, ignore_backface_check=False, is_watertight=is_watertight, is_inverted=is_inverted)
        noised_tensor = super().quality(state, noised_grasps, is_watertight=is_watertight, is_inverted=is_inverted)
        # expected quality
        qual_tensor = torch.mean(noised_tensor)
        # if mode == 'actual':
        #     qual_tensor = torch.mean(noised_tensor)
        # elif mode == 'quality_increase':
        #     qual_tensor = 1-torch.nn.functional.relu(-(torch.mean(noised_tensor) - 1))
        # elif mode == 'quality_decrease':
        #     qual_tensor = 1-torch.nn.functional.relu((torch.mean(noised_tensor) - 1))

        # if mode == 'actual':
        #     qual_tensor = torch.mean((noised_tensor > self.min_quality).float())
        # elif mode == 'quality_increase':
        #     qual_tensor = torch.mean(1-torch.nn.functional.relu(-(noised_tensor - self.min_quality)))
        # elif mode == 'quality_decrease':
        #     qual_tensor = torch.mean(1-torch.nn.functional.relu((noised_tensor - self.min_quality)))
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
        hull_points = multi_gather_tris(points, hull.contiguous()) # simplices, verts, dims
        
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
    def computePlanes(points,hull):
        # generalized cross product to create planes
        # https://mathoverflow.net/questions/109949/how-to-efficiently-compute-the-generalized-cross-product

        # alternative: put points in cross product, but inefficient if plane compared repeatedly 
        # compare all simplices, as points to all points
        # 

        det_A, A_matrices = hullTorch.hull_det(points, hull)
        b = torch.zeros_like(A_matrices[...,0:1])
        b[...,-1,-1] = torch.sign(det_A)
        
        planes = torch.linalg.solve(A_matrices,b,left=True).squeeze(-1)

        return planes




class qHullTorch(torch.autograd.Function):
    @staticmethod
    def forward(_, miniG):
        # both fingers hit a backface, so convex hull will throw an error
        if torch.all(miniG == 0):
            return torch.tensor([range(6)],dtype=torch.long)
        # I don't remember why I was afraid of this, hasn't happened in memory
        if torch.any(torch.isnan(miniG)):
            breakpoint()
        miniGnumpy = miniG.numpy(force=True)
        hull = ConvexHull(miniGnumpy) # ,qhull_options='QJ'
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


def multi_indexing(index: torch.Tensor, shape: torch.Size, dim=-2):
    shape = list(shape)
    back_pad = len(shape) - index.ndim
    for _ in range(back_pad):
        index = index.unsqueeze(-1)
    expand_shape = shape
    expand_shape[dim] = -1
    return index.expand(*expand_shape)


def multi_gather(values: torch.Tensor, index: torch.Tensor, dim=-2):
    # take care of batch dimension of, and acts like a linear indexing in the target dimention
    # we assume that the index's last dimension is the dimension to be indexed on
    return values.gather(dim, multi_indexing(index, values.shape, dim))


def multi_gather_tris(v: torch.Tensor, f: torch.Tensor, dim=-2) -> torch.Tensor:
    # compute faces normals w.r.t the vertices (considering batch dimension)
    if v.ndim == (f.ndim + 1):
        f = f[None].expand(v.shape[0], *f.shape)
    # assert verts.shape[0] == faces.shape[0]
    shape = torch.tensor(v.shape)
    remainder = shape.flip(0)[:(len(shape) - dim - 1) % len(shape)]
    return multi_gather(v, f.view(*f.shape[:-2], -1), dim=dim).view(*f.shape, *remainder)  # B, F, 3, 3

def moller_trumbore(ray_o, ray_d, tris , eps=1e-8):
    """
    The Moller Trumbore algorithm for fast ray triangle intersection
    Naive batch implementation (m rays and n triangles at the same time)
    O(n_rays * n_faces) memory usage, parallelized execution
    Parameters
    ----------
    ray_o : torch.Tensor, (n_rays, 3)
    ray_d : torch.Tensor, (n_rays, 3)
    tris  : torch.Tensor, (n_faces, 3, 3)
    """
    E1 = tris[:, 1] - tris[:, 0]  # vector of edge 1 on triangle (n_faces, 3)
    E2 = tris[:, 2] - tris[:, 0]  # vector of edge 2 on triangle (n_faces, 3)

    # batch cross product
    N = torch.cross(E1, E2)  # normal to E1 and E2, automatically batched to (n_faces, 3)
    # TODO, should this be a solve instead? need to batch u,v,t into one matrix?
    invdet = 1. / -(torch.einsum('md,nd->mn', ray_d, N) + eps)  # inverse determinant (n_faces, 3)

    A0 = ray_o[:, None] - tris[None, :, 0]  # (n_rays, 3) - (n_faces, 3) -> (n_rays, n_faces, 3) automatic broadcast
    DA0 = torch.cross(A0, ray_d[:, None].expand(*A0.shape))  # (n_rays, n_faces, 3) x (n_rays, 3) -> (n_rays, n_faces, 3) no automatic broadcast

    u = torch.einsum('mnd,nd->mn', DA0, E2) * invdet
    v = -torch.einsum('mnd,nd->mn', DA0, E1) * invdet
    t = torch.einsum('mnd,nd->mn', A0, N) * invdet  # t >= 0.0 means this is a ray

    return u, v, t

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

    com_qual_func = CannyFerrariQualityFunction(config_dict)

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
    graspObj = GraspTorch(center3D, axis3D=axis3D, width=0.05,
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
    center3D.retain_grad()
    optimizer = optim.Rprop([axis3D, center3D], lr=0.00001)
    #optimizer = optim.SGD([center3D], lr=0.25, momentum=0.0)
    print('original-grasp', axis3D.squeeze().numpy(force=True), center3D.squeeze().numpy(force=True))
    for i in range(20):
        optimizer.zero_grad()
        graspObj = GraspTorch(center3D, axis3D=axis3D, width=0.05,
                        friction_coef=config_dict["friction_coef"], torque_scaling=config_dict["torque_scaling"])
        noised_grasps = graspObj.generateNoisyGrasps(25).apply_to_mesh(mesh, ignore_backface_check=True)
        noised_tensor = com_qual_func.quality(mesh, noised_grasps)
        com_qual_func.savemat(f'robust_sgd_iterates{i}.mat')
        qual_tensor = torch.sum(torch.nn.functional.relu(-(noised_tensor - 0.002)))
        # com_qual_func.savemat(f'quality_out{i}.mat')
        qual_tensor.backward()
        print('iteration: ', i)
        print('raw cf: ',noised_tensor.squeeze().numpy(force=True))
        print('count success: ',np.sum(noised_tensor.squeeze().numpy(force=True) > 0.002))
        print('loss score:',qual_tensor.squeeze().numpy(force=True))
        print('grasp-update', axis3D.grad.squeeze().numpy(force=True), center3D.grad.squeeze().numpy(force=True))
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
    unconnectivity = mesh_properties(mesh)
    # with record_function("distfunction"):
        # print(self_collision(mesh, unconnectivity,forceNormalDist=False))
    with record_function("distfunction_n"):
        print(self_collision(mesh, unconnectivity,forceNormalDist=False))
    config_dict = {
        "torque_scaling":1000,
        "soft_fingers":1,
        "friction_coef": 0.8, # TODO use 0.8 in practice
        "antipodality_pctile": 1.0 
    }
    print("mesh vol:", compute_mesh_volume(mesh).numpy(force=True))
    print("mesh bb vol:", compute_mesh_bounding_volume(mesh).numpy(force=True))
    print("mesh hull vol:", compute_mesh_hull_volume(mesh).numpy(force=True))
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

        test_grasps_compute.append(GraspTorch(center3D, axis3D=axis3D, width=0.05,
                                               friction_coef=config_dict["friction_coef"], torque_scaling=config_dict["torque_scaling"]))
        test_grasps_compute[-1] = test_grasps_compute[-1].apply_to_mesh(mesh)
        print("contact points:", i)
        print(test_grasps_compute[-1].contact_points.squeeze().numpy(force=True))
        print(np.array(dicts[-1]['contact_points']))
        print("contact normals:", i)
        print(test_grasps_compute[-1].contact_normals.squeeze().numpy(force=True))
        print(-torch.nn.functional.normalize(torch.tensor(dicts[-1]['normals_1'],device=device).transpose(0,1).double(),dim=-1).numpy(force=True)) # .json has inward normal
        test_grasps_set.append(GraspTorch(center3D, axis3D=axis3D, width=0.05,
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
    graspObj = GraspTorch(center3D, axis3D=axis3D, width=0.05,
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

    grasp1 = GraspTorch(center2d, angle, depth, width, renderer.rasterizer.cameras,friction_coef=config_dict["friction_coef"], torque_scaling=config_dict["torque_scaling"]) 
        

    center3D = torch.tensor([[ 0.027602000162005424, 0.017583999782800674, -9.273400064557791e-05]], device=device)
    axis3D   = torch.tensor([[-0.9384999871253967, 0.2660999894142151, -0.22010000050067902]], device=device)

    grasp2 = GraspTorch(center3D, axis3D=axis3D, width=width, camera_intr=renderer.rasterizer.cameras,friction_coef=config_dict["friction_coef"], torque_scaling=config_dict["torque_scaling"]) 
    grasp2.make2D(updateCamera=False)

    center3D = torch.tensor([[-0.03714486211538315, -0.029467197135090828, 0.01168159581720829]], device=device)
    axis3D   = torch.tensor([[-0.974246621131897, -0.19650164246559143, -0.11059238761663437]], device=device)

    grasp3 = GraspTorch(center3D, axis3D=axis3D, width=width, camera_intr=renderer.rasterizer.cameras,friction_coef=config_dict["friction_coef"], torque_scaling=config_dict["torque_scaling"]) 
    # Call ComForceClosureParallelJawQualityFunction init with parameters from gqcnn (from gqcnn/cfg/examples/replication/dex-net_2.1.yaml 

    # with record_function("FastAntipodalityFunction"):
    #     com_qual_func = ComForceClosureParallelJawQualityFunction(config_dict)

    # # Call quality with the Grasp2D and mesh
    #     com_qual_func.quality(mesh, grasp3)

    with record_function("CannyFerrari"):
        com_qual_func = CannyFerrariQualityFunction(config_dict)

    # Call quality with the Grasp2D and mesh
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
        device = torch.device("cuda:1")
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
    unconnectivity = mesh_properties(mesh)
    distances,barys = self_collision(mesh, unconnectivity,forceNormalDist=False)
    print(distances)

if __name__ == "__main__":
    #minHull.apply(torch.tensor(dict['G']).transpose(0,1).reshape((20,1,1,6)))
    np.set_printoptions(edgeitems=30, linewidth=100)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=False) as prof:
        with record_function("test_quality"):
            #model(inputs)
            with torch.enable_grad():        
                # test_stein()
                test_quality()
                test_wine()
                test_dist()
                
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))
    prof.export_chrome_trace("trace.json")