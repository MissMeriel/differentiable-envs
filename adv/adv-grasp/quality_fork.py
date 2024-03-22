import torch
import torch.optim as optim
import os
from torch.masked import masked_tensor
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import math
import json
import cvxopt as cvx
import cvxpy as cp
import pyhull.convex_hull as cvh
from pytorch3d.io import load_obj
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
from ll4ma_opt.problems.problem import Problem
from ll4ma_opt.problems import SteinWrapper
from ll4ma_opt.solvers import GradientDescent,BFGSMethod

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

    @staticmethod
    def normal_diagonal_3D(reference, var_triple, sampleCount, meanZero=False):
        output_size = [sampleCount] + list(reference.shape)
        #reference = torch.unsqueeze(reference,0)
        reference = reference.expand(output_size)
        output_samples = torch.zeros_like(reference)
        for i in range(3):
            if meanZero:
                output_samples[...,i] = torch.normal(output_samples[...,i], var_triple[i] ** 2, generator=torch.cuda.manual_seed(i))
            else:
                output_samples[...,i] = torch.normal(reference[...,i], var_triple[i] ** 2, generator=torch.cuda.manual_seed(i+10))
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
        center_noised_in_noise_frame = self.normal_diagonal_3D(center_in_noise_frame, t_var, sampleCount)
        # R_sample_sigma is treated as 1 x (1 grasp dim times) x 3 x 3
        center3D = torch.squeeze(R_sample_sigma.matmul(torch.unsqueeze(center_noised_in_noise_frame,-1)),-1)

        axis_in_noise_frame = torch.matmul(torch.transpose(R_sample_sigma,-1,-2), torch.unsqueeze(self.axis3D,-1))
        randRotVel = self.normal_diagonal_3D(self.axis3D, r_var, sampleCount, meanZero=True)
        randRotVel = torch.reshape(randRotVel, (-1,3))
        randRelRotMat = tf.so3_exp_map(randRotVel)
        randRelRotMat = torch.reshape(randRelRotMat, list(center3D.shape)+[3])
        # R_sample_sigma is treated as 1 x (1 grasp dim times) x 3 x 3
        axis3D = torch.squeeze(torch.matmul(R_sample_sigma, torch.matmul(randRelRotMat, axis_in_noise_frame)),-1)
        return GraspTorch(center=center3D, axis3D=axis3D, 
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
        
    @staticmethod
    def compute_mesh_COM(mesh): 
        """Compute the center of mass for a mesh, assume uniform density."""

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
        
        # det([[x1,y1,z1,1],[x2,y2,z2,1],[x3,y3,z3,1],[x4,y4,z4,1]]) / 6 
        # technically a scaled volume, since we dropped the division by 6
        # does det on last 2 dims, considers at least first 1 to be batch dim
        vol_per_triangle = torch.reshape(torch.linalg.det(mesh_tetra),(mesh_tetra.shape[0],1))

        com = torch.sum(com_per_triangle * vol_per_triangle,dim=0) / torch.sum(vol_per_triangle)

        return com

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
        finger_radius=0.005

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

        
    def apply_to_mesh(self, mesh): 
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
        mesh_unwrapped = multi_gather_tris(mesh.verts_packed(), mesh.faces_packed())
        opposite_dir_rays = self.ray_directions
        ray_o = self.center3D - (opposite_dir_rays * self.width / 2)
        ray_d = opposite_dir_rays # [axis3d, -axis3d]
        # ray is n, 3
        
        target_shape = ray_o.shape
        # moller_trumbore assumes flat list of rays (2d Nx3)
        ray_o_flat = torch.flatten(ray_o, end_dim=-2)
        ray_d_flat = torch.flatten(ray_d, end_dim=-2)

        u, v, t = moller_trumbore(ray_o_flat, ray_d_flat, mesh_unwrapped.double())

        u = torch.unflatten(u, 0, target_shape[:-1])
        v = torch.unflatten(v, 0, target_shape[:-1])
        t = torch.unflatten(t, 0, target_shape[:-1])
        # correct dir, not too far, actually hits triangle
        inside1 = ((t >= 0.0) * (t < self.width/2) * (u >= 0.0) * (v >= 0.0) * ((u + v) <= 1.0)).bool()  # (n_rays, n_faces)
        t[torch.logical_not(inside1)] = float('Inf')
        # (n_rays, n_faces)
        min_out = torch.min(t, -1,keepdim=True)

        intersectionPoints = ray_o + ray_d * min_out.values
        faces_index = min_out.indices
        contactFound = torch.all(torch.logical_not(torch.isinf(min_out.values)))
        self.contact_points = intersectionPoints
        
        verts = mesh.verts_packed()[mesh.faces_packed()[faces_index,:]]
        # experimental, weighted vertex normals
        vertex_normals = mesh.verts_normals_packed()[mesh.faces_packed()[faces_index,:]]
        u_vals = torch.gather(u, 2, faces_index).unsqueeze(3).unsqueeze(4)
        v_vals = torch.gather(v, 2, faces_index).unsqueeze(3).unsqueeze(4)
        w_vals = 1 - u_vals - v_vals
        weights = torch.cat((w_vals,u_vals,v_vals),-2)
        minReturn=torch.min(weights,dim=-2)
        normsVert = torch.sum(torch.multiply(vertex_normals,weights),dim=-2).squeeze(-2)
        # verts[[0,1],:,:,minReturn.indices.squeeze(),:] = torch.mean(verts, dim=-2,keepdim=False)
        # vertex_normals[[0,1],:,:,minReturn.indices.squeeze(),:] = mesh.faces_normals_packed()[faces_index,:]
        # (idxs_face, masks, sphereDirs) = GraspTorch.sphereSamples(self.contact_points, mesh)
        normsContinuous = GraspTorch.avgSphereArcNormal(mesh, self.contact_points, ray_d)
        # normsDexnet = self.svdSpherePoints(self.contact_points, sphereDirs, masks, ray_d)
        # normsSphereAvg = self.avgSpherePoints(state, idxs_face, masks, self.contact_points)
        #torch.mean(verts, dim=-2,keepdim=True)
        #vertex_normals.scatter_( state.faces_normals_packed()[faces_index,:])
        # TODO, update to use built in https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/interp_face_attrs.html
        #self.contact_normals = (torch.sum(torch.multiply(vertex_normals,weights),dim=-2).squeeze(-2) + torch.squeeze(state.faces_normals_packed()[faces_index,:],-2))/2
        normsFace = torch.squeeze(mesh.faces_normals_packed()[faces_index,:],-2)
        self.face_normals = normsFace
        self.applied_to_object = contactFound.item()
        self.contact_normals = normsContinuous
        self.mesh = mesh
        self.object_com = GraspTorch.compute_mesh_COM(mesh)
        self.faces_index = faces_index
        return self

    @staticmethod
    def dexnetRadius(mesh, gridDist=1.5):
        # matches min scaling from:
        # https://github.com/BerkeleyAutomation/dex-net/blob/cccf93319095374b0eefc24b8b6cd40bc23966d2/src/dexnet/database/mesh_processor.py#L281
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
    def avgSphereArcNormal(mesh, surface_point, in_rays):
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
        arc_com_total = torch.sum(arc_mass.unsqueeze(-1) * arc_com_3D,dim=(-3,-2)) / torch.sum(arc_mass,dim=(-1,-2)).unsqueeze(-1) 
        #arc_com_total = surface_point

        # alternative: could also do weighted average normal
        arc_mass_per_face = torch.sum(arc_mass,dim=-1,keepdim=True)
        normalContribution = mesh.faces_normals_packed().reshape(rm_shape[:-1]) * arc_mass_per_face
        surface_normals_avg = torch.nn.functional.normalize(torch.sum(normalContribution,dim=-2), dim=-1)


        # moment of inertia -> Cov -> SVD
        moment_of_inertia_vec = torch.cat((moment_of_inertia_1.unsqueeze(-1),moment_of_inertia_2.unsqueeze(-1),moment_of_inertia_3.unsqueeze(-1)),dim=-1)
        moment_of_inertia_local = torch.diag_embed(moment_of_inertia_vec)
        moment_of_inertia_local[torch.logical_not(mid_point_in_tri)] = 0
        moment_of_inertia_local = moment_of_inertia_local * (radius_2D*radius_2D).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # transform to shared CoM 
        # https://en.m.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor_of_rotation
        arc_displacement = (arc_com_total.unsqueeze(-2).unsqueeze(-2) - arc_com_3D)
        moment_of_inertia_local_aligned = torch.matmul(torch.matmul(fac_rot, moment_of_inertia_local),  fac_rot_inverse)
        # outter product
        m_o_i_2 = torch.matmul(arc_displacement.unsqueeze(-1), arc_displacement.unsqueeze(-2))
        # inner product
        m_o_i_1 = torch.matmul(arc_displacement.unsqueeze(-2), arc_displacement.unsqueeze(-1)).squeeze(-1)
        m_o_i_1 = torch.diag_embed(m_o_i_1.expand(list(m_o_i_1.shape)[:-1]+[3]))
        moment_of_inertia_global = moment_of_inertia_local_aligned + m_o_i_1 - m_o_i_2
        moment_of_inertia_global[torch.logical_not(mid_point_in_tri)] = 0

        moment_of_inertia_total = torch.sum(moment_of_inertia_global,dim=(-4,-3))
        moment_of_inertia_total_diag = torch.diagonal(moment_of_inertia_total,dim1=-1,dim2=-2)
        moment_of_inertia_total_trace = torch.sum(moment_of_inertia_total_diag,keepdim=True,dim=-1)
        moment_of_inertia_total_trace_mat = torch.diag_embed(moment_of_inertia_total_trace.expand(list(moment_of_inertia_total_trace.shape)[:-1]+[3]))
        cov = moment_of_inertia_total_trace_mat/2-moment_of_inertia_total
        eig_return = torch.linalg.eigh(cov)
        surface_normals = torch.real(eig_return.eigenvectors[...,0])

        # savemat('segments.mat',{'segments':segments_3D[linearized_segments_mask].numpy(force=True),'eigvec':torch.real(eig_return[1]).numpy(force=True),'eigval':torch.real(eig_return[0]).numpy(force=True)})

        #return surface_normals * -torch.sign(torch.sum(surface_normals * in_rays, dim=-1,keepdim=True))
        return surface_normals_avg

    @staticmethod
    def avgSpherePointsNormal(self, mesh, idxs_face, masks, surface_point):
        normals = torch.zeros_like(surface_point).reshape((len(masks),3))
        for maskInd in range(len(masks)):
            normals[maskInd,:] = torch.mean(mesh.faces_normals_packed()[idxs_face.squeeze()[maskInd, masks[maskInd].squeeze()],:], dim=0)
        normals = normals.reshape(surface_point.shape)
        return normals

    @staticmethod
    def svdSpherePointsNormal(self, surface_point, sphereDirs, masks, in_rays):

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


class CannyFerrariQualityFunction(ParallelJawQualityFunction):
    """Measures the distance to the estimated center of mass for antipodal
    parallel-jaw grasps."""
    def __init__(self, config):
        ParallelJawQualityFunction.__init__(self, config)


    def quality(self, state, actions):
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
        if actions.applied_to_object is False :
            with record_function("solveForIntersection"):
                actions = actions.apply_to_mesh(state)
        if actions.applied_to_object is False :
            return torch.zeros_like(actions.axis3D[...,0])
        self.Mesh = state
        self.Grasps = actions
        self.G = actions.grasp_matrix
        with record_function("minHull"):
            closest = CannyFerrariQualityFunction.find_min_dist_to_hull(self.G)
        self.quality_cache = closest
        return closest
    
    def savemat(self, path, other_items=None):
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
            ep0, ep1 = self.Grasps.endpoints3D
            dict_to_save['endpoints3D'] =torch.cat((ep0.unsqueeze(0),ep1.unsqueeze(0)),dim=0).numpy(force=True)

        if other_items is not None:
            dict_to_save = dict_to_save | other_items

        dict_to_save['quality'] = self.quality_cache.numpy(force=True)

        savemat(path, dict_to_save)

        return
    
    @staticmethod
    def qp_wrap(facets):
        ### TODO solve QP over G to figure out sign
        # square facet matrix
        Gsquared  = torch.linalg.matmul(facets,torch.permute(facets, [0,2,1]))
        wrench_regularizer=1e-10
        regulizer_mat = (wrench_regularizer * torch.eye(Gsquared.shape[1], device = facets.device, dtype=facets.dtype))
        n_dim = Gsquared.shape[1]
        n_batch = Gsquared.shape[0]
        P = 2 * (Gsquared + regulizer_mat).transpose(1,2)
        q = torch.zeros((n_batch,n_dim), device = facets.device, dtype=facets.dtype)
        G = -torch.eye(n_dim, device = facets.device, dtype=facets.dtype).unsqueeze(0).expand((n_batch,n_dim,n_dim))
        h = torch.zeros((n_batch,n_dim), device = facets.device, dtype=facets.dtype)
        A = torch.ones((n_batch,1,n_dim), device = facets.device, dtype=facets.dtype)
        b = torch.ones((n_batch,1),device = facets.device, dtype=facets.dtype)

        x = QPFunction(check_Q_spd=True)(P, q, G, h, A , b)
        dist = torch.sqrt(torch.matmul(x.unsqueeze(1), torch.matmul(P, x.unsqueeze(2)))/2)
        return dist, x, P
    
    @staticmethod
    def min_norm_vector_in_facet(facet, wrench_regularizer=1e-10):
        """ Finds the minimum norm point in the convex hull of a given facet (aka simplex) by solving a QP.

        Parameters
        ----------
        facet : 6xN :obj:`numpy.ndarray`
            vectors forming the facet
        wrench_regularizer : float
            small float to make quadratic program positive semidefinite

        Returns
        -------
        float
            minimum norm of any point in the convex hull of the facet
        Nx1 :obj:`numpy.ndarray`
            vector of coefficients that achieves the minimum
        """
        dim = facet.shape[1] # num vertices in facet

        # create alpha weights for vertices of facet
        G = facet.T.dot(facet)
        grasp_matrix = G + wrench_regularizer * np.eye(G.shape[0])

        # Solve QP to minimize .5 x'Px + q'x subject to Gx <= h, Ax = b
        P = cvx.matrix(2 * grasp_matrix)   # quadratic cost for Euclidean dist
        q = cvx.matrix(np.zeros((dim, 1)))
        G = cvx.matrix(-np.eye(dim))       # greater than zero constraint
        h = cvx.matrix(np.zeros((dim, 1)))
        A = cvx.matrix(np.ones((1, dim)))  # sum constraint to enforce convex
        b = cvx.matrix(np.ones(1))         # combinations of vertices

        sol = cvx.solvers.qp(P, q, G, h, A, b)
        v = np.array(sol['x'])
        min_norm = np.sqrt(sol['primal objective'])

        return abs(min_norm), v, 2 * grasp_matrix

    @staticmethod
    def distWrap(G_unwrapped):
        facets_local = []
        lengths = []
        with record_function("ConvexHull-Loop"):
            for batch_idx in range(G_unwrapped.shape[1]):
                miniG = G_unwrapped[:,batch_idx,:]
                simplices = qHullTorch.apply(miniG)
                facet = miniG[simplices,:]
                facets_local.append(facet)
                lengths.append(facet.shape[0])
        facets = torch.cat(facets_local,dim=0)
        ### reassemble simplices into batch may be too many and need to serialize
        ## maybe we only (re)compute the important one in pytorch, drop others for memory
        with record_function("qp_wrap"):
            dist,x,P = CannyFerrariQualityFunction.qp_wrap(facets)
        return dist, lengths, x, facets
    
    @staticmethod
    def find_min_dist_to_hull(G):       
        original_shape = G.shape
        G_unwrapped = G.view((original_shape[0]*original_shape[1], -1, 6))
        dist, lengths, x, facets = CannyFerrariQualityFunction.distWrap(G_unwrapped)

        start_ind = 0
        closest = torch.zeros(len(lengths),1, dtype=torch.float64)
        for index,length in enumerate(lengths):
            end_ind = start_ind + length
            dist_local = dist[start_ind:end_ind]
            minReturn = torch.min(dist_local[torch.logical_not(torch.logical_or(torch.isnan(dist_local),dist_local<0))],dim=0)
            closest[index] = minReturn.values
            start_ind = end_ind
        return closest

    dtype=torch.float64

class qHullTorch(torch.autograd.Function):
    @staticmethod
    def forward(_, miniG):
        if torch.any(torch.isnan(miniG)):
            breakpoint()
        miniGnumpy = miniG.numpy(force=True)
        hull = ConvexHull(miniGnumpy)  
        return torch.tensor(hull.simplices,dtype=torch.long)
    @staticmethod
    def backward(_, grad_output):
        # miniG, = ctx.saved_tensors
        # grad = torch.zeros(size=)
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
    
def test_quality():
    # load PyTorch3D mesh from .obj file
    renderer, device = pytorch_setup()
    with record_function("load_obj"):
        verts, faces_idx, _ = load_obj("data/new_barclamp.obj")
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

    # Test intersection finding
    test_grasps_compute = []
    test_grasps_set = []
    dicts = []
    for i in range(0,12):
        f = open('data/data/data'+str(i)+'.json')
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
    #     dists,_,_,_ = minHull.distWrap(G_unwrapped)
    #     np.testing.assert_allclose(dists.reshape((-1)).numpy(force=True),
    #             np.array(dicts[i]['dists']).T,
    #             atol=0.04,rtol=test[1])

    # Test final grasp quality
    tests = [[1e-3, 1e-5, test_grasps_set], [1e-2, 1e-3, test_grasps_compute]]
    print("atol", tests[0][0], "rtol", tests[0][1],"atol", tests[1][0], "rtol", tests[1][1])
    print("torch (w/o col), torch (w/ col), dexnet")
    for i in range(len(dicts)):            
            com_qual_func = CannyFerrariQualityFunction(config_dict)
            # print("set")
            torch_quality_no_col = com_qual_func.quality(mesh, tests[0][2][i]).numpy(force=True)[0]
            np.testing.assert_allclose(torch_quality_no_col,
                    np.array(dicts[i]['ferrari_canny_fc08']).T,
                    atol=tests[0][0],rtol=tests[0][1])
            # print("compute")
            torch_quality_col = com_qual_func.quality(mesh, tests[1][2][i]).numpy(force=True)[0]
            np.testing.assert_allclose(torch_quality_col,
                    np.array(dicts[i]['ferrari_canny_fc08']).T,
                    atol=tests[1][0],rtol=tests[1][1])
            print(torch_quality_no_col[0],",", torch_quality_col[0],",",
            dicts[i]['ferrari_canny_fc08'],";")

    #torch.autograd.set_detect_anomaly(True)
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


    optimizer = optim.SGD([axis3D, center3D], lr=1, momentum=0.0)
    for i in range(10):
        optimizer.zero_grad()
        graspObj = GraspTorch(center3D, axis3D=axis3D, width=0.05,
                        friction_coef=config_dict["friction_coef"], torque_scaling=config_dict["torque_scaling"]).apply_to_mesh(mesh)
        noised_grasps = graspObj.generateNoisyGrasps(25)
        noised_tensor = com_qual_func.quality(mesh, noised_grasps)   
        qual_tensor = torch.sum(torch.nn.functional.relu(-(noised_tensor - 0.002)))
        # com_qual_func.savemat(f'quality_out{i}.mat')
        qual_tensor.backward()
        print('iteration: ', i)
        print('raw cf: ',noised_tensor.squeeze().numpy(force=True))
        print('count success: ',np.sum(noised_tensor.squeeze().numpy(force=True) > 0.002))
        print('loss score:',qual_tensor.squeeze().numpy(force=True))
        print('grasp-update', axis3D.grad.squeeze().numpy(force=True), center3D.grad.squeeze().numpy(force=True))
        print('new-grasp', axis3D.squeeze().numpy(force=True), center3D.squeeze().numpy(force=True))
        print()
        optimizer.step()

    optimizer = optim.SGD([axis3D, center3D], lr=0.001, momentum=0.0)
    for i in range(10):
        optimizer.zero_grad()
        graspObj = GraspTorch(center3D, axis3D=axis3D, width=0.05,
                        friction_coef=config_dict["friction_coef"], torque_scaling=config_dict["torque_scaling"]).apply_to_mesh(mesh)
   
        qual_tensor = -com_qual_func.quality(mesh, graspObj)
        # com_qual_func.savemat(f'quality_out{i}.mat')
        qual_tensor.backward()
        print('quality:',-qual_tensor.squeeze().numpy(force=True))
        print('grad', axis3D.grad.squeeze().numpy(force=True), center3D.grad.squeeze().numpy(force=True))
        print('value', axis3D.squeeze().numpy(force=True), center3D.squeeze().numpy(force=True))
        optimizer.step()
    

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

    with record_function("FastAntipodalityFunction"):
        com_qual_func = ComForceClosureParallelJawQualityFunction(config_dict)

    # Call quality with the Grasp2D and mesh
        com_qual_func.quality(mesh, grasp3)

    with record_function("CannyFerrari"):
        com_qual_func = CannyFerrariQualityFunction(config_dict)

    # Call quality with the Grasp2D and mesh
        com_qual_func.quality(mesh, grasp3)

def test_stein():

    renderer, device = pytorch_setup()
    with record_function("load_obj"):
        verts, faces_idx, _ = load_obj("data/new_barclamp.obj")
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
    i=0
    f = open('data/data/data'+str(i)+'.json')
    grasp_dict = json.load(f)

    center3D = torch.tensor([grasp_dict['pytorch_w_center']],device=device,requires_grad=True)
    axis3D = torch.tensor([grasp_dict['pytorch_w_axis']],device=device,requires_grad=True)
    axis3D.retain_grad() 
    center3D.retain_grad()

    com_qual_func = CannyFerrariQualityFunction(config_dict)
    
    GT = lambda center,axis:GraspTorch(center, axis3D=axis, width=0.05,friction_coef=config_dict["friction_coef"], torque_scaling=config_dict["torque_scaling"]).apply_to_mesh(mesh)
    graspObj = GraspTorch(center3D, axis3D=axis3D, width=0.05,friction_coef=config_dict["friction_coef"], torque_scaling=config_dict["torque_scaling"]).apply_to_mesh(mesh)

    grasp = GT(center3D,axis3D)
    qual = com_qual_func.quality(mesh, grasp)
    qual.backward(inputs=[axis3D, center3D])
    print('quality', qual)
    print('center', center3D, center3D.grad)
    print('axis3D', axis3D, axis3D.grad)

    prob = graspQualityOpt(com_qual_func,GT,mesh)
    num_particles = 10
    stein_problem = SteinWrapper(prob,num_particles,repulsive_weight=1e-3)


    solver = BFGSMethod(stein_problem, alpha=0.01, rho=0.2,min_alpha=1e-5)
    initial_solution = torch.cat((axis3D, center3D),dim=1).T.numpy(force=True)
    particles = np.random.normal(initial_solution,1e-4,(num_particles,6,1))
    stacked_particles = particles.reshape((-1,1))
    result = solver.optimize(stacked_particles,max_iterations=10)
    result.display()
    new_particles = result.iterates.reshape((result.iterates.shape[0],-1,6,1))
    tensor_iterates = prob.make_tensor(new_particles)
    for iterate_index in range(new_particles.shape[0]):
        for particle_index in range(new_particles.shape[1]):
            axis = tensor_iterates[iterate_index,particle_index, :3].T
            center = tensor_iterates[iterate_index,particle_index, 3:].T
            grasp = GT(center,axis)
            com_qual_func.quality(mesh, grasp)
            com_qual_func.savemat(f'stein_iterates{iterate_index}_{particle_index}.mat')

class graspQualityOpt(Problem):
    def __init__(self, qualityObj=None, grasp=None, mesh=None):
        self.grasp = grasp
        self.device = mesh.device
        self.qualityObj = qualityObj
        self.mesh = mesh
        super().__init__()
        self.make_tensor = self.test
    def test(self, x):
        return torch.DoubleTensor(x).to(self.device)
    def size(self):
        return 6
    def tensor_batch_cost(self, x):
        axis = x[:,:3,0]
        center = x[:,3:,0]
        return -self.qualityObj.quality(self.mesh,self.grasp(center,axis)).squeeze(-1)
    def tensor_cost(self, x):
        axis = x[:3].T
        center = x[3:].T
        return -self.qualityObj.quality(self.mesh,self.grasp(center,axis))

if __name__ == "__main__":
    #minHull.apply(torch.tensor(dict['G']).transpose(0,1).reshape((20,1,1,6)))
    np.set_printoptions(edgeitems=30, linewidth=100)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=False) as prof:
        with record_function("test_quality"):
            #model(inputs)
            with torch.enable_grad():        
                # test_stein()
                test_quality()
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))
    # prof.export_chrome_trace("trace.json")