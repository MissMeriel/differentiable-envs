import torch
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import math
import json
import cvxopt as cvx
import cvxpy as cp
import pyhull.convex_hull as cvh
from pytorch3d.io import load_obj
import pytorch3d.transforms as tf
from scipy.spatial import ConvexHull
from pytorch3d.structures import Meshes
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

# class Grasp(Object)
#
# grasp2D(self, updateRotation=False)


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
                 width=0.0,
                 camera_intr=None,
                 contact_points=None,
                 contact_normals=None,
                 axis3D=None):
        self.width = width
        self.camera_intr = camera_intr
        if(center.shape[-1] == 3): # 3D grasp
            self.center3D = center.double()
            self.axis3D = axis3D.double()


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
        self.friction_cone = None

    def make2D(self, updateCamera=False,camera_intr=None):
        if camera_intr==None:
            camera_intr = self.camera_intr
        if camera_intr==None:
            # TODO error
            return None

        if updateCamera: # in order to keep specified axis, we need to update the camera
            
            axis3D = camera_intr.get_world_to_view_transform().transform_normals(self.axis3D.float()).double() # in world
            
            cameraDir = torch.tensor([0.,0.,1.],device=self.axis3D.device,dtype=axis3D.dtype) # camera Z
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
    
    def torques(self, forces, com):
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
        
        momentArm = self.contact_points - com
        momentArm = momentArm.expand([forces.shape[0]]+list(momentArm.shape))
        torques = torch.linalg.cross(momentArm, forces, dim=-1)
        return torques
        
    def normal_force_magnitude(self):
        """ Returns the component of the force that the contact would apply along the normal direction.

        Returns
        -------
        float
            magnitude of force along object surface normal
        """

        in_direction_norm = torch.nn.functional.normalize(self.ray_directions,dim=-1)

        in_normal = -self.contact_normals

        normal_force_mag = torch.sum(torch.mul(in_normal, in_direction_norm),-1)
        return torch.nn.functional.relu(normal_force_mag)

    @staticmethod
    def grasp_matrix(forces, torques, normals, torque_scaling, soft_fingers=False,
                     finger_radius=0.005, friction_coef=0.5):
        """ Computes the grasp map between contact forces and wrenchs on the object in its reference frame.

        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            set of forces on object in object basis
        torques : 3xN :obj:`numpy.ndarray`
            set of torques on object in object basis
        normals : 3xN :obj:`numpy.ndarray`
            surface normals at the contact points
        soft_fingers : bool
            whether or not to use the soft finger contact model
        finger_radius : float
            the radius of the fingers to use
        params : :obj:`GraspQualityConfig`
            set of parameters for grasp matrix and contact model

        Returns
        -------
        G : 6xM :obj:`numpy.ndarray`
            grasp map
        """
        G = torch.cat([forces, torques*torque_scaling], dim=-1)
        if soft_fingers:

            torsion = np.pi * finger_radius**2 * friction_coef * normals * torque_scaling
            G_torsion = torch.zeros(torsion.shape, device=torsion.device, dtype=torsion.dtype)
            G_torsion = torch.cat((G_torsion, torsion),-1)
            G_torsion = G_torsion.unsqueeze(0)
            G = torch.cat((G, G_torsion, -G_torsion), 0)

        return G


    def compute_friction_cone(self, state, num_cone_faces=8, friction_coef=0.5):
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

        if self.friction_cone is not None:
            return self.friction_cone

        if self.contact_points is None or self.contact_normals is  None:
            # TODO return warning that normals are needed first
            return self.friction_cone
        
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
            
        yvec = reshape_local(yvec, num_cone_faces)
        xvec = reshape_local(xvec, num_cone_faces)

        sampleAngles = torch.linspace(0, 2 * math.pi, num_cone_faces+1, device=self.contact_normals.device)
        sampleAngles = sampleAngles[:-1]
        sampleAngles = torch.reshape(sampleAngles, [num_cone_faces] + [1] * (len(xvec.shape)-1))

        tan_vec = torch.mul(xvec, torch.cos(sampleAngles)) + torch.mul(yvec, torch.sin(sampleAngles))
        self.friction_cone = -self.contact_normals + friction_coef * tan_vec
        return self.friction_cone

        
    def solveForIntersection(self, state): 
        """Compute where grasp contacts a mesh, state is meshes pytorch3D object"""
        # TODO
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
        mesh_unwrapped = multi_gather_tris(state.verts_packed(), state.faces_packed())
        opposite_dir_rays = self.ray_directions
        ray_o = self.center3D + (opposite_dir_rays * self.width / 2)
        ray_d = -opposite_dir_rays # [axis3d, -axis3d]
        # ray is n, 3
        
        target_shape = ray_o.shape
        # moller_trumbore assumes flat list of rays (2d Nx3)
        ray_o_flat = torch.flatten(ray_o, end_dim=-2).float()
        ray_d_flat = torch.flatten(ray_d, end_dim=-2).float()

        u, v, t = moller_trumbore(ray_o_flat, ray_d_flat, mesh_unwrapped)

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
        contactFound = torch.logical_not(torch.isinf(min_out.values))
        self.contact_points = intersectionPoints
        self.contact_normals = torch.squeeze(state.faces_normals_packed()[faces_index,:],-2)
        return self, faces_index, contactFound




    # @property
    # def approach_axis(self):
    #     return np.array([0, 0, 1])

    # @property
    # def approach_angle(self):
    #     """The angle between the grasp approach axis and camera optical axis.
    #     """
    #     return 0.0

    # @property
    # def frame(self):
    #     """The name of the frame of reference for the grasp."""
    #     if self.camera_intr is None:

    #         raise ValueError("Must specify camera intrinsics")
    #     return self.camera_intr.frame





    @property
    def feature_vec(self):
        """Returns the feature vector for the grasp.

        `v = [p1, p2, depth]` where `p1` and `p2` are the jaw locations in
        image space.
        """
        p1, p2 = self.endpoints
        return np.r_[p1, p2, self.depth]


class GraspQualityFunction():	#ABC):
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


class CannyFerrariQualityFunction(ParallelJawQualityFunction):
    """Measures the distance to the estimated center of mass for antipodal
    parallel-jaw grasps."""
    def __init__(self, config):
        self._soft_fingers = config["soft_fingers"]
        self._torque_scaling = config["torque_scaling"]
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
        if actions.contact_normals == None or actions.contact_points == None:
            with record_function("solveForIntersection"):
                actions, faces_index, contactFound = actions.solveForIntersection(state)
        
        # Compute object center of mass.
        object_com = ParallelJawQualityFunction.compute_mesh_COM(state)

        bounding_box = state.get_bounding_boxes()
        bounding_lengths = torch.diff(bounding_box, dim=-1 )
        if self._torque_scaling == None:
            median_length = torch.median(bounding_lengths)
            torque_scaling = torch.pow(median_length, -1)
        else:
            torque_scaling = self._torque_scaling

        n_force = actions.normal_force_magnitude().unsqueeze(-1)
        normals = torch.mul(-actions.contact_normals , n_force)

        n_force = n_force.unsqueeze(0)
        cone = torch.mul(actions.compute_friction_cone(state,friction_coef=self._friction_coef), n_force)
        torques = torch.mul(actions.torques(cone, object_com), n_force)

            # def grasp_matrix(forces, torques, normals, torque_scaling, soft_fingers=False,
            #          finger_radius=0.005, friction_coef=0.5):
        G = actions.grasp_matrix(cone, torques, normals, torque_scaling, 
                                 soft_fingers=self._soft_fingers, finger_radius=0.005, 
                                 friction_coef=self._friction_coef)
        with record_function("minHull"):
            closest = minHull.apply(G)

        return closest
    
    # taken from: https://gist.github.com/dendenxu/ee5008acb5607195582e7983a384e644#file-moller_trumbore_winding_number_inside_mesh-py-L318
class minHull(torch.autograd.Function):
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

        x = QPFunction(check_Q_spd=False)(P, q, G, h, A , b)
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

    def forward_single_np(Q, p, G, h, A, b):
        nz, neq, nineq = p.shape[0], A.shape[0] if A is not None else 0, G.shape[0]

        z_ = cp.Variable(nz)

        obj = cp.Minimize(0.5 * cp.quad_form(z_, Q) + p.T @ z_)
        eqCon = A @ z_ == b if neq > 0 else None
        if nineq > 0:
            slacks = cp.Variable(nineq)
            ineqCon = G @ z_ + slacks == h
            slacksCon = slacks >= 0
        else:
            ineqCon = slacks = slacksCon = None
        cons = [x for x in [eqCon, ineqCon, slacksCon] if x is not None]
        prob = cp.Problem(obj, cons)
        prob.solve()  # solver=cp.SCS, max_iters=5000, verbose=False)
        # prob.solve(solver=cp.SCS, max_iters=10000, verbose=True)
        assert('optimal' in prob.status)
        zhat = np.array(z_.value).ravel()
        nu = np.array(eqCon.dual_value).ravel() if eqCon is not None else None
        if ineqCon is not None:
            lam = np.array(ineqCon.dual_value).ravel()
            slacks = np.array(slacks.value).ravel()
        else:
            lam = slacks = None

        return prob.value, zhat, nu, lam, slacks

    @staticmethod
    def forward(ctx, G):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        original_shape = G.shape
        G_unwrapped = G.reshape((original_shape[0]*original_shape[1], -1, 6))
        ctx.save_for_backward(input)
        numpyG = G_unwrapped.numpy(force=True)
        
  
        facets_local = []
        lengths = []
        with record_function("ConvexHull-Loop"):
            for batch_idx in range(numpyG.shape[1]):
                miniGnumpy = numpyG[:,batch_idx,:]
                miniG = G_unwrapped[:,batch_idx,:]
                hull = ConvexHull(miniGnumpy)
                facets_local.append(miniG[hull.simplices,:])
                lengths.append(hull.nsimplex)
        facets = torch.cat(facets_local,dim=0)
        ### reassemble simplices into batch may be too many and need to serialize
        ## maybe we only (re)compute the important one in pytorch, drop others for memory
        with record_function("qp_wrap"):
            dist,x,P = minHull.qp_wrap(facets)


        start_ind = 0
        closest = torch.zeros(len(lengths),1, dtype=torch.float64)
        for index,length in enumerate(lengths):
        # todo reshape back to batch dim
            end_ind = start_ind + length
            dist_local = dist[start_ind:end_ind]
            closest[index] = torch.min(dist_local[torch.logical_not(torch.logical_or(torch.isnan(dist_local),dist_local<0))],dim=0).values
            start_ind = end_ind
        return closest

    dtype=torch.float64
    
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
        object_com = ParallelJawQualityFunction.compute_mesh_COM(state)

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
        
        dist = torch.norm(actions.center3D - object_com,dim=-1)
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


def linear_indexing(index: torch.Tensor, shape: torch.Size, dim=0):
    assert index.ndim == 1
    shape = list(shape)
    dim = dim if dim >= 0 else len(shape) + dim
    front_pad = dim
    back_pad = len(shape) - dim - 1
    for _ in range(front_pad):
        index = index.unsqueeze(0)
    for _ in range(back_pad):
        index = index.unsqueeze(-1)
    expand_shape = shape
    expand_shape[dim] = -1
    return index.expand(*expand_shape)


def linear_gather(values: torch.Tensor, index: torch.Tensor, dim=0):
    # only taking linea indices as input
    return values.gather(dim, linear_indexing(index, values.shape, dim))


def linear_scatter(target: torch.Tensor, index: torch.Tensor, values: torch.Tensor, dim=0):
    return target.scatter(dim, linear_indexing(index, values.shape, dim), values)


def linear_scatter_(target: torch.Tensor, index: torch.Tensor, values: torch.Tensor, dim=0):
    return target.scatter_(dim, linear_indexing(index, values.shape, dim), values)

def ray_stabbing(pts: torch.Tensor, verts: torch.Tensor, faces: torch.Tensor, multiplier: int = 1):
    """
    Check whether a bunch of points is inside the mesh defined by verts and faces
    effectively calculating their occupancy values
    Parameters
    ----------
    ray_o : torch.Tensor(float), (n_rays, 3)
    verts : torch.Tensor(float), (n_verts, 3)
    faces : torch.Tensor(long), (n_faces, 3)
    """
    n_rays = pts.shape[0]
    pts = pts[None].expand(multiplier, n_rays, -1)
    pts = pts.reshape(-1, 3)
    ray_d = torch.rand_like(pts)  # (n_rays, 3)
    ray_d = normalize(ray_d)  # (n_rays, 3)
    # 251,252
    u, v, t = moller_trumbore(pts, ray_d, multi_gather_tris(verts, faces))  # (n_rays, n_faces, 3)
    inside = ((t >= 0.0) * (u >= 0.0) * (v >= 0.0) * ((u + v) <= 1.0)).bool()  # (n_rays, n_faces)
    inside = (inside.count_nonzero(dim=-1) % 2).bool()  # if mod 2 is 0, even, outside, inside is odd
    inside = inside.view(multiplier, n_rays, -1)
    inside = inside.sum(dim=0) / multiplier  # any show inside mesh
    return inside


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
        fl = torch.tensor([[525.0]])
        pp = torch.tensor([[319.5, 239.5]])
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
		verts, faces_idx, _ = load_obj("adv/adv-grasp/data/bar_clamp.obj")
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
		"friction_coef": 0.5, # TODO use 0.8 in practice
		"antipodality_pctile": 1.0 
	}

	object_com = ParallelJawQualityFunction.compute_mesh_COM(mesh)
	test_grasps = []
	dicts = []
	for i in range(12):
		f = open('adv/adv-grasp/data/data/data'+str(i)+'.json')
		dicts.append(json.load(f))
		center3D = torch.tensor([dicts[i]['pytorch_w_center']],device=device)
		axis3D = torch.tensor([dicts[i]['pytorch_w_axis']],device=device)        
		test_grasps.append(GraspTorch(center3D, axis3D=axis3D))

		test_grasps[i], _, _ = test_grasps[i].solveForIntersection(mesh)
		print("contact points:",test_grasps[i].contact_points,dicts[i]['contact_points'],i)
		print("contact normals:",test_grasps[i].contact_normals,dicts[i]['normals_1'],i)

	for i in range(len(dicts)):
		test_grasps[i].contact_points = torch.tensor(dicts[i]['contact_points'],device=device).unsqueeze(1).double()
		test_grasps[i].contact_normals = -torch.nn.functional.normalize(torch.tensor(dicts[i]['normals_1'],device=device).transpose(0,1).unsqueeze(1).double(),dim=-1)
		n_force = test_grasps[i].normal_force_magnitude().unsqueeze(-1)
		normals = torch.mul(-test_grasps[i].contact_normals , n_force)
		cone = torch.mul(test_grasps[i].compute_friction_cone(mesh,friction_coef=0.5), n_force)
		torques = torch.mul(test_grasps[i].torques(cone, object_com), n_force)
		
		# print("n_force:",list(n_force.squeeze().numpy(force=True)),(dicts[i]['n_0'],dicts[i]['n_1']),i)
		# print("cone:", i)
		# print(cone.transpose(0,1).numpy(force=True).reshape(16,3))
		# print(np.array(dicts[i]['forces_1']).T)
		np.testing.assert_allclose(cone.transpose(0,1).numpy(force=True).reshape(16,3), 
                            np.array(dicts[i]['forces_1']).T,
                            atol=1e-7,rtol=float('inf'))
		
		# print("torqes:", i)
		# print(torques.transpose(0,1).numpy(force=True).reshape(16,3))
		# print(np.array(dicts[i]['torques_1']).T)
		np.testing.assert_allclose(torques.transpose(0,1).numpy(force=True).reshape(16,3),
                            np.array(dicts[i]['torques_1']).T,
                            atol=1e-7,rtol=float('inf'))

	for i in range(len(dicts)):
		com_qual_func = CannyFerrariQualityFunction(config_dict)
		torch_quality = com_qual_func.quality(mesh, test_grasps[i]).numpy(force=True)[0]
		print(torch_quality,
        dicts[i]['rfc_quality'])
		np.testing.assert_allclose(torch_quality,
                    np.array(dicts[i]['rfc_quality']).T,
                    atol=1e-2, rtol=.5)
    
	# load Grasp2D 
	# camera_intr = CameraIntrinsics.load("data/primesense.intr") 
	# center2d = torch.tensor([[299.4833, 182.4288]],device=device)
	# angle = torch.tensor([[0]],device=device)
	# depth = torch.tensor([[0.583332717]],device=device)
	# width = torch.tensor([[0.05]],device=device)

	center2d = torch.tensor([[344.3809509277344, 239.4164276123047]],device=device)
	angle = torch.tensor([[0.3525843322277069 + math.pi]],device=device)
	depth = torch.tensor([[0.5824159979820251]],device=device)
	width = torch.tensor([[0.05]],device=device)

	grasp1 = GraspTorch(center2d, angle, depth, width, renderer.rasterizer.cameras) 
        

	center3D = torch.tensor([[ 0.027602000162005424, 0.017583999782800674, -9.273400064557791e-05]], device=device)
	axis3D   = torch.tensor([[-0.9384999871253967, 0.2660999894142151, -0.22010000050067902]], device=device)

	grasp2 = GraspTorch(center3D, axis3D=axis3D, width=width, camera_intr=renderer.rasterizer.cameras) 
	grasp2.make2D(updateCamera=False)

	center3D = torch.tensor([[-0.03714486211538315, -0.029467197135090828, 0.01168159581720829]], device=device)
	axis3D   = torch.tensor([[-0.974246621131897, -0.19650164246559143, -0.11059238761663437]], device=device)

	grasp3 = GraspTorch(center3D, axis3D=axis3D, width=width, camera_intr=renderer.rasterizer.cameras) 
	# Call ComForceClosureParallelJawQualityFunction init with parameters from gqcnn (from gqcnn/cfg/examples/replication/dex-net_2.1.yaml 

	with record_function("FastAntipodalityFunction"):
		com_qual_func = ComForceClosureParallelJawQualityFunction(config_dict)

	# Call quality with the Grasp2D and mesh
		com_qual_func.quality(mesh, grasp3)

	with record_function("CannyFerrari"):
		com_qual_func = CannyFerrariQualityFunction(config_dict)

	# Call quality with the Grasp2D and mesh
		com_qual_func.quality(mesh, grasp3)

if __name__ == "__main__":
    #minHull.apply(torch.tensor(dict['G']).transpose(0,1).reshape((20,1,1,6)))
    np.set_printoptions(edgeitems=30, linewidth=100)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=False) as prof:
        with record_function("test_quality"):
            #model(inputs)
            test_quality()
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))
    # prof.export_chrome_trace("trace.json")