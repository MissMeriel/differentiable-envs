import torch
import numpy as np
from pytorch3d.io import load_obj
import pytorch3d.transforms as tf
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


# Grasp2D class copied from: https://github.com/BerkeleyAutomation/gqcnn/blob/master/gqcnn/grasping/grasp.py
class Grasp2D(object):
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
                 contact_normals=None):
        self.center = center
        self.angle = angle
        self.depth = depth
        self.width = width

        # If `camera_intr` is none use default primesense camera intrinsics.
        """
	# don't have CameraIntrinsics import, but can get info from PyTorch camera
        if not camera_intr:
            self.camera_intr = CameraIntrinsics("primesense_overhead",
                                                fx=525,
                                                fy=525,
                                                cx=319.5,
                                                cy=239.5,
                                                width=640,
                                                height=480)
        else:
            self.camera_intr = camera_intr
	"""
        self.camera_intr = camera_intr
        

        self.contact_points = contact_points
        self.contact_normals = contact_normals

        """
        frame = "image"
        if camera_intr is not None:
            frame = camera_intr.frame
        if isinstance(center, np.ndarray):
            self.center = Point(center, frame=frame)
        """

    @property
    def axis(self):
        """Returns the grasp axis."""
        return torch.cat((torch.cos(self.angle), torch.sin(self.angle)), dim=1)
    
    @property
    def axis3D(self):
        """Returns the grasp axis."""
        axis_in_camera = torch.cat((self.axis, torch.zeros([self.axis.shape[0],1],device=self.axis.device)), dim=1)
        return self.camera_intr.get_world_to_view_transform().inverse().transform_normals(axis_in_camera)
    
    @property
    def center3D(self):
        """Returns the grasp axis."""
        center_in_camera = torch.cat((self.center, self.depth), dim=1)
        return self.camera_intr.unproject_points(center_in_camera, world_coordinates=True)
    
    @property
    def tform_to_camera(self):
        """Returns the grasp axis."""
        return self.camera_intr.get_full_projection_transform()

    @property
    def approach_axis(self):
        return np.array([0, 0, 1])

    @property
    def approach_angle(self):
        """The angle between the grasp approach axis and camera optical axis.
        """
        return 0.0

    @property
    def frame(self):
        """The name of the frame of reference for the grasp."""
        if self.camera_intr is None:

            raise ValueError("Must specify camera intrinsics")
        return self.camera_intr.frame

    @property
    def width_px(self):
        """Returns the width in pixels."""
        if self.camera_intr is None:
            missing_camera_intr_msg = ("Must specify camera intrinsics to"
                                       " compute gripper width in 3D space.")
            raise ValueError(missing_camera_intr_msg)
        # Form the jaw locations in 3D space at the given depth.
        p1 =torch.cat((torch.zeros([self.depth.shape[0],2],device=self.depth.device), self.depth), dim=1)
        p2 =torch.cat((self.depth, torch.zeros([self.depth.shape[0],1],device=self.depth.device), self.depth), dim=1)

        # Project into pixel space.
        u1 = self.camera_intr.transform_points(p1)
        u2 = self.camera_intr.transform_points(p2)
        return torch.norm(u1 - u2,dim=-1)

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

        # Read parameters.
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
        axisBatched = torch.unsqueeze(action.axis3D,0)
        opposite_dir_rays = torch.cat([-axisBatched, axisBatched], 0)
        dot_prod = torch.sum(torch.mul(action.contact_normals,opposite_dir_rays),-1) 
        # not sure if necessary, should already be bounded -1 to 1. 
        dot_prod = torch.minimum(torch.maximum(dot_prod, torch.tensor(-1.0)), torch.tensor(1.0))
        angle = torch.arccos(dot_prod)
        max_angle = torch.max(angle,0).values

        return max_angle

    def force_closure(self, action):
        """Determine if the grasp is in force closure."""
        return (self.friction_cone_angle(action) <
                self._max_friction_cone_angle)


    def compute_mesh_COM(mesh): 
        """Compute the center of mass for a mesh, assume uniform density"""
        # TODO combine each triangle with innerPoint in loop
        # to form pyramids (tetrahedrons)
        #https://forums.cgsociety.org/t/how-to-calculate-center-of-mass-for-triangular-mesh/1309966
        mesh_unwrapped = multi_gather_tris(mesh.verts_packed(), mesh.faces_packed())
        # B, F, 3, 3
        # assume Faces, verts, coords
        totalCoords = torch.sum(mesh_unwrapped, 1)
        meanVert = torch.sum(totalCoords,0) / (totalCoords.shape[0] * totalCoords.shape[1])

        totalCoords = totalCoords + meanVert
        com_per_triangle = totalCoords / 4

        meanVert_expand = torch.reshape(meanVert,[1,1, 3])
        meanVert_expand = meanVert_expand.expand(mesh_unwrapped.shape[0],1,3)

        mesh_tetra = torch.cat([mesh_unwrapped, meanVert_expand], 1)
        mesh_tetra = torch.cat([mesh_tetra, torch.ones([mesh_tetra.shape[0],4,1],device=mesh_unwrapped.device)], -1)
        # det([[x1,y1,z1,1],[x2,y2,z2,1],[x3,y3,z3,1],[x4,y4,z4,1]]) / 6 
        # does det on last 2 dims, considers at least first 1 to be batch dim
        vol_per_triangle = torch.reshape(torch.linalg.det(mesh_tetra),(mesh_tetra.shape[0],1))

        com = torch.sum(com_per_triangle * vol_per_triangle,dim=0) / torch.sum(vol_per_triangle)

        return com
        


    def solveForIntersection(self, state, actions): 
        """Compute where grasp contacts a mesh, state is meshes, action is grasp2d"""
        # TODO
        # for grasp in grasp 
        # for each ray (2)
        # for each triangle (vectorize for parallel)
        #  compute intersection
        #https://en.m.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
        mesh_unwrapped = multi_gather_tris(state.verts_packed(), state.faces_packed())
        axisBatched = torch.unsqueeze(actions.axis3D,0)
        opposite_dir_rays = torch.cat([-axisBatched, axisBatched], 0)
        ray_o = actions.center3D + (opposite_dir_rays * actions.width / 2)
        ray_d = -opposite_dir_rays
        # ray is n, 3
        
        target_shape = ray_o.shape
        # moller_trumbore assumes flat list of rays (2d Nx3)
        ray_o_flat = torch.flatten(ray_o, end_dim=-2)
        ray_d_flat = torch.flatten(ray_d, end_dim=-2)

        u, v, t = moller_trumbore(ray_o_flat, ray_d_flat, mesh_unwrapped, eps=1e-8)

        u = torch.unflatten(u, 0, target_shape[:-1])
        v = torch.unflatten(v, 0, target_shape[:-1])
        t = torch.unflatten(t, 0, target_shape[:-1])
        # correct dir, not too far, actually hits triangle
        inside1 = ((t >= 0.0) * (t < actions.width/2) * (u >= 0.0) * (v >= 0.0) * ((u + v) <= 1.0)).bool()  # (n_rays, n_faces)
        t[torch.logical_not(inside1)] = float('Inf')
        # (n_rays, n_faces)
        min_out = torch.min(t, -1,keepdim=True)

        intersectionPoints = ray_o + ray_d * min_out.values
        faces_index = min_out.indices
        contactFound = torch.logical_not(torch.isinf(min_out.values))
        actions.contact_points = intersectionPoints
        actions.contact_normals = torch.squeeze(state.faces_normals_packed()[faces_index,:],-2)
        return actions, faces_index, contactFound


        # find intersection
        # 1. Inside triangle
        # 2. In direction for ray
        # 3. Closest to start point (gripper closes until contact)
        # that intersection is contact_point
        # angle between ray and normal is contact_normal

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
        actions, faces_index, contactFound = ParallelJawQualityFunction.solveForIntersection(self, state, actions)

        # Compute antipodality.
        antipodality_q = ParallelJawQualityFunction.friction_cone_angle(self, actions)


        # Compute object center of mass.
        object_com = ParallelJawQualityFunction.compute_mesh_COM(state)

        # Can rank grasps, instead of compute absolute score. Only makes sense if seeding many grasps
        # antipodality_thresh = abs(
        #     np.percentile(antipodality_q, 100 - self._antipodality_pctile))
        
        max_q = torch.norm(torch.diff(state.get_bounding_boxes(),1,2))
        quality = torch.ones(actions.depth.shape,device=actions.depth.device) * max_q
        
        in_force_closure = ParallelJawQualityFunction.force_closure(self, actions)
        
        dist = torch.norm(actions.center3D - object_com,dim=-1)
        quality[in_force_closure] = dist[in_force_closure]
        # some kind of damped sigmoid, not sure where it comes from
        e_inverse = torch.exp(torch.tensor(-1))
        quality = (torch.exp(-quality / max_q) - e_inverse) / (1 - e_inverse)

        return quality
# taken from: https://gist.github.com/dendenxu/ee5008acb5607195582e7983a384e644#file-moller_trumbore_winding_number_inside_mesh-py-L318

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

	# camera with info from gqcnn primesense
        R, T = look_at_view_transform(dist=0.6, elev=90, azim=0)        # camera located above object, pointing down
        fl = torch.tensor([[525.0]])
        pp = torch.tensor([[319.5, 239.5]])
        im_size = torch.tensor([[480, 640]])

        camera = PerspectiveCameras(focal_length=fl, principal_point=pp, in_ndc=False, image_size=im_size, device=device, R=R, T=T)[0]

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

	verts, faces_idx, _ = load_obj("data/bar_clamp.obj")
	faces = faces_idx.verts_idx
	verts_rgb = torch.ones_like(verts)[None]
	textures = TexturesVertex(verts_features=verts_rgb.to(device))

	mesh = Meshes(
		verts=[verts.to(device)],
		faces=[faces.to(device)],
		textures=textures
	)


	# load Grasp2D 
	# camera_intr = CameraIntrinsics.load("data/primesense.intr") 
	# center2d = torch.tensor([[299.4833, 182.4288]],device=device)
	# angle = torch.tensor([[0]],device=device)
	# depth = torch.tensor([[0.583332717]],device=device)
	# width = torch.tensor([[0.05]],device=device)

	center2d = torch.tensor([[344.3809509277344, 239.4164276123047]],device=device)
	angle = torch.tensor([[0.3525843322277069]],device=device)
	depth = torch.tensor([[0.5824159979820251]],device=device)
	width = torch.tensor([[0.05]],device=device)

	grasp = Grasp2D(center2d, angle, depth, width, renderer.rasterizer.cameras) 

	# Call ComForceClosureParallelJawQualityFunction init with parameters from gqcnn (from gqcnn/cfg/examples/replication/dex-net_2.1.yaml 
	config_dict = {
		"friction_coef": 0.8,
		"antipodality_pctile": 1.0 
	}
	
	com_qual_func = ComForceClosureParallelJawQualityFunction(config_dict)

	# Call quality with the Grasp2D and mesh
	com_qual_func.quality(mesh, grasp)

if __name__ == "__main__":
	test_quality()

