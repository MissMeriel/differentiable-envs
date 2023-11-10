import torch
import numpy as np
from pytorch3d.io import load_obj
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
        return np.array([np.cos(self.angle), np.sin(self.angle)])

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
        p1 = Point(np.array([0, 0, self.depth]), frame=self.frame)
        p2 = Point(np.array([self.width, 0, self.depth]), frame=self.frame)

        # Project into pixel space.
        u1 = self.camera_intr.project(p1)
        u2 = self.camera_intr.project(p2)
        return np.linalg.norm(u1.data - u2.data)

    @property
    def endpoints(self):
        """Returns the grasp endpoints."""
        p1 = self.center.data - (self.width_px / 2) * self.axis
        p2 = self.center.data + (self.width_px / 2) * self.axis
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
    """Abstract wrapper class for parallel jaw quality functions (only image
    based metrics for now)."""

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
        dot_prod1 = min(max(action.contact_normals[0].dot(-action.axis), -1.0),
                        1.0)
        angle1 = np.arccos(dot_prod1)
        dot_prod2 = min(max(action.contact_normals[1].dot(action.axis), -1.0),
                        1.0)
        angle2 = np.arccos(dot_prod2)
        return max(angle1, angle2)

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
        totalCoords = torch.sum(mesh_unwrapped, 0)
        meanVert = torch.sum(totalCoords,0) / (totalCoords.shape[0] * totalCoords.shape[1])

        totalCoords = totalCoords + meanVert
        com_per_triangle = totalCoords / 4

        meanVert_expand = meanVert.expand([totalCoords.shape[0], 1, 1])

        mesh_tetra = torch.cat([mesh_unwrapped, meanVert_expand], 1)
        # det([[x1,y1,z1,1],[x2,y2,z2,1],[x3,y3,z3,1],[x4,y4,z4,1]]) / 6 
        # does det on last 2 dims, considers at least first 1 to be batch dim
        vol_per_triangle = torch.linalg.det(mesh_tetra)

        com = com_per_triangle * vol_per_triangle / torch.sum(vol_per_triangle)

        return com
        


    def solveForIntersection(self, state, grasp): 
        """Compute where grasp contacts a mesh, state is meshes, action is grasp2d"""
        # TODO
        # for grasp in grasp 
        # for each ray (2)
        # for each triangle (vectorize for parallel)
        #  compute intersection
        #https://en.m.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
        mesh_unwrapped = multi_gather_tris(mesh.verts_packed(), mesh.faces_packed())
        
        finger1_o = grasp.center - grasp.axis * grasp.jaw_width / 2
        ray_o = grasp.center + (torch.cat([-grasp.axis, grasp.axis], 0) * grasp.jaw_width / 2)
        ray_d = torch.cat([grasp.axis, -grasp.axis], 0)
        # ray is n, 3
        u, v, t = moller_trumbore(ray_o, ray_d, mesh_unwrapped, eps=1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # correct dir, not too far, actually hits triangle
        inside1 = ((t >= 0.0) * (t < grasp.jaw_width/2) * (u >= 0.0) * (v >= 0.0) * ((u + v) <= 1.0)).bool()  # (n_rays, n_faces)
        t(torch.not(inside1)) = float('Inf')
        # (n_rays, n_faces)
        min_out = min(t, 1)

        intersectionPoints = ray_o + ray_d * min_out.values
        faces_index = min_out.indices
        contactFound = torch.not(torch.or(torch.isinf(t)))


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
        """Create a best-fit planarity suction metric."""
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
        actions = ParallelJawQualityFunction.solveForIntersection(self, state, actions)

        # Compute antipodality.
        antipodality_q = [
            ParallelJawQualityFunction.friction_cone_angle(self, action)
            for action in actions
        ]

        # Compute object center of mass.
        object_com = ParallelJawQualityFunction.compute_mesh_COM(self, state)

        # Compute negative SSE from the best fit plane for each grasp.
        antipodality_thresh = abs(
            np.percentile(antipodality_q, 100 - self._antipodality_pctile))
        qualities = []
        max_q = max(state.rgbd_im.height, state.rgbd_im.width)
        for i, action in enumerate(actions):
            q = max_q
            friction_cone_angle = antipodality_q[i]
            force_closure = ParallelJawQualityFunction.force_closure(
                self, action)
            if force_closure or friction_cone_angle < antipodality_thresh:
                grasp_center = np.array([action.center.y, action.center.x])


                q = np.linalg.norm(grasp_center - object_com)


            q = (np.exp(-q / max_q) - np.exp(-1)) / (1 - np.exp(-1))
            qualities.append(q)

        return np.array(qualities)

// taken from: https://gist.github.com/dendenxu/ee5008acb5607195582e7983a384e644#file-moller_trumbore_winding_number_inside_mesh-py-L318

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
    u, v, t = moller_trumbore(pts, ray_d, multi_gather_tris(verts, faces))  # (n_rays, n_faces, 3)
    inside = ((t >= 0.0) * (u >= 0.0) * (v >= 0.0) * ((u + v) <= 1.0)).bool()  # (n_rays, n_faces)
    inside = (inside.count_nonzero(dim=-1) % 2).bool()  # if mod 2 is 0, even, outside, inside is odd
    inside = inside.view(multiplier, n_rays, -1)
    inside = inside.sum(dim=0) / multiplier  # any show inside mesh
    return inside


def moller_trumbore(ray_o: torch.Tensor, ray_d: torch.Tensor, tris: torch.Tensor, eps=1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


// Our code

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
	grasp = Grasp2D(np.array([299.4833, 182.4288]), 0, 0.583332717, 0.05, 0) 

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

