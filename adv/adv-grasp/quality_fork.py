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

    def solveForIntersection(self, state, action): 
        """Compute where grasp contacts a mesh"""
        # TODO
        # for action in actions 
        # for each ray (2)
        # for each triangle (vectorize for parallel)
        #  compute intersection
        #https://en.m.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
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
        action: :obj:`Grasp2D`
            A suction grasp in image space that encapsulates center, approach
            direction, depth, camera_intr.
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
	grasp = Grasp2D(np.array([416, 286]), -2.896613990462929, 0.607433762324266, 0.05, 0) 

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

