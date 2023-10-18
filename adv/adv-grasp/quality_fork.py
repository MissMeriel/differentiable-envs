# imports for gqcnn/autolab functions
from autolab_core import Point
from perception import CameraIntrinsics
from gqcnn.grasping import Grasp2D

# imports from our own code
from render import *

class GraspQualityFunction():	#ABC):
    """Abstract grasp quality class."""

    def __init__(self):
        # Set up logger.
        self._logger = Logger.get_logger(self.__class__.__name__)

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


def test_quality():
	# load PyTorch3D mesh from .obj file
	renderer1 = Renderer()
	mesh, image = renderer1.render_object("data/bar_clamp.obj", display=False)

	# load Grasp2D 
	camera_intr = CameraIntrinsics.load("data/primesense.intr") 
	grasp = Grasp2D(Point(np.array([416, 286])), -2.896613990462929, 0.607433762324266, 0.05, camera_intr) 

	# Call ComForceClosureParallelJawQualityFunction init with parameters from gqcnn
	config = YamlConfig("data/dexnet-21-cfg.yaml") 		# file from gqcnn/cfg/examples/replication/dex-net_2.1.yaml	
	com_qual_func = ComForceClosureParallelQualityFunction(config["policy"]["metric"])

	# Call quality with the Grasp2D and mesh
	com_qual_func.quality(mesh, grasp)

if __name__ == "__main__":
	test_quality()

