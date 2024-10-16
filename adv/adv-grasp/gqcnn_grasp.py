# -*- coding: utf-8 -*-
"""
Copied from: https://github.com/BerkeleyAutomation/gqcnn/blob/master/gqcnn/grasping/grasp.py

Copyright ©2017. The Regents of the University of California (Regents).
All Rights Reserved. Permission to use, copy, modify, and distribute this
software and its documentation for educational, research, and not-for-profit
purposes, without fee and without a signed licensing agreement, is hereby
granted, provided that the above copyright notice, this paragraph and the
following two paragraphs appear in all copies, modifications, and
distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150
Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201,
otl@berkeley.edu,
http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

Classes to encapsulate parallel-jaw grasps in image space.

Author
------
Jeff Mahler
"""
import numpy as np

# from autolab_core import Point, RigidTransform, CameraIntrinsics


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

        frame = "image"
        # if camera_intr is not None:
        #    frame = camera_intr.frame
        # if isinstance(center, np.ndarray):
        #    self.center = Point(center, frame=frame)

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

    @staticmethod
    def from_feature_vec(v, width=0.0, camera_intr=None):
        """Creates a `Grasp2D` instance from a feature vector and additional
        parameters.

        Parameters
        ----------
        v : :obj:`numpy.ndarray`
            Feature vector, see `Grasp2D.feature_vec`.
        width : float
            Grasp opening width, in meters.
        camera_intr : :obj:`autolab_core.CameraIntrinsics`
            Frame of reference for camera that the grasp corresponds to.
        """
        # Read feature vec.
        p1 = v[:2]
        p2 = v[2:4]
        depth = v[4]

        # Compute center and angle.
        center_px = (p1 + p2) // 2
        center = Point(center_px, camera_intr.frame)
        axis = p2 - p1
        if np.linalg.norm(axis) > 0:
            axis = axis / np.linalg.norm(axis)
        if axis[1] > 0:
            angle = np.arccos(axis[0])
        else:
            angle = -np.arccos(axis[0])
        return Grasp2D(center,
                       angle,
                       depth,
                       width=width,
                       camera_intr=camera_intr)

    def pose(self, grasp_approach_dir=None):
        """Computes the 3D pose of the grasp relative to the camera.

        If an approach direction is not specified then the camera
        optical axis is used.

        Parameters
        ----------
        grasp_approach_dir : :obj:`numpy.ndarray`
            Approach direction for the grasp in camera basis (e.g. opposite to
            table normal).

        Returns
        -------
        :obj:`autolab_core.RigidTransform`
            The transformation from the grasp to the camera frame of reference.
        """
        # Check intrinsics.
        if self.camera_intr is None:
            raise ValueError(
                "Must specify camera intrinsics to compute 3D grasp pose")

        # Compute 3D grasp center in camera basis.
        grasp_center_im = self.center.data
        center_px_im = Point(grasp_center_im, frame=self.camera_intr.frame)
        grasp_center_camera = self.camera_intr.deproject_pixel(
            self.depth, center_px_im)
        grasp_center_camera = grasp_center_camera.data

        # Compute 3D grasp axis in camera basis.
        grasp_axis_im = self.axis
        grasp_axis_im = grasp_axis_im / np.linalg.norm(grasp_axis_im)
        grasp_axis_camera = np.array([grasp_axis_im[0], grasp_axis_im[1], 0])
        grasp_axis_camera = grasp_axis_camera / np.linalg.norm(
            grasp_axis_camera)

        # Convert to 3D pose.
        grasp_rot_camera, _, _ = np.linalg.svd(grasp_axis_camera.reshape(3, 1))
        grasp_x_camera = grasp_approach_dir
        if grasp_approach_dir is None:
            grasp_x_camera = np.array([0, 0, 1])  # Align with camera Z axis.
        grasp_y_camera = grasp_axis_camera
        grasp_z_camera = np.cross(grasp_x_camera, grasp_y_camera)
        grasp_z_camera = grasp_z_camera / np.linalg.norm(grasp_z_camera)
        grasp_y_camera = np.cross(grasp_z_camera, grasp_x_camera)
        grasp_rot_camera = np.array(
            [grasp_x_camera, grasp_y_camera, grasp_z_camera]).T
        if np.linalg.det(grasp_rot_camera) < 0:  # Fix reflections due to SVD.
            grasp_rot_camera[:, 0] = -grasp_rot_camera[:, 0]
        T_grasp_camera = RigidTransform(rotation=grasp_rot_camera,
                                        translation=grasp_center_camera,
                                        from_frame="grasp",
                                        to_frame=self.camera_intr.frame)
        return T_grasp_camera

    @staticmethod
    def image_dist(g1, g2, alpha=1.0):
        """Computes the distance between grasps in image space.

        Uses Euclidean distance with alpha weighting of angles

        Parameters
        ----------
        g1 : :obj:`Grasp2D`
            First grasp.
        g2 : :obj:`Grasp2D`
            Second grasp.
        alpha : float
            Weight of angle distance (rad to meters).

        Returns
        -------
        float
            Distance between grasps.
        """
        # Point to point distances.
        point_dist = np.linalg.norm(g1.center.data - g2.center.data)

        # Axis distances.
        dot = max(min(np.abs(g1.axis.dot(g2.axis)), 1.0), -1.0)
        axis_dist = np.arccos(dot)
        return point_dist + alpha * axis_dist


