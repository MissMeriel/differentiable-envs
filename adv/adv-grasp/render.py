import torch
import numpy as np
import matplotlib.pyplot as plt

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
	look_at_view_transform,
	FoVPerspectiveCameras,
	MeshRenderer,
	MeshRasterizer,
	SoftPhongShader,
	RasterizationSettings,
	PointLights,
	TexturesVertex
)

class Renderer:

	def __init__(self, renderer=None, rasterizer=None, raster_settings=None, camera=None, lights=None):

		# set PyTorch device, use cuda if available
		if torch.cuda.is_available():
			self.device = torch.device("cuda:0")
			torch.cuda.set_device(self.device)
		else:
			print("cuda not available")
			self.device = torch.device("cpu")

		if lights:
			self.lights = lights
		else:
			self.lights = PointLights(device=self.device, location=[[0.0, 0.0, -3.0]])
		
		if camera:
			self.camera = camera
		else:
			R, T = look_at_view_transform(dist=0.7, elev=90, azim=180)
			self.camera = FoVPerspectiveCameras(device=self.device, R=R, T=T)

		if raster_settings:
			self.raster_settings = raster_settings
		else:
			self.raster_settings = RasterizationSettings(
				image_size=512,
				blur_radius=0.0,
				faces_per_pixel=1
			)

		if rasterizer:
			self.rasterizer = rasterizer
		else:
			self.rasterizer = MeshRasterizer(
				cameras = self.camera,
				raster_settings = self.raster_settings
			)

		if renderer:	
			self.renderer = renderer
		else:
			self.renderer = MeshRenderer(
				rasterizer = self.rasterizer,
				shader = SoftPhongShader(
					device = self.device,
					cameras = self.camera,
					lights = self.lights
				)
			)	

	def render_object(self, obj_file, display=True, title=None):
		"""
		Render mesh object and optionally display
		Parameters
		----------
		obj_file: String
			Path to a .obj file to be rendered
		display: Boolean
			If True, displays image of rendered object via matplotlib	
		title: String
			If display is True and title is not None, tile of plot is title
		Returns
		-------
		pytorch3d.structures.Meshes, numpy.ndarray
			PyTorch3D mesh of the object and associated numpy array	
		"""

		verts, faces_idx, _ = load_obj(obj_file)
		faces = faces_idx.verts_idx

		verts_rgb = torch.ones_like(verts)[None]
		textures = TexturesVertex(verts_features=verts_rgb.to(self.device))

		mesh = Meshes(
			verts=[verts.to(self.device)],
			faces=[faces.to(self.device)],
			textures=textures
		)

		image = self.render_mesh(mesh, display=display, title=title)
		dis_image = image[0, ..., :3].cpu().detach().numpy()

		return mesh, dis_image

	def render_mesh(self, mesh, display=True, title=None):
		"""
		Renders a Mesh object and optionally displays.
		Parameters
		----------
		mesh: pytorch3d.structures.meshes.Mesh
			Mesh object to be rendered
		display: Boolean
			If True, the rendered mesh is displayed
		title: String
			If display is True and title is not None, title is set as title of image.
		Returns
		-------
		torch.tensor
			Tensor of rendered Mesh
		"""

		if not isinstance(mesh, Meshes):
			print("render_mesh input given not a mesh.")
			return None
		
		ret_tens = self.renderer(mesh, cameras=self.camera, lights=self.lights)		
		if display:
			self.display(ret_tens[0, ..., :3].cpu().detach().numpy(), title)

		return ret_tens 

	def display(self, obj, title=None):
		"""
		Displays a mesh or image, optionally with a title.
		Parameters
		----------
		obj: pytorch3d.structures.meshes.Meshes or numpy.ndarray
			Mesh object or numpy array of object to be displayed
		title: String
			If not none, title of matplotlib figure
		Returns
		-------
		None
		"""

		if isinstance(obj, Meshes):
			image = self.render_mesh(obj, display=False)	
			image = image[0, ..., :3].cpu().detach().numpy()
		elif isinstance(obj, np.ndarray):
			image = obj
		else:
			print("display_im only takes Meshes object or numpy array")
			return None

		plt.figure(figsize=(6,6))
		plt.imshow(image)
		plt.axis("off")
		if title:
			plt.title(title)
		plt.show()

	def mesh_to_depth_im(self, mesh, display=True, title=None):
		"""
		Converts a Mesh to a noramlized 512 x 512 numpy depth image (values between 0 and 1) and optionally displays it.
		Parameters
		----------
		mesh: pytorch.structures.meshes.Meshes
			Mesh object to be converted to depth image
		display: Boolean
			If True, displays converted depth image with matplotlib
		title: String
			If display is True and title is not None, title of the matplotlib image
		Returns
		-------
		numpy.ndarray
			512 x 512 numpy array representing normalized depth image (values between 0 and 1 inclusive)
		"""	

		if not isinstance(mesh, Meshes):
			print("mesh_to_depth_im: input not a mesh.")
			return None
	
		depth_im = self.rasterizer(mesh).zbuf.squeeze(-1).squeeze(0)
		depth_im = depth_im.cpu().detach().numpy()

		# normalize
		depth_im = (depth_im-np.min(depth_im))/(np.max(depth_im)-np.min(depth_im))

		if display:
			self.display(depth_im, title=title)

		return depth_im

def test_renderer():
	# instantiate Renderer object with default parameters
	renderer1 = Renderer()

	# test render_obj -> render_mesh -> display mesh
	mesh, image = renderer1.render_object("data/bar_clamp.obj", display=True, title="testing render_obj -> render_mesh")

	# test mesh_to_depth_im -> display np.array
	depth_im = renderer1.mesh_to_depth_im(mesh, display=True, title="testing mesh_to_depth_im")

	return "success"

if __name__ == "__main__":
	print(test_renderer())



