import torch
import numpy as np
import matplotlib.pyplot as plt

from pytorch3d.utils import ico_sphere
from pytorch3d.io import load_obj, save_obj
from pytorch3d.transforms import Translate
from pytorch3d.structures import Meshes, join_meshes_as_scene
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
			self.lights = PointLights(device=self.device, location=[[0.0, 0.0, 3.0]])
		
		if camera:
			self.camera = camera
		else:
			# R, T = look_at_view_transform(dist=0.7, elev=90, azim=180)
			R, T = look_at_view_transform(dist=0.6, elev=90, azim=0)	# camera located above object, pointing down

			# camera intrinsics
			fl = torch.tensor([[525.0]])
			pp = torch.tensor([[319.5, 239.5]]) 
			im_size = torch.tensor([[480, 640]])
 
			self.camera = PerspectiveCameras(focal_length=fl, principal_point=pp, in_ndc=False, image_size=im_size, device=self.device, R=R, T=T)[0]

		if raster_settings:
			self.raster_settings = raster_settings
		else:
			self.raster_settings = RasterizationSettings(
				image_size=(480, 640), 	# image size (H, W) in pixels 
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
		elif isinstance(obj, torch.Tensor):
			image = obj.squeeze().cpu().detach().numpy()
		elif isinstance(obj, np.ndarray):
			image = obj
		else:
			print("display_im only takes Meshes object or numpy array")
			return None

		plt.figure(figsize=(8,8))
		plt.imshow(image)
		plt.axis("off")
		if title:
			plt.title(title)
		plt.show()

	def mesh_to_depth_im(self, mesh, display=True, title=None):
		"""
		Converts a Mesh to a noramlized 480 x 640 numpy depth image and optionally displays it.
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
			480 x 640 numpy array representing normalized depth image (values between 0 and 1 inclusive)
		"""	

		if not isinstance(mesh, Meshes):
			print("mesh_to_depth_im: input not a mesh.")
			return None
	
		depth_im = self.rasterizer(mesh).zbuf.squeeze(-1).squeeze(0)
		depth_im = depth_im.cpu().detach().numpy()

		# normalize
		max_depth = np.max(depth_im)	
		depth_im[depth_im == -1] = max_depth 	
 
		if display:
			self.display(depth_im, title=title)

		return depth_im

	def grasp_sphere(self, center, grasp_obj, fname):
		"""Generate an ico_sphere mesh to visualize a particular grasp
		Parameters
		----------
		center: torch.Tensor of size 3 or tuple of two torch.Tensors of size 3
			3D coordinates of center of the grasp to display or the two contact points of the grasp
		grasp_obj: torch.structures.meshes.Meshes
			Mesh object being grasped
		fname: String
			filepath to save the visualized grasp object
		Returns
		-------
		torch.structures.meshes.Meshes
			mesh including both the grasping object and a sphere visualizing the grasp
		"""

		# instantiate sphere
		grasp_sphere = ico_sphere(4, self.device)
		vertex_colors = torch.full(grasp_sphere.verts_packed().shape, 0.5) 
		vertex_colors = vertex_colors.to(device=torch.device(self.device))
		grasp_sphere.textures = TexturesVertex(verts_features=vertex_colors.unsqueeze(0))

		# translate sphere(s) to match grasp and join grasping object with sphere(s)
		if isinstance(center, torch.Tensor):
			grasp_sphere = grasp_sphere.scale_verts(0.025)
			grasp_sphere.offset_verts_(center)
			mesh = join_meshes_as_scene([grasp_obj, grasp_sphere])
		else:
			grasp_sphere = grasp_sphere.scale_verts(0.010)
			c0 = grasp_sphere
			c1 = c0.clone()
			c0.offset_verts_(center[0])
			c1.offset_verts_(center[1])
			mesh = join_meshes_as_scene([grasp_obj, c0, c1])
		

		save_obj(fname, verts=mesh.verts_list()[0], faces=mesh.faces_list()[0])
		
		return mesh

def test_renderer():
	# instantiate Renderer object with default parameters
	renderer1 = Renderer()

	# test render_obj -> render_mesh -> display mesh
	mesh, image = renderer1.render_object("data/bar_clamp.obj", display=False, title="testing render_obj -> render_mesh")

	# test mesh_to_depth_im -> display np.array
	depth_im = renderer1.mesh_to_depth_im(mesh, display=True, title="testing mesh_to_depth_im")
	print("depth im values:", np.max(depth_im), np.min(depth_im))
	print(depth_im)

	depth_ex = np.load("data/depth_0.npy")
	print("\ngqcnn depth im:", np.max(depth_ex), np.min(depth_ex))
	print(depth_ex)
	renderer1.display(depth_ex, title="gqcnn depth_0")

	return "success"

if __name__ == "__main__":
	# print(test_renderer())

	renderer1 = Renderer()
	grasp_obj, _ = renderer1.render_object("data/bar_clamp.obj", display=False)
	# center = torch.tensor([-0.01691197, -0.02238275, 0.04196089]).to(renderer1.device)
	center = torch.tensor([0.04, 0.04, 0.04]).to(renderer1.device)
	sphere = renderer1.grasp_sphere(center, grasp_obj, "vis_grasps/test.obj")




