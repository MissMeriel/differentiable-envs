import logging
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import trimesh
from PIL import Image

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

	# SET UP LOGGING
	logger = logging.getLogger('render')
	logger.setLevel(logging.DEBUG)
	if not logger.handlers:
		ch = logging.StreamHandler()
		ch.setLevel(logging.DEBUG)
		formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		ch.setFormatter(formatter)
		logger.addHandler(ch)

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
			# R, T = look_at_view_transform(dist=0.6, elev=90, azim=0)	# camera located above object, pointing down
			eye = torch.tensor([[0.0, 0.6, 0.0]])	# 
			up = torch.tensor([[0.0, 0.0, 1.0]])
			at = torch.tensor([[0.0, 0.0, 0.0]])
			R, T = look_at_view_transform(eye=eye, up=up, at=at)	# camera located above object, pointing down

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

		# check that vertices are unique, and fix if not
		unique_vals, inverse_indices = torch.unique(verts, sorted=False, return_inverse=True, return_counts=False, dim=0)
		if (len(unique_vals) != len(verts)):
			faces_flat = faces_idx.verts_idx.flatten()
			new_faces_flat = torch.index_select(inverse_indices, 0, faces_flat)
			faces = new_faces_flat.reshape(faces_idx.verts_idx.shape)
			verts = unique_vals

		verts_rgb = torch.ones_like(verts)[None]
		textures = TexturesVertex(verts_features=verts_rgb.to(self.device))

		mesh = Meshes(
			verts=[verts.to(self.device)],
			faces=[faces.to(self.device)],
			textures=textures
		)

		image = self.render_mesh(mesh, display=display, title=title)
		dis_image = image[0, ..., :3].cpu().detach().numpy()

		return mesh, image

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
			self.display(ret_tens[0, ..., :3].cpu().detach().numpy(), title=title)

		return ret_tens 

	def display(self, images, shape=None, title=None, save="", crop=False):
		"""
		Display multiple images in one figure
		Parameters
		----------
		images: List of 2D/3D Torch.tensors or np.ndarrays or one 4D Torch.tensor
			List of images to display
		shape: Tuple of ints
			Shape to plot images, ex: (2, 3) indicates two rows of three images, default is in one row
		title: List of Strings
			Titles of images to display on plot
		save: String
			If not empty, save the figure to file `save`
			If empty, do not save
		"""

		# check input images
		if isinstance(images, torch.Tensor):
			if images.dim() == 4:	# batch of images
				if images.shape[3] == 4:	# RGB-D images, cut to 3 pixel channels
					images = images[..., :3]
				images = images.cpu().detach().numpy()
				images = np.split(images, images.shape[0])
				images = [np.squeeze(image, axis=0) for image in images]
			else:
				images = [images]
		elif isinstance(images, np.ndarray):
			if len(images.shape) == 4:
				print("check1")
				images = np.split(images, images.shape[0])
				images = [np.squeeze(image, axis=0) for image in images]
			else:
				images = [images]
		elif isinstance(images, Meshes):
			images = images
		elif not isinstance(images, list):
			Renderer.logger.error("display - only takes List, torch.Tensor, np.ndarray, and Meshes objects")
			return None


		# format rows and columns
		num_ims = len(images)
		if isinstance(shape, tuple) and (shape[0] * shape[1] >= num_ims):
			rows, cols = shape[0], shape[1]
		else:
			cols = round(math.sqrt(num_ims))
			rows = math.ceil(num_ims / cols)
		fig = plt.figure(figsize=(8*cols, 8*rows))

		# check titles
		if isinstance(title, str):
			fig.suptitle(title, fontsize=15)
		elif title and (not isinstance(title, list) or len(title) != num_ims):
			title = None

		# plot images
		for i in range(num_ims):
			image = images[i]

			# check type of image
			if isinstance(image, Meshes):
				image = self.render_mesh(image, display=False)	
				image = image[0, ..., :3].cpu().detach().numpy()
			elif isinstance(image, torch.Tensor):
				if image.dim() == 4:
					image = image.squeeze(0)
				if image.shape[-1] == 4:	# RGB-D
					image = image[..., :3]
				if image.shape[0] == 1:
					image = image.squeeze(0)
				image = image.cpu().detach().numpy()
			elif isinstance(image, np.ndarray):
				if image.shape[0] == 1:
					print("check2")
					image = np.squeeze(image, axis=0)
			elif image != "":
				Renderer.logger.error("display - List elements of 'images' must be Torch.tensors, np.ndarrays, and/or Meshes objects")
				return None
			
			if crop:
				x, y = image.shape[0], image.shape[1]
				if x != y:
					if x > y:
						x_start = (x - y) // 2
						if len(image.shape) == 3: image = image[x_start:x_start+y, :, :]
						else: image = image[x_start:x_start+y, :]
					else:
						y_start = (y - x) // 2
						if len(image.shape) == 3: image = image[:, y_start:y_start+x, :]
						else: image = image[:, y_start:y_start+x]

			# # if image != "":
			# if len(image.shape) == 2:
			# 	ax = fig.add_subplot(rows, cols, i+1)
			# 	ax.axis('off')
			# 	plot = ax.imshow(image, cmap="Spectral")
			# 	fig.colorbar(plot)

			# else:
			fig.add_subplot(rows, cols, i+1)
			plt.axis('off')
			plt.imshow(image)

			if isinstance(title, list) and isinstance(title[i], str):
				plt.title(title[i])

		if save != "":
			plt.savefig(save)
		else:
			plt.show()

		plt.close()

	def draw_grasp(self, obj, contact0, contact1, title=None, save="", display=True):

		if isinstance(obj, Meshes):
			image = self.render_mesh(obj, display=False)	
			image = image[0, ..., :3].cpu().detach().numpy()
		elif isinstance(obj, torch.Tensor):
			image = obj.squeeze().cpu().detach().numpy()
		elif isinstance(obj, np.ndarray):
			image = obj
		else:
			print("display_im only takes Meshes object, pytorch tensor, or numpy array")
			return None

		# calculate grasp line from 3D contact points
		if contact0.dim() == 1 and contact1.dim() == 1:
			contacts = torch.stack((contact0, contact1))
		else:
			contacts = torch.cat((contact0, contact1), 0)

		contacts = -1 * contacts	# multiply by -1 bc matplotlib has opposite 2D coordinate system
		im_contacts = self.camera.transform_points(contacts)
		for i in range(im_contacts.shape[0]):		# fix depth value
			im_contacts[i][2] = 1/im_contacts[i][2]
		im_contacts = im_contacts[..., :2]

		# plot image
		fig = plt.figure()
		fig.add_subplot(111)
		plt.imshow(image)
		plt.axis("off")
		if title:
			plt.title(title)

		# add grasp line
		endpoint0 = im_contacts[0].cpu().detach().numpy()
		endpoint1 = im_contacts[1].cpu().detach().numpy()
		plt.plot([endpoint0[0], endpoint1[0]], [endpoint0[1], endpoint1[1]], color='red', linewidth=2)

		if save:
			plt.savefig(save)
		
		if display:
			plt.show()

		fig.canvas.draw()
		data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
		data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

		plt.close()
		return data

	def mesh_to_depth_im(self, mesh, display=True, title=None, save=""):
		"""
		Converts a Mesh to a noramlized 480 x 640 depth image and optionally displays it.
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
			480 x 640 torch.tensor representing depth image
		"""	

		if not isinstance(mesh, Meshes):
			print("mesh_to_depth_im: input not a mesh.")
			return None
	
		depth_im = self.rasterizer(mesh).zbuf.squeeze(-1)

		# ADD TABLE
		max_depth = torch.max(depth_im)	
		depth_im = torch.where(depth_im == -1, max_depth, depth_im)

		if display:
			self.display(depth_im, title=title, save=save)

		return depth_im

	def grasp_sphere(self, center, grasp_obj, display=True, title=None, save=""):
		"""Generate an ico_sphere mesh to visualize a particular grasp
		Parameters
		----------
		center: torch.Tensor of size 3 or tuple of two torch.Tensors of size 3
			3D coordinates of center of the grasp to display or the two contact points of the grasp
		grasp_obj: torch.structures.meshes.Meshes
			Mesh object being grasped
		fname: String
			filepath to save the visualized grasp object
		display: Boolean
			True: display a rendering of the visualized grasp
			False: (default) don't display rendering
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
			c0.offset_verts_(center[0].squeeze(0))
			c1.offset_verts_(center[1].squeeze(0))
			mesh = join_meshes_as_scene([grasp_obj, c0, c1])

		if save:
			fname = save.split(".")[0] + ".obj"
			save_obj(fname, verts=mesh.verts_list()[0], faces=mesh.faces_list()[0])

		if display:
			if save:
				save = save.split(".")[0] + "-g-sphere.png"
			image = self.display(mesh, title=title, save=save)
		
		return mesh

	@staticmethod
	def volume_diff(meshf1, meshf2):
		"""Calculate the total volume displacement between two meshes"""
		mesh1 = trimesh.load(meshf1, "obj", force="mesh")
		mesh2 = trimesh.load(meshf2, "obj", force="mesh")
		if not mesh1.is_volume or not mesh2.is_volume:
			Renderer.logger.error("At least 1 mesh input to volume_diff does not have the properties to compute a volume")
			return None
		
		mesh1_vol, mesh2_vol = mesh1.volume, mesh2.volume
		return mesh2_vol / mesh1_vol
		
	@staticmethod
	def vertex_diff(mesh1, mesh2, abs=True):
		"""Calculate the average Euclidean distance between vertices of two meshes"""
		verts1 = mesh1.verts_list()[0]
		verts2 = mesh2.verts_list()[0]
		if not verts1.shape == verts2.shape:
			Renderer.logger.error("vertex_diff method requires both meshes to have the same number of vertices.")
			return None
		
		if abs:
			return torch.mean(torch.abs(torch.sub(verts2, verts1)))
		else:
			return torch.mean(torch.sub(verts2, verts1))

def test_renderer():
	Renderer.logger.debug("Running test_renderer...")

	# instantiate Renderer object with default parameters
	renderer1 = Renderer()

	# test render_obj -> render_mesh -> display mesh
	print("")
	Renderer.logger.debug("Testing render_obj -> render_mesh -> display...")
	mesh, image = renderer1.render_object("data/bar_clamp.obj", display=True, title="original barclamp obj")
	mesh2, image2 = renderer1.render_object("data/new_barclamp.obj", display=True, title="new barclamp obj")

	# test render_mesh
	Renderer.logger.debug("Testing render_mesh...")
	print("")
	tens = renderer1.render_mesh(mesh, display=True, title="testing mesh -> render_mesh")
	tens2 = renderer1.render_mesh(mesh2, display=True, title="testing mesh -> render_mesh 2")

	# test display with all possible input types
	print("")
	Renderer.logger.debug("Testing batch display with several input types...")
	depth_im = renderer1.mesh_to_depth_im(mesh, display=True, title="testing mesh_to_dim -> display")
	depth_im2 = renderer1.mesh_to_depth_im(mesh2, display=True, title="testing mesh_to_dim -> display2")
	display_images = [mesh, mesh2, image, image2, np.load("data/depth_0.npy"), depth_im, depth_im2]
	display_titles = ["testing mesh -> display", "testing mesh -> display 2", "testing torch.tensor -> display", "testing torch.tensor -> display 2", "testing np.ndarray -> display", "original_barclamp", "new_barclamp"]
	renderer1.display(display_images, title=display_titles)

	print("")
	Renderer.logger.debug("Finished running test_renderer.")

def test_draw_grasp():
	# instantiate Renderer with default parameters
	r = Renderer()
	mesh, image = r.render_object("data/bar_clamp.obj", display=False)

	# define a basic grasp axis and grasp center
	c0 = torch.Tensor([0.0441, 0.0129, 0.0038]).to(r.device)
	c1 = torch.Tensor([ 0.0112,  0.0222, -0.0039]).to(r.device)
	c2 = torch.Tensor([-0.037973009049892426, -0.04009656608104706, 0.02387666516005993]).to(r.device)
	c3 = torch.Tensor([-0.020172616466879845, -0.02000114694237709, 0.015976890921592712]).to(r.device)
	c4 = torch.Tensor([-0.036394111812114716, 0.029987281188368797, 0.0016563538229092956]).to(r.device)
	c5 = torch.Tensor([-0.008031980134546757, 0.020576655864715576, -0.002032859018072486]).to(r.device)
	contacts = [c0, c1, c2, c3, c4, c5]

	# testing on images
	display_images = [r.draw_grasp(image, c0, c1, display=False), r.draw_grasp(image, c2, c3, display=False), r.draw_grasp(image, c4, c5, display=False)]

	# testing on meshes
	display_images = display_images + [r.draw_grasp(mesh, c0, c1, display=False), r.draw_grasp(mesh, c2, c3, display=False), r.draw_grasp(mesh, c4, c5, display=False)]
	r.display(display_images, title="draw_grasp on images and meshes")

	for i in range(0, 6, 2):
		c_0 = contacts[i]
		c_1 = contacts[i+1]
		title_num = i // 2
		r.draw_grasp(mesh, c_0, c_1, title="testing "+str(title_num))

def test_display_batch():

	Renderer.logger.debug("Running test_display_batch...")

	r = Renderer()
	mesh, image = r.render_object("data/bar_clamp.obj", display=False)
	d_image = r.mesh_to_depth_im(mesh, display=True)

	images = [image, image, image, image]
	images2 = images + images[0:2]
	image = image.repeat(4, 1, 1, 1)
	assert len(images2) == 6
	dim = d_image.repeat(8, 1, 1, 1)
	dim8 = [d_image, d_image, d_image, d_image, d_image, d_image, d_image, d_image]
	titles = ["image 1", "image 2", "image 3", "image 4"]

	print("")
	Renderer.logger.debug("Testing list of four images with four titles")
	r.display(images, title=titles)

	print("")
	Renderer.logger.debug("Testing list of four images with shape (1,4) and four titles")
	r.display(images, shape=(1,4), title=titles)

	print("")
	Renderer.logger.debug("Testing batch of four images with four titles")
	r.display(image, title=titles)

	print("")
	Renderer.logger.debug("Testing batch of four images with shape (4,1)")
	r.display(image, shape=(4,1), title="column of four images")

	print("")
	Renderer.logger.debug("Testing batch of six images with two titles")
	r.display(images2, title=titles[:2])

	print("")
	Renderer.logger.debug("Testing batch of 8 depth images")
	r.display(dim, title="batch of 8 depth images")

	print("")
	Renderer.logger.debug("Testing list of 8 depth images with shape (2,4)")
	r.display(dim8, shape=(2,4), title="list of 8 depth images with shape (2,4)")

	print("")
	Renderer.logger.debug("Finished running test_display_batch.")

if __name__ == "__main__":
	# test_renderer()
	# test_draw_grasp()
	# test_display_batch()

	# renderer1 = Renderer()
	# grasp_obj, _ = renderer1.render_object("data/bar_clamp.obj", display=False)
	# # center = torch.tensor([-0.01691197, -0.02238275, 0.04196089]).to(renderer1.device)
	# center = torch.tensor([0.04, 0.04, 0.04]).to(renderer1.device)
	# sphere = renderer1.grasp_sphere(center, grasp_obj, title="vis_grasps/test.obj")
 
	r = Renderer()
	for i in range(5):
		print(f"\nlr-{i}: ")
		for j in range(3):
			# vol1 = Renderer.volume_diff("data/new_barclamp.obj", f"exp-results2/no-oracle/lr-{i}/grasp-{j}/it-100.obj")
			# if vol1 is not None: r.render_object(f"exp-results2/no-oracle/lr-{i}/grasp-{j}/it-100.obj", title=(str(vol1 * 100.0) + " percent of original mesh"))

			vol2 = Renderer.volume_diff("data/new_barclamp.obj", f"exp-results2/no-oracle-grad/lr-{i}/grasp-{j}/it-100.obj")
			if vol2 is not None: r.render_object(f"exp-results2/no-oracle-grad/lr-{i}/grasp-{j}/it-100.obj", title=(str(vol2 * 100.0) + " percent of original mesh"))

			if i < 4:
				vol3 = Renderer.volume_diff("data/new_barclamp.obj", f"exp-results2/oracle-grad/lr-{i}/grasp-{j}/it-100.obj")
				if vol3 is not None: r.render_object(f"exp-results2/oracle-grad/lr-{i}/grasp-{j}/it-100.obj", title=(str(vol3 * 100.0) + " percent of original mesh"))



