import Pyro4
import os
import sys
import torch
import numpy as np

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
	look_at_view_transform,
	FoVPerspectiveCameras,
	MeshRenderer,
	MeshRasterizer,
	SoftSilhouetteShader,
	SoftPhongShader,
	RasterizationSettings,
	PointLights,
	TexturesVertex
)

import matplotlib.pyplot as plt

SHARED_DIR = "dex_shared_dir"

class PyTorchObject:
	def __init__(self, obj, mesh=None):
		""" initialize pytorch object """
		
		# initialize device
		if torch.cuda.is_available():
			self.device = torch.device("cuda:0")
			torch.cuda.set_device(self.device)
		else:
			print("cuda not available")
			self.device = torch.device("cpu")
			
		self.obj = obj		# string - path to obj file
		self.mesh = mesh
		self.lights = PointLights(device=self.device, location=[[0.0, 0.0, -3.0]])
		R, T = look_at_view_transform(dist=1.0, elev=90, azim=180)
		self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
		self.raster_settings = RasterizationSettings(
			image_size=512,
			blur_radius=0.0,
			faces_per_pixel=1
		)
		self.rasterizer = MeshRasterizer(
			cameras=self.cameras,
			raster_settings=self.raster_settings
		)
		self.renderer = MeshRenderer(
			rasterizer=self.rasterizer,
			shader=SoftPhongShader(
				device=self.device,
				cameras=self.cameras,
				lights=self.lights
			)
		)

	def render_obj_file(self, obj_file=None, display=True):
		""" render mesh object and return rendering as numpy array """
		
		if obj_file == None:
			obj_file = self.obj
		
		verts, faces_idx, _ = load_obj(obj_file)
		faces = faces_idx.verts_idx

		verts_rgb = torch.ones_like(verts)[None]
		textures = TexturesVertex(verts_features=verts_rgb.to(self.device))

		mesh = Meshes(
			verts=[verts.to(self.device)],
			faces=[faces.to(self.device)],
			textures=textures
		)
		
		self.mesh = mesh

		image = self.renderer(mesh, cameras=self.cameras, lights=self.lights)
		dis_image = image[0, ..., :3].cpu().detach().numpy()
		
		# plot if display is True
		if display:
			plt.figure(figsize=(6,6))
			plt.imshow(dis_image)
			plt.axis("off")
			plt.show()
		
		return dis_image
		
	def save_nparr(self, image, filename):
		""" save numpy.ndarray image in the shared dir """
		filepath = os.path.join(SHARED_DIR, filename)
		np.save(filepath, image)
		
	def mesh_to_depth(self, mesh=None, display=True):
		""" convert mesh to depth image, then plot and return depth image as numpy array """
		
		if mesh==None:
			mesh = self.mesh
		if mesh == None:
			print("ERROR - mesh_to_depth: no mesh to convert to depth image")
		
		depth_im = self.rasterizer(mesh).zbuf.squeeze(-1).squeeze(0)
		depth_im = depth_im.cpu().detach().numpy()
		
		if display:
			plt.imshow(depth_im)
			plt.axis("off")
			plt.show()
		
		return depth_im
	
# render barclamp object
barclamp_obj = PyTorchObject("bar_clamp.obj")
image = barclamp_obj.render_obj_file(display=False)
d_im = barclamp_obj.mesh_to_depth(display=False)
# barclamp_obj.save_nparr(d_im, "barclamp.npy")

# communication with server in dex3 docker container
Pyro4.config.COMMTIMEOUT = None

# uri = input("What is the Pytro uri of the server object? ").strip()
server = Pyro4.Proxy("PYRO:Server@0.0.0.0:5000")
# print(server.sample_grasps('bar_clamp.obj'))
# print(server.depth_im("barclamp.npy"))

print(server.gqcnn_sample_grasps("depth_0.npy", 100))



