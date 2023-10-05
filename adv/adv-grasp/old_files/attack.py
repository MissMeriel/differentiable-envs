import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gqcnn_pytorch import KitModel

import cv2
import PIL.Image as PImage
import skimage.transform as skt
from skimage.util import compare_images


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
	
def render_object(obj_file, renderer, cameras, lights, device, display=True):
	"""render mesh object and optionally display"""

	verts, faces_idx, _ = load_obj(obj_file)
	faces = faces_idx.verts_idx

	verts_rgb = torch.ones_like(verts)[None]
	textures = TexturesVertex(verts_features=verts_rgb.to(device))

	mesh = Meshes(
		verts=[verts.to(device)],
		faces=[faces.to(device)],
		textures=textures
	)

	image = renderer(mesh, cameras=cameras, lights=lights)
	dis_image = image[0, ..., :3].cpu().detach().numpy()
		
	# plot if display is True
	if display:
		plt.figure(figsize=(6,6))
		plt.imshow(dis_image)
		plt.axis("off")
		plt.show()
		
	return mesh, dis_image

def mesh_to_depth_im(mesh, rasterizer, display=True):
	"""transform mesh to npy depth image and optionally display"""
	depth_im = rasterizer(mesh).zbuf.squeeze(-1).squeeze(0)
	depth_im = depth_im.cpu().detach().numpy()
	
	# normalize??
	depth_im = (depth_im-np.min(depth_im))/(np.max(depth_im)-np.min(depth_im))
		
	if display:
		plt.imshow(depth_im)
		plt.axis("off")
		plt.show()
		
	return depth_im
	
def get_input_tensors(depth_im, grasp_depth, grasp_angle, grasp_center_x, grasp_center_y):
	pose_tensor = np.zeros([1, 1])
	pose_tensor[0] = grasp_depth
	pose_tensor = torch.from_numpy(pose_tensor).float()
	
	image_tensor = np.zeros([1, 32, 32, 1])
	scale = 32 / 96
	
	depth_height = depth_im.shape[0]
	depth_width = depth_im.shape[1]
	
	translation = scale * np.array([
		(depth_height / 2) - grasp_center_x,
		(depth_width / 2) - grasp_center_y
	])
	
	# image_tensor = process_depth_im(depth_im, scale, grasp_angle, 32, 32, translation)
	image_tensor = np.load('dex_shared_dir/image_tensor.npy')
	image_tensor = torch.from_numpy(image_tensor).float().permute(0,3,1,2)
	
	return pose_tensor, image_tensor
	
def process_depth_im(image, scale, grasp_angle, im_height, im_width, translation):
	"""pre-process the depth image before input to model and return"""
	# lots of code from autolab_core/image.py
	# THIS FUNC DOESN'T WORK AT ALL RN
	
	center_x = image.shape[0] / 2
	center_y = image.shape[1] / 2
	
	# resize depth image according to scale
	image = image.squeeze()
	np_shape = np.asarray(image.shape).astype(np.float32)
	np_shape[0:2] *= scale
	out_shape = tuple(np_shape.astype(int))
	
	scaled_image = skt.resize(
		image,
		out_shape,
		order=1,
		anti_aliasing=False,
		mode="constant"
	)
	scaled_image = scaled_image[:, :, np.newaxis]	# 3 dimensional here
	print("scaled image:", scaled_image)
	
	# transform image based on translation and grasp angle
	angle = np.rad2deg(grasp_angle)
	trans_map = np.float32(
		[[1, 0, translation[1]], [0, 1, translation[0]]]
	)
	rot_map = cv2.getRotationMatrix2D(
		(center_x, center_y), angle, 1
	)
	trans_map_aff = np.r_[trans_map, [[0, 0, 1]]]
	rot_map_aff = np.r_[rot_map, [[0, 0, 1]]]
	full_map = rot_map_aff.dot(trans_map_aff)
	full_map = full_map[:2, :]
	image = cv2.warpAffine(scaled_image, full_map, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST)
	image = image.astype(np.float32)
	image = image[:, :, np.newaxis]
	
	# crop depth image according to im_height and im_width
	start_row = int(np.floor(center_x - float(im_height) / 2))
	end_row = int(np.floor(center_x + float(im_height) / 2))
	start_col = int(np.floor(center_y - float(im_width) / 2))
	end_col = int(np.floor(center_y + float(im_width) / 2))
	pil_im = PImage.fromarray(image.squeeze())
	cropped_pil = pil_im.crop((start_col, start_row, end_col, end_row))
	crop_image = np.array(cropped_pil)
	crop_image = crop_image[:, :, np.newaxis]
	
	# permute tensor for pytorch and return
	image_tensor = torch.from_numpy(crop_image).float().unsqueeze(0).permute(0, 3, 1, 2)
	
	return image_tensor

# initalize PyTorch3D objects
if torch.cuda.is_available():
	device = torch.device("cuda:0")
	torch.cuda.set_device(device)
else:
	print("cuda not available")
	device = torch.device("cpu")
			
lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
R, T = look_at_view_transform(dist=0.7, elev=90, azim=180)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
raster_settings = RasterizationSettings(
	image_size=512,
	blur_radius=0.0,
	faces_per_pixel=1
)
rasterizer = MeshRasterizer(
	cameras=cameras,
	raster_settings=raster_settings
)
renderer = MeshRenderer(
	rasterizer=rasterizer,
	shader=SoftPhongShader(
		device=device,
		cameras=cameras,
		lights=lights
	)
)

# render object
# mesh, _ = render_object("data/bar_clamp.obj", renderer, cameras, lights, device, display=False)
# depth_im = mesh_to_depth_im(mesh, rasterizer)
# np.save("data/bar_clamp.npy", depth_im)
# print("depth_im shape:", depth_im.shape)
# print("depth_im:")
# print(depth_im)
# print("depth im min:", np.min(depth_im))
# print("depth im max:", np.max(depth_im))

# load in depth image
depth_im = np.load("data/depth_0.npy")

# get image and pose tensors from depth image or from grasp - manual values for depth_0.npy
grasp_depth=0.607433762324266
grasp_angle=-2.896613990462929
center_x=416
center_y=286
pose_tensor, image_tensor = get_input_tensors(depth_im, grasp_depth, grasp_angle, grasp_center_x=center_x, grasp_center_y=center_y)

# print("pose tensor:", pose_tensor.detach().numpy())
# print("image tensor:", image_tensor.detach().numpy())
# print("image tensor max:", np.max(image_tensor.detach().numpy()))

# initialize model
model = KitModel("weights.npy")	# ("573931b36e8e4fdab6218f6c598e100d.npy")
model.eval()

# test prediction
x1 = torch.from_numpy(np.load('data/pose_tensor1_raw.npy')).float()  # pose - 64x1
x2 = torch.from_numpy(np.load('data/image_tensor1_raw.npy')).float().permute(0,3,1,2)    # image - 64x32x32x1
# print("input pose shape", x1.shape)
# print("input image shape", x2.shape)

output_arr = model(pose_tensor, image_tensor)
print("output:", output_arr.shape)
print(output_arr)

def calc_loss(pose_tensor, image_tensor, model):
	""" Minimize grasp quality prediction for an initially successful grasp """
	output = model(pose_tensor, image_tensor)
	output = output[:, -1]
	return output

def perturb(pose_tensor, param, model):
	""" Modify image array/depth image to change shape of object being grasped """
	new_image = param
	loss = calc_loss(pose_tensor, new_image, model)
	return loss, new_image

# start adversarial attack on depth image
# param = image_tensor
"""
orig_tensor = image_tensor.clone().detach().squeeze().numpy()
loss = calc_loss(pose_tensor, image_tensor, model)
print("orig tensor:\n", orig_tensor, "\nshape:", orig_tensor.shape)


image_tensor.requires_grad=True
optimizer = torch.optim.SGD([image_tensor], lr=1e-5, momentum=0.99)

num_plots = 6
steps_per_plot = 5
num_steps = num_plots * steps_per_plot
_, arr = plt.subplots(2, num_plots)

# plot original tensor before attack starts
cmap='viridis'
arr[0, 0].imshow(orig_tensor, cmap=cmap)
arr[0, 0].axis("off")
title = "step:0 conf:" + str(round(float(loss[0].data),3))
arr[0, 0].set_title(title)
arr[1, 0].imshow(compare_images(orig_tensor, orig_tensor), cmap=cmap)
arr[1, 0].axis("off")

# from scipy.io import savemat
# mat_dict = {}

for i in range(num_steps):
	optimizer.zero_grad()
	loss, image_tensor = perturb(pose_tensor, image_tensor, model)
	loss.backward()
	optimizer.step()
	print("step:", i, "\tloss:", float(loss[0].data))
	if (i%steps_per_plot == 0) and (i != 0):
	 	tensor = image_tensor.clone().detach().squeeze().numpy()
	 	index = i // steps_per_plot
	 	print("i:", i, "index:", index)
	 	arr[0, index].imshow(tensor, cmap=cmap)
	 	title = "step:" + str(i) + " conf:" + str(round(float(loss[0].data),3))
	 	arr[0, index].set_title(title)
	 	arr[0, index].axis("off")
	 	arr[1, index].imshow(compare_images(orig_tensor, tensor), cmap=cmap)
	 	arr[1, index].axis("off")
	 	# mat_dict["step" + str(i)] = tensor
	 	
	
# display new depth image
final_depth_im = image_tensor.detach().squeeze().numpy()

# arr[1].imshow(final_depth_im)
plt.show()

# savemat("data/advgrasps.mat", mat_dict)
"""


