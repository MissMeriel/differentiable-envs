import os
import sys
import torch
import pytorch3d
import numpy as np

import os
import torch
import matplotlib.pyplot as plt

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform, look_at_rotation,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    HardPhongShader,
    HardGouraudShader,
    TexturesUV,
    TexturesVertex
)

# add path for demo utils functions 
import sys
import os
sys.path.append(os.path.abspath(''))


# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Load obj
DATA_DIR = "./data"
# obj_filename = os.path.join(DATA_DIR, "cow_mesh/cow.obj")
obj_filename = os.path.join(DATA_DIR, "roadrunnertest2.obj")
# obj_filename = os.path.join(DATA_DIR, "kidneycircle.obj")
# obj_filename = os.path.join(DATA_DIR, "trafficcone_mesh/absperrhut.obj")
# obj_filename = "/home/meriel/Documents/RoadRunner/Exports/kidneycircle.obj"
mesh = load_objs_as_meshes([obj_filename], device=device)

# Initialize a camera.
# With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction. 
# So we move the camera by 180 in the azimuth direction so it is facing the front of the cow. 
# vvv works for kidneycircle
# R, T = look_at_view_transform(2.5, -10, 90, degrees=True)
# R, T = look_at_view_transform(60, 0, 180, at=((-192.488525, 99.366821, -5.828361),), up=((0,1,0),), degrees=True)
# vvv roadrunnertest2.obj
R, T = look_at_view_transform(80, 0, 180, at=((10.741272, -357.137512, 0.1),), up=((0,1,0),), degrees=True)
# vvvv traffic cone test
# R, T = look_at_view_transform(2.5, 0, 90, at=((0, 0, 0),), up=((0, 0, 1),), degrees=True)
# camera_pos = torch.from_numpy(np.array([-100, 100, -50], dtype=np.float32)).to(device)
# R = look_at_rotation(camera_pos[None, :], device=device)
# T = -torch.bmm(R.transpose(1, 2), camera_pos[None, :, None])[:, :, 0]   # (1, 3)


cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

# Define the settings for rasterization and shading. Here we set the output image to be of size
# 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
# and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
# the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
# explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
# the difference between naive and coarse-to-fine rasterization. 
raster_settings = RasterizationSettings(
    image_size=512, 
    blur_radius=0.0, 
    faces_per_pixel=1,
    # bin_size=2,
    # max_faces_per_bin=1
)

# Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
# -z direction. 
lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

# Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
# interpolate the texture uv coordinates for each vertex, sample from a texture image and 
# apply the Phong lighting model
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=cameras,
        lights=lights
    )
)

images = renderer(mesh)
plt.figure(figsize=(10, 10))
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.pause(5)
plt.axis("off")
plt.close()