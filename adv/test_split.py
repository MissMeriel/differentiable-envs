from pytorch3d.utils import ico_sphere
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    HardFlatShader,
    SoftGouraudShader,
    HardPhongShader,
    TexturesUV,
    TexturesVertex
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import laplacian
import matplotlib.pyplot as plt
import imageio
import numpy as np
from skimage import img_as_ubyte
import math
import os

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0")
print(f"Using {device}")
# Use an ico_sphere mesh and load a mesh from an .obj e.g. model.obj
# cow_mesh = ico_sphere(level=3)
# cow_mesh = load_objs_as_meshes([f"{os.getcwd()}/data/cow_mesh/cow.obj"], device=device)
cow_mesh = load_objs_as_meshes([f"{os.getcwd()}/data/trafficcone_mesh/absperrhut.obj"], device=device)
print(cow_mesh.textures._maps_list)
verts = cow_mesh.verts_packed()
verts = -2 * verts;
verts[:,0] = 0;
verts[:,2] = 0;
cow_mesh.offset_verts_(verts)
offset = torch.zeros((3,), device=device, dtype=torch.float32)
offset[2] = 5
cow_mesh = cow_mesh.offset_verts_(offset)

test_mesh = load_objs_as_meshes([f"{os.getcwd()}/data/roadrunnertest2.obj"], device=device)
# test_mesh = load_objs_as_meshes([f"{os.getcwd()}/data/Mega_City.obj"], device=device)
print(test_mesh.verts_packed())
print("successfully loaded meshes")

# pytorch3d is sensitive to file extension
# saving using .ply does not save .mtl and .png
from pytorch3d.io import IO
IO().save_mesh(data=test_mesh, path="./test_save_scene.obj")

mesh = join_meshes_as_scene([test_mesh,cow_mesh], include_textures=True)
print("joined meshes as scene")

print("rendering mesh...")
images = renderer(mesh)
print("successfully rendered mesh")
plt.figure(figsize=(10, 10))
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.axis("off");
plt.savefig("render_joined_meshes.png")