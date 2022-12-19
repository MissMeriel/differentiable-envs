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

def load_new_blender_ob(filename):
    from pytorch3d.renderer.mesh import TexturesUV, TexturesVertex, TexturesAtlas
    verts, faces, aux = load_obj(filename)
    verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
    faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)
    tex_maps = aux.texture_images

    # tex_maps is a dictionary of {material name: texture image}.
    # Take the first image:
    texture_image = list(tex_maps.values())[0]
    texture_image = texture_image[None, ...]  # (1, H, W, 3)

    # Create a textures object
    tex = TexturesUV(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image)

    # Initialise the mesh with textures
    meshes = Meshes(verts=[verts], faces=[faces.verts_idx], textures=tex)
    return meshes

device = torch.device("cuda:0")
print(f"Using {device}")
# Use an ico_sphere mesh and load a mesh from an .obj e.g. model.obj
# cow_mesh = ico_sphere(level=3)
# cow_mesh = load_objs_as_meshes([f"{os.getcwd()}/data/cow_mesh/cow.obj"], device=device)
cow_mesh = load_objs_as_meshes([f"{os.getcwd()}/data/trafficcone_mesh/absperrhut2.obj"], device=device)
print(cow_mesh.textures._maps_list)
verts = cow_mesh.verts_packed()
verts = -2 * verts;
verts[:,0] = 0;
verts[:,2] = 0;
cow_mesh.offset_verts_(verts)
offset = torch.zeros((3,), device=device, dtype=torch.float32)
offset[2] = 5
cow_mesh = cow_mesh.offset_verts_(offset)


# test_mesh = load_objs_as_meshes([f"{os.getcwd()}/data/trafficcone_mesh/absperrhut.obj"], device=device)
# test_mesh = load_objs_as_meshes([f"{os.getcwd()}/data/untitled/simple.obj"], device=device)
test_mesh = load_objs_as_meshes([f"{os.getcwd()}/data/roadrunnertest2.obj"], device=device)
print(test_mesh.verts_packed())
print("successfully loaded meshes")

# test_mesh = load_objs_as_meshes([f"{os.getcwd()}/data/Mega_City.obj"], device=device)
# mesh = join_meshes_as_scene([test_mesh,cow_mesh], include_textures=True)
# print("joined meshes as scene")
mesh = test_mesh

#R, T = look_at_view_transform(2.7, 0, 180) R=R, T=T, K=K
# cameras = FoVPerspectiveCameras(device=device, K=K)
# R, T = look_at_view_transform(1.0, 0.0, 0.0)
T = torch.Tensor([[32.5993, 82.1822, -311.538]])
cameras = FoVPerspectiveCameras(device=device, fov=100, T=T)

# Define the settings for rasterization and shading. Here we set the output image to be of size
# 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
# and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
# the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
# explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
# the difference between naive and coarse-to-fine rasterization. 
raster_settings = RasterizationSettings(
    image_size=[480,720], 
    blur_radius=0.0, 
    faces_per_pixel=1, 
)

# Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
# -z direction. 
lights = PointLights(device=device, location=[[32.5993, 82.1822, -311.538]])

# Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
# interpolate the texture uv coordinates for each vertex, sample from a texture image and 
# apply the Phong lighting model
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=HardPhongShader(
        device=device, 
        cameras=cameras,
        lights=lights
    )
)

print("rendering mesh...")
images = renderer(mesh)
print("successfully rendered mesh")
plt.figure(figsize=(10, 10))
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.axis("off");
plt.savefig("render1.png")

class Model(nn.Module):
    def __init__(self, test_mesh, cow_mesh, renderer):
        super().__init__()
        self.test_mesh = test_mesh
        self.cow_mesh = cow_mesh
        self.device = cow_mesh.device
        self.renderer = renderer
        
        # Create an optimizable parameter for the x, y, z position of the camera. 
        self.cowWorldTranslation = nn.Parameter(
            torch.from_numpy(np.array([0, 0], dtype=np.float32)).to(cow_mesh.device))
        print(self.cowWorldTranslation.shape)
        angle = 0.506145
        self.yscale = math.sin(-angle)
        self.zscale = math.cos(angle)
    def forward(self):

        cowTranslation = torch.stack([self.cowWorldTranslation[0],
          self.cowWorldTranslation[1]*self.yscale,
          self.cowWorldTranslation[1]*self.zscale])
        cow_mesh = self.cow_mesh.clone().offset_verts(cowTranslation)
        mesh = join_meshes_as_scene([self.test_mesh.clone(),cow_mesh])
        
        image = self.renderer(meshes_world=mesh)
        
        # calculate ammount of green in image
        #lapl = laplacian(image, 3)
        #loss = -torch.sum(torch.square(lapl));

        loss = torch.sum((-image[0, :, :, 1]+-image[0, :, :, 2]+-image[0, :, :, 0]) )

        # plt.figure(figsize=(10, 10))
        # plt.imshow(image[0, ..., :3].clone().detach().cpu().numpy())
        # plt.axis("off");
        # plt.savefig("renderInLoop.png")
        return loss, image

# We will save images periodically and compose them into a GIF.
filename_output = "./cow_attack_optimization_demo.gif"
writer = imageio.get_writer(filename_output, mode='I', duration=0.3)

# Initialize a model using the renderer, mesh and reference image
model = Model(test_mesh=test_mesh, cow_mesh=cow_mesh, renderer=renderer).to(device)

# Create an optimizer. Here we are using Adam and we pass in the parameters of the model
optimizer = torch.optim.SGD(model.parameters(), lr=0.00005)


loop = range(200)
for i in loop:
    
    optimizer.zero_grad()
    loss, _ = model()
    loss.backward()
    optimizer.step()
    
    
    # Save outputs to create a GIF. 
    if i % 10 == 0:
        cowTranslation = torch.stack([model.cowWorldTranslation[0],
          model.cowWorldTranslation[1]*model.yscale,
          model.cowWorldTranslation[1]*model.zscale])
        cow_mesh = model.cow_mesh.clone().offset_verts(cowTranslation)
        #cow_mesh = model.cow_mesh.clone().offset_verts_(model.cowTranslation)
        mesh = join_meshes_as_scene([test_mesh,cow_mesh])
        image = model.renderer(meshes_world=mesh)
        image = image.detach().squeeze().cpu().numpy()
        image = img_as_ubyte(image)
        writer.append_data(image)
        
    
writer.close()