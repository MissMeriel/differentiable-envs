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
    TexturesUV,
    TexturesVertex
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import imageio
from skimage import img_as_ubyte

device = torch.device("cuda:0")
# Use an ico_sphere mesh
sphere_mesh = ico_sphere(level=5)

R, T = look_at_view_transform(2.7, 0, 180) 
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

# Define the settings for rasterization and shading. Here we set the output image to be of size
# 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
# and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
# the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
# explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
# the difference between naive and coarse-to-fine rasterization. 
raster_settings = RasterizationSettings(
    image_size=720, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
)

# Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
# -z direction. 
lights = PointLights(device=device, location=[[0.0, 0.0, 0.0]])

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

ncolors = 10
numDups = 3
r = 0.3
rampSlope = 1


def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(npoints, ndim)
    norm = np.linalg.norm(vec, axis=1);
    vec = np.divide(vec,np.expand_dims(norm,1))
    return vec

class Model(nn.Module):
    def __init__(self, renderer):
        super().__init__()
        self.device = torch.device("cuda:0")
        self.renderer = renderer


        pointsCart = sample_spherical(ncolors * numDups)
        pointsSphere = np.stack([np.arccos(pointsCart[:,2]),np.arctan2(pointsCart[:,1],pointsCart[:,0])],1)

        self.pointsSphere = nn.Parameter(torch.from_numpy(pointsSphere).to(torch.float).to(device))

        colors = np.random.uniform(size=[ncolors, 3])
        self.colors = torch.from_numpy(colors).to(torch.float).to(device)

        isoVert,isoFace = sphere_mesh.get_mesh_verts_faces(0)
        self.isoVert = isoVert.to(device)
        self.isoFace = isoFace.to(device)
#verts_rgb_colors = torch.from_numpy(np.random.uniform(size=[1,isoVert.shape[0],isoVert.shape[1]])).to(device)
#verts_rgb_colors = verts_rgb_colors.to(torch.float)
        
    def forward(self):
        verts_rgb_colors = torch.zeros([1,self.isoVert.shape[0],self.isoVert.shape[1]]).to(device)
        points = torch.stack([torch.sin(self.pointsSphere[:,0]) * torch.cos(self.pointsSphere[:,1]),
            torch.sin(self.pointsSphere[:,0]) * torch.sin(self.pointsSphere[:,1]),
            torch.cos(self.pointsSphere[:,0])],1)
        for i in range(self.isoVert.shape[0]):
            for j in range(points.shape[0]):
                dist = torch.linalg.norm(points[j,:]-self.isoVert[i,:])
                if dist < r:
                    verts_rgb_colors[0,i,:] = self.colors[j%ncolors,:] - dist * rampSlope * (self.colors[j%ncolors,:] - verts_rgb_colors[0,i,:])

        textures = TexturesVertex(verts_rgb_colors)

        mesh = Meshes([self.isoVert], [self.isoFace], textures)
        image = renderer(mesh)
        loss = torch.sum((-image[0, :, :, 1]+-image[0, :, :, 2]+-image[0, :, :, 0]) )

        return loss, image
# plt.figure(figsize=(10, 10))
# plt.imshow(images[0, ..., :3].cpu().numpy())
# plt.axis("off");
# plt.savefig("rendersphere.png")


filename_output = "./sphere_brighten_optimization_demo.gif"
writer = imageio.get_writer(filename_output, mode='I', duration=0.3)

# Initialize a model using the renderer, mesh and reference image
model = Model(renderer=renderer).to(device)

# Create an optimizer. Here we are using Adam and we pass in the parameters of the model
optimizer = torch.optim.SGD(model.parameters(), lr=0.00005)


loop = range(20)
for i in loop:
    optimizer.zero_grad()
    loss, _ = model()
    loss.backward()
    optimizer.step()
    
    
    # Save outputs to create a GIF. 
    if True: #i % 10 == 0:
        _, image = model()
        image = image.detach().squeeze().cpu().numpy()
        image = img_as_ubyte(image)
        writer.append_data(image)
        
    
writer.close()