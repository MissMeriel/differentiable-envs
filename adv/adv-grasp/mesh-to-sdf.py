import mesh_to_sdf
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes

if torch.cuda.is_available():
	device = torch.device("cuda:0")
	torch.cuda.set_device(device)
else:
	print("cuda not available")
	device = torch.device("cpu")
	
mesh = load_objs_as_meshes(["bar_clamp.obj"], device=device)

sdf = mesh_to_sdf.mesh_to_sdf(mesh, query_points, surface_point_method='scan', sign_method='normal', bounding_radius=None, scan_count=100, scan_resolution=400, sample_point_count=10000000, normal_sample_count=11)


