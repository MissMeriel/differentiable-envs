import logging
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import itertools
from torch.profiler import profile, record_function, ProfilerActivity

from PIL import Image
# used by block print
import os
import sys
from tqdm import tqdm
from qpth.qp import QPFunction
from pytorch3d.utils import ico_sphere
from pytorch3d.io import load_obj, save_obj
from pytorch3d.transforms import Translate
from pytorch3d.ops import mesh_face_areas_normals
from pytorch3d.structures import Meshes, join_meshes_as_scene
from scipy.spatial import ConvexHull
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

    def __init__(self, renderer=None, rasterizer=None, raster_settings=None, camera=None, lights=None, device=None):

        if device is None:
            # set PyTorch device, use cuda if available
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                torch.cuda.set_device(device)
            else:
                print("cuda not available")
                device = torch.device("cpu")
        self.device = device
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

    def pre_process_object(self, verts, faces_idx, faces_to_remove=None):
        if isinstance(verts, np.ndarray):
            verts = torch.from_numpy(verts).float().to(self.device)
            faces_idx = torch.from_numpy(faces_idx).to(self.device)

        if faces_to_remove is not None:
            faces_mask = torch.logical_not(torch.sum(torch.nn.functional.one_hot(faces_to_remove, faces_idx.shape[0]),dim=0))
            mesh_tri = trimesh.Trimesh(faces=faces_idx.numpy(force=True),
                    vertices=verts.numpy(force=True),process=False, validate=False)
            mesh_tri.update_faces(faces_mask.numpy(force=True))
            mesh_tri.remove_unreferenced_vertices()
            faces_idx = torch.tensor(mesh_tri.faces, device=self.device)
            verts = torch.tensor(mesh_tri.vertices, device=self.device, dtype=verts.dtype)

        unique_vals, inverse_indices = torch.unique(verts, sorted=False, return_inverse=True, return_counts=False, dim=0)
        if (len(unique_vals) != len(verts)):
            faces_flat = faces_idx.flatten()
            new_faces_flat = torch.index_select(inverse_indices, 0, faces_flat)
            faces_idx = new_faces_flat.reshape(faces_idx.shape)
            verts = unique_vals

        uniques = torch.unique(torch.sort(faces_idx,dim=-1).values,dim=0)
        if uniques.shape[0] != faces_idx.shape[0]:
            # need to strip duplicate faces, which is easiest in trimesh
            mesh_tri = trimesh.Trimesh(faces=faces_idx.numpy(force=True),
                              vertices=verts.numpy(force=True),process=False, validate=True)
            faces_idx = torch.tensor(mesh_tri.faces, device=self.device)
            verts = torch.tensor(mesh_tri.vertices, device=self.device, dtype=verts.dtype)

        #mask[grouping.unique_rows(np.sort(self.faces, axis=1))[0]] = True

        verts_rgb = torch.ones_like(verts)[None]

        textures = TexturesVertex(verts_features=verts_rgb.to(self.device))

        mesh = Meshes(
            verts=[verts.to(self.device)],
            faces=[faces_idx.to(self.device)],
            textures=textures
        )
        return mesh

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
        
        mesh = self.pre_process_object(verts, faces_idx.verts_idx)

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


class mesh_properties(nn.Module):
    def __init__(self, mesh, forceNormalDist=True, far_dist:float=0) -> None:
        """Find properties of a mesh, including topology, watertightness, and handedness
           topology is all pairs of non-neighbor triangles, edges, and vertices in a mesh"""
        # on initilization creates lists of indices to compare, including
        # triangle to triangle, triangle to edge, and triangle to vertex comparisons
        #
        # forceNormalDist is whether we want the euclidian distance between triangles (False) or the distance in the normal
        #   direction of each triangle (true). Since distance along the normal depends on which triangle use use for the normal, 
        #   this is not symmetric for triangle to triangle comparisons. Therefore, (nearly) twice as many distances will be returned 
        #   in the (True) case. Triangle-edge and triangle-vertex are not affected and are never constrained to normal
        #
        # used as a function object (implemented in forward) will compute distance to self collision
        #
        # for minor speedup when running repeatedly on the same mesh, you can compile it with torch.jit.script(mesh_properties(mesh))
        super().__init__()
        """Find properties of a mesh, including topology, watertightness, and handedness
           topology is all pairs of non-neighbor triangles, edges, and vertices in a mesh"""
        self.forceNormalDist = nn.Parameter(torch.tensor(forceNormalDist),requires_grad=False)
        
        self.compute_connectivity(mesh)
        # initially set using a manifold check during connectivity. This also checks for self collision, but that is very expensive for large meshes
        # self.is_watertight = self.is_watertight and self_collision_min(mesh.detach(), self, forceNormalDist=0) > 0

        self.far_dist = far_dist

        if self.is_watertight:
            volume = self.compute_mesh_volume(mesh, ignore_backface_check=False)
            bounding_volume = self.compute_mesh_bounding_volume(mesh)
            if abs(volume) < bounding_volume * 1e-8 or abs(volume) > bounding_volume:
                self.is_watertight = False
                self.is_inverted = False
            else:
                self.is_inverted = volume.item() < 0
        else:
            self.is_inverted = False
        
    def compute_connectivity(self, mesh):
        # find mapping between vertices and all faces they're part of
        faces = mesh.faces_packed()
        inverseFaceMap = [[] for _ in range(mesh._V)]
        for rowInd in range(faces.shape[0]):
            for ind in range(3):
                inverseFaceMap[faces[rowInd,ind]].append((rowInd, ind))
            
        # find mapping between edges and all faces they're part of
        edges_per_face =mesh.faces_packed_to_edges_packed()
        inverseFaceEdgeMap = [[] for _ in range(mesh.num_edges_per_mesh())]
        for rowInd in range(edges_per_face.shape[0]):
            for ind in range(3):
                inverseFaceEdgeMap[edges_per_face[rowInd,ind]].append((rowInd,ind))

        # For all edges, mark which faces they come from AND what index they are in that face
        connectivity_edge_ind = torch.full((mesh._F,mesh._F), -1, dtype=torch.int64, device=mesh.device)
        for vert_idx in range(len(inverseFaceEdgeMap)):
            for pair in itertools.combinations(inverseFaceEdgeMap[vert_idx],2):
                if pair[0][0] != pair[1][0]:
                    connectivity_edge_ind[pair[0][0],pair[1][0]] = pair[0][1]
                    connectivity_edge_ind[pair[1][0],pair[0][0]] = pair[1][1]
            
        # For all verts, mark which non-edge faces they come from AND what index they are in that face
        connectivity_vert_ind = torch.full((mesh._F,mesh._F), -1, dtype=torch.int64, device=mesh.device)
        for vert_idx in range(len(inverseFaceMap)):
            for pair in itertools.combinations(inverseFaceMap[vert_idx],2):
                if pair[0][0] != pair[1][0] and connectivity_edge_ind[pair[0][0],pair[1][0]] == -1:
                    connectivity_vert_ind[pair[0][0],pair[1][0]] = pair[0][1]
                    connectivity_vert_ind[pair[1][0],pair[0][0]] = pair[1][1]
        
        connectivity_edge_bool = connectivity_edge_ind > -1
        connectivity_vert_bool = connectivity_vert_ind > -1

        # use mask to get all unconnected triangles (symmetric, so only triu)
        tri_u_ind = torch.triu_indices(mesh._F,mesh._F, device=mesh.device)
        connectivity_bool = torch.logical_or(connectivity_edge_bool, torch.logical_or(connectivity_vert_bool,torch.eye(mesh._F,dtype=bool,device=mesh.device)))
        unconnectivity_flat_mask = torch.logical_not(connectivity_bool[tri_u_ind[0],tri_u_ind[1]])
        tri_unconnectivity = tri_u_ind[:,unconnectivity_flat_mask]
        if self.forceNormalDist:
            tri_unconnectivity = torch.cat((tri_unconnectivity,torch.flip(tri_unconnectivity,[0])),dim=1)

        self.tri_unconnectivity = torch.nn.Parameter(tri_unconnectivity,requires_grad=False)

        # convert to tensor of the form (face_index, unconnected_edge_index) and (face_index, unconnected_vert_index)
        # verts are indexed opposite edges, so edge 0 does not include vert 0
        edge_faces_indices = torch.nonzero(connectivity_edge_bool, as_tuple=True)
        vert_indices_in_face = connectivity_edge_ind[edge_faces_indices]
        vert_indices = faces[edge_faces_indices[0], vert_indices_in_face]
        self.vert_unconnectivity = torch.nn.Parameter(torch.cat((edge_faces_indices[1].unsqueeze(0), vert_indices.unsqueeze(0)),dim=0),requires_grad=False)

        vert_faces_indices = torch.nonzero(connectivity_vert_bool, as_tuple=True)
        edge_indices_in_face = connectivity_vert_ind[vert_faces_indices]
        edge_indices = edges_per_face[vert_faces_indices[0], edge_indices_in_face]
        self.edge_unconnectivity = torch.nn.Parameter(torch.cat((vert_faces_indices[1].unsqueeze(0), edge_indices.unsqueeze(0)),dim=0),requires_grad=False)

        self.is_watertight = all([len(edge_faces) % 2==0 for edge_faces in inverseFaceEdgeMap])

        self.faces = torch.nn.Parameter(faces,requires_grad=False)
        self.edges = torch.nn.Parameter(mesh.edges_packed(),requires_grad=False)
        self.face_indices = torch.nn.Parameter(multi_indexing(faces.view(*faces.shape[:-2], -1), mesh.verts_packed().shape, -2),requires_grad=False)
        self.face_size = tuple(self.faces.shape + (-1,))
        self.face_size_sphere = tuple(self.faces.shape + (4,))

        self.edge_indices = torch.nn.Parameter(multi_indexing(self.edges.view(*self.edges.shape[:-2], -1), mesh.verts_packed().shape, -2),requires_grad=False)
        self.edge_size = tuple(self.edges.shape + (-1,))

        self.face_edge_indices_4 = torch.nn.Parameter(multi_indexing(edges_per_face.reshape(*edges_per_face.shape[:-2], -1), (mesh.edges_packed().shape[0],4), -2),requires_grad=False)
        self.face_edge_indices_3 = torch.nn.Parameter(multi_indexing(edges_per_face.reshape(*edges_per_face.shape[:-2], -1), (mesh.edges_packed().shape[0],3), -2),requires_grad=False)

    def _gather_faces(self, verts):
        return verts.gather(-2, self.face_indices).reshape(self.face_size)

    def _gather_edges(self, verts):
        return verts.gather(-2, self.edge_indices).reshape(self.edge_size)
    
    def _gather_face_edges_4(self, edges):
        return edges.gather(-2, self.face_edge_indices_4).reshape(self.face_size)

    def _gather_face_edges_3(self, edges):
        return edges.gather(-2, self.face_edge_indices_3).reshape(self.face_size)

    @staticmethod
    def compute_mesh_tri_areas(mesh):
        """Compute the area of each mesh triangle, using mean as 4th point. Used in CoM/Volume for non-watertight"""
        
        verts_tri = multi_gather_tris(mesh.verts_packed(), mesh.faces_packed())
        com_per_triangle = torch.mean(verts_tri, 1) 

        area_per_triangle = mesh_face_areas_normals(mesh.verts_packed(), mesh.faces_packed())[0].unsqueeze(-1)

        return area_per_triangle, com_per_triangle

    @staticmethod
    def compute_mesh_tetra_volumes(mesh):
        """Compute the volume of each mesh tetra, using mean as 4th point. Used in CoM/Volume for watertight"""
        #https://forums.cgsociety.org/t/how-to-calculate-center-of-mass-for-triangular-mesh/1309966
        mesh_unwrapped = multi_gather_tris(mesh.verts_packed(), mesh.faces_packed())
        # B, F, 3, 3
        # assume Faces, verts, coords
        totalCoords = torch.sum(mesh_unwrapped, 1) # used in both mean and CoM, so split out

        meanVert = torch.sum(totalCoords,0) / (totalCoords.shape[0] * totalCoords.shape[1])

        totalCoords = totalCoords + meanVert
        com_per_triangle = totalCoords / 4

        # add dims and expand "average vertex" to match mesh. Will be used to go from triangles to
        # tetrahedrons
        meanVert_expand = torch.reshape(meanVert, [1, 1, 3]).expand(mesh_unwrapped.shape[0],1,3)

        mesh_tetra = torch.cat([mesh_unwrapped, meanVert_expand], 1)
        mesh_tetra = torch.cat([mesh_tetra, torch.ones([mesh_tetra.shape[0],4,1],device=mesh_unwrapped.device,dtype=mesh_unwrapped.dtype)], -1)
        vol_per_triangle = torch.reshape(torch.linalg.det(mesh_tetra),(mesh_tetra.shape[0],1))
        # det([[x1,y1,z1,1],[x2,y2,z2,1],[x3,y3,z3,1],[x4,y4,z4,1]]) / 6 
        # technically a scaled volume, since we dropped the division by 6
        # does det on last 2 dims, considers at least first 1 to be batch dim
        return vol_per_triangle, com_per_triangle

    @staticmethod
    def get_volumes(mesh):
        vol_bounding = mesh_properties.compute_mesh_bounding_volume(mesh)
        vol_mesh = mesh_properties.compute_mesh_volume(mesh)
        vol_hull = mesh_properties.compute_mesh_hull_volume(mesh)
        return (vol_mesh, vol_bounding, vol_hull)

    @staticmethod
    def compute_mesh_bounding_volume(mesh):
        bb = mesh.get_bounding_boxes()
        ranges = torch.diff(bb,dim=-1).squeeze(-1)[0]
        return torch.prod(ranges,dim=-1)

    @staticmethod
    def compute_mesh_volume(mesh, ignore_backface_check=False):
        """Compute the volume for a mesh, assume uniform density."""
        vol_per_triangle,_ = mesh_properties.compute_mesh_tetra_volumes(mesh)
        if ignore_backface_check:
            vol_per_triangle = torch.abs(vol_per_triangle)
        return torch.sum(vol_per_triangle)/6

    @staticmethod
    def compute_mesh_hull_volume(mesh):
        verts = mesh.verts_packed()
        verts_np = verts.numpy(force=True)
        hull = ConvexHull(verts_np).simplices
        # get unique from hull, update hull to use unique indices
        indices_kept, hull_new = np.unique(hull.reshape(-1),return_inverse=True)
        hull_new = np.reshape(hull_new, hull.shape)
        faces = torch.tensor(hull_new, dtype=torch.long, device=mesh.device)
        verts = verts[indices_kept,:]
        mesh_hull = Meshes([verts],[faces])
        # sub sample verts with unique indices
        return mesh_properties.compute_mesh_volume(mesh_hull,ignore_backface_check=True)
        # construct mesh

    @staticmethod
    def compute_mesh_COM(mesh, is_watertight=True): 
        """Compute the center of mass for a mesh, assume uniform density."""
        if is_watertight:
            vol_per_triangle, com_per_triangle = mesh_properties.compute_mesh_tetra_volumes(mesh)
            total_vol = torch.sum(vol_per_triangle)
            if abs(total_vol) < 1e-10:
                print('WARNING: "watertight" mesh has volume 0, treating as non-watertight')
        
        if not is_watertight or total_vol < 1e-10: 
            vol_per_triangle, com_per_triangle = mesh_properties.compute_mesh_tri_areas(mesh)
            total_vol = torch.sum(vol_per_triangle)

        com = torch.sum(com_per_triangle * vol_per_triangle,dim=0) / total_vol

        return com
    
    @staticmethod
    def self_collision(mesh_properties, mesh):
        # helper function for running self collision on a pytorch3d mesh object, pulling the verts out for you
        verts = mesh.verts_packed()
        verts.requires_grad_(True)

        return mesh_properties.forward(verts)

    def get_colliding_faces(self, dist, bary, min_dist = 1e-8):
        invalid_bary = torch.logical_or(torch.all(bary < 0,dim=1),
                            torch.all(bary[:,[0,2]] + bary[:,[1,3]] > 1,dim=1))
        dists_filtered = dist
        dists_filtered[invalid_bary] = float('Inf')
        dist_interest = dist < min_dist
        dist_interest_tri = dist_interest[:self.tri_unconnectivity.shape[1]]
        dist_interest_edge_vert = dist_interest[self.tri_unconnectivity.shape[1]:]
        faces_coliding_face = self.tri_unconnectivity[:,dist_interest_tri]
        faces_coliding_edge_vert = torch.cat((self.edge_unconnectivity[0,:],self.vert_unconnectivity[0,:]))[dist_interest_edge_vert]
        faces_coliding = torch.cat((faces_coliding_face.flatten(),faces_coliding_edge_vert))
        return torch.unique(faces_coliding)


    def forward(self, input):
        # function to compute the distance between all specified triangles pairs in a mesh.
        # can return nan or 0 if triangles are in collision. Also returns barycentric coordinates
        # that resulted in those distance
        #


        # useful for debugging
        # torch.autograd.set_detect_anomaly(True)
    
        # since forceNormalDist is not symmetric, we need to duplicate the list of pairs, with the order swapped

        quadratic_term_tri, linear_term_tri, const_term_tri, \
            quadratic_term_edg, linear_term_edg, const_term_edg, \
                quadratic_term_vtx, linear_term_vtx, const_term_vtx, \
                    equality_A_tri, equality_b_tri, dist_spheres = self._perform_triangle_comparisons(input)
        # build inequality constraints: solution falls in triangle (or along edge)

        # modified barycentric, 0,0,0 is center and 2/3, -1/3, -1/3  is corner0 
        bary_G_tri = torch.cat((-torch.eye(4,device=input.device),
                            torch.tensor([[1,1,0,0]],device=input.device),
                            torch.tensor([[0,0,1,1]],device=input.device)), dim=0)
        bary_h_tri = torch.tensor([1/3,1/3,1/3,1/3,1/3,1/3],device=input.device)

        bary_G_edg = torch.cat((-torch.eye(3,device=input.device),
                            torch.tensor([[1,1,0]],device=input.device),
                            torch.tensor([[0,0,1]],device=input.device)), dim=0)
        # center for edge is at [0.5, 0], so slight tweak
        bary_h_edg = torch.tensor([1/3,1/3,1/2,1/3,1/2],device=input.device)

        bary_G_vtx = torch.cat((-torch.eye(2,device=input.device),
                            torch.tensor([[1,1]],device=input.device)), dim=0)
        bary_h_vtx = torch.tensor([1/3,1/3,1/3],device=input.device)

        equality_A_edg =  torch.zeros((0),device=input.device) 
        equality_b_edg =  torch.zeros((0),device=input.device) 

        equality_A_vtx =  torch.zeros((0),device=input.device) 
        equality_b_vtx =  torch.zeros((0),device=input.device) 

        # solve qp with our custom wrapper. It returns distance, in addition to solution
        # it also, in the forceNormalDist case, replaces distance with inf when solution does not exist
        dist_all_tri, bary_coords_tri = self._qp_bary_feasible(quadratic_term_tri, linear_term_tri, const_term_tri, equality_A_tri, equality_b_tri, bary_G_tri, bary_h_tri)

        dist_all_edg, bary_coords_edg = self._qp_bary_feasible(quadratic_term_edg, linear_term_edg, const_term_edg, equality_A_edg, equality_b_edg, bary_G_edg, bary_h_edg)
        
        dist_all_vtx, bary_coords_vtx = self._qp_bary_feasible(quadratic_term_vtx, linear_term_vtx, const_term_vtx, equality_A_vtx, equality_b_vtx, bary_G_vtx, bary_h_vtx)

        # for ease of use outside of this function, we go back to traditional bary-centric coordinates before returning
        # modified barycentric, 0,0,0 is center and 2/3, -1/3, -1/3  is corner0 
        # traiditional barycentric 1/3,1/3,1/3 is center and 0,0,0 is corner0

        bary_coords_tri_corrected = bary_coords_tri + 1/3

        # center for edge is at [0.5, 0], so slight tweak
        # correct first two by 1/3, 3rd by 1/2, add zero column
        bary_coords_edg_corrected = torch.cat((bary_coords_edg + torch.tensor([[1/3,1/3,1/2]],device=bary_coords_edg.device), torch.zeros_like(bary_coords_edg[:,0:1])),dim=1)

        # correct first two by 1/3, add 2 zero columns
        bary_coords_vtx_corrected =  torch.cat((bary_coords_vtx + 1/3, torch.zeros_like(bary_coords_vtx)),dim=1)

        # cat all before returning
        dist_all = torch.cat((dist_all_tri,dist_all_edg,dist_all_vtx),dim=0)
        bary_coords_corrected = torch.cat((bary_coords_tri_corrected, bary_coords_edg_corrected, bary_coords_vtx_corrected),dim=0)

        if self.far_dist > 0:
            triangle_dist = dist_spheres < self.far_dist
            dist_return = dist_spheres
            dist_return[triangle_dist] = dist_all
            bary_return = torch.full_like(dist_return.unsqueeze(1).expand([-1,4]),float('Inf'))
            bary_return[triangle_dist] = bary_coords_corrected
            
        else:
            dist_return = dist_all
            bary_return = bary_coords_corrected

        return dist_return, bary_return
    
    @torch.jit.export
    def _compute_sphere(self, edges_unwrapped, E_edge, tris, face_areas):
        edge_sphere_diagonal_sq = torch.sum(torch.square(E_edge),dim=-1,keepdim=True)
        edge_spheres = self._compute_sphere_edge(edges_unwrapped, squared_diameter=edge_sphere_diagonal_sq)
        face_edge_spheres = self._gather_face_edges_4(edge_spheres) # ..., 3 edges in face, 4( radius + xyz center )
        face_edge_sphere_diagonal_sq = edge_sphere_diagonal_sq[self.face_edge_indices_4[...,0]].reshape([-1,3])
        square_length_longest, indices_big_edge = torch.max(face_edge_sphere_diagonal_sq, dim=-1)
        is_tri_not_obtuse = 0 < torch.sum(face_edge_sphere_diagonal_sq, dim=-1) - 2 * square_length_longest
        face_edge_E = self._gather_face_edges_3(E_edge)

        
        # gather? on indices
        tri_spheres = face_edge_spheres.gather(-2, indices_big_edge.unsqueeze(-1).unsqueeze(-1).expand((-1,-1,4))).reshape((self.face_size[0],-1))

        # compute full triangle with all remaining
        tri_spheres[is_tri_not_obtuse] = self._compute_sphere_tri(tris[is_tri_not_obtuse], face_edge_spheres[...,0][is_tri_not_obtuse]*2, face_edge_E[is_tri_not_obtuse], face_areas[is_tri_not_obtuse])
        return tri_spheres, edge_spheres

    @torch.jit.export
    def _compute_sphere_edge(self,edges, squared_diameter):
        radius = torch.sqrt(squared_diameter)/2
        center = torch.mean(edges, dim=-2)
        return torch.cat((radius, center), dim=-1)

    @torch.jit.export
    def _compute_sphere_tri(self, tris, face_edge_lengths, E_edge, face_areas):
        # https://en.wikipedia.org/wiki/Circumcircle#Cartesian_coordinates_from_cross-_and_dot-products
        radius = torch.prod(face_edge_lengths, dim=-1) / face_areas * 4
        outer = torch.linalg.matmul(E_edge, torch.transpose(E_edge,-1,-2))
        weights = (torch.cat((outer[...,0,:1]*outer[...,1,2:],outer[...,1,1:2]*outer[...,0,2:],outer[...,2,2:]*outer[...,0,1:2]),dim=1) \
            / (torch.square(face_areas * 2).unsqueeze(1)*2)).unsqueeze(-1)
        center = torch.sum( weights * tris, dim=-2)
        return torch.cat((radius.unsqueeze(-1), center), dim=-1)

    @torch.jit.export
    def _sphere_sphere_distance(self, sphere1, sphere2):
        # assume radius, center
        return torch.linalg.vector_norm(sphere1[...,1:]-sphere2[...,1:],dim=1) - sphere1[...,0] - sphere2[...,0]

    @torch.jit.export
    def _perform_triangle_comparisons(self, verts):   

        # compute all edge vectors
        # NOTE minor repeated calcs with shared edges
        
        tris = self._gather_faces(verts) # n_faces, corner, xyz

        E1 = tris[:, 1] - tris[:, 0]  # vector of edge 1 on triangle (n_faces, 3)
        E2 = tris[:, 2] - tris[:, 0]  # vector of edge 2 on triangle (n_faces, 3)

        edges = self._gather_edges(verts) # n_edges, end, xyz


        E_edge = edges[:, 1] - edges[:, 0]  # vector of edge 1 on triangle (n_faces, 3)
        if self.forceNormalDist or self.far_dist > 0:
            face_areas, face_normals = self._normal_wrapper(verts, self.faces)
        else:
            face_areas = None

        if self.far_dist > 0:
            tri_spheres, edge_spheres = self._compute_sphere( edges, E_edge, tris, face_areas)
            vert_spheres = torch.nn.functional.pad(verts, (1,0))
            tri_tri_dist_sphere = self._sphere_sphere_distance(tri_spheres[self.tri_unconnectivity[0]],tri_spheres[self.tri_unconnectivity[1]])
            tri_edge_dist_sphere = self._sphere_sphere_distance(tri_spheres[self.edge_unconnectivity[0]],edge_spheres[self.edge_unconnectivity[1]])
            tri_vert_dist_sphere = self._sphere_sphere_distance(tri_spheres[self.vert_unconnectivity[0]],vert_spheres[self.vert_unconnectivity[1]])
            sphere_dist = torch.cat((tri_tri_dist_sphere,tri_edge_dist_sphere,tri_vert_dist_sphere),dim=0)
            tri_unconnectivity = self.tri_unconnectivity[:,tri_tri_dist_sphere < self.far_dist]
            edge_unconnectivity = self.edge_unconnectivity[:,tri_edge_dist_sphere < self.far_dist]
            vert_unconnectivity = self.vert_unconnectivity[:,tri_vert_dist_sphere < self.far_dist]
        else:
            tri_unconnectivity = self.tri_unconnectivity
            edge_unconnectivity = self.edge_unconnectivity
            vert_unconnectivity = self.vert_unconnectivity
            sphere_dist = None

        


        # triangle center to triangle center, when no vertices or edges are shared
        w_diff_vec_tri = torch.mean(tris[tri_unconnectivity[1]],dim=1) - torch.mean(tris[tri_unconnectivity[0]],dim=1)  # n_tri_unconnect, xyz

        # triangle center to (non-neighbor) edge center, when vertex is shared
        w_diff_vec_edg = torch.mean(edges[edge_unconnectivity[1]],dim=1) - torch.mean(tris[edge_unconnectivity[0]],dim=1) # n_edg_unconnect, xyz

        # triangle center to (non-neighbor) vertex, when edge is shared
        w_diff_vec_vtx = verts[vert_unconnectivity[1]] - torch.mean(tris[vert_unconnectivity[0]],dim=1) # n_edg_unconnect, xyz

        # expand edge vector directions to full size (per face or edge they appear in) for linear and quadratic terms
        TR1_E1_full_tri = E1[tri_unconnectivity[0]]
        TR1_E2_full_tri = E2[tri_unconnectivity[0]]
        TR2_E1_full_tri = E1[tri_unconnectivity[1]]
        TR2_E2_full_tri = E2[tri_unconnectivity[1]]

        TR1_E1_full_edg = E1[edge_unconnectivity[0]]
        TR1_E2_full_edg = E2[edge_unconnectivity[0]]

        TR2_E_full_edg = E_edge[edge_unconnectivity[1]]

        TR1_E1_full_vtx = E1[vert_unconnectivity[0]]
        TR1_E2_full_vtx = E2[vert_unconnectivity[0]]

        # linear terms, 

        #  concatenate all relevant edge vectors

        # tri
        triangle_vecs_tri = torch.cat(
                        (-TR1_E1_full_tri.unsqueeze(1),
                            -TR1_E2_full_tri.unsqueeze(1),
                            TR2_E1_full_tri.unsqueeze(1),
                            TR2_E2_full_tri.unsqueeze(1)),
                            dim=1)
        
        # edge
        triangle_vecs_edg = torch.cat(
                        (-TR1_E1_full_edg.unsqueeze(1),
                            -TR1_E2_full_edg.unsqueeze(1),
                            TR2_E_full_edg.unsqueeze(1)),
                            dim=1)
        
        # vert
        triangle_vecs_vtx = torch.cat(
                        (-TR1_E1_full_vtx.unsqueeze(1),
                        -TR1_E2_full_vtx.unsqueeze(1)),
                            dim=1)
        
        # compare triangle edge directions to vector between objects to compare
        linear_term_tri = 2 * triangle_vecs_tri @ w_diff_vec_tri.unsqueeze(2)

        linear_term_edg = 2 * triangle_vecs_edg @ w_diff_vec_edg.unsqueeze(2)

        linear_term_vtx = 2 * triangle_vecs_vtx @ w_diff_vec_vtx.unsqueeze(2)


        # const terms, not used in quadratic program, only to correct distance afterwards.
        # only depend on distance between "centers"

        const_term_tri = torch.linalg.vecdot(w_diff_vec_tri,w_diff_vec_tri)

        const_term_edg = torch.linalg.vecdot(w_diff_vec_edg,w_diff_vec_edg)

        const_term_vtx = torch.linalg.vecdot(w_diff_vec_vtx,w_diff_vec_vtx)


        # quadratic terms

        # edge - edge products

        # within triangle terms, compute for each triangle then expand

        # within "first" edges
        E1_E1 = torch.linalg.vecdot(E1,E1)
        TR1_TR1_E1_E1_full_tri = E1_E1[tri_unconnectivity[0]].unsqueeze(1)
        TR2_TR2_E1_E1_full_tri = E1_E1[tri_unconnectivity[1]].unsqueeze(1)

        TR1_TR1_E1_E1_full_edg = E1_E1[edge_unconnectivity[0]].unsqueeze(1)

        TR1_TR1_E1_E1_full_vtx = E1_E1[vert_unconnectivity[0]].unsqueeze(1)

        # within "second" edges
        E2_E2 = torch.linalg.vecdot(E2,E2)
        TR1_TR1_E2_E2_full_tri = E2_E2[tri_unconnectivity[0]].unsqueeze(1)
        TR2_TR2_E2_E2_full_tri = E2_E2[tri_unconnectivity[1]].unsqueeze(1)

        TR1_TR1_E2_E2_full_edg = E2_E2[edge_unconnectivity[0]].unsqueeze(1)

        TR1_TR1_E2_E2_full_vtx = E2_E2[vert_unconnectivity[0]].unsqueeze(1)

        # between "first" edges and "second edges"
        E1_E2 = torch.linalg.vecdot(E1,E2)
        TR1_TR1_E1_E2_full_tri = E1_E2[tri_unconnectivity[0]].unsqueeze(1)
        TR2_TR2_E1_E2_full_tri = E1_E2[tri_unconnectivity[1]].unsqueeze(1)

        TR1_TR1_E1_E2_full_edg = E1_E2[edge_unconnectivity[0]].unsqueeze(1)

        TR1_TR1_E1_E2_full_vtx = E1_E2[vert_unconnectivity[0]].unsqueeze(1)

        # single edge terms, can be 1st, 2nd, or even 3rd edge
        E_E = torch.linalg.vecdot(E_edge,E_edge)
        TR2_TR2_E_E_full_edg = E_E[edge_unconnectivity[1]].unsqueeze(1)

        # between triangle terms, use expanded version, repeated calcs when forceNormalDist is true
        # TODO, detect forceNormalDist and share computation
        TR1_TR2_E1_E1_full_tri = -torch.linalg.vecdot(TR1_E1_full_tri,TR2_E1_full_tri).unsqueeze(1)
        TR1_TR2_E1_E2_full_tri = -torch.linalg.vecdot(TR1_E1_full_tri,TR2_E2_full_tri).unsqueeze(1)
        TR1_TR2_E2_E1_full_tri = -torch.linalg.vecdot(TR1_E2_full_tri,TR2_E1_full_tri).unsqueeze(1)
        TR1_TR2_E2_E2_full_tri = -torch.linalg.vecdot(TR1_E2_full_tri,TR2_E2_full_tri).unsqueeze(1)

        quadratic_term_tri = 2 * torch.cat((TR1_TR1_E1_E1_full_tri, TR1_TR1_E1_E2_full_tri, TR1_TR2_E1_E1_full_tri, TR1_TR2_E1_E2_full_tri,
                        TR1_TR1_E1_E2_full_tri, TR1_TR1_E2_E2_full_tri, TR1_TR2_E2_E1_full_tri, TR1_TR2_E2_E2_full_tri,
                        TR1_TR2_E1_E1_full_tri, TR1_TR2_E2_E1_full_tri, TR2_TR2_E1_E1_full_tri, TR2_TR2_E1_E2_full_tri,
                        TR1_TR2_E1_E2_full_tri, TR1_TR2_E2_E2_full_tri, TR2_TR2_E1_E2_full_tri, TR2_TR2_E2_E2_full_tri),dim=1).reshape([-1,4,4])

        # triangle-edge terms, use expanded version
        TR1_TR2_E1_E_full_edg = -torch.linalg.vecdot(TR1_E1_full_edg,TR2_E_full_edg).unsqueeze(1)
        TR1_TR2_E2_E_full_edg = -torch.linalg.vecdot(TR1_E2_full_edg,TR2_E_full_edg).unsqueeze(1)

        quadratic_term_edg = 2 * torch.cat(
                (TR1_TR1_E1_E1_full_edg, TR1_TR1_E1_E2_full_edg, TR1_TR2_E1_E_full_edg,
                    TR1_TR1_E1_E2_full_edg, TR1_TR1_E2_E2_full_edg, TR1_TR2_E2_E_full_edg, 
                    TR1_TR2_E1_E_full_edg, TR1_TR2_E2_E_full_edg, TR2_TR2_E_E_full_edg, ),dim=1).reshape([-1,3,3])
        
        # triangle-vertex terms, use expanded version. Note that vertex does not actually appear!
        quadratic_term_vtx = 2 * torch.cat(
                (TR1_TR1_E1_E1_full_vtx, TR1_TR1_E1_E2_full_vtx,
                    TR1_TR1_E1_E2_full_vtx, TR1_TR1_E2_E2_full_vtx, ),dim=1).reshape([-1,2,2])
        
            # build equality constraints, if needed
        if self.forceNormalDist:
            # need additional constraint that solution is in normal direction of 1st triangle
            # encoded as an equality constraint in the optimization
            
            normal_vecs_tri = face_normals[tri_unconnectivity[0]]

            TR_1_N_tri = face_normals[tri_unconnectivity[0]]

            # compute projection of edge vectors to containing triangle normal.
            # perform only once per triangle and expand later
            TR_1_E1_N = torch.linalg.vecdot( face_normals , E1)
            TR_1_E2_N = torch.linalg.vecdot( face_normals , E2)

            # compute projection of second triangle edge vectors onto first triangle normal
            TR_2_E1_N_tri = torch.linalg.vecdot( TR_1_N_tri , TR2_E1_full_tri)
            TR_2_E2_N_tri = torch.linalg.vecdot( TR_1_N_tri , TR2_E2_full_tri)

            # use projection of edge onto normal to compute rejection of normal from vector
            # perform only once per triangle, expand later
            vec_reject_TR1_E1 = (E1 - TR_1_E1_N.unsqueeze(1) * face_normals)
            vec_reject_TR1_E2 = (E2 - TR_1_E2_N.unsqueeze(1) * face_normals)

            equality_A_tri = torch.cat(
                        (vec_reject_TR1_E1[tri_unconnectivity[0]].unsqueeze(-1),
                        vec_reject_TR1_E2[tri_unconnectivity[0]].unsqueeze(-1),
                        ((TR_2_E1_N_tri.unsqueeze(1) * TR_1_N_tri) - TR2_E1_full_tri).unsqueeze(-1) ,
                        ((TR_2_E2_N_tri.unsqueeze(1) * TR_1_N_tri) - TR2_E2_full_tri).unsqueeze(-1)),dim=-1)
            
            # compute rejection of vector normal from the vector between triangles
            equality_b_tri = w_diff_vec_tri - (torch.linalg.vecdot(w_diff_vec_tri,normal_vecs_tri).unsqueeze(1) * normal_vecs_tri)
            


            # Don't ever do normal constraint for neighbors, since they can pass through each others sides, not just faces
            #
            # normal_vecs_edg = mesh.faces_normals_packed()[edge_unconnectivity[0]]
            # normal_vecs_vtx = mesh.faces_normals_packed()[vert_unconnectivity[0]]
            # TR_1_N_edg = mesh.faces_normals_packed()[edge_unconnectivity[0]]
            # TR_2_E_N_edg = torch.linalg.vecdot( TR_1_N_edg , TR2_E_full_edg)
            #
            # equality_A_edg = torch.cat(
            #               (vec_reject_TR1_E1[edge_unconnectivity[0]].unsqueeze(-1),
            #                vec_reject_TR1_E2[edge_unconnectivity[0]].unsqueeze(-1),
            #                ((TR_2_E_N_edg.unsqueeze(1) * TR_1_N_edg) - TR2_E_full_edg).unsqueeze(-1)),dim=-1)
            # equality_A_vtx = torch.cat(
            #               (vec_reject_TR1_E1[vert_unconnectivity[0]].unsqueeze(-1),
            #                vec_reject_TR1_E2[vert_unconnectivity[0]].unsqueeze(-1)),dim=-1)
            #
            # equality_b_edg = -(torch.linalg.vecdot(w_diff_vec_edg,normal_vecs_edg).unsqueeze(1) * normal_vecs_edg - w_diff_vec_edg)
            # equality_b_vtx = -(torch.linalg.vecdot(w_diff_vec_vtx,normal_vecs_vtx).unsqueeze(1) * normal_vecs_vtx - w_diff_vec_vtx)

        else:
            # no equality constraint needed, so empty tensor
            equality_A_tri = torch.zeros((0),device=verts.device) 
            equality_b_tri = torch.zeros((0),device=verts.device)

        
        return quadratic_term_tri, linear_term_tri, const_term_tri, quadratic_term_edg, linear_term_edg, \
    const_term_edg, quadratic_term_vtx, linear_term_vtx, const_term_vtx, equality_A_tri, equality_b_tri, sphere_dist

    @torch.jit.ignore
    def _normal_wrapper(self, verts, faces):
        return mesh_face_areas_normals(verts, faces)

    @staticmethod
    def _applyQuad(bary_coords, quadratic_term, linear_term, const_term):
        # computes distance for a batch of quadratic programs and their solutions
        # needed because QPFunction only returns the coordinates of the closest points on the triangles
        bc1 = bary_coords.unsqueeze(1)
        bc2 = bary_coords.unsqueeze(2)

        # for numeric reasons, this can be negative when it should be 0
        dist_raw = (torch.matmul(bc1, torch.matmul(quadratic_term, bc2))/2).squeeze(2).squeeze(1) + torch.linalg.vecdot(linear_term.squeeze(2), bary_coords) + const_term
        # replace negative values with 0, indicating a collision has taken place
        dist_relu = torch.nn.functional.relu(dist_raw)
        return dist_relu
        
    @torch.jit.ignore
    def _qp_bary_feasible(self, quadratic_term, linear_term, const_term, equality_A, equality_b, inequality_G, inequality_h):
        # optimization is not possible if quadratic term is not semi-positive-definite (spd)
        # this means the parabola must have single minimum (non flat and facing up)

        # there is probably a better way to enforce this, but I force it to be SPD by repeadtedly adding a scaled identiy matrix.
        # scaled identity matrix is equivalent to L2 regularization on the triangle coordinates. ie, the optimization
        # is solving for a scaled objective that is distance + l2norm(coordinates). Intuitively, when there are 
        # infinit solutions, like when two triangles are at least partially parallel, this will cause us to find
        # the coordinates that are "smallest" along the parallel part. "Smallest" means closest to some origin. If
        # useRefPointBias is True, this origin is at one of the triangle corners. If it is false, this origin
        # is at the triangle center. 
        #
        # Solving for the identity matrix that will make our matrix spd seemed complicated, so for now
        # we compute a lower bound, the size of the smallest (largest negative) eigenvalue, compared to a small
        # positive epsilon (spd_target). We double the size of this target every time we fail to create an spd
        # matrix, then try again, up to max_spd_iter
        with record_function("find spd"):
            quadratic_term_diagonal = torch.diagonal(quadratic_term,dim1=-1,dim2=-2)
            quadratic_term_orig = quadratic_term_diagonal.clone()
            quadratic_term_reg = quadratic_term_diagonal.clone()
            
            quadratic_term_diagonal
            spd_min_eig=1e-6
            spd_target = spd_min_eig
            max_spd_iter = 4
            
            while max_spd_iter > 0:
                max_spd_iter-=1
                quadratic_term_diagonal.copy_(quadratic_term_reg)
                # eigs = torch.real(torch.linalg.eigvals(quadratic_term.detach()))
                # not_spd = torch.any(eigs < spd_min_eig,dim=1)

                chol,info = torch.linalg.cholesky_ex(quadratic_term.detach())
                # not true, I made this up
                # not_spd = torch.logical_or(info>0,torch.min(torch.diagonal(chol,dim1=-1,dim2=-2),dim=-1).values < 0.01)
                _,singular_values,_ = torch.linalg.svd(quadratic_term.detach())
                not_spd = torch.logical_or(info>0,  singular_values[:,-1] < spd_min_eig)

                # print(f'found {torch.where(not_spd)[0].shape[0]} non spd quadratic terms of {quadratic_term.shape[0]}')
                if not torch.any(not_spd):
                    break
                spd_target = spd_target * 2
                
                correction = 0 # if using singular vectors, we don't know which one is negative, so no extra correction available
                # correction = -torch.min(eigs[not_spd],dim=1)[0]
                dist_regularizer = spd_target + correction
                quadratic_term_reg[not_spd] += dist_regularizer #.unsqueeze(-1)
                # regulizer_mat = (dist_regularizer.unsqueeze(1).unsqueeze(2) * torch.eye(quadratic_term.shape[-1], device = quadratic_term.device).unsqueeze(0))

                # quadratic_term_reg[not_spd] = quadratic_term_reg[not_spd] + regulizer_mat
            # tqdm.write(f'found spd in {3-max_spd_iter} iterations')
        with record_function("initial qp"):
        # solve all, including infeasible
            qp = QPFunction(check_Q_spd=False)
            mesh_properties._blockPrint()
            bary_coords = qp(quadratic_term, linear_term.squeeze(2), # optimization
                                                        inequality_G, inequality_h, #inequality constraints
                                                        equality_A,equality_b) # equality constraints)
            mesh_properties._enablePrint()
            
            # we can check that our inequality constraint was met (Gz <= h), which indicates bary coordinates fall outside triangle
            valid_bary = torch.all(torch.matmul(bary_coords, inequality_G.transpose(0,1)) <= inequality_h.unsqueeze(0),dim=1)

        

        # if any invalid solutions are found, we need to repeat the computation without them
        # pytorch fails to compute any derivatives if an invalid distance exists

        with record_function("secondary qp"):        
            if equality_A.size(0) == 0:
                equality_A_filt = equality_A
                equality_b_filt = equality_b
            else:
                equality_A_filt = equality_A[valid_bary]
                equality_b_filt = equality_b[valid_bary]
            bary_coords =  torch.full_like(bary_coords,float('Inf'))
            dist_all = torch.full_like(const_term,float('Inf'))
            if torch.any(valid_bary):
                mesh_properties._blockPrint()
                bary_coords_feasible = qp(quadratic_term[valid_bary], linear_term[valid_bary].squeeze(2), # optimization
                                                        inequality_G, inequality_h, #inequality constraints
                                                        equality_A_filt,equality_b_filt) # equality constraints)
                mesh_properties._enablePrint()

                bary_coords[valid_bary] = bary_coords_feasible
                quadratic_term_diagonal.copy_(quadratic_term_orig)
                dist_feasible = mesh_properties._applyQuad(bary_coords_feasible, quadratic_term[valid_bary], linear_term[valid_bary], const_term[valid_bary])
                dist_all[valid_bary] = dist_feasible


        # causes anomalies
        # else:
        #     tqdm.write('all valid')
        #     with record_function("apply solution"):
        #         dist_all = mesh_properties._applyQuad(bary_coords, quadratic_term.clone(), linear_term, const_term)

        return dist_all, bary_coords

    # Disable
    @staticmethod
    def _blockPrint():
        sys.stdout = open(os.devnull, 'w')

    # Restore
    @staticmethod
    def _enablePrint():
        sys.stdout = sys.__stdout__

    @staticmethod
    def check_bary_validity(bary_coords):
        invalid_bary = torch.logical_not(torch.logical_and(torch.all(bary_coords >= 0,dim=1),
                            torch.all(bary_coords[:,[0,2]] + bary_coords[:,[1,3]] <= 1,dim=1)))
        return invalid_bary

    @staticmethod
    def make_invalid_dist_inf(dist, bary_coords):
        invalid_bary = mesh_properties.check_bary_validity(bary_coords=bary_coords)
        dists_filtered = dist
        dists_filtered[invalid_bary] = float('Inf')
        return dists_filtered

    @staticmethod
    def self_collision_min(dists_raw, bary_coords_raw):
        # wrapper on self_collision that returns the minimum valid distance
        dists_filtered = mesh_properties.make_invalid_dist_inf(dist=dists_raw, bary_coords=bary_coords_raw)
        return torch.min(dists_filtered)


def multi_indexing(index: torch.Tensor, shape: torch.Size, dim=-2):
    shape = list(shape)
    back_pad = len(shape) - index.ndim
    for _ in range(back_pad):
        index = index.unsqueeze(-1)
    expand_shape = shape
    expand_shape[dim] = -1
    return index.expand(*expand_shape)


def multi_gather(values: torch.Tensor, index: torch.Tensor, dim=-2):
    # take care of batch dimension of, and acts like a linear indexing in the target dimention
    # we assume that the index's last dimension is the dimension to be indexed on
    return values.gather(dim, multi_indexing(index, values.shape, dim))

@torch.jit.ignore
def multi_gather_tris(v: torch.Tensor, f: torch.Tensor, dim:int=-2) -> torch.Tensor:
    # compute faces normals w.r.t the vertices (considering batch dimension)
    if v.ndim == (f.ndim + 1):
        f = f[None].expand(v.shape[0], *f.shape)
    # assert verts.shape[0] == faces.shape[0]
    shape = torch.tensor(v.shape)
    remainder = shape.flip(0)[:(len(shape) - dim - 1) % len(shape)]
    return multi_gather(v, f.view(*f.shape[:-2], -1), dim=dim).view(*f.shape, *remainder)  # B, F, 3, 3

def moller_trumbore(ray_o, ray_d, mesh , eps=1e-8):
    """
    The Moller Trumbore algorithm for fast ray triangle intersection
    Naive batch implementation (m rays and n triangles at the same time)
    O(n_rays * n_faces) memory usage, parallelized execution
    Parameters
    ----------
    ray_o : torch.Tensor, (n_rays, 3)
    ray_d : torch.Tensor, (n_rays, 3)
    tris  : torch.Tensor, (n_faces, 3, 3)
    """
    tris = multi_gather_tris(mesh.verts_packed(), mesh.faces_packed()).double()
    E1 = tris[:, 1] - tris[:, 0]  # vector of edge 1 on triangle (n_faces, 3)
    E2 = tris[:, 2] - tris[:, 0]  # vector of edge 2 on triangle (n_faces, 3)

    # batch cross product
    N = torch.cross(E1, E2)  # normal to E1 and E2, automatically batched to (n_faces, 3)
    # TODO, should this be a solve instead? need to batch u,v,t into one matrix?
    invdet = 1. / -(torch.einsum('md,nd->mn', ray_d, N) + eps)  # inverse determinant (n_faces, 3)

    A0 = ray_o[:, None] - tris[None, :, 0]  # (n_rays, 3) - (n_faces, 3) -> (n_rays, n_faces, 3) automatic broadcast
    DA0 = torch.cross(A0, ray_d[:, None].expand(*A0.shape))  # (n_rays, n_faces, 3) x (n_rays, 3) -> (n_rays, n_faces, 3) no automatic broadcast

    u = torch.einsum('mnd,nd->mn', DA0, E2) * invdet
    v = -torch.einsum('mnd,nd->mn', DA0, E1) * invdet
    t = torch.einsum('mnd,nd->mn', A0, N) * invdet  # t >= 0.0 means this is a ray

    return u, v, t

def moller_tumbore_solve(ray_o, ray_d, mesh):
    tris = multi_gather_tris(mesh.verts_packed(), mesh.faces_packed())
    # verts = mesh.verts_packed()
    # edges = multi_gather_tris(verts, mesh.edges_packed())
    # edge_vectors = edges[..., 1,:] - edges[..., 0,:]
    # edges_per_face = mesh.faces_packed_to_edges_packed()
    # face_edges_1 = multi_gather_tris(edge_vectors, edges_per_face[...,1:2])
    # face_edges_2 = -multi_gather_tris(edge_vectors, edges_per_face[...,2:3])
    face_edges_1 = tris[:, 1:2] - tris[:, 0:1]
    face_edges_2 = tris[:, 2:3] - tris[:, 0:1] 
    face_edges = torch.cat((face_edges_1,face_edges_2),dim=-2).unsqueeze(0)

    # shapes rays, faces, [1,3], 3
    ray_o_expand = ray_o.unsqueeze(-2).unsqueeze(-2)
    ray_d_expand = ray_d.unsqueeze(-2).unsqueeze(-2)
    shared_vert = tris[:,0:1].unsqueeze(0)

    ray_o_expand,ray_d_expand,face_edges,shared_vert = torch.broadcast_tensors(ray_o_expand,ray_d_expand,face_edges,shared_vert)
    A = torch.cat((-ray_d_expand[...,0:1,:],face_edges),dim=-2)
    b = ray_o_expand[...,0:1,:] - shared_vert[...,0:1,:]
    # tuv,info = torch.linalg.solve_ex(A,b.squeeze(-2).unsqueeze(-1),left=True)
    # tuv = tuv.squeeze(-1)
    tvu,info = torch.linalg.solve_ex(A,b,left=False)
    t, v, u = torch.split(tvu, 1, dim=-1)
    # reorder to match cramer implementation
    #((t >= 0.0) * (t < self.width) * (u >= 0.0) * (v >= 0.0) * ((u + v) <= 1.0)).bool()
    return u.squeeze(-1).squeeze(-1), v.squeeze(-1).squeeze(-1), t.squeeze(-1).squeeze(-1)


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



