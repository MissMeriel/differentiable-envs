from pytorch3d.utils import ico_sphere
from pytorch3d.io import load_obj, load_objs_as_meshes, IO, save_obj
from pytorch3d.structures import Meshes, join_meshes_as_scene, join_meshes_as_batch
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    OpenGLPerspectiveCameras, 
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
from kornia.filters import laplacian
import matplotlib.pyplot as plt
import imageio
import numpy as np
from skimage import img_as_ubyte
import math
import sys
import os

from pytorch3d.renderer.mesh.utils import pack_unique_rectangles
from typing import List, NamedTuple, Tuple

import argparse

def parse_cmdline_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--objfile", type=str, default="/media/raid/Home/Data/HPSTA/adv/data/exportedOrig/roadrunnerScene.obj")
    args = parser.parse_args()
    return args

args = parse_cmdline_args()

# Maybe subdivide faces and verts based on verts2D(faces2D) lengths
# shift verts2D to fit into max size grid
def splitAndOffsetOneFace(verts_3D_tensor_in, verts_2D_tensor_in, verts_normals_tensor_in, max_grid_size):
    def findTextureGridSize(verts_2D):
        max_return = torch.max(torch.ceil(verts_2D),0)
        return max_return.values

    # split a single triangle on one side, return explicitly for 2 resulting triangles
    def splitTriangle(verts_2D, verts_3D, verts_normals):
        # remainder and roll work together, remainder should be negation of roll dist
        length_squared = torch.sum( torch.square( verts_2D - torch.roll(verts_2D,1,0) ) , 1)
        max_length_side_0 = torch.argmax(length_squared, 0)
        max_length_side_1 = torch.remainder(max_length_side_0-1,3)

        # midpoint of max length in 2D and 3D
        midpoint_2D = torch.mean(verts_2D[(max_length_side_0,max_length_side_1),:],0)
        midpoint_3D = torch.mean(verts_3D[(max_length_side_0,max_length_side_1),:],0)
        midpoint_normal_unnorm = torch.mean(verts_normals_tensor_in[(max_length_side_0,max_length_side_1),:],0)
        midpoint_normal = midpoint_normal_unnorm / torch.linalg.vector_norm(midpoint_normal_unnorm) # make unit length

        # new triangles
        verts_2D_0 = verts_2D.clone()
        verts_2D_1 = verts_2D.clone()

        verts_2D_0[max_length_side_0,:] = midpoint_2D
        verts_2D_1[max_length_side_1,:] = midpoint_2D

        verts_3D_0 = verts_3D.clone()
        verts_3D_1 = verts_3D.clone()

        verts_3D_0[max_length_side_0,:] = midpoint_3D
        verts_3D_1[max_length_side_1,:] = midpoint_3D

        verts_normals_0 = verts_normals.clone()
        verts_normals_1 = verts_normals.clone()

        verts_normals_0[max_length_side_0,:] = midpoint_normal
        verts_normals_1[max_length_side_1,:] = midpoint_normal

        return verts_2D_0, verts_2D_1, verts_3D_0, verts_3D_1, verts_normals_0, verts_normals_1

    # offset triangle and check grid size
    max_return = torch.min(torch.floor(verts_2D_tensor_in),0)
    verts_2D_tensor = verts_2D_tensor_in - max_return.values # make triangle strictly positive and close to origin
    curr_grid_size = findTextureGridSize(verts_2D_tensor)

    if (torch.any(torch.gt(curr_grid_size, max_grid_size)).data):
        # split needed, so make two triangles and we'll recursively check them
        verts_2D_0, verts_2D_1, verts_3D_0, verts_3D_1, verts_normals_0, verts_normals_1 = splitTriangle(
            verts_2D_tensor, verts_3D_tensor_in, verts_normals_tensor_in)
        # recurse on each
        verts_3D_0, verts_2D_0, verts_normals_0 = splitAndOffsetOneFace(
            verts_3D_0, verts_2D_0, verts_normals_0, max_grid_size)
        verts_3D_1, verts_2D_1, verts_normals_1 = splitAndOffsetOneFace(
            verts_3D_1, verts_2D_1, verts_normals_1, max_grid_size)
        
        # collect resulting triangles
        verts_2D_tensor = torch.cat((verts_2D_0,verts_2D_1), 0)
        verts_3D_tensor = torch.cat((verts_3D_0,verts_3D_1), 0)
        verts_normals_tensor = torch.cat((verts_normals_0, verts_normals_1),0)
    else:
        verts_3D_tensor = verts_3D_tensor_in
        verts_normals_tensor = verts_normals_tensor_in

    return verts_3D_tensor, verts_2D_tensor, verts_normals_tensor


def unwrapTextureUV(faces_3D_tensor_in, verts_3D_tensor_in,faces_2D_tensor_in, verts_2D_tensor_in, verts_normals_tensor_in, image_in):
    max_grid_tuple = (1, 5, 5, 1)
    max_grid_tensor = torch.tensor([max_grid_tuple[1:2]])
    
    # initialize lists for later cat operation
    faces_3D_tensor_list = []
    verts_3D_tensor_list = []
    faces_2D_tensor_list = []
    verts_2D_tensor_list = []
    verts_normals_tensor_list = []
    count_verts_so_Far = 0

    image_unwrapped = image_in.repeat(max_grid_tuple) # tensor with copies of image_in
    # TODO consider vectorizing initial area computation
    for faceIndex in range(faces_3D_tensor_in.shape[0]): # loop over faces 
        
        faces_2D_local = faces_2D_tensor_in[faceIndex, :] # index into indices, wow!
        faces_3D_local = faces_3D_tensor_in[faceIndex, :]

        verts_2D_local = verts_2D_tensor_in[faces_2D_local, :] # unpack verts into 3(v)x3(uv) triangle
        verts_3D_local = verts_3D_tensor_in[faces_3D_local, :] # unpack verts into 3(v)x3(xyz) triangle
        # normals are in meshes object and assume same face indices 
        verts_normals_local = verts_normals_tensor_in[faces_3D_local, :] # unpack normals into 3(v)x3(xyz) triangle 

        # this function always shifts to strictly positive. May also split if triangle outside grid
        verts_3D_local, verts_2D_local, verts_normals_local = splitAndOffsetOneFace(verts_3D_local, verts_2D_local, verts_normals_local, max_grid_tensor) # assume triangle is dense
        
        # reindex naively, no duplication in this representation (dependent variables)
        # overwrite intermediate version, not needed anymore
        faces_3D_local =  torch.reshape(
            torch.arange(
                count_verts_so_Far, count_verts_so_Far + verts_3D_local.shape[0]),
                [-1,3]) # -1 means "figure it out for me"
        faces_2D_local = faces_3D_local


        # add to lists for later cat
        faces_3D_tensor_list.append( faces_3D_local )
        verts_3D_tensor_list.append( verts_3D_local )
        faces_2D_tensor_list.append( faces_2D_local )
        verts_2D_tensor_list.append( verts_2D_local )
        verts_normals_tensor_list.append( verts_normals_local )
        count_verts_so_Far += verts_3D_local.shape[0]

        
    # cat all lists
    faces_3D_tensor = torch.cat(faces_3D_tensor_list,0)
    verts_3D_tensor = torch.cat(verts_3D_tensor_list,0)
    faces_2D_tensor = torch.cat(faces_2D_tensor_list,0)  
    # scale to new dimensions
    verts_2D_tensor = torch.divide( torch.cat(verts_2D_tensor_list,0), max_grid_tensor) 
    verts_normals_tensor = torch.cat(verts_normals_tensor_list)

    return  faces_3D_tensor, verts_3D_tensor, faces_2D_tensor, verts_2D_tensor, verts_normals_tensor, image_unwrapped

def load_objs_as_meshes_mine(
    files: list,
    device:  None,
    load_textures: bool = True,
    create_texture_atlas: bool = False,
    texture_atlas_size: int = 4,
):
    """
    Load meshes from a list of .obj files using the load_obj function, and
    return them as a Meshes object. This only works for meshes which have a
    single texture image for the whole mesh. See the load_obj function for more
    details. material_colors and normals are not stored.

    Args:
        files: A list of file-like objects (with methods read, readline, tell,
            and seek), pathlib paths or strings containing file names.
        device: Desired device of returned Meshes. Default:
            uses the current device for the default tensor type.
        load_textures: Boolean indicating whether material files are loaded
        create_texture_atlas, texture_atlas_size, texture_wrap: as for load_obj.
        path_manager: optionally a PathManager object to interpret paths.

    Returns:
        New Meshes object.
    """
    mesh_list = []
    filenames = []
    materials = []
    mesh_list_raw = []
    mesh_list_unwrapped = []
    print('inside load' + str(files) + ' device: ' + str(device), flush=True)
    for f_obj in files:

        verts, faces, aux = load_obj(
            f_obj,
            load_textures=load_textures,
            create_texture_atlas=create_texture_atlas,
            texture_atlas_size=texture_atlas_size,
        )
        
        
        print('after load_obj', flush=True)
        tex = None
        if False & create_texture_atlas:
            # TexturesAtlas type
            tex = TexturesAtlas(atlas=[aux.texture_atlas.to(device)])
        else:
            # TexturesUV type
            tex_maps = aux.texture_images
            print(aux)
            if tex_maps is not None and len(tex_maps) > 0:
                # add file and material names for easy use later
                filenames.append([os.path.basename(f_obj)] * len(tex_maps))
                materials.append(tex_maps.keys())

                # initialize many lists
                textures_per_mat = []
                verts_tensor_list = []
                faces_3D_tensor_list = []
                verts_normals_tensor_list = []
                
                # copy to device
                faces_3D_tensor_in = faces.verts_idx.to(device) # (F, 3)
                verts_tensor_in = verts.to(device)
                verts_uvs_in = aux.verts_uvs.to(device)
                # if faces have different uv indices than xyz indices, grab them too. Otherwise copy
                if hasattr(faces, 'textures_idx'):
                    faces_2D_tensor_in = faces.textures_idx.to(device) # (F, 3)
                else:
                    # TODO, may be able to optimize more in this case
                    faces_2D_tensor_in = faces_3D_tensor_in

                # if faces have different normal indices than xyz indices, grab them too. Otherwise copy
                if hasattr(faces, 'normals_idx'):
                    faces_normals_tensor_in = faces.normals_idx.to(device) # (F, 3)
                else:
                    
                    faces_normals_tensor_in = faces_3D_tensor_in

                if hasattr(aux, 'normals'):
                    verts_normals_tensor_in = aux.normals
                else:
                    verts_normals_tensor_in = None

                count_verts_3D = 0
                for tex_map_index in range(len(tex_maps)):
                    # TODO merge into 1 function
                    def sample_single_material(faces_tensor_in, verts_tensor_in, faces_mask):
                        faces_tensor_in = faces_tensor_in[faces_mask,:]
                        unique_return = torch.unique(faces_tensor_in,return_inverse=True)
                        verts_index_tensor = unique_return[0]
                        verts_tensor = torch.index_select(verts_tensor_in,0,verts_index_tensor) # slicing but preserved shape
                        faces_tensor = unique_return[1]
                        return faces_tensor, verts_tensor


                    def sample_single_material_shared_indexing(faces_tensor_0, verts_tensor_0, faces_tensor_1, verts_tensor_1,  faces_mask):
                        combinedFaces = torch.cat([torch.unsqueeze(faces_tensor_0[faces_mask,:],2),
                         torch.unsqueeze(faces_tensor_1[faces_mask],2)],2)
                        unique_return = torch.unique(torch.reshape(combinedFaces,[-1,2]),dim=0,return_inverse=True)
                        faces_tensor = torch.reshape(unique_return[1],[-1,3])
                        
                        verts_index_0_tensor = unique_return[0][:,0]
                        verts_index_1_tensor = unique_return[0][:,1]
                        verts_3D_tensor = torch.index_select(verts_tensor_0,0,verts_index_0_tensor) # slicing but preserved shape
                        verts_norm_tensor = torch.index_select(verts_tensor_1,0,verts_index_1_tensor) # slicing but preserved shape
                        
                        return faces_tensor, verts_3D_tensor, verts_norm_tensor
                                      

                    # TODO make indexing cleaner
                    # copy to device
                    image = list(tex_maps.values())[tex_map_index].to(device)[None]     

                    # only some faces are in this material
                    faces_used_mask = (faces.materials_idx == tex_map_index).to(device)

                    faces_3D_tensor, verts_3D_tensor, verts_normals_tensor = sample_single_material_shared_indexing(
                        faces_3D_tensor_in, verts_tensor_in, faces_normals_tensor_in, verts_normals_tensor_in, faces_used_mask)

                    faces_2D_tensor, verts_2D_tensor= sample_single_material(
                        faces_2D_tensor_in, verts_uvs_in, faces_used_mask)



                    print('nwrap verts 2D: ', verts_2D_tensor.shape)
                    print("nwrap faces 3D:", faces_3D_tensor.shape)
                    print('nwrap verts 3D: ', verts_3D_tensor.shape)
                    print("nwrap faces 2D:", faces_2D_tensor.shape, flush=True)
                    print('nwrap verts nm: ', verts_normals_tensor.shape)
                    
                    # Store original meshes to return for optimization, maybe not compatible with renderer.
                    mesh_list_raw.append(Meshes([verts_3D_tensor], [faces_3D_tensor],
                    textures=TexturesUV(
                        verts_uvs=[verts_2D_tensor], 
                        faces_uvs=[faces_2D_tensor], maps=image
                    ), verts_normals=[verts_normals_tensor]))
                 
                    # do a bunch of recursive splitting until vts between 0 and 1
                    if torch.any(torch.gt(verts_2D_tensor, 1)).data or torch.any(torch.lt(verts_2D_tensor, 0)).data:
                        faces_3D_tensor, verts_3D_tensor, faces_2D_tensor, verts_2D_tensor, verts_normals_tensor, image = unwrapTextureUV(
                            faces_3D_tensor, verts_3D_tensor,faces_2D_tensor, verts_2D_tensor, verts_normals_tensor, image)

                    mesh_list_unwrapped.append(Meshes([verts_3D_tensor], [faces_3D_tensor],
                    textures=TexturesUV(
                        verts_uvs=[verts_2D_tensor], 
                        faces_uvs=[faces_2D_tensor], maps=image
                    ), verts_normals=[verts_normals_tensor] ))      

                    print("nwrap verts 3D: ", verts_3D_tensor.shape)
                    print("nwrap faces 3D:", faces_3D_tensor.shape)
                    print("nwrap verts 2D: ", verts_2D_tensor.shape)
                    print("nwrap faces 2D:", faces_2D_tensor.shape, flush=True)
                    print("nwrap verts nm: ", verts_normals_tensor.shape)

                    # verts_3D naively cat-ed, need to start faces index from last size
                    faces_3D_tensor +=  count_verts_3D
                    count_verts_3D += verts_3D_tensor.shape[0]
                    
                    # collect into lists for cat later
                    textures_per_mat.append( TexturesUV(
                        verts_uvs=[verts_2D_tensor], 
                        faces_uvs=[faces_2D_tensor], maps=image
                    ) )
                    faces_3D_tensor_list.append(faces_3D_tensor)
                    verts_tensor_list.append(verts_3D_tensor)
                    verts_normals_tensor_list.append(verts_normals_tensor)

                if len(tex_maps) > 1:
                    tex = textures_per_mat[0].join_batch(textures_per_mat[1:])
                    tex = join_scene_mine(tex) # TODO figure out how to include from pytorch3D
                    faces_tensor = torch.cat(faces_3D_tensor_list)
                    verts_tensor = torch.cat(verts_tensor_list)
                    verts_normals_tensor = torch.cat(verts_tensor_list)
                else:
                    tex = textures_per_mat[0]
                    faces_tensor = faces_3D_tensor_list[0]
                    verts_tensor = verts_tensor_list[0]
                    verts_normals_tensor = verts_normals_tensor_list[0]
                
        # TODO other aux properties lost: normals, material colors, 
        print("verts 3D: ", verts_tensor.shape)
        print("faces 3D:", faces_tensor.shape)
        print("verts 2D: ", tex.verts_uvs_padded().shape)
        print("faces 2D:", tex.faces_uvs_padded().shape)
        print("max 2d face index: ", torch.max(tex.faces_uvs_padded()))
        print("max 3d face index: ", torch.max(faces_tensor))
        print(tex)
        mesh = Meshes(
            verts=verts_tensor.unsqueeze(0), faces=faces_tensor.unsqueeze(0), 
            textures=tex, verts_normals=[verts_normals_tensor]
        )
        mesh_list.append(mesh)
    meshes_batch = join_meshes_as_batch(mesh_list)
    # TODO redundant?
    if len(mesh_list) == 1:
        meshes_batch = mesh_list[0]
    else:
        meshes_batch = join_meshes_as_batch(mesh_list)
    meshes_batch.mesh_list = mesh_list_raw
    meshes_batch.mesh_list_unwrapped = mesh_list_unwrapped
    meshes_batch.filenames = filenames
    meshes_batch.materials = materials
    print(meshes_batch.mesh_list)
    print(meshes_batch.materials)
    print(meshes_batch.filenames)
    
    return meshes_batch

class Rectangle(NamedTuple):
    xsize: int
    ysize: int
    identifier: int


class PackedRectangle(NamedTuple):
    x: int
    y: int
    flipped: bool
    is_first: bool


class PackedRectangles(NamedTuple):
    total_size: Tuple[int, int]
    locations: List[PackedRectangle]



def join_scene_mine(self) -> "TexturesUV":
    """
    Return a new TexturesUV amalgamating the batch.

    We calculate a large single map which contains the original maps,
    and find verts_uvs to point into it. This will not replicate
    behavior of padding for verts_uvs values outside [0,1].

    If align_corners=False, we need to add an artificial border around
    every map.

    We use the function `pack_unique_rectangles` to provide a layout for
    the single map. This means that if self was created with a list of maps,
    and to() has not been called, and there were two maps which were exactly
    the same tensor object, then they will become the same data in the unified map.
    _place_map_into_single_map is used to copy the maps into the single map.
    The merging of verts_uvs and faces_uvs is handled locally in this function.
    """
    maps = self.maps_list()
    heights_and_widths = []
    extra_border = 0 if self.align_corners else 2
    for map_ in maps:
        heights_and_widths.append(
            Rectangle(
                map_.shape[0] + extra_border, map_.shape[1] + extra_border, id(map_)
            )
        )
    merging_plan = pack_unique_rectangles(heights_and_widths)
    C = maps[0].shape[-1]
    single_map = maps[0].new_zeros((*merging_plan.total_size, C))
    verts_uvs = self.verts_uvs_list()
    verts_uvs_merged = []

    for map_, loc, uvs in zip(maps, merging_plan.locations, verts_uvs):
        new_uvs = uvs.clone()
        if loc.is_first:
            self._place_map_into_single_map(single_map, map_, loc)
        do_flip = loc.flipped
        x_shape = map_.shape[1] if do_flip else map_.shape[0]
        y_shape = map_.shape[0] if do_flip else map_.shape[1]

        if do_flip:
            # Here we have flipped / transposed the map.
            # In uvs, the y values are decreasing from 1 to 0 and the x
            # values increase from 0 to 1. We subtract all values from 1
            # as the x's become y's and the y's become x's.
            new_uvs = 1.0 - new_uvs[:, [1, 0]]
            if TYPE_CHECKING:
                new_uvs = torch.Tensor(new_uvs)

        # If align_corners is True, then an index of x (where x is in
        # the range 0 .. map_.shape[1]-1) in one of the input maps
        # was hit by a u of x/(map_.shape[1]-1).
        # That x is located at the index loc[1] + x in the single_map, and
        # to hit that we need u to equal (loc[1] + x) / (total_size[1]-1)
        # so the old u should be mapped to
        #   { u*(map_.shape[1]-1) + loc[1] } / (total_size[1]-1)

        # Also, an index of y (where y is in
        # the range 0 .. map_.shape[0]-1) in one of the input maps
        # was hit by a v of 1 - y/(map_.shape[0]-1).
        # That y is located at the index loc[0] + y in the single_map, and
        # to hit that we need v to equal 1 - (loc[0] + y) / (total_size[0]-1)
        # so the old v should be mapped to
        #   1 - { (1-v)*(map_.shape[0]-1) + loc[0] } / (total_size[0]-1)
        # =
        # { v*(map_.shape[0]-1) + total_size[0] - map.shape[0] - loc[0] }
        #        / (total_size[0]-1)

        # If align_corners is False, then an index of x (where x is in
        # the range 1 .. map_.shape[1]-2) in one of the input maps
        # was hit by a u of (x+0.5)/(map_.shape[1]).
        # That x is located at the index loc[1] + 1 + x in the single_map,
        # (where the 1 is for the border)
        # and to hit that we need u to equal (loc[1] + 1 + x + 0.5) / (total_size[1])
        # so the old u should be mapped to
        #   { loc[1] + 1 + u*map_.shape[1]-0.5 + 0.5 } / (total_size[1])
        #  = { loc[1] + 1 + u*map_.shape[1] } / (total_size[1])

        # Also, an index of y (where y is in
        # the range 1 .. map_.shape[0]-2) in one of the input maps
        # was hit by a v of 1 - (y+0.5)/(map_.shape[0]).
        # That y is located at the index loc[0] + 1 + y in the single_map,
        # (where the 1 is for the border)
        # and to hit that we need v to equal 1 - (loc[0] + 1 + y + 0.5) / (total_size[0])
        # so the old v should be mapped to
        #   1 - { loc[0] + 1 + (1-v)*map_.shape[0]-0.5 + 0.5 } / (total_size[0])
        #  = { total_size[0] - loc[0] -1 - (1-v)*map_.shape[0]  }
        #         / (total_size[0])
        #  = { total_size[0] - loc[0] - map.shape[0] - 1 + v*map_.shape[0] }
        #         / (total_size[0])

        # We change the y's in new_uvs for the scaling of height,
        # and the x's for the scaling of width.
        # That is why the 1's and 0's are mismatched in these lines.
        one_if_align = 1 if self.align_corners else 0
        one_if_not_align = 1 - one_if_align
        denom_x = merging_plan.total_size[0] - one_if_align
        scale_x = x_shape - one_if_align
        denom_y = merging_plan.total_size[1] - one_if_align
        scale_y = y_shape - one_if_align
        new_uvs[:, 1] *= scale_x / denom_x
        new_uvs[:, 1] += (
            merging_plan.total_size[0] - x_shape - loc.x - one_if_not_align
        ) / denom_x
        new_uvs[:, 0] *= scale_y / denom_y
        new_uvs[:, 0] += (loc.y + one_if_not_align) / denom_y

        verts_uvs_merged.append(new_uvs)

    faces_uvs_merged = []
    offset = 0
    for faces_uvs_, verts_uvs_ in zip(self.faces_uvs_list(), verts_uvs):
        faces_uvs_merged.append(offset + faces_uvs_)
        offset += verts_uvs_.shape[0]
        print('merging faces with offset: ', offset)

    print(merging_plan)

    returnTexture = self.__class__(
        maps=[single_map],
        verts_uvs=[torch.cat(verts_uvs_merged)],
        faces_uvs=[torch.cat(faces_uvs_merged)],
        align_corners=self.align_corners,
        padding_mode=self.padding_mode,
        sampling_mode=self.sampling_mode,
    )    
    returnTexture.plan = merging_plan
    return returnTexture


# conda activate pytorch3d3.8 && python testpytorch3d.py

#device = torch.device("cuda:0")
device = torch.device("cpu")
# Use an ico_sphere mesh and load a mesh from an .obj e.g. model.obj
#sphere_mesh = ico_sphere(level=3)
# cow_mesh = load_objs_as_meshes(["/media/raid/Home/Data/HPSTA/adv/data/cow_mesh/cow.obj"], device=device)
# verts = cow_mesh.verts_packed()
# verts = -2 * verts;
# verts[:,0] = 0;
# verts[:,2] = 0;
# cow_mesh.offset_verts_(verts)
#cow_mesh = load_objs_as_meshes(["/media/raid/Home/Data/HPSTA/adv/data/barrel_mesh/big_cone.obj"], device=device)
# offset = torch.zeros((3,), device=device, dtype=torch.float32)
# offset[2] = -5
# cow_mesh = cow_mesh.offset_verts_(offset)


# test_mesh = load_objs_as_meshes(["/media/raid/Home/Data/HPSTA/adv/201408272252_09.obj"], device=device)

print("entering load")
test_mesh = load_objs_as_meshes_mine(["/media/raid/Home/Data/HPSTA/adv/data/exportedOrig/roadrunnerScene.obj"], device=device)
#test_mesh = load_objs_as_meshes_mine(["/media/raid/Home/Data/HPSTA/adv/data/trianglegrasspatch_preproc/trianglegrasspatch_proc.obj"], device=device)

print("entering save")
mapsnumpy = np.squeeze(test_mesh.textures.maps_padded().cpu().numpy())
print(mapsnumpy.shape)
imageio.imwrite('maps1.png',mapsnumpy)



# test_mesh = load_objs_as_meshes([
#     "/media/raid/Home/Data/HPSTA/adv/data/exportedOrig/roadrunnerScene/roadrunnerScene_0.obj",
#     "/media/raid/Home/Data/HPSTA/adv/data/exportedOrig/roadrunnerScene/roadrunnerScene_1.obj",
# "/media/raid/Home/Data/HPSTA/adv/data/exportedOrig/roadrunnerScene/roadrunnerScene_2.obj",
# "/media/raid/Home/Data/HPSTA/adv/data/exportedOrig/roadrunnerScene/roadrunnerScene_3.obj",
# "/media/raid/Home/Data/HPSTA/adv/data/exportedOrig/roadrunnerScene/roadrunnerScene_4.obj"
# ], device=device)



# mesh = join_meshes_as_scene([test_mesh,cow_mesh])
#mesh = join_meshes_as_scene([test_mesh,cow_mesh])
mesh = test_mesh;

K = torch.zeros((1, 4, 4), device=device, dtype=torch.float32)
K[:,0,0] =       -1
#K[:,0,2] =       3.5483500e+02
K[:,1,1] =       -1
#K[:,1,2] =       2.3208600e+02
K[:,2,2] =       1
K[:,2,3] =       -0.001
K[:,3,2] =       1
print(K)

print(mesh);
#IO().save_mesh(mesh, "final_model.obj")
save_obj("final_model.obj", verts=mesh.verts_list()[0], faces=mesh.faces_list()[0],
    normals=mesh.verts_normals_list()[0], faces_normals_idx=mesh.faces_list()[0],
    faces_uvs=mesh.textures.faces_uvs_list()[0], verts_uvs=mesh.textures.verts_uvs_list()[0], texture_map=mesh.textures.maps_list()[0])



#R, T = look_at_view_transform(2.7, 0, 180) R=R, T=T, K=K
dist = 20
cameras = FoVPerspectiveCameras(device=device, 
K=K, T=torch.tensor([[0, 0, dist ]]), zfar=torch.tensor([dist * 1.5]))
#cameras = OpenGLPerspectiveCameras(device=device, T=torch.tensor([[32.5993, 82.1822, -198.781 ]]))
#cameras = OpenGLPerspectiveCameras(device=device)

# Define the settings for rasterization and shading. Here we set the output image to be of size
# 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
# and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
# the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
# explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
# the difference between naive and coarse-to-fine rasterization. 
raster_settings = RasterizationSettings(
    image_size=[2068, 2068],  #image_size=[480,720], 
    blur_radius=0.0, 
    faces_per_pixel=5, 
)

# Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
# -z direction. 
lights = PointLights(device=device, location=[[0.0, 0.0, dist+10]])

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
imageio.imwrite("render1.png",images[0, ..., :3].cpu().numpy())

sys.exit()

plt.figure(figsize=(10, 10))
mapsnumpy = np.squeeze(mesh.textures.maps_padded().cpu().numpy())
print(mapsnumpy.shape)
imageio.imwrite('maps1.png',mapsnumpy)

sys.exit()

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