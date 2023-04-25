mesh = join_meshes_as_scene([self.test_mesh.clone(),cow_mesh])

from pytorch3d.io import IO
IO().save_mesh(data=mesh, path="./test_save_mesh_cow.obj")

from pytorch3d.io import save_obj
save_obj(f="./test_save_obj_cow.obj", 
    verts=cow_mesh.verts_packed(), 
    faces=cow_mesh.faces_packed(),
    texture_map=cow_mesh.textures
)
