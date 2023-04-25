Based on our previous discussion, I am going to try to
1. render a scene
2. render something in the scene, maybe a person
3. compute something ground truth like, maybe a bounding box, for the person in the scene.
4. pull in a network from a model zoo and run it on the scene
5. compare the ground-truth to the network prediction and compute a single frame loss like thing
6. take the derivative of the loss, with respect to object position
7. gradient ascent on loss, see if I can make the network do worse

Mesh from [NREC ag project](https://www.nrec.ri.cmu.edu/solutions/agriculture/other-agriculture-projects/human-detection-and-tracking.html)

from my first project at nrec, creating a big offroad person dataset

this is a scene without any people, so I wanted to insert one

Used a pretty slow structure from motion library to turn it into a mesh, hoping one day I could do something like this

cruft

pytorch3d-supported filetypes are either .obj or .ply

The problem with exporting single simulator environments is that I need to do it through Blender or Unreal or something similar, and to do that I have to regenerate the environment in some programmatic way in that mesh editor. I might have luck with RoadRunner which Carla uses to generate maps but I need the university to make a license available to me

roadrunner environment is consumable by pytorch3D as a .obj 

Using meshlab to inspect environment meshes

Mesh and texture troubleshooting
- https://forum.keyshot.com/index.php?topic=832.0
- http://madscientistsecretbase.com/?page_id=34
- https://pytorch3d.readthedocs.io/en/latest/modules/renderer/cameras.html#pytorch3d.renderer.cameras.FoVPerspectiveCameras

what about extending to another robot task, maybe in gazebo
https://berkeleyautomation.github.io/dex-net/ SUT
https://github.com/PickNikRobotics/deep_grasp_demo gazebo environment
That's a great idea, that might preclude us from coming up with more complex environments; by "more complex" I think I really mean larger

I'm going to try to debug pytorch3d this week
list of changes, so that I don't forget as I go through figuring out what matters:
1. opengl camera
2. set camera at the origin
3. removed the normals from the faces
4. removed the bump and Ks from the mtl file

small update w.r.t rendering: RoadRunner meshes are not centered at the origin, and pytorch3d's `look_at_view_transform` assumes you are looking at the origin. There's a bug in the `dist` parameter that they [claim to have fixed](https://github.com/facebookresearch/pytorch3d/issues/191), but doesn't seem fixed. I don't think they fixed it because if it were fixed, we wouldn't need the following change. To view the non-origin-centered mesh you have to add parameters `at` and `up` similar to this:
```R, T = look_at_view_transform(80, 0, 180, at=((10.741272, -357.137512, 0.1),), up=((0,1,0),), degrees=True)```
I picked a random vertex from the roadrunnertest2.obj file for the `at` parameter. The default `up` parameter seems to hold for both pytorch3d and roadrunner.

I've tried to render textures properly for meshes that use procedural textures (like the meshes produced by RoadRunner). That is a little more difficult to reason about.  But, at least we have a better idea of why the camera has been rendering the scene so weirdly.


## Useful documentation on .obj files

http://paulbourke.net/dataformats/obj/

http://paulbourke.net/dataformats/mtl/



## Useful references for how pytorch3D interfaces with .obj files

https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/io/obj_io.py#L667

https://github.com/facebookresearch/pytorch3d/issues/1277

https://github.com/facebookresearch/pytorch3d/issues/913

https://pytorch3d.readthedocs.io/en/latest/modules/renderer/mesh/textures.html?highlight=TexturesUV#pytorch3d.renderer.mesh.textures.TexturesUV

https://pytorch3d.readthedocs.io/en/v0.6.0/modules/io.html#pytorch3d.io.IO.save_mesh

https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/structures/meshes.html#join_meshes_as_batch

https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/renderer/mesh/textures.py

https://pytorch3d.readthedocs.io/en/latest/modules/structures.html#pytorch3d.structures.Meshes.submeshes

https://discuss.pytorch.org/t/ptorch3d-loss-on-projected-image/82560

https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/renderer/cameras.html#look_at_view_transform

https://pytorch3d.readthedocs.io/en/latest/modules/renderer/cameras.html

https://www.andrew.cmu.edu/user/chenhao2/

https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/render_textured_meshes.ipynb

## Useful resources on gradients

https://pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html