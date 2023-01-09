Based on our previous discussion, I am going to try to
1. render a scene
2. render something in the scene, maybe a person
3. compute something ground truth like, maybe a bounding box, for the person in the scene.
4. pull in a network from a model zoo and run it on the scene
5. compare the ground-truth to the network prediction and compute a single frame loss like thing
6. take the derivative of the loss, with respect to object position
7. gradient ascent on loss, see if I can make the network do worse


Mesh from [NREC ag project](https://www.nrec.ri.cmu.edu/solutions/agriculture/other-agriculture-projects/human-detection-and-tracking.html)

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