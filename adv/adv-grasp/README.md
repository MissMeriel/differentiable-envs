## Running Attacks using run_gqcnn.py

#### 1. Load in a pytorch mesh of type `pytorch3d.structures.meshes.Mesh` for a grasp object to be perturbed.

Example: use Renderer instance from render.py with a path `DATA_FILE` to a .obj file of the object.

`r = Renderer()`

`mesh, _ = r.render_object(DATA_FILE, display=False)`

#### 2. Select a grasp (or batch of grasps) of the mesh, specifically as an instance of the `Grasp` class from select_grasp.py.

Example 1: select a new grasp (or batch of grasps) using select_grasp.py.

`g = Grasp.sample_grasps(obj_f=DATA_FILE, num_samples=1, renderer=r, min_qual=0.002, max_qual=0.005)`

Example 2: read in a known/existing grasp(s) using select_grasp.py.

`g = Grasp.read("grasp-dataset2/grasp-batch.json")`

#### 3. Intialize an instance of the `Attack` class from run_gqcnn.py.

Example:
`model = KitModel("weights.npy")`

`model.eval()`

`run1 = Attack(num_plots=10, steps_per_plot=10, model=model, renderer=r, oracle_method="pytorch")`

Note on parameters:

- `num_plots` is the number of plots that will be visualized from the attack.
- `steps_per_plot` is the number of steps between each plot in the attack. Thus, the maximum number of steps for an attack is `num_plots` * `steps_per_plot`.
- `model` is the model being attacked, namely the GQCNN model loaded by `KitModel` from gqcnn_pytorch.py. 
- `renderer` is an instance of the `Renderer` class from render.py.
- `oracle_method` defaults to `pytorch` to use the PyTorch oracle implementation, but can also be `dexnet` to use the dex-net oracle instead.

#### 4. Call attack method on Attack instance.

Example:

`run1.attack(mesh=mesh, grasp=g, dir=dir, lr=1e-05, momentum=None, loss_alpha=None, method=AttackMethod.ORACLE_GRAD, epsilon=None)`

Note on parameters:

- `dir` is a path to a directory where the visualization results, logging file, etc. should be saved.
- `loss_alpha` is an optional float value between 0.0 and 1.0 for how heavily to weight the oracle evaluation compared to the model prediction. 
- `method` is an AttackMethod from run_gqcnn.py specifying what method of attack to use. 
- `epsilon` is an optional float value between 0.0 and 1.0 for attack methods ORACLE_GRAD and NO_ORACLE_GRAD. When the difference between the model prediction and the oracle evaluation reaches or surpasses this value, the adversarial loop exits. 

**Also see main.py for an example on how to run attacks.**

pip install Pyro4
pip install trimesh

main.py 
grasp
    momentum/learning rate
        attack method

TODO: 

compute: volume, convex hull volume, mesh normal consistency
options:
1. every time - maybe calc_loss, in run_gqcnn
2. snapshots (every 10) - snapsot, in run_gqcnn

select_grasp is responsible for converting type of grasps and imports my library
can put a wrapper for volume into select grasp that takes pytorch meshes

check how much space we're using