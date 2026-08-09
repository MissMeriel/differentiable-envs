from ll4ma_opt.problems import Constraint, Problem
from run_gqcnn import AttackMethod
import torch
from functools import partial
from tqdm import tqdm
import numpy as np

class opt_w_coll(Problem):
    def __init__(self, initial, loss_func, loss_keys, size):
            
        self.size_val = size
        super().__init__()   
        self.initial_solution = torch.flatten(initial).reshape([-1,1])
        self.loss_func = loss_func
        self.loss_keys_raw = loss_keys
        if AttackMethod.SELF_COLLISION_LOSS_DOWN in loss_keys:
            self.loss_keys = [key for key in loss_keys if AttackMethod.SELF_COLLISION_LOSS_DOWN != key]
            self.inequality_constraints=[(coll_constraint(self))]
            self.equality_constraints = []
        else:
            self.loss_keys = loss_keys
            self.equality_constraints = []
        
        self.index = -1
        self.unmake_tensor = self.do_nothing
        self.make_tensor = self.do_nothing
        self.max_bounds = torch.full_like(self.initial_solution,0.01)
        self.min_bounds = torch.full_like(self.initial_solution,-0.01)
        self.grad_cache = {}

    def size(self):
        return self.size_val

    def tensor_cost(self, x):
        self.x = x
        self.grad_cache = {}
        self.loss_dict, self.perturbed_by_param, self.eps_check = self.loss_func(param=x.clone().reshape([-1,3]), index=self.index)
        loss_stack = torch.stack([self.loss_dict[key].flatten() for key in self.loss_keys])
        self.loss = torch.sum(loss_stack,dim=0)
        if torch.any(loss_stack == 0):
            self.loss = torch.full_like(self.loss,10e9) # big value
        return self.loss
    
    def populate_grad_cost(self):
        for key in self.loss_keys:
            self.grad_cache[key] = torch.autograd.grad(
                [self.loss_dict[key]], self.x, allow_unused=True,retain_graph=True)[0]
            if self.grad_cache[key] is None:
                self.grad_cache[key] = torch.zeros_like(self.x)
            
    def populate_grad_cons(self):
        self.grad_cache[AttackMethod.MIN_DIST_COLL] = torch.autograd.grad(
            [-self.loss_dict[AttackMethod.MIN_DIST_COLL]], self.x, allow_unused=True,retain_graph=True)[0]

    def do_nothing(self,x):
        return x
    
    def cost_gradient(self, x):
        if not hasattr( self,'x') or self.x is not x:
            self.tensor_cost(x)

        if not set(self.loss_keys).issubset(self.grad_cache.keys()):
            self.populate_grad_cost()

        return torch.sum(torch.stack([self.grad_cache[key].flatten() for key in self.loss_keys]),dim=0).reshape([-1,1])
    
    def cons_gradient(self, x):
        if not hasattr( self,'x') or self.x is not x:
            self.tensor_cost(x)

        if not AttackMethod.MIN_DIST_COLL in self.grad_cache.keys():
            self.populate_grad_cons()

        return self.grad_cache[AttackMethod.MIN_DIST_COLL].reshape([-1,1]) * 1/self.loss_dict[AttackMethod.REF_DIST_COLL] 
    
class coll_constraint(Constraint):
# create one of these in problem constructor
# update the cached input and tensor from tensor_error
# check equality of input with "is"
    def __init__(self, problem):
        self.problem = problem
        super().__init__()
    def tensor_error(self, x):
        if not hasattr( self.problem,'x') or self.problem.x is not x:
            self.problem.tensor_cost(x)
        dist = self.problem.loss_dict[AttackMethod.MIN_DIST_COLL]
        if dist <= self.problem.loss_dict[AttackMethod.REF_DIST_COLL]/1000:
            return torch.full_like(dist,10e9) # big number, very violated
        else:
            return 1 - dist/self.problem.loss_dict[AttackMethod.REF_DIST_COLL]
    def error_gradient(self, x):
        return self.problem.cons_gradient(x)
    
class logging_manager():
    def __init__(self, attack, loss_keys, grasp, mesh, dir, orig_pdim, logfile):
        param = torch.zeros(mesh.verts_packed().shape, device=mesh.device, requires_grad=True)
        size = torch.numel(mesh.verts_packed())
        
        
        self.grasp = grasp
        self.attack = attack
        loss_func = partial(self.attack.perturb,mesh=mesh, grasp=grasp, method=loss_keys)
        self.problem = opt_w_coll(param, loss_func, loss_keys, size)
        self.mesh = mesh
        self.dir = dir
        self.orig_pdim = orig_pdim
        self.logfile = logfile
        self.pbar = tqdm(total=self.attack.num_steps,desc='attack outer iterations',leave=False)
        self.update(-1,param,torch.tensor(0),torch.tensor(0),torch.tensor(0))

    def update(self,i,x, mult, quad_co, alpha, final=False):
        if not hasattr( self.problem,'x') or self.problem.x is not x:
            self.problem.tensor_cost(x)
        self.pbar.update(1)
        if AttackMethod.SELF_COLLISION_LOSS_DOWN in self.problem.loss_keys_raw:

            tqdm.write(f'dist {self.problem.loss_dict[AttackMethod.MIN_DIST_COLL]} ref {self.problem.loss_dict[AttackMethod.REF_DIST_COLL]}')
        # #### DELETE ME
        # all_dist, coords = self.attack.mesh_properties(x.reshape([-1,3]))
        # np_dist = all_dist.numpy(force=True)
        # np.savetxt(f'distances_{i+1}.txt',np_dist)
        # ####

        self.attack.loss_mag.append(torch.zeros((1,len(self.problem.loss_keys_raw)+3), device=self.mesh.device))
        self.problem.populate_grad_cost()
        self.attack.param_grad_list = []

        self.attack.optim_status.append(np.array([[self.problem.index+1, self.problem.index+1, self.problem.index+1, 0]]))
        for ind,key in enumerate(self.problem.loss_keys_raw):
            self.attack.loss_mag[-1][0,ind] = self.problem.loss_dict[key]
            if key == AttackMethod.SELF_COLLISION_LOSS_DOWN:
                self.problem.populate_grad_cons()
                grad = self.problem.grad_cache[AttackMethod.MIN_DIST_COLL]
            else:
                grad = self.problem.grad_cache[key]
            self.attack.param_grad_list.append(grad)
        
        if hasattr(self,'old_x'):
            difference =torch.linalg.vector_norm(x.flatten()-self.old_x.flatten())
        else:
            difference = torch.linalg.vector_norm(x)
        self.old_x = x
        self.attack.update_scale.append(self.attack.loss_mag[-1])
        self.attack.qual_measures_raw.append(torch.tensor(((self.attack.track_qual["gqcnn prediction"][-1],
                                        self.attack.track_qual["minWeight quality"][-1], self.attack.track_qual["oracle quality"][-1], 
                                        self.attack.track_qual["dist loss"][-1], self.attack.track_qual["did hit object"][-1], 
                                        self.attack.track_qual["min dist"][-1], self.attack.track_qual["l2 norm"][-1], 
                                        self.attack.track_qual["laplacian smoothing"][-1],
                                        difference.item(), mult.item(), quad_co,alpha.item()),) ,
                                        device = self.mesh.device))
        if final or ((self.problem.index+1) % self.attack.steps_per_plot == 0 and (self.problem.index+1)>0) or (self.problem.index == self.attack.num_steps):
            mesh2 = self.problem.perturbed_by_param.clone()
            self.attack.snapshot(mesh=mesh2, grasp=self.grasp, dir=self.dir, iteration=self.problem.index, orig_pdim=self.orig_pdim, logfile=self.logfile, final_snapshot=final)
            if self.problem.index == self.attack.num_steps:
                self.pbar.close 
        if not hasattr(self,'best_loss') or self.problem.loss < self.best_loss:
            self.best_loss = self.problem.loss
            mesh2 = self.problem.perturbed_by_param.clone()
            self.attack.snapshot(mesh=mesh2, grasp=self.grasp, dir=self.dir, iteration=self.problem.index, orig_pdim=self.orig_pdim, logfile=self.logfile, name='best')
        
        self.problem.index = i+2
