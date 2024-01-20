from AdvectionSubnets import SubNetwork
from IBVP import *
import torch
from torch import nn
from tqdm import trange
import time
from NetworkViz import *
        
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
torch.set_default_dtype(torch.float64)

# exact solution
def uexact(z):
    x, t = z.split(1, dim=1)
    return torch.exp(-(x-t)**2)
def pde_src(z):
    return 0 * uexact(z)

        
N_layers = np.array([2, 8, 8, 8, 1])
activations = [nn.SiLU(), nn.SiLU(), nn.SiLU()]

# domain discretization/decomposition
δx = 0.1
grid = np.arange(0, 1+δx, δx)
sub_domains = [[grid[i], grid[i+1]] for i in range(len(grid)-1)]


  
# each subnet should be given to a different processor  
# 0th process handles boundary data at x=0 
networks = []
for k in sub_domains:
    networks.append(SubNetwork(N_layers, activations, k, [0.0, 1.0]))
    
# each process should initialize subnet boundary conditions   
for k in range(len(networks)):
    if k==0:
        networks[k].set_lft_bdry(f=uexact, pts=24)
    else:
        networks[k].set_lft_bdry(pts=24)
    networks[k].set_interior(pde_src, pts=96)
    networks[k].set_interface(pts=24)

# each process should use the subnet to generate boundary data
for k in networks:
    k.sample_domains()    


# need to pass interface data to neighboring network to use as 
# data on their left boundary.
# Data pass occurs at each training iteration
# before evaluating the loss function.
# should be able to pass the data with MPI
for i in range(len(networks[:-1])):
    new_data = networks[i].IC.wrap_data()
    networks[i+1].update_bdry_data(new_data)


total_time = 0
 
for i in trange(20):
    t0 = time.time()
    
    for k in networks:
        k.sample_domains()    

    # need to pass interface data to neighboring network to use as 
    # data on their left boundary
    for i in range(len(networks[:-1])):
        new_data = networks[i].IC.wrap_data()
        networks[i+1].update_bdry_data(new_data)
    
    for k in range(len(networks)):
        networks[k].train()
    
    t1 = time.time()
    
    err = []
    for k in networks:
        eval_pts = k.PDE.sample_domain(n_samples=1000)
        e = torch.nn.MSELoss()(k.N(eval_pts), uexact(eval_pts))
        err.append(e.cpu().detach().numpy())
    
    for k in networks:
        idx = networks.index(k)
        print("MSE of subnet {:d}: {:.3e}".format(idx, err[idx-1]))
    print("Training run time: {:.3f}s".format(t1-t0))
    print("--------------------------")
    total_time += t1-t0
print("Total training time: {:.3f}s".format(total_time))    



    
t0, t1 = [0.0, 1.0]
x0, x1 = [0.0, 1.0]
      
t_grid = torch.arange(t0, t1, 0.005).unsqueeze(1).to(device)

x_grid = torch.arange(x0, x1, 0.005).unsqueeze(1).to(device)

visualize_soln(networks, uexact, x_grid, t_grid)


