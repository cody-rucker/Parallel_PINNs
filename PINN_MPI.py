from AdvectionSubnets import SubNetwork
from IBVP import *
import torch
from torch import nn
#from tqdm import trange
import time
from NetworkViz import *
from mpi4py import MPI
import pickle
import sys
import os



if __name__ == '__main__':
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

            
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
    δx = 1 / comm.size
        
    grid = np.arange(0, 1+δx, δx)
    sub_domains = [[grid[i], grid[i+1]] for i in range(len(grid)-1)]

    subnet = SubNetwork(N_layers, activations, sub_domains[rank], [0.0, 1.0])

    if rank==0:
        subnet.set_lft_bdry(f=uexact, pts=24)
    else:
        subnet.set_lft_bdry(pts=24)

    subnet.set_interior(pde_src, pts=96)
    subnet.set_interface(pts=24)    


    total_time = 0
    commTotal = 0

    runTime = 0



    for i in range(20):
        t0 = time.time()
        commTime = 0
        
        comm.Barrier()
        start = MPI.Wtime() 
        if rank==0:
            print(comm.size)
            subnet.sample_domains()
            data = subnet.IC.wrap_data()
                
            commTime -= MPI.Wtime()    
            comm.send(data, dest=1)
            commTime += MPI.Wtime()
            
            commTotal += commTime
            
            subnet.train()
            
            
        if rank==(comm.size-1):
            subnet.sample_domains()
            
            commTime -= MPI.Wtime()
            bdry_data = comm.recv(source=(rank-1))
            commTime += MPI.Wtime()
            
            commTotal += commTime
            
            subnet.update_bdry_data(bdry_data)
            subnet.train()
            
            
        if 0 < rank < (comm.size-1):
            subnet.sample_domains()
            data = subnet.IC.wrap_data()
            
            commTime -= MPI.Wtime()
            comm.send(data, dest=(rank+1))
            bdry_data = comm.recv(source=(rank-1))
            commTime += MPI.Wtime()
            
            commTotal += commTime
            
            subnet.update_bdry_data(bdry_data)
            subnet.train()
            
        end = MPI.Wtime() 
        runTime += end-start    
        t1 = time.time()
        eval_pts = subnet.PDE.sample_domain(n_samples=1000)
        e = torch.nn.MSELoss()(subnet.N(eval_pts), uexact(eval_pts))
        total_time += t1-t0
        
        print("MSE of subnet {:d}: {:.3e}, runtime={:.3f}s".format(rank, e, t1-t0))
        comm.Barrier()  


    runTime = comm.reduce(runTime, op=MPI.MAX, root = 0 )
    commTotal = comm.reduce(commTotal, op=MPI.MAX, root = 0 )


            
    def read_or_new_pickle(path):
        default = {}
        for k in 2**np.arange(2, 9, 1):
            default[str(k)] = [] 
            
        if os.path.isfile(path):
            with open(path, "rb") as f:
                try:
                    return pickle.load(f)
                except Exception: # so many things could go wrong, can't be more specific.
                    pass 
        with open(path, "wb") as f:
            pickle.dump(default, f)
        return default
        
    '''    
    if rank==0:
        print("total time: {}".format(runTime))   
        print("total comm time: {}".format(commTotal))  
        
        num_procs = comm.size
        T = runTime
        T_comm = commTotal  
        
        fname = './outputs/weak_scaling_data.pickle'
        os.makedirs(os.path.dirname(fname), exist_ok=True)

        data = read_or_new_pickle(fname)
        
        data[str(num_procs)].append(T)
        data[str(num_procs)].append(T_comm)
        #data[1].append(T)
        #data[3].append(T_comm) 
        
        # save
        print('Saving to ', fname)

        with open(fname, 'wb') as f:
            pickle.dump(data, f)

    '''
    #fname = './outputs/weak_scaling_data.pickle'
    #os.makedirs(os.path.dirname(fname), exist_ok=True)

    #default = [1, 2, 3]

    #data = read_or_new_pickle(fname)

    #for i in range(len(data)):
    #    data[i].append(time_data[i])  
        
    # save

    #print('Saving to ', fname)

    #with open(fname, 'wb') as f:
    #    pickle.dump([uht, cht, sht, ave], f)


    t0, t1 = [0.0, 1.0]
    x0, x1 = [0.0, 1.0]
          
    t_grid = torch.arange(t0, t1, 0.005).unsqueeze(1).to(device)

    x_grid = torch.arange(x0, x1, 0.005).unsqueeze(1).to(device)

    uNN = []
    uE = []
    uerr = []


    
    for i in range(len(t_grid)):
        if comm.rank==0:
            sol = np.zeros_like(x_grid.numpy())
        else:
            sol = None
        
        t = torch.ones(x_grid.shape).to(device) * t_grid[i]

        z_grid = torch.cat([x_grid, t], dim=1).to(device)
        
        u_net = subnet.N(z_grid) * subnet.indicator(z_grid)
        u_net = u_net.cpu().detach().numpy()
        comm.Barrier()
        comm.Reduce(
            [u_net, MPI.DOUBLE],
            [sol, MPI.DOUBLE],
            op=MPI.SUM,
            root = 0
            )
        
        if rank==0:
            uex = uexact(z_grid).cpu().detach().numpy()
            #L2_err = np.square(sol - uex)
            #L2_err = np.sum(L2_err)
            #L2_err = np.sqrt(L2_err)
            
            uNN.append(sol)
            uE.append(uex)
            uerr.append(sol - uex)
            
    if rank==0:
        t0, t1 = [0.0, 1.0]
        x0, x1 = [0.0, 1.0]
              
        t_grid = torch.arange(t0, t1, 0.005).unsqueeze(1).to(device)

        x_grid = torch.arange(x0, x1, 0.005).unsqueeze(1).to(device)

    
        Main.xfine = x_grid.cpu().detach().numpy()#x2_grid
        Main.tfine = t_grid.cpu().detach().numpy()

        Main.uNet = uNN
        Main.uExact = uE
        Main.err = uerr

        # visualize solution
        Main.eval("""
        using Plots
        pyplot()
        theme(:dark)    # the most powerful theme

        i = 200
           plot(xfine, uNet[i], size=(750, 500),dpi=300,
            ylims=(0.25, 1.2), lw=1.5, color=0, legend=:topright,
            label = "network", xaxis=(""), linestyle=:dashdot)#,
            #right_margin = 30Plots.mm)
           plot!(xfine, uExact[i], label="exact", color=3)
           
           plot!(twinx(),xfine, err[i], ylims = (-0.005, 0.005),
                 yaxis=("Error\n |u_exact - u_approx|"), color=:red,
                 y_guidefontcolor=:red, legend=false)
                 
              
        savefig("advection_200s.png")
        """)

    
