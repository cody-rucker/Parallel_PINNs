from FeedForward import *
from geometry_2D import BoundaryCurve
from IBVP import *
import torch
from ptflops import get_model_complexity_info
from pympler import asizeof 

# exact solution
def uexact(z):
    x, t = z.split(1, dim=1)
    return torch.exp(-(x-t)**2)
def pde_src(z):
    return 0 * uexact(z)
    
class SubNetwork:
    def __init__(self, N_layers, activations, x_int, t_int):
        self.N_net = Feedforward(N_layers, activations).to(device)

        # pytorch opbjects for tracking and updating network parameters
        self.params = list(self.N_net.parameters())
        self.opt = torch.optim.LBFGS(self.params, lr=0.1, max_iter=250,
                                     max_eval=None, tolerance_grad=1e-05,
                                     tolerance_change=1e-09, history_size=150,
                                     line_search_fn='strong_wolfe')
        #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='min', patience=5)

        # domain intervals x_int =[x0, x1] t_int=[ti, tf]
        self.xs = x_int
        self.ts = t_int

        # domain objects for generating training points
        lft, rgt = [torch.tensor([[x]]) for (x) in self.xs]
        self.lft_bdry = BoundaryCurve(['line', lft, lft])
        self.rgt_bdry = BoundaryCurve(['line', rgt, rgt])
        self.interior = BoundaryCurve(['line', lft, rgt])
    
    def subnet_complexity(self):
        ''' Method for computing network complexity metrics.
        
        Size is given in bytes and FLOPs are given in terms of 
        MAC (Multiply-ACCumulate MAC=a*b+c counts as two FLOPs)
        '''
        self.size = asizeof.asizeof(self.N_net)
        self.flops, self.params = get_model_complexity_info(self.N_net, (1,2), as_strings=True, print_per_layer_stat=True)
        
    def N(self, z):
        ''' Network function enforcing hard initial conditions. '''
        x, t = z.split(1, dim=1)
        t0 = torch.zeros(x.shape).to(device)
        z0 = torch.cat([x, t0], dim=1)
        u0 = uexact(z0)

        return u0 + t*self.N_net(z)
        
    def set_interface(self, pts=1):
        self.rgt_bdry_pts = pts
        self.IC = InterfaceCondition(self.rgt_bdry)
    
    def set_lft_bdry(self, f=None, pts=1):
        self.lft_bdry_pts = pts   
        self.BC = BoundaryCondition(self.lft_bdry, NNet=self.N, data_function=f, bc='dirichlet', t_initial=self.ts[0], t_final=self.ts[1]) 
        
    def set_interior(self, pde_src, pts=1):
        self.interior_pts = pts
        self.PDE = PDE(self.interior, NNet=self.N, data_function=pde_src, t_initial=self.ts[0], t_final=self.ts[1]) 
    
    
    def sample_domains(self):
        if self.BC.data_function is not None:
            self.BC.sample_domain(n_samples=self.lft_bdry_pts)
        
        self.IC.sample_domain(n_samples=self.rgt_bdry_pts)
        self.IC.test_data = self.N(self.IC.sampled_points).detach()
        self.PDE.sample_domain(n_samples=self.interior_pts)
    
    def update_bdry_data(self, wrapped_data):
        pts = wrapped_data['pts']
        data = wrapped_data['data']
        self.BC.sampled_points = pts
        self.BC.test_data = data    
            
    
    def myloss(self):
        '''Define loss function for training. '''
        loss = self.PDE.loss().view(1)
        loss = loss + self.BC.loss().view(1)

        return loss

    def closure(self):
        '''Closure necessary for LBFGS in pytorch. '''
        self.opt.zero_grad(set_to_none=True)
        loss = self.myloss()

        loss.backward()
        return loss

    def train(self):
        '''A single training step. '''
        #self.BC.sample_domain(n_samples=self.bdry_pts)
        #self.PDE.sample_domain(n_samples=self.interior_pts)

        running_loss = 0.0

        self.opt.step(self.closure)
        L = self.closure()
        running_loss += L.item()

        #self.scheduler.step(L)
        
        #print('running loss:', running_loss)
        #print('learning rate:', self.scheduler.state_dict()['_last_lr'])
        
    def indicator(self, z):
        '''Indicator function for isolating subnetwork over entire spatial domain. '''
        x, t = z.split(1, dim=1)
        return torch.where((x >= self.xs[0]) & (x < self.xs[1]), 1.0, 0.0)
