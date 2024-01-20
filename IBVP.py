from geometry_2D import *
import torch

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

mse = torch.nn.MSELoss()


class Condition:
    """A class for automating tasks used to train the PINN loss function."""

    def __init__(self, domain, data_function=None, t_initial=0.0, t_final=1.0):
        self.domain = domain
        self.data_function = data_function
        self.test_data = None
        self.t_initial = t_initial
        self.t_final = t_final
        self.prev_loss = [torch.tensor(0.0)]
        self.sampled_points = None

    def sample_domain(self, n_samples=1, uniform=False):
        """Generate input data on a subdomain and generate test data."""

        pt = self.domain.get_point(n_samples, uniform=uniform)
        τ = torch.ones((n_samples, 1)).uniform_(
            self.t_initial, self.t_final)

        pt = torch.cat([pt, τ], dim=1)

        if self.data_function is not None:
            self.test_data = self.data_function(pt).to(device).detach()

        self.sampled_points = pt.to(device)

        return pt.to(device)

    def loss(self):
        """Compute the error given of a condition function w.r.t test_data."""

        output = self.condition(self.sampled_points)
        loss = mse(output, self.test_data)
        self.prev_loss.append(loss.detach())
        return loss


class InterfaceCondition(Condition):
    ''' A subclass of Condition for generating interface data.

    This condition stores generated data at the right boundary interface
    which will be passed to neighboring networks for use in subnet training.
    '''
    def __init__(self, domain, t_initial=0.0, t_final=1.0):
        Condition.__init__(self, domain, None, t_initial, t_final)

        #self.NNet = NNet
        #self.condition = self.NNet

    def set_bdry_data(self, f):
        '''Create interface data for passing to neighboring subnetworks. '''
        if self.sampled_points is not None:
            self.test_data = f(self.sampled_points)
            
    def set_condition(self, f):
        self.condition = f
        
    def wrap_data(self):
        '''Reduce the package size.?'''
        bdry_samples = {'pts': self.sampled_points, 'data': self.test_data}
        return bdry_samples


class BoundaryCondition(Condition):
    ''' A Condition subclass for enforcing Dirichlet and Neumann b.c.'''
    def __init__(self, domain, NNet=None, data_function=None, bc='dirichlet',t_initial=0.0, t_final=1.0):
        Condition.__init__(self, domain, data_function, t_initial, t_final)

        self.bc = bc
        self.NNet = NNet

        if self.bc == 'dirichlet':
            self.condition = self.NNet

        elif self.bc == 'neumann':
            self.condition = self.neumann


    def neumann(self, z):
        z.requires_grad = True

        output = self.NNet(z)
        output = output.sum()
        grad, = torch.autograd.grad(output, z, create_graph=True)

        dx, dy, dt = grad.split(1, dim=1)

        n = self.domain.normal().to(device)
        nx, ny = n.split(1, dim=1)

        τ = nx*dx + ny*dy

        return μ * τ


class PDE(Condition):
    def __init__(self, domain, NNet=None, data_function=None, t_initial=0.0, t_final=1.0):
        Condition.__init__(self, domain, data_function=data_function, t_initial=t_initial, t_final=t_final)
        self.NNet = NNet
        #self.domain = domain
        #self.ti, self.tf = self.domain.t_bounds
        #self.tf = self.domain.tf

    def condition(self, z):
        z.requires_grad = True

        u = self.NNet(z)
        u = u.sum()

        grad, = torch.autograd.grad(u, z, create_graph=True)
        dx, dt = grad.split(1, dim=1)  # gradient can be split into parts

        Δ = dt + dx
        return Δ
