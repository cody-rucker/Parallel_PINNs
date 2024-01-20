from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
import torch

device = torch.device("cpu")
torch.set_default_dtype(torch.float64)

def visualize_soln(subnets, exact_soln, x_grid, t_grid):

    
    uNN = []
    uE = []
    uerr = []

    for i in range(len(t_grid)):
        t = torch.ones(x_grid.shape).to(device) * t_grid[i]

        z_grid = torch.cat([x_grid, t], dim=1).to(device)
        u_net = 0
    
        for k in subnets:
            u_net = u_net + k.N(z_grid) * k.indicator(z_grid)

        u_net = u_net.cpu().detach().numpy()
        uex = exact_soln(z_grid).cpu().detach().numpy()
    
        uNN.append(u_net)
        uE.append(uex)
        uerr.append(u_net - uex)
    


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

    anim = @animate for i=1:2:length(tfine)

           plot(xfine, uNet[i], size=(1000, 750),
            ylims=(-0.2, 1.2), lw=1.5, color=0, legend=:topright,
            label = "network", xaxis=(""), linestyle=:dashdot)#,
            #right_margin = 30Plots.mm)
       plot!(xfine, uExact[i], label="exact", color=3)
       
       plot!(twinx(),xfine, err[i], ylims = (-0.01, 0.01),
             yaxis=("Error\n |u_exact - u_approx|"), color=:red,
             y_guidefontcolor=:red, legend=false)
             
       end

    gif(anim, "./advection_1D_char_fps30_LBFGS.gif", fps = 30)
    """)

