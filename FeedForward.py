import torch
from torch import nn


class Feedforward(torch.nn.Module):
    """Compute the forward pass of a Feed-Forward Neural Net."""

    def __init__(self, layers, activations):
        """Initialize the network."""
        super().__init__()
        self.layers = layers
        self.activation = nn.ModuleList(list(activations))
        self.softplus = nn.Softplus()
        self.mse = nn.MSELoss(reduction='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1])
                                      for i in range(len(layers)-1)])

        self.hidden_dim = len(self.layers) - 2

        for i in range(len(self.layers)-1):
            nn.init.xavier_uniform_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)

    def forward(self, z):
        """Compute the network's forward pass."""
        for i in range(self.hidden_dim):
            w = self.linears[i](z)
            z = self.activation[i](w)
        z = self.linears[-1](z)
        return z
