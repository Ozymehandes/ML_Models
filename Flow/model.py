import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class AffineLayerBase(nn.Module):
    def __init__(self, in_features, hidden_dim=8):
        super(AffineLayerBase, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, in_features)
        )

    def forward(self, z):
        return self.model(z)

class AffineCouplingLayer(nn.Module):
    def __init__(self, l_dim):
        super(AffineCouplingLayer, self).__init__()
        self.l_dim = l_dim
        self.b = AffineLayerBase(l_dim)
        self.log_s = AffineLayerBase(l_dim)

    def forward(self, z):
        z_l, z_r = z.chunk(2, dim=-1)

        b = self.b(z_l)
        log_s = self.log_s(z_l)

        y_r = z_r * torch.exp(log_s) + b
        y_l = z_l

        return torch.cat([y_l, y_r], dim=-1)

    def inverse(self, y):
        y_l, y_r = y.chunk(2, dim=-1)

        b = self.b(y_l)
        log_s = self.log_s(y_l)

        z_r = (y_r - b) * torch.exp(-log_s)
        z_l = y_l

        return torch.cat([z_l, z_r], dim=-1)
    
    def log_det_jacobian(self, y):
        y_l, _ = y.chunk(2, dim=-1)
        log_s = self.log_s(y_l)
        return torch.sum(-log_s, dim=-1)


class PermutationLayer(nn.Module):
    def __init__(self, input_features):
        super(PermutationLayer, self).__init__()
        self.register_buffer('permutation', torch.randperm(input_features))
        self.register_buffer('inverse_permutation', torch.argsort(self.permutation))

    def forward(self, z):
        return z[:, self.permutation]

    def inverse(self, y):
        return y[:, self.inverse_permutation]

class NormalizingFlowModel(nn.Module):
    def __init__(self, input_dim, affine_layers_n=15):
        super(NormalizingFlowModel, self).__init__()
        self.input_dim = input_dim
        self.l_dim = input_dim // 2

        layers = []
        for _ in range(affine_layers_n-1):
            layers.extend([
                AffineCouplingLayer(self.l_dim),
                PermutationLayer(input_dim)
            ])
        layers.append(AffineCouplingLayer(self.l_dim))
        self.normalizing_flow = nn.ModuleList(layers)

    def forward(self, z):
        for layer in self.normalizing_flow:
            z = layer(z)
            if torch.isinf(z).any() or torch.isnan(z).any():
                print("Inf or NaN detected in forward")
                sys.exit("Terminating program due to Inf or NaN values.")
        return z

    def inverse(self, y):
        for layer in reversed(self.normalizing_flow):
            y = layer.inverse(y)
            if torch.isinf(y).any() or torch.isnan(y).any():
                print("Inf or NaN detected in inverse")
                sys.exit("Terminating program due to Inf or NaN values.")
        return y

    def log_det_jacobian(self, y):
        log_det_jacobian = 0
        for layer in reversed(self.normalizing_flow):
            if isinstance(layer, AffineCouplingLayer):
                log_det_jacobian += layer.log_det_jacobian(y)
            if torch.isinf(log_det_jacobian).any() or torch.isnan(log_det_jacobian).any():
                print("Inf or NaN detected in log_det_jacobian")
                print(log_det_jacobian)
                sys.exit("Terminating program due to Inf or NaN values.")

            y = layer.inverse(y)

        return log_det_jacobian
    

class UnconditionalFlow(nn.Module):
    def __init__(self, input_dim, hidden_dim=128) -> None:
        super(UnconditionalFlow, self).__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim), # Account for the time dimension
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, y, t):
        t = t.view(-1, 1)
        return self.model(torch.cat([y, t], dim=-1))
    

class ConditionalFlow(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_dim=64) -> None:
        super(ConditionalFlow, self).__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.embedding_dim = 3
        self.embedding = nn.Embedding(n_classes, self.embedding_dim)
        self.model = nn.Sequential(
            nn.Linear(input_dim + 1 + self.embedding_dim, hidden_dim), # Account for the time dimension and the classes
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, y, t, c):
        t = t.view(-1, 1)
        c = self.embedding(c)
        return self.model(torch.cat([y, t, c], dim=-1))
    

