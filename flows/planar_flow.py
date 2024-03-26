import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATION = {"elu": F.elu, "tanh": torch.tanh, "linear": lambda x: x}

ACTIVATION_DERIVATIVES = {
    "elu": lambda x: torch.ones_like(x) * (x >= 0) + torch.exp(x) * (x < 0),
    "tanh": lambda x: 1 - torch.tanh(x) ** 2,
    "linear": lambda x: 1,
}


class PlanarFlow(nn.Module):
    def __init__(self, z_dim, activation):
        super(PlanarFlow, self).__init__()

        self.input_dim = z_dim
        self.w = nn.Parameter(torch.randn(1, self.input_dim))
        self.u = nn.Parameter(torch.randn(1, self.input_dim))
        self.b = nn.Parameter(torch.randn(1))
        self.activation = ACTIVATION[activation]
        self.activation_derivative = ACTIVATION_DERIVATIVES[activation]

        nn.init.normal_(self.w)
        nn.init.normal_(self.u)
        nn.init.normal_(self.b)

    def forward(self, z: torch.Tensor):
        lin = z @ self.w.T + self.b
        f = z + self.u * self.activation(lin)
        phi = self.activation_derivative(lin) @ self.w
        log_det = torch.log(torch.abs(1 + phi @ self.u.T) + 1e-4).squeeze()

        return f, log_det