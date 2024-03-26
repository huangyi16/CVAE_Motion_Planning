import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedLinear(nn.Linear):
    """Masked Linear Layer inheriting from `~torch.nn.Linear` class and applying a mask to consider
    only a selection of weight.
    """

    def __init__(self, in_features, out_features, mask):
        nn.Linear.__init__(self, in_features=in_features, out_features=out_features)

        self.register_buffer("mask", mask)

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)

class MADE(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim, ordering='sequential'):
        super(MADE, self).__init__()
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.output_dim = output_dim
        self.ordering = ordering

        self.net = []
        self.m = {}

        hidden_sizes = [self.input_dim] + self.hidden_sizes + [self.output_dim]
        masks = self._make_mask(ordering=self.ordering)

        for inp, out, mask in zip(hidden_sizes[:-1], hidden_sizes[1:-1], masks[:-1]):
            self.net.extend([MaskedLinear(inp, out, mask), nn.ReLU()])

        # outputs mean and logvar
        self.net.extend(
            [
                MaskedLinear(
                    self.hidden_sizes[-1], 2 * self.output_dim, masks[-1].repeat(2, 1)
                )
            ]
        )

        self.net = nn.Sequential(*self.net)

    def _make_mask(self, ordering):
        # Get degrees for mask creation

        if ordering == "sequential":
            self.m[-1] = torch.arange(self.input_dim)
            for i in range(len(self.hidden_sizes)):
                self.m[i] = torch.arange(self.hidden_sizes[i]) % (self.input_dim - 1)

        else:
            self.m[-1] = torch.randperm(self.input_dim)
            for i in range(len(self.hidden_sizes)):
                self.m[i] = torch.randint(
                    self.m[-1].min(), self.input_dim - 1, (self.hidden_sizes[i],)
                )

        masks = []
        for i in range(len(self.hidden_sizes)):
            masks += [(self.m[i].unsqueeze(-1) >= self.m[i - 1].unsqueeze(0)).float()]

        masks.append(
            (
                self.m[len(self.hidden_sizes) - 1].unsqueeze(0)
                < self.m[-1].unsqueeze(-1)
            ).float()
        )

        return masks
    
    def forward(self, x: torch.Tensor, **kwargs):
        """The input data is transformed toward the prior

        Args:
            inputs (torch.Tensor): An input tensor

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """
        net_output = self.net(x.reshape(x.shape[0], -1))

        mu = net_output[:, : self.input_dim]
        log_var = net_output[:, self.input_dim :]

        return mu, log_var

class IAF(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_hidden_in_made, n_made_blocks, ordering='sequential'):
        super(IAF, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_hidden_in_made = n_hidden_in_made
        self.ordering = ordering
        self.n_made_blocks = n_made_blocks

        self.hidden_sizes = [self.hidden_dim] * self.n_hidden_in_made

        self.net = []
        self.m = {}

        for i in range(self.n_made_blocks):
            self.net.extend([MADE(self.input_dim, self.hidden_sizes, self.input_dim, self.ordering)])

        self.net = nn.ModuleList(self.net)

    def forward(self, x: torch.Tensor, **kwargs):
        sum_log_abs_det_jac = torch.zeros(x.shape[0]).to(x.device)
        y = torch.zeros_like(x)
        for layer in self.net:
            y = torch.zeros_like(x)
            for i in range(self.input_dim):
                m, s = layer(y.clone())

                y[:, i] = (x[:, i] - m[:, i]) * (-s[:, i]).exp()

                sum_log_abs_det_jac += -s[:, i]

            x = y
            x = x.flip(dims=(1,))
        
        return x, sum_log_abs_det_jac
    
    def inverse(self, y: torch.Tensor, **kwargs):
        """The prior is transformed toward the input data (f)

        Args:
            inputs (torch.Tensor): An input tensor

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """
        y = y.reshape(y.shape[0], -1)
        sum_log_abs_det_jac = torch.zeros(y.shape[0]).to(y.device)

        for layer in self.net[::-1]:
            y = y.flip(dims=(1,))
            m, s = layer(y)
            y = y * s.exp() + m
            sum_log_abs_det_jac += s.sum(dim=-1)


        return y, sum_log_abs_det_jac
