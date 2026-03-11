###############
#     Change           #
###############

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_base_activation(name):
    name = str(name).lower()
    if name == 'silu':
        return nn.SiLU()
    if name == 'relu':
        return nn.ReLU()
    if name == 'gelu':
        return nn.GELU()
    if name == 'identity':
        return nn.Identity()
    raise ValueError(f'Unsupported base activation: {name}')


class KANBasisLinear(nn.Module):
    """A lightweight KAN-inspired layer.

    The layer models each output as a sum of per-input univariate functions:
    - Spline branch: piecewise linear hat bases on a fixed 1D grid.
    - Base branch: optional residual linear mapping on base-activated inputs.
    """

    def __init__(
            self,
            in_features,
            out_features,
            num_grids=8,
            grid_range=(-2.0, 2.0),
            base_activation='silu',
            use_base=True,
            bias=True):
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError('in_features and out_features must be positive.')
        if num_grids < 2:
            raise ValueError('num_grids must be >= 2.')

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_grids = int(num_grids)
        self.num_knots = self.num_grids + 1
        self.grid_min = float(grid_range[0])
        self.grid_max = float(grid_range[1])
        self.use_base = bool(use_base)
        self.base_activation = _get_base_activation(base_activation)

        grid = torch.linspace(self.grid_min, self.grid_max, steps=self.num_knots)
        self.register_buffer('grid', grid)
        self.register_buffer('grid_delta', torch.tensor((self.grid_max - self.grid_min) / self.num_grids))

        # spline weights: [out_features, in_features, num_knots]
        self.spline_weight = nn.Parameter(torch.empty(self.out_features, self.in_features, self.num_knots))

        if self.use_base:
            self.base_weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        else:
            self.register_parameter('base_weight', None)

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.spline_weight, std=0.02)
        if self.base_weight is not None:
            nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def _spline_basis(self, x):
        # x: [batch, in_features]
        # output basis: [batch, in_features, num_knots]
        x = x.clamp(min=self.grid_min, max=self.grid_max)
        basis = 1.0 - torch.abs((x.unsqueeze(-1) - self.grid) / self.grid_delta)
        return torch.relu(basis)

    def forward(self, x):
        if x.size(-1) != self.in_features:
            raise ValueError(
                f'Expected last dim={self.in_features}, got {x.size(-1)}.')

        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)

        basis = self._spline_basis(x_flat)
        out = torch.einsum('bik,oik->bo', basis, self.spline_weight)

        if self.base_weight is not None:
            out = out + F.linear(self.base_activation(x_flat), self.base_weight, None)

        if self.bias is not None:
            out = out + self.bias

        return out.reshape(*orig_shape, self.out_features)


class KANBasisMLP(nn.Module):
    """Stacked KANBasisLinear layers for MLP-style heads."""

    def __init__(
            self,
            layer_dims,
            dropout=0.0,
            num_grids=8,
            grid_range=(-2.0, 2.0),
            base_activation='silu',
            final_activation=None):
        super().__init__()
        if len(layer_dims) < 2:
            raise ValueError('layer_dims must contain at least input and output dims.')

        blocks = []
        for i in range(len(layer_dims) - 1):
            in_dim = int(layer_dims[i])
            out_dim = int(layer_dims[i + 1])
            blocks.append(
                KANBasisLinear(
                    in_features=in_dim,
                    out_features=out_dim,
                    num_grids=num_grids,
                    grid_range=grid_range,
                    base_activation=base_activation,
                    use_base=True,
                    bias=True))
            is_last = (i == len(layer_dims) - 2)
            if not is_last:
                blocks.append(nn.ReLU())
                if dropout > 0:
                    blocks.append(nn.Dropout(dropout))
            elif final_activation is not None:
                blocks.append(final_activation)
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)

###############
#     Change End    #
###############
