"""KAN-inspired layers for Particle Transformer integration.

Provides:
  - KANBasisLinear: univariate spline + residual linear branch.
  - KANClassificationHead: drop-in KAN replacement for the classification head.
"""

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


class KANClassificationHead(nn.Module):
    """KAN-based classification head, drop-in replacement for the MLP head."""

    def __init__(
            self,
            input_dim,
            num_classes,
            fc_params,
            num_grids=8,
            grid_range=(-2.0, 2.0),
            base_activation='silu'):
        super().__init__()
        hidden_dims = [out_dim for out_dim, _ in fc_params]
        layer_dims = [input_dim] + hidden_dims + [num_classes]
        self.kan_layers = nn.ModuleList([
            KANBasisLinear(
                in_features=layer_dims[i],
                out_features=layer_dims[i + 1],
                num_grids=num_grids,
                grid_range=grid_range,
                base_activation=base_activation)
            for i in range(len(layer_dims) - 1)
        ])
        self.hidden_dropout = nn.ModuleList([
            nn.Dropout(drop_rate) for _, drop_rate in fc_params
        ])

    def forward(self, x):
        for layer, drop in zip(self.kan_layers[:-1], self.hidden_dropout):
            x = drop(layer(x))
        return self.kan_layers[-1](x)


class KANMonitor:
    """Monitors input distributions of all KANBasisLinear layers via forward hooks.

    Periodically samples statistics (mean, std, min, max, clamp ratio) and
    writes them to a JSON file at the end of training.
    """

    def __init__(self, model, log_interval=500):
        import logging
        self._logger = logging.getLogger('weaver')
        self.log_interval = log_interval
        self._batch_count = 0
        self._records = []
        self._hooks = []
        self._layers = {}
        self._pending = {}

        for name, module in model.named_modules():
            if isinstance(module, KANBasisLinear):
                self._layers[name] = module
                hook = module.register_forward_hook(self._make_hook(name, module))
                self._hooks.append(hook)
        self._logger.info('KANMonitor: tracking %d KANBasisLinear layers, log_interval=%d',
                          len(self._layers), log_interval)

    def _make_hook(self, layer_name, layer):
        is_first_layer = (layer_name == next(iter(self._layers)))

        def hook_fn(module, input, output):
            if is_first_layer:
                self._batch_count += 1
            if self._batch_count % self.log_interval != 0:
                return
            if layer_name in self._pending:
                return
            x = input[0].detach()
            x_flat = x.reshape(-1, x.size(-1)).float()
            total = x_flat.numel()
            clamped = ((x_flat < layer.grid_min) | (x_flat > layer.grid_max)).sum().item()
            stats = {
                'batch': self._batch_count,
                'layer': layer_name,
                'grid_range': [layer.grid_min, layer.grid_max],
                'mean': x_flat.mean().item(),
                'std': x_flat.std().item(),
                'min': x_flat.min().item(),
                'max': x_flat.max().item(),
                'clamp_ratio': clamped / total if total > 0 else 0.0,
            }
            self._records.append(stats)
            self._pending[layer_name] = True
            if len(self._pending) == len(self._layers):
                for name in self._pending:
                    r = next(r for r in reversed(self._records) if r['layer'] == name)
                    self._logger.info(
                        'KANMonitor [batch %d] %s: mean=%.3f std=%.3f min=%.3f max=%.3f clamp=%.2f%%',
                        self._batch_count, name, r['mean'], r['std'],
                        r['min'], r['max'], r['clamp_ratio'] * 100)
                self._pending.clear()
        return hook_fn

    def save(self, path):
        """Write all collected records to a JSON file."""
        import json
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump({'log_interval': self.log_interval, 'samples': self._records}, f, indent=2)
        self._logger.info('KANMonitor: saved %d records to %s', len(self._records), path)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
