"""Full KAN layers with B-spline basis for Particle Transformer.

Compared to kan_basis_layers.py (piecewise linear hat functions), this module uses:
  - B-spline basis of configurable order (default: cubic, order 3)
  - C^(k-1) continuity (cubic -> C2 smooth, better gradient flow)
  - Per-feature adaptive grid via update_grid()
  - L1 + entropy regularization for KAN-style sparsity
  - Extended knot vector for proper boundary handling

Reference: Liu et al., "KAN: Kolmogorov-Arnold Networks", 2024.
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


class KANSplineLinear(nn.Module):
    """KAN layer with B-spline basis (default: cubic, order 3).

    output_j = sum_i [ spline_ij(x_i) + base_weight_ij * base_act(x_i) ] + bias_j

    where spline_ij(x) = sum_k c_{j,i,k} * B_{k,order}(x) uses the Cox-de Boor
    B-spline recursion on an extended knot vector.

    Key differences from KANBasisLinear (hat/tent functions):
      - Cubic B-spline: C2 continuous, much smoother than C0 hat functions
      - Extended knot vector: proper boundary handling without hard clamping
      - Adaptive grid: update_grid() adjusts knots to data distribution
      - More expressive per grid interval (cubic vs linear)
    """

    def __init__(
            self,
            in_features,
            out_features,
            num_grids=8,
            spline_order=3,
            grid_range=(-2.0, 2.0),
            base_activation='silu',
            use_base=True,
            bias=True,
            grid_eps=0.02):
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError('in_features and out_features must be positive.')
        if num_grids < 2:
            raise ValueError('num_grids must be >= 2.')

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_grids = int(num_grids)
        self.spline_order = int(spline_order)
        self.grid_range = (float(grid_range[0]), float(grid_range[1]))
        self.grid_eps = float(grid_eps)
        self.use_base = bool(use_base)
        self.base_activation = _get_base_activation(base_activation)

        # Extended knot vector: G + 2k + 1 knots total
        # k extra knots on each side for proper B-spline boundary support
        h = (self.grid_range[1] - self.grid_range[0]) / self.num_grids
        grid = (
            torch.arange(
                -self.spline_order, self.num_grids + self.spline_order + 1,
                dtype=torch.float32)
            * h + self.grid_range[0]
        )
        # Per-input-feature grid allows independent adaptive updates
        grid = grid.unsqueeze(0).expand(self.in_features, -1).contiguous()
        self.register_buffer('grid', grid)

        # B-spline basis has G + k functions
        num_coeffs = self.num_grids + self.spline_order
        self.spline_weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, num_coeffs))

        if self.use_base:
            self.base_weight = nn.Parameter(
                torch.empty(self.out_features, self.in_features))
        else:
            self.register_parameter('base_weight', None)

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            # Fit spline to small noise at sample points for stable initialization
            device = self.spline_weight.device
            num_samples = self.num_grids + self.spline_order + 1
            x_sample = torch.linspace(
                self.grid_range[0], self.grid_range[1], num_samples,
                device=device)
            x_sample = x_sample.unsqueeze(1).expand(-1, self.in_features)

            noise = (
                (torch.rand(
                    self.out_features, self.in_features, num_samples,
                    device=device) - 0.5)
                * 0.1 / self.num_grids
            )
            self.spline_weight.copy_(self._curve2coeff(x_sample, noise))

            if self.base_weight is not None:
                nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in = self.in_features
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def b_splines(self, x):
        """Evaluate B-spline basis functions via Cox-de Boor recursion.

        Args:
            x: (batch, in_features)

        Returns:
            (batch, in_features, num_grids + spline_order)
        """
        grid = self.grid  # (in_features, G + 2k + 1)
        x = x.unsqueeze(-1)  # (batch, in_features, 1)

        # Order 0: piecewise constant
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        # Cox-de Boor recursion: order 1 through spline_order
        for p in range(1, self.spline_order + 1):
            left_num = x - grid[:, :-(p + 1)]
            left_den = grid[:, p:-1] - grid[:, :-(p + 1)]
            right_num = grid[:, (p + 1):] - x
            right_den = grid[:, (p + 1):] - grid[:, 1:-p]

            bases = (
                (left_num / left_den.clamp(min=1e-10)) * bases[:, :, :-1]
                + (right_num / right_den.clamp(min=1e-10)) * bases[:, :, 1:]
            )

        return bases

    def _curve2coeff(self, x_samples, y_target):
        """Compute spline coefficients that best fit given data points.

        Args:
            x_samples: (num_samples, in_features) - evaluation points
            y_target:  (out_features, in_features, num_samples) - target values

        Returns:
            coeffs: (out_features, in_features, num_grids + spline_order)
        """
        bases = self.b_splines(x_samples)  # (num_samples, in_features, G+k)

        A = bases.permute(1, 0, 2)         # (in_features, num_samples, G+k)
        B = y_target.permute(1, 2, 0)      # (in_features, num_samples, out_features)

        solution = torch.linalg.lstsq(A, B).solution  # (in_features, G+k, out_features)
        return solution.permute(2, 0, 1)               # (out_features, in_features, G+k)

    def forward(self, x):
        if x.size(-1) != self.in_features:
            raise ValueError(
                f'Expected last dim={self.in_features}, got {x.size(-1)}.')

        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)

        # Spline branch: cubic B-spline
        bases = self.b_splines(x_flat)
        out = torch.einsum('bik,oik->bo', bases, self.spline_weight)

        # Base branch: residual linear on activated inputs
        if self.base_weight is not None:
            out = out + F.linear(self.base_activation(x_flat), self.base_weight)

        if self.bias is not None:
            out = out + self.bias

        return out.reshape(*orig_shape, self.out_features)

    @torch.no_grad()
    def update_grid(self, x):
        """Adapt grid knots to the input data distribution.

        Recomputes knot positions as a blend of data quantiles and uniform
        spacing, then re-fits spline coefficients to preserve the learned
        function on the new grid.

        Args:
            x: (batch, in_features) or (batch, ..., in_features)
        """
        x_flat = x.reshape(-1, self.in_features)
        batch_size = x_flat.size(0)

        x_sorted = torch.sort(x_flat, dim=0).values

        num_interior = self.num_grids + 1
        indices = torch.linspace(
            0, batch_size - 1, num_interior, device=x.device).long()
        grid_adaptive = x_sorted[indices, :].t()  # (in_features, num_interior)

        grid_uniform = torch.stack([
            torch.linspace(
                x_sorted[0, i].item(), x_sorted[-1, i].item(),
                num_interior, device=x.device)
            for i in range(self.in_features)
        ], dim=0)

        grid_interior = (
            grid_adaptive * (1 - self.grid_eps)
            + grid_uniform * self.grid_eps
        )

        # Densely sample the current function BEFORE updating the grid
        num_samples = 5 * (self.num_grids + 1)
        x_eval = torch.stack([
            torch.linspace(
                grid_interior[i, 0].item(), grid_interior[i, -1].item(),
                num_samples, device=x.device)
            for i in range(self.in_features)
        ], dim=1)  # (num_samples, in_features)

        bases_old = self.b_splines(x_eval)
        y_old = torch.einsum(
            'bik,oik->oib', bases_old, self.spline_weight)  # (out, in, num_samples)

        # Build new extended knot vector
        h = (grid_interior[:, -1:] - grid_interior[:, :1]) / self.num_grids
        ext_left = (
            grid_interior[:, :1]
            - h * torch.arange(
                self.spline_order, 0, -1,
                device=x.device, dtype=torch.float32).unsqueeze(0)
        )
        ext_right = (
            grid_interior[:, -1:]
            + h * torch.arange(
                1, self.spline_order + 1,
                device=x.device, dtype=torch.float32).unsqueeze(0)
        )
        new_grid = torch.cat([ext_left, grid_interior, ext_right], dim=1)

        self.grid.copy_(new_grid)

        # Re-fit coefficients on the new grid via least squares
        bases_new = self.b_splines(x_eval)
        A = bases_new.permute(1, 0, 2)    # (in, num_samples, G+k)
        B = y_old.permute(1, 2, 0)        # (in, num_samples, out)
        coeffs = torch.linalg.lstsq(A, B).solution  # (in, G+k, out)
        self.spline_weight.copy_(coeffs.permute(2, 0, 1))

    def regularization_loss(self, l1_weight=1.0, entropy_weight=1.0):
        """KAN regularization: L1 on activation magnitudes + entropy for sparsity."""
        mag = self.spline_weight.abs().mean(dim=-1)  # (out, in)
        reg_l1 = mag.sum()
        p = mag / (mag.sum(dim=0, keepdim=True) + 1e-10)
        reg_entropy = -(p * p.clamp(min=1e-10).log()).sum()
        return l1_weight * reg_l1 + entropy_weight * reg_entropy


class KANSplineClassificationHead(nn.Module):
    """KAN-based classification head using B-spline layers."""

    def __init__(
            self,
            input_dim,
            num_classes,
            fc_params,
            num_grids=8,
            spline_order=3,
            grid_range=(-2.0, 2.0),
            base_activation='silu',
            grid_eps=0.02):
        super().__init__()
        hidden_dims = [out_dim for out_dim, _ in fc_params]
        layer_dims = [input_dim] + hidden_dims + [num_classes]
        self.kan_layers = nn.ModuleList([
            KANSplineLinear(
                in_features=layer_dims[i],
                out_features=layer_dims[i + 1],
                num_grids=num_grids,
                spline_order=spline_order,
                grid_range=grid_range,
                base_activation=base_activation,
                grid_eps=grid_eps)
            for i in range(len(layer_dims) - 1)
        ])
        self.hidden_dropout = nn.ModuleList([
            nn.Dropout(drop_rate) for _, drop_rate in fc_params
        ])

    def forward(self, x):
        for layer, drop in zip(self.kan_layers[:-1], self.hidden_dropout):
            x = drop(layer(x))
        return self.kan_layers[-1](x)


class KANSplineMonitor:
    """Monitors KANSplineLinear layers and optionally performs grid adaptation.

    Forward hooks track input statistics (mean, std, min, max, out-of-grid ratio).
    After a configurable warmup, a one-time adaptive grid update is performed
    so that grid knots cover the actual data distribution.
    """

    def __init__(self, model, log_interval=500, grid_update_step=200):
        import logging
        self._logger = logging.getLogger('weaver')
        self.log_interval = log_interval
        self._grid_update_step = grid_update_step
        self._batch_count = 0
        self._records = []
        self._hooks = []
        self._layers = {}
        self._pending = {}
        self._grid_updated = False
        self._grid_update_pending = False
        self._grid_update_data = {}

        for name, module in model.named_modules():
            if isinstance(module, KANSplineLinear):
                self._layers[name] = module
                pre_hook = module.register_forward_pre_hook(
                    self._make_pre_hook(name, module))
                hook = module.register_forward_hook(
                    self._make_hook(name, module))
                self._hooks.extend([pre_hook, hook])
        self._logger.info(
            'KANSplineMonitor: tracking %d KANSplineLinear layers, '
            'log_interval=%d, grid_update_step=%d',
            len(self._layers), log_interval, grid_update_step)

    def _make_pre_hook(self, layer_name, layer):
        """Execute deferred grid update BEFORE the forward pass starts.

        This avoids the inplace-modification error: at this point the
        computation graph for this batch hasn't been built yet, so
        copy_() on spline_weight is safe.
        """
        def pre_hook_fn(module, input):
            if not self._grid_update_pending:
                return
            if layer_name not in self._grid_update_data:
                return
            data = self._grid_update_data.pop(layer_name)
            module.update_grid(data)
            self._logger.info(
                'KANSplineMonitor: grid updated for %s (batch %d), '
                'new range=[%.3f, %.3f]',
                layer_name, self._batch_count,
                module.grid[:, module.spline_order].min().item(),
                module.grid[:, -module.spline_order - 1].max().item())
            if not self._grid_update_data:
                self._grid_update_pending = False
                self._grid_updated = True
        return pre_hook_fn

    def _make_hook(self, layer_name, layer):
        is_first_layer = (layer_name == next(iter(self._layers)))

        def hook_fn(module, input, output):
            if is_first_layer:
                self._batch_count += 1

            # Collect data for deferred grid update
            if (not self._grid_updated
                    and not self._grid_update_pending
                    and self._batch_count == self._grid_update_step
                    and layer_name not in self._grid_update_data):
                x = input[0].detach().clone()
                self._grid_update_data[layer_name] = x.reshape(-1, x.size(-1))
                if len(self._grid_update_data) == len(self._layers):
                    self._grid_update_pending = True

            # Periodic logging
            if self._batch_count % self.log_interval != 0:
                return
            if layer_name in self._pending:
                return
            x = input[0].detach()
            x_flat = x.reshape(-1, x.size(-1)).float()
            grid = layer.grid  # (in_features, n_knots)
            gmin = grid[:, layer.spline_order].min().item()
            gmax = grid[:, -layer.spline_order - 1].max().item()
            total = x_flat.numel()
            clamped = ((x_flat < gmin) | (x_flat > gmax)).sum().item()
            stats = {
                'batch': self._batch_count,
                'layer': layer_name,
                'grid_range': [gmin, gmax],
                'mean': x_flat.mean().item(),
                'std': x_flat.std().item(),
                'min': x_flat.min().item(),
                'max': x_flat.max().item(),
                'out_of_grid_ratio': clamped / total if total > 0 else 0.0,
            }
            self._records.append(stats)
            self._pending[layer_name] = True
            if len(self._pending) == len(self._layers):
                for name in self._pending:
                    r = next(r for r in reversed(self._records) if r['layer'] == name)
                    self._logger.info(
                        'KANSplineMonitor [batch %d] %s: mean=%.3f std=%.3f '
                        'min=%.3f max=%.3f oog=%.2f%%',
                        self._batch_count, name, r['mean'], r['std'],
                        r['min'], r['max'], r['out_of_grid_ratio'] * 100)
                self._pending.clear()

        return hook_fn

    def save(self, path):
        """Write all collected records to a JSON file."""
        import json
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                'log_interval': self.log_interval,
                'grid_update_step': self._grid_update_step,
                'samples': self._records,
            }, f, indent=2)
        self._logger.info(
            'KANSplineMonitor: saved %d records to %s',
            len(self._records), path)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
