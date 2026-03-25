import torch
from weaver.nn.model.ParticleTransformerKANSpline import ParticleTransformer
from weaver.nn.model.kan_spline_layers import KANSplineMonitor
from weaver.utils.logger import _logger

'''
s4: Full KAN (cubic B-spline) — all components (head + main FFN + CLS FFN).
Same replacement scope as v4, but using order-3 B-spline basis instead of
piecewise-linear hat functions, with adaptive grid and proper initialization.
'''


class ParticleTransformerWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mod = ParticleTransformer(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(features, v=lorentz_vectors, mask=mask)


def get_model(data_config, **kwargs):

    cfg = dict(
        input_dim=len(data_config.input_dicts['pf_features']),
        num_classes=len(data_config.label_value),
        # network configurations
        pair_input_dim=4,
        use_pre_activation_pair=False,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        activation='gelu',
        # KAN spline switches — same scope as v4 (all components)
        use_kan_head=True,
        use_kan_main_ffn=True,
        use_kan_cls_ffn=True,
        kan_num_grids=8,
        kan_spline_order=3,
        kan_grid_range=(-2.0, 2.0),
        kan_base_activation='silu',
        kan_grid_eps=0.02,
        # misc
        trim=True,
        for_inference=False,
    )
    cfg.update(**kwargs)

    kan_num_grids = cfg.pop('kan_num_grids')
    kan_spline_order = cfg.pop('kan_spline_order')
    kan_grid_range = cfg.pop('kan_grid_range')
    kan_base_activation = cfg.pop('kan_base_activation')
    kan_grid_eps = cfg.pop('kan_grid_eps')
    cfg.setdefault('kan_head_num_grids', kan_num_grids)
    cfg.setdefault('kan_head_spline_order', kan_spline_order)
    cfg.setdefault('kan_head_grid_range', kan_grid_range)
    cfg.setdefault('kan_head_base_activation', kan_base_activation)
    cfg.setdefault('kan_head_grid_eps', kan_grid_eps)
    cfg.setdefault('kan_ffn_num_grids', kan_num_grids)
    cfg.setdefault('kan_ffn_spline_order', kan_spline_order)
    cfg.setdefault('kan_ffn_grid_range', kan_grid_range)
    cfg.setdefault('kan_ffn_base_activation', kan_base_activation)
    cfg.setdefault('kan_ffn_grid_eps', kan_grid_eps)
    _logger.info('Model config (s4 — spline all): %s' % str(cfg))

    model = ParticleTransformerWrapper(**cfg)
    model.kan_monitor = KANSplineMonitor(model)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
