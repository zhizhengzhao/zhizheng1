# KAN Integration Changelog

This file records the current KAN experiment layout for Particle Transformer.

## Rules Applied

- No direct overwrite of original model files.
- Modified copies were created with explicit feature-oriented names.
- New utilities were added as new files.
- Every edited code region is marked with:
  - `###############`
  - `#     Change           #`
  - `###############`
  - ...
  - `###############`
  - `#     Change End    #`
  - `###############`

## Unified Experiment Layout

### Core files

- `weaver/nn/model/kan_basis_layers.py`
  - `KANBasisLinear`: KAN-inspired univariate spline + residual base branch.
  - `KANClassificationHead`: KAN-based classifier head.
  - `KANMonitor`: hook-based input distribution monitor for all KAN layers.
- `weaver/nn/model/ParticleTransformerKANHybrid.py`
  - Unified configurable ParT variant supporting:
    - `use_kan_head`
    - `use_kan_main_ffn`
    - `use_kan_cls_ffn`
    - `kan_head_num_grids`
    - `kan_head_grid_range`
    - `kan_head_base_activation`
    - `kan_ffn_num_grids`
    - `kan_ffn_grid_range`
    - `kan_ffn_base_activation`

### Network configs

- `particle_transformer/networks/example_ParticleTransformer_v1.py`
  - Only replace the final classification head with KAN.
- `particle_transformer/networks/example_ParticleTransformer_v2.py`
  - Only replace the CLS blocks FFN with KAN.
- `particle_transformer/networks/example_ParticleTransformer_v3.py`
  - Only replace the main transformer blocks FFN with KAN.
- `particle_transformer/networks/example_ParticleTransformer_v4.py`
  - Replace head + CLS blocks FFN + main blocks FFN with KAN.

### Shared config knobs

All `v1-v4` configs expose:

- `kan_num_grids`
- `kan_grid_range`
- `kan_base_activation`

These are mapped to both head and FFN KAN settings unless explicitly overridden.

### Precision control

`particle_transformer/train_JetClass.sh` uses `USE_AMP` to control mixed precision:

- `USE_AMP=0` (default): disable AMP / use full precision
- `USE_AMP=1`: enable AMP

## Suggested Run Commands

Run in `particle_transformer`:

```bash
./train_JetClass.sh ParT full --network-config networks/example_ParticleTransformer_v1.py
```

```bash
./train_JetClass.sh ParT full --network-config networks/example_ParticleTransformer_v2.py
```

```bash
./train_JetClass.sh ParT full --network-config networks/example_ParticleTransformer_v3.py
```

```bash
./train_JetClass.sh ParT full --network-config networks/example_ParticleTransformer_v4.py
```

```bash
USE_AMP=0 ./train_JetClass.sh ParT full --network-config networks/example_ParticleTransformer_v2.py
```

If your script already sets `--network-config`, pass the new config path directly to `weaver`.

## Environment Note

To ensure new `weaver` modules are used, install this repo in editable mode:

```bash
cd kan_weaver-core
pip install -e .
```
