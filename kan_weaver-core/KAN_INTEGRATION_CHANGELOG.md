# KAN Integration Changelog

This file records incremental changes for KAN-inspired modifications.

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

## Version v1 (Small Scope)

### Goal

Replace only the **final classification head** in ParticleTransformer with a KAN-inspired head.

### Files Added / Modified

- Added: `weaver/nn/model/kan_basis_layers.py`
  - `KANBasisLinear`: KAN-inspired univariate spline + residual base branch.
  - `KANBasisMLP`: stacked KAN MLP utility.
- Added/Modified copy: `weaver/nn/model/ParticleTransformerKANHead.py`
  - Added `KANClassificationHead`.
  - Added `ParticleTransformer` args:
    - `use_kan_head`
    - `kan_head_num_grids`
    - `kan_head_grid_range`
    - `kan_head_base_activation`
  - If `use_kan_head=True`, replace original `nn.Linear` head with `KANClassificationHead`.

### Network Config (for particle_transformer)

- Added copy: `particle_transformer/networks/example_ParticleTransformer_kan_head.py`
  - Imports `weaver.nn.model.ParticleTransformerKANHead`.
  - Enables `use_kan_head=True`.

## Version v2 (Larger Scope)

### Goal

Keep v1 KAN head and additionally replace FFN with KAN in a controlled range.

### Files Added / Modified

- Added/Modified copy: `weaver/nn/model/ParticleTransformerKANHybrid.py`
  - `Block` adds options:
    - `use_kan_ffn`
    - `kan_ffn_num_grids`
    - `kan_ffn_grid_range`
    - `kan_ffn_base_activation`
  - If `use_kan_ffn=True`, FFN `fc1/fc2` use `KANBasisLinear`.
  - `ParticleTransformer` adds options:
    - `use_kan_main_ffn`
    - `use_kan_cls_ffn`
    - `kan_ffn_num_grids`
    - `kan_ffn_grid_range`
    - `kan_ffn_base_activation`
  - Default v2 behavior in provided config:
    - Main blocks FFN: off
    - CLS blocks FFN: on

### Network Config (for particle_transformer)

- Added copy: `particle_transformer/networks/example_ParticleTransformer_kan_hybrid.py`
  - Imports `weaver.nn.model.ParticleTransformerKANHybrid`.
  - Enables:
    - `use_kan_head=True`
    - `use_kan_main_ffn=False`
    - `use_kan_cls_ffn=True`

## Suggested Run Commands

Run in `particle_transformer`:

```bash
./train_JetClass.sh ParT full --network-config networks/example_ParticleTransformer_kan_head.py
```

```bash
./train_JetClass.sh ParT full --network-config networks/example_ParticleTransformer_kan_hybrid.py
```

If your script already sets `--network-config`, pass the new config path directly to `weaver`.

## Environment Note

To ensure new `weaver` modules are used, install this repo in editable mode:

```bash
cd /Users/zhaozhizheng/Desktop/article/kan_weaver-core
pip install -e .
```
