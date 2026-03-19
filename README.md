# Graduation Thesis — KAN × Particle Transformer

将 KAN (Kolmogorov-Arnold Networks) 集成到 Particle Transformer 中，用于高能物理 jet tagging 任务。

## 项目结构

```
├── kan_weaver-core/       # 核心改动：KAN 版 weaver-core
│   └── weaver/nn/model/
│       ├── kan_basis_layers.py                 # KAN 基础层
│       └── ParticleTransformerKANHybrid.py     # 统一的可配置 KAN ParT 实现
├── particle_transformer/  # 实验编排层（训练脚本、数据配置、网络 wrapper）
│   └── networks/
│       ├── example_ParticleTransformer.py      # Baseline
│       ├── example_ParticleTransformer_v1.py   # v1: 只改 head
│       ├── example_ParticleTransformer_v2.py   # v2: 只改 CLS blocks
│       ├── example_ParticleTransformer_v3.py   # v3: 只改 main blocks
│       └── example_ParticleTransformer_v4.py   # v4: head + CLS + main 全改
└── visualize_results.py   # 训练曲线可视化脚本
```

## 环境搭建

```bash
conda create -n zhizheng1 python=3.10 -y
conda activate zhizheng1
pip install torch torchvision torchaudio -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
git clone git@github.com:zhizhengzhao/Graduation_Thesis.git
cd Graduation_Thesis/kan_weaver-core
pip install -e .
pip install requests
```

数据路径在 `particle_transformer/env.sh` 中配置，`train_JetClass.sh` 启动时会自动 `source env.sh`，无需额外设置。

## 数据下载

数据集已在服务器上准备好，如需重新下载：

```bash
cd particle_transformer
python get_datasets.py JetClass
```

> **注**：`test_20M` 数据量约 19 GB，下载较慢。训练时默认关闭测试集评估，可通过 `USE_TEST=1` 开启（见下文）。

## 训练

```bash
cd particle_transformer

# Baseline: 原版 ParT
CUDA_VISIBLE_DEVICES=4,5,6,7 DDP_NGPUS=4 ./train_JetClass.sh ParT full

# v1: 只改分类头
CUDA_VISIBLE_DEVICES=4,5,6,7 DDP_NGPUS=4 ./train_JetClass.sh ParT full --network-config networks/example_ParticleTransformer_v1.py

# v2: 只改 CLS blocks
CUDA_VISIBLE_DEVICES=4,5,6,7 DDP_NGPUS=4 ./train_JetClass.sh ParT full --network-config networks/example_ParticleTransformer_v2.py

# v3: 只改 main blocks
CUDA_VISIBLE_DEVICES=4,5,6,7 DDP_NGPUS=4 ./train_JetClass.sh ParT full --network-config networks/example_ParticleTransformer_v3.py

# v4: head + CLS blocks + main blocks 全改
CUDA_VISIBLE_DEVICES=4,5,6,7 DDP_NGPUS=4 ./train_JetClass.sh ParT full --network-config networks/example_ParticleTransformer_v4.py
```

### 精度控制

`train_JetClass.sh` 通过环境变量 `USE_AMP` 控制是否启用混合精度（AMP）。
默认关闭，使用全精度（对 KAN 层数值稳定性更安全）：

```bash
# 默认：USE_AMP=0（全精度，推荐）
CUDA_VISIBLE_DEVICES=4,5,6,7 DDP_NGPUS=4 ./train_JetClass.sh ParT full

# 显式开启 AMP（加速训练，但 KAN 层在 float16 下可能不稳定）
USE_AMP=1 CUDA_VISIBLE_DEVICES=4,5,6,7 DDP_NGPUS=4 ./train_JetClass.sh ParT full
```

### 测试集控制

`test_20M` 数据下载完成后，通过 `USE_TEST=1` 开启测试集评估：

```bash
# 默认：USE_TEST=0（不跑测试集）
CUDA_VISIBLE_DEVICES=4,5,6,7 DDP_NGPUS=4 ./train_JetClass.sh ParT full

# 开启测试集评估
USE_TEST=1 CUDA_VISIBLE_DEVICES=4,5,6,7 DDP_NGPUS=4 ./train_JetClass.sh ParT full
```

### KAN 超参数控制

`v1-v4` 的 network config 在各自的 `cfg = dict(...)` 中统一提供以下超参数：

```python
kan_num_grids = 20
kan_grid_range = (-5.0, 5.0)
kan_base_activation = 'silu'
```

## 结果可视化

```bash
python visualize_results.py
```

生成 `validation_accuracy_curves.png` 和 `best_accuracy_comparison.png`。

## 实验方案

| 版本 | 改动范围 | 配置文件 |
|------|---------|---------|
| Baseline | 原版 ParT | `example_ParticleTransformer.py` |
| v1 | 只改分类头 | `example_ParticleTransformer_v1.py` |
| v2 | 只改 CLS blocks | `example_ParticleTransformer_v2.py` |
| v3 | 只改 main blocks | `example_ParticleTransformer_v3.py` |
| v4 | head + CLS blocks + main blocks 全改 | `example_ParticleTransformer_v4.py` |
