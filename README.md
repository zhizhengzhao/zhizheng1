# Graduation Thesis — KAN × Particle Transformer

将 KAN (Kolmogorov-Arnold Networks) 集成到 Particle Transformer 中，用于高能物理 jet tagging 任务。

## 项目结构

```
├── kan_weaver-core/       # 核心改动：KAN 版 weaver-core
│   └── weaver/nn/model/
│       ├── kan_basis_layers.py                # KAN 基础层
│       ├── ParticleTransformerKANHead.py       # v1: KAN 分类头
│       └── ParticleTransformerKANHybrid.py     # v2: KAN 分类头 + CLS FFN
├── particle_transformer/  # 实验编排层（训练脚本、数据配置、网络 wrapper）
│   └── networks/
│       ├── example_ParticleTransformer_kan_head.py    # v1 wrapper
│       └── example_ParticleTransformer_kan_hybrid.py  # v2 wrapper
└── backups/               # 参考代码（不修改）
    ├── pykan/             # KAN 论文原始实现
    └── weaver-core/       # 原版 weaver-core（基线对照）
```

## 环境搭建

```bash
conda create -n zhizheng1 python=3.10 -y
conda activate zhizheng1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
git clone git@github.com:zhizhengzhao/Graduation_Thesis.git
cd Graduation_Thesis/kan_weaver-core
pip install -e .
pip install requests
```

## 训练

```bash
cd particle_transformer

# Baseline: 原版 ParT
./train_JetClass.sh ParT full

# v1: KAN 分类头
./train_JetClass.sh ParT full --network-config networks/example_ParticleTransformer_kan_head.py

# v2: KAN 分类头 + CLS FFN
./train_JetClass.sh ParT full --network-config networks/example_ParticleTransformer_kan_hybrid.py

# 多卡训练（例如用 4 张 GPU）
DDP_NGPUS=4 ./train_JetClass.sh ParT full --network-config networks/example_ParticleTransformer_kan_head.py
```

## 实验方案

| 版本 | 改动范围 | 配置文件 |
|------|---------|---------|
| Baseline | 原版 ParT | `example_ParticleTransformer.py` |
| v1 | 分类头 → KAN | `example_ParticleTransformer_kan_head.py` |
| v2 | 分类头 + CLS FFN → KAN | `example_ParticleTransformer_kan_hybrid.py` |
