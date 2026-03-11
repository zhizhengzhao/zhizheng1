# Particle Transformer 项目 Refine 深度报告（联网增强版）

- 报告对象：`/Users/zhaozhizheng/Desktop/article/particle_transformer`
- 本地代码基线：`main@2925bdb249e8ef78560cc2b9b651eda3615da8c7`
- 联网核验日期：**2026-02-27**
- 报告目标：在上一版本地解构报告基础上，补上论文原始细节、公开数据源、上游框架信息，并给出更可执行的改造路线。

---

## 0. 先给结论（面向你后续改论文）

这个仓库可以被准确理解成三层：

1. **论文思想层（ParT / JetClass）**：定义了方法论与指标目标。  
2. **上游框架层（weaver-core）**：实现训练引擎、数据管线抽象、主干模型实现。  
3. **当前仓库层（particle_transformer）**：做实验编排（YAML+脚本+wrapper+预训练权重）。

所以后续改进要先判断你在改哪一层：

- 只改本仓库即可完成：特征工程、训练策略、微调策略、实验组织。
- 必须改上游 `weaver-core`：ParT attention 内核、Pair embedding 主干结构、算子优化。

---

## 1. 联网事实核验（相对上一版新增）

## 1.1 论文与官方实现状态

- ICML 正式条目仍是 2022 版 `Particle Transformer for Jet Tagging`（PMLR 162:18281-18292）。[R1]
- 官方代码仓库仍为 `jet-universe/particle_transformer`，README 中训练入口与本地仓库结构一致。[R2]
- 在线页面显示该仓库历史提交规模为 `13 commits`（抓取时刻快照），与你本地仓库 `13 commits` 对齐。[R2]

## 1.2 JetClass 数据集元数据

- JetClass Zenodo 条目为 `10.5281/zenodo.6619768`，发布日期 `2022-06-16`，版本 `1.0.0`。[R3]
- 规模为 `100M train + 5M val + 20M test`，`10` 类 jets。[R3]
- 文件总量约 `189.8 GB`，并公开每个 tar 的 md5；本仓库 `get_datasets.py` 里的 md5 与 Zenodo一致（我已逐项核对）。[R3]

## 1.3 上游 weaver-core 状态

- PyPI 页面显示 `weaver-core` 最新版本为 `0.4.17`，发布日期 `2024-08-06`。[R4]
- 项目要求 `Python>=3.7`，并继续把 YAML data config / python model config 作为标准接口。[R4]
- GitHub 页面显示 weaver-core 是活跃主仓（抓取时显示 `133 commits` 快照）。[R5]

## 1.4 下游基准数据集来源核验

- QG 数据来源是 `Pythia8 Quark and Gluon Jets for Energy Flow` (`10.5281/zenodo.3164691`)。[R6]
- EnergyFlow 文档明确该数据常用标准切分是 `1.6M/200k/200k`（train/val/test），与 ParT 论文写法一致。[R7]
- Top 基准来自 `The Machine Learning Landscape of Top Taggers`，SciPost 正式 DOI 为 `10.21468/SciPostPhys.7.1.014`。[R8]

---

## 2. 论文方法到本仓库代码的“对齐图”

## 2.1 ParT 的核心数学对象（论文）

论文定义了 ParT 的两个核心增强：

1. **pairwise interaction features**（式(3)）
- 对粒子对 `(a,b)` 构造 `Δ, kT, z, m²`，实际使用其对数 `lnΔ, lnkT, lnz, lnm²`。[R1]

2. **P-MHA**（式(4)）
- 在注意力 softmax 前加 interaction matrix：
- `SoftMax(QK^T/sqrt(dk) + U)V`。[R1]

论文还给出 baseline 结构：

- `8` 个 particle attention blocks + `2` 个 class attention blocks。  
- 粒子 embedding 维度 `d=128`，embedding MLP 为 `(128,512,128)`。  
- 交互特征编码使用 `(64,64,64,16)` pointwise conv。  
- 注意力头数 `8`，每头 `d'=16`。[R1]

## 2.2 本仓库如何落地该方法

`networks/example_ParticleTransformer.py` 对应配置：

- `embed_dims=[128, 512, 128]`
- `pair_embed_dims=[64, 64, 64]`
- `pair_input_dim=4`
- `num_heads=8`
- `num_layers=8`
- `num_cls_layers=2`

这和论文 baseline 在“层数/维度/head数”上是同一设计族。

实现关系是：

- 当前仓库 wrapper 负责声明配置与输入输出协议；
- 具体 ParT 主干在 `weaver-core` 里实现（本仓库文件本身也显式写了链接指向）。

## 2.3 一个关键一致性细节：JetClass 训练“迭代预算”

论文写法：

- batch=512, 共 `1M` iterations，约等于 `512M` 样本，即 `~5` 个 effective epochs（100M 训练集）。[R1]

仓库脚本写法（`train_JetClass.sh`）：

- `samples_per_epoch = 10,240,000`
- `epochs = 50`
- 总样本预算 = `512,000,000`（与论文几乎等价）

所以它看起来是“50 epoch”，本质上是用 `samples_per_epoch` 把总训练量约束到论文同级预算。

---

## 3. 训练系统深度剖析（从命令到梯度）

## 3.1 训练入口设计

三个脚本各自管理一个数据域：

- `train_JetClass.sh`
- `train_QuarkGluon.sh`
- `train_TopLandscape.sh`

共同机制：

- 先 `source env.sh`
- 通过 `--data-config` 绑定 YAML 特征工程
- 通过 `--network-config` 绑定模型 wrapper
- 其余参数交给 `weaver` CLI

## 3.2 Optimizer 与学习率策略

### JetClass

- optimizer 固定 `ranger`（即 Lookahead + RAdam），论文也是同路线。[R1]
- 模型别名与超参组合复现论文基准：
- ParT: `batch 512, lr 1e-3`
- PN: `batch 512, lr 1e-2`
- PFN/PCNN: `batch 4096, lr 2e-2`

### QuarkGluon / TopLandscape

- ParT 默认附加 `weight_decay 0.01`。
- FineTune 模式采用 `lr_mult` 提升新分类头学习率：
- `ParT-FineTune` 用 `("fc.*",50)`
- `PN-FineTune` 用 `("fc_out.*",50)`

这正对应论文 fine-tune 叙述：预训练权重用小 LR，新增头用大 LR（在论文 top 任务里约 0.0001 vs 0.005）。[R1]

## 3.3 多卡机制

- DataParallel：直接 `--gpus 0,1,...`
- DDP：`DDP_NGPUS>1` 时用 `torchrun --standalone --nproc_per_node=$NGPUS $(which weaver) --backend nccl`

JetClass 脚本显式实现了 DDP 分支；QG/Top 默认是单命令 `weaver`。

---

## 4. 数据与特征工程：项目真正的“可改造核心”

## 4.1 YAML 接口语义（与 weaver 官方文档一致）

`weaver` 官方说明 data config 由以下段落组成：[R4][R5]

- `selection`
- `new_variables`
- `inputs`
- `labels`
- `observers`
- `weights`

你的项目中这些段落全部具备，且是主实验入口。

## 4.2 JetClass 三档输入（kin / kinpid / full）

- `kin`：7维纯运动学。
- `kinpid`：13维，加入电荷+5类 PID。
- `full`：17维，再加入 `d0/dz` 位移与误差项。

这与论文 Table 2 的“JETCLASS full=17粒子特征”严格一致。[R1]

## 4.3 QG 数据工程“二次语义编码”

`qg_kinpidplus.yaml` 里不是直接 one-hot hadron，而是加权映射：

- `part_isCHad = pi + 0.5*K + 0.2*p`
- `part_isNHad = K0 + 0.2*n`

这是一种“软物理先验压缩”，有两点研究价值：

1. 可能减少维度与噪声；
2. 也可能引入手工偏置，影响可解释性。

## 4.4 Top 数据工程与源数据边界

Top 数据脚本从 HDF 读前 200 粒子四动量列，构建 `part_deta/dphi`，没有 PID 分支，这与该 benchmark 数据提供形态一致。[R8]

---

## 5. 论文结果与项目默认配置的可复现性检查

## 5.1 论文关键数值（供你写 Related Work / Baseline）

以下数字来自 ParT 论文表格：

- JetClass（Table 1）
- ParticleNet: Accuracy `0.844`, AUC `0.9849`
- ParT: Accuracy `0.861`, AUC `0.9877`

- 训练集规模效应（Table 3）
- ParT: `2M -> 10M -> 100M` 时，Accuracy `0.836 -> 0.850 -> 0.861`
- 说明 ParT 对大规模数据更敏感。[R1]

- 模型规模与 FLOPs（Table 4）
- ParticleNet: `370k params, 540M FLOPs`
- ParT: `2.14M params, 340M FLOPs`
- 参数更多但 FLOPs 未必更高（实现路径不同）。[R1]

- Top benchmark（Table 5）
- ParT-f.t.: Accuracy `0.944`, AUC `0.9877`, Rej30 `2766±130`
- LorentzNet: Rej30 `2195±173`

- QG benchmark（Table 6）
- ParT-f.t.full: Accuracy `0.852`, AUC `0.9230`, Rej30 `138.7±1.3`
- ParTfull: Accuracy `0.849`, AUC `0.9203`, Rej30 `129.5±0.9`

## 5.2 你当前仓库与论文训练策略的一致度

高一致项：

- 特征维度定义（特别是 JetClass full=17）。
- optimizer 家族（ranger）。
- batch/lr 组合。
- fine-tune 头部高学习率策略。
- 基准模型集合（PFN/PCNN/ParticleNet/ParT）。

可能产生偏差项：

- 论文明确写了 JetClass 训练的 LR 衰减细节（70%常数+后续指数衰减）；脚本本身没有显式写调度细节，依赖 weaver 内部默认或CLI覆盖。
- 脚本默认 `num-workers` 与 `fetch-step` 在不同数据集上差异较大，可能影响吞吐和随机性。
- QG 的训练/验证切分在脚本中通过 `--train-val-split 0.8889` 在线切分，而非固定预切文件。

---

## 6. 与上游 weaver-core 的边界（你改代码前必须明确）

## 6.1 本仓库能做什么

- 换输入特征（YAML）
- 换训练策略（shell参数）
- 换微调头结构（`*_finetune.py`）
- 加实验组织（日志、目录、评估脚本）

## 6.2 本仓库做不到什么（除非改上游）

- 修改 P-MHA 内核公式
- 修改 pair embedding 的底层编码层实现
- 修改 attention/block 内核细节

论文也明确指出“full pairwise matrix”有时间和内存代价，这是 ParT 的先天瓶颈之一。[R1]

---

## 7. 2024-2026 相关工作给你的改进启发（联网补充）

## 7.1 一条明确路线：在交互分支做“更高效表达”

Chinese Physics C 2025 的 MIParT 工作声称：

- 通过更强调 interaction 分支（MIA），在部分指标超过 ParT；
- 同时减少参数与计算复杂度；
- 并在 JetClass 预训练 + 下游微调场景继续获益。[R9]

这条路线与你当前仓库结构高度兼容：

- 先在 wrapper/YAML 层做可插拔实验；
- 若趋势成立，再下沉到 weaver-core 主干做正式结构改动。

## 7.2 一条工程路线：加速与可解释联合

还有工作在研究 ParT 的可解释性/加速方向（attention pattern、硬件友好实现）。[R10]

对毕业论文更务实的落地方式是：

- 先做轻量可解释分析（attention map、feature ablation）；
- 再做结构瘦身（头数、pair维度、稀疏近似）。

---

## 8. 我建议你下一步直接做的实验包（可立刻开工）

## 8.1 基线固定

先固定这条基线，避免变量污染：

- 数据：JetClass full
- 模型：ParT
- 训练总样本预算：保持 512M 不变（和论文同量级）

## 8.2 三条最有论文产出的改造轴

1. **Pair 特征轴**：
- 保持 backbone 不动，只改 interaction feature 设计与归一化。
- 预期收益：指标改善 + 物理解释更明确。

2. **FineTune 轴**：
- 比较 `全参数微调` vs `部分冻结 + lr_mult` vs `逐层lr衰减`。
- 预期收益：小数据集泛化稳定性提升。

3. **效率轴**：
- 控制 FLOPs/吞吐：`pair_embed_dims`、`num_heads`、粒子截断长度。
- 预期收益：在接近性能下显著降算力。

## 8.3 评价指标建议

建议同时报告：

- Accuracy / AUC（通用）
- Rej@固定TPR（HEP关键）
- 参数量 / FLOPs / 吞吐（训练与推理）
- 稳定性（多seed均值和方差）

这样论文会同时具备“物理价值 + 工程价值”。

---

## 9. 风险清单（答辩时会被问）

1. `dataloader.py` 不是训练同款流水线，不能直接拿它的输入结果当 paper baseline。  
2. `weights:` 为空意味着默认不做类别重加权，长尾类指标要单独监控。  
3. QG/Top 的数据预处理是“转换后数据集”，与原始公开数据有中间处理层；论文里要写清楚转换假设。  
4. 当前仓库无环境锁文件，复现实验要自己固化 Python/PyTorch/weaver 版本。  

---

## 10. 报告附录：关键“论文-代码”映射点

- 论文式(3) 4个 pair 特征（`lnΔ, lnkT, lnz, lnm²`） -> 本仓库 `pair_input_dim=4`。  
- 论文式(4) `+U` 注意力偏置 -> wrapper 指向 weaver-core ParT 实现。  
- 论文 JetClass full 17粒子特征 -> `data/JetClass/JetClass_full.yaml`。  
- 论文 fine-tune 新头高学习率 -> `train_QuarkGluon.sh` / `train_TopLandscape.sh` 的 `lr_mult`。  

---

## 参考来源（联网）

- [R1] ICML/PMLR 论文主页与 PDF：Particle Transformer for Jet Tagging  
  https://proceedings.mlr.press/v162/qu22b  
  https://proceedings.mlr.press/v162/qu22b/qu22b.pdf

- [R2] 官方代码仓库：jet-universe/particle_transformer  
  https://github.com/jet-universe/particle_transformer

- [R3] JetClass Zenodo 数据集（10.5281/zenodo.6619768）  
  https://zenodo.org/records/6619768

- [R4] weaver-core PyPI（版本 0.4.17, 2024-08-06）  
  https://pypi.org/project/weaver-core/

- [R5] weaver-core GitHub 仓库与 README  
  https://github.com/hqucms/weaver-core

- [R6] QG Zenodo 数据集（10.5281/zenodo.3164691）  
  https://zenodo.org/records/3164691

- [R7] EnergyFlow datasets 文档（QG 推荐切分 1.6M/200k/200k）  
  https://energyflow.network/docs/datasets/

- [R8] Top benchmark 论文（SciPost Phys. 7, 014, 2019）  
  https://scipost.org/SciPostPhys.7.1.014

- [R9] MIParT（Chinese Physics C, 2025）  
  https://cpc.ihep.ac.cn/article/doi/10.1088/1674-1137/ad7f3d

- [R10] Interpreting and Accelerating Transformers for Jet Tagging（OSTI 索引）  
  https://www.osti.gov/biblio/2474973

