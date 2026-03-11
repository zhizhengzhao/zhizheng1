# Particle Transformer 项目深度解构报告

- 报告目标：帮助你在“改代码前”先建立完整心智模型（代码组织、数据流、训练逻辑、边界与可改造位点）。
- 代码基线：`particle_transformer` 仓库当前 `HEAD` 为 `2925bdb249e8ef78560cc2b9b651eda3615da8c7`（`main`）。
- 本报告范围：仅基于当前目录内源码；不会臆测未出现于代码中的行为。

## 1. 一句话结论

这个仓库**不是完整深度学习框架实现本体**，而是一个“实验工程封装层”：

- 核心模型实现（ParT/ParticleNet 主体）在外部依赖 `weaver-core`。
- 本仓库主要负责：
- 数据下载与格式转换。
- 特征工程配置（YAML）。
- 模型包装器与微调头定义。
- 训练命令编排（不同数据集/模型组合）。
- 提供预训练权重与教程 notebook。

你后续要做论文改进，最关键是先决定改动落在以下哪一层：

- 仓库内可直接改：数据表示、训练策略、微调方式、实验脚本化。
- 仓库外（weaver-core）才是主干：ParT 注意力与 Pair 特征细节、ParticleNet EdgeConv 细节。

## 2. 仓库快照

## 2.1 目录与体量

- tracked 文件：35 个。
- 仓库总大小（含 `.git` 和模型）：约 `59M`。
- 关键目录：
- `data/`：数据配置（YAML）
- `networks/`：模型包装脚本
- `utils/`：下载与格式转换
- `models/`：预训练权重（`.pt`）
- `notebooks/`：入门分析 notebook

## 2.2 代码角色分层

- L0 文档层：`README.md`。
- L1 数据层：`get_datasets.py`、`utils/*`、`dataloader.py`、`data/*.yaml`。
- L2 模型层：`networks/*.py`（包装与接口，不是完整主干实现）。
- L3 训练编排层：`train_*.sh`。

## 2.3 版本历史特征

- 总提交可见：从 `Initial commit` 到当前约 13 次。
- 后期更新集中在：
- 数据下载 URL 与转换脚本。
- Notebook 和 dataloader 辅助函数。
- 与 `weaver-core>=0.4` 对齐。

## 3. 端到端运行路径（非常关键）

一次完整实验典型路径如下：

1. 配置数据路径：`env.sh`。  
2. 下载/准备数据：`get_datasets.py`（或手动转换）。  
3. 选择训练脚本：`train_JetClass.sh` / `train_QuarkGluon.sh` / `train_TopLandscape.sh`。  
4. 训练脚本把参数传给 `weaver`：
- `--data-config data/...yaml`
- `--network-config networks/...py`
- 学习率、batch、epoch、优化器等
5. `weaver` 读取 YAML 做特征构造与标准化，调用 network 脚本中的 `get_model/get_loss`。
6. network 脚本再调用 `weaver.nn.model.ParticleTransformer` 或 `ParticleNet`（真实主干）。
7. 输出 artifacts 到 `training/`、`logs/`、`pred.root`、tensorboard 名称空间。

这条链决定了改造入口：

- 改“输入语义”优先改 YAML/转换脚本。
- 改“模型骨干”要进 `weaver-core`。
- 改“实验策略”改 `train_*.sh` 与 network wrapper。

## 4. README 与项目定位

`README.md` 明确仓库定位与能力边界：

- 官方实现 + 预训练模型 + JetClass 数据说明。
- 训练依赖 `weaver-core>=0.4`。
- 支持四类模型名：`ParT`、`PN`、`PFN`、`PCNN`。
- 支持三个数据集：`JetClass`、`QuarkGluon`、`TopLandscape`。

关键点：README 已经暗示“本仓库的 ParT 代码是通过 weaver 调用”，不是在本仓库自己重写完整 Transformer。

## 5. 数据系统详解

## 5.1 `env.sh`

文件非常简单，只有三个环境变量占位：

- `DATADIR_JetClass`
- `DATADIR_TopLandscape`
- `DATADIR_QuarkGluon`

训练脚本都会先 `source env.sh`，然后：

- 若对应变量为空，则 fallback 到 `./datasets/<DatasetName>`。

## 5.2 `get_datasets.py`：下载与 env 回写

功能结构：

- `datasets` 字典定义 URL + MD5。
- `download_dataset(...)` 下载并解压，然后把 `env.sh` 中对应 `DATADIR_*` 行替换为新路径。

三个数据集策略：

- JetClass：
- 10 个训练分片 tar（`part0` 到 `part9`）。
- 1 个验证 tar（5M）+ 1 个测试 tar（20M）。
- TopLandscape：1 个 tar（已转换版本）。
- QuarkGluon：1 个 tar（已转换版本）。

实现细节：

- `--force` 会先删目录再重下。
- 只对“本次下载动作”做解压：若文件已存在且 hash 通过，不会再解压。
- 解压路径用 `extract_archive(fpath, path=os.path.join(datadir, subdir))`。

观察：Top/Quark 的 `subdir` 写成 `'../'`，意味着解压目标是 `datadir/..`（即上一级目录），依赖压缩包内部目录结构。

## 5.3 `utils/dataset_utils.py`：下载基础设施

能力：

- 流式下载（`requests + tqdm`）。
- md5/sha256 校验。
- tar/zip 自动识别与解压。
- 缓存命中时先校验 hash。

行为细节：

- 默认 hash 算法 md5。
- 下载失败会删除不完整文件。
- 校验不通过会删除并抛错。

潜在工程注意点：

- `extractall` 没有路径穿越防护（通用 tar 风险点）。
- 无重试、无超时显式配置。
- `requests.exceptions.RequestException` 里用 `e.msg`，不同异常对象不一定有该字段。

## 5.4 `dataloader.py`：JetClass 单文件读取器

这是一个“轻量读取接口”，用于把 ROOT 文件转为常规 numpy 张量：

- 输入：`filepath`。
- 输出：
- `x_particles`: `(N, C_particle, Pmax)`
- `x_jets`: `(N, C_jet)`
- `y`: `(N, n_classes)`

关键逻辑：

- 读取 `tree.arrays()`。
- 用 `part_px/py/pz/energy` 现算 `part_pt/eta/phi`。
- `_pad(...)` 负责截断/补零到 `max_num_particles`（默认 128）。
- `np.stack(..., axis=1)` 把多变量拼成通道维。

重要边界：

- 默认 `particle_features` 只有 4 项（`pt,eta,phi,energy`），与训练 YAML 中的高阶特征集合不是同一套。
- 不做 log/clip/标准化等训练变换。

## 5.5 数据转换脚本

### 5.5.1 `utils/convert_qg_datasets.py`

输入来源：`QG_jets*.npz`（EnergyFlow 数据）。

处理流程：

- `X` 形状注释为 `(num_data, max_num_particles, 4)`，四列 `(pt, y, phi, pid)`。
- 先按 `pt` 降序重排每个 jet 的粒子。
- 通过 `pt>0` 形成真实粒子 mask，构造成 jagged array。
- 用 `vector` 从 `(pt,eta,phi,mass=0)` 变到 `(px,py,pz,E)`。
- 构造 jet 级量与粒子级量（`part_deta/dphi`、PID one-hot 派生）。
- 写成 parquet（LZ4 压缩）。

切分策略：

- `--train-test-split` 默认 0.9。
- 是“按文件级”分 train/test，不是逐事件随机混合。

### 5.5.2 `utils/convert_top_datasets.py`

输入来源：TopLandscape 的 `train.h5/val.h5/test.h5`。

处理流程：

- 每事件最多 200 粒子列（`PX_i/PY_i/PZ_i/E_i`）。
- `E>0` 判定有效粒子。
- 生成 p4、jet 级量、`part_deta/dphi`。
- 标签来自 `is_signal_new`。
- 输出为 `train_file.parquet`、`val_file.parquet`、`test_file.parquet`。

## 6. YAML 特征工程体系（核心）

所有 YAML 共享骨架：

- `selection`
- `new_variables`
- `preprocess`
- `inputs`
- `labels`
- `observers`
- `weights`（当前均为空）

## 6.1 输入张量约定

四组输入固定命名：

- `pf_points`：二维几何点（通常 `part_deta`, `part_dphi`）
- `pf_features`：模型主特征
- `pf_vectors`：四动量向量分量
- `pf_mask`：粒子有效位

长度与 padding：

- 长度统一 `128`。
- `pf_points/pf_features/pf_vectors` 用 `pad_mode: wrap`。
- `pf_mask` 用 `pad_mode: constant`。

## 6.2 预处理策略

- `preprocess.method: manual`
- 给出显式平移缩放参数，例如：
- `part_pt_log: [subtract 1.7, multiply 0.7]`
- `part_deltaR: [subtract 0.2, multiply 4.0]`

含义：标准化参数不是在线估计，而是固定手工常量，保证可复现一致性。

## 6.3 JetClass 三套特征

### 6.3.1 `JetClass_kin.yaml`

`pf_features` 共 7 维：

- `part_pt_log`
- `part_e_log`
- `part_logptrel`
- `part_logerel`
- `part_deltaR`
- `part_deta`
- `part_dphi`

### 6.3.2 `JetClass_kinpid.yaml`

在 `kin` 基础上增加 PID 与电荷相关，变成 13 维。

### 6.3.3 `JetClass_full.yaml`

在 `kinpid` 基础上增加轨迹位移相关：

- `part_d0 = tanh(part_d0val)`
- `part_dz = tanh(part_dzval)`
- `part_d0err`、`part_dzerr` 被 clip 到 `[0,1]`

总计 17 维。

标签：10 类 one-hot：

- `QCD, Hbb, Hcc, Hgg, H4q, Hqql, Zqq, Wqq, Tbqq, Tbl`

## 6.4 QuarkGluon 三套特征

### 6.4.1 `qg_kin.yaml`

- 二分类标签：`jet_isQ=label`, `jet_isG=1-label`
- 7 维运动学特征。

### 6.4.2 `qg_kinpid.yaml`

- 13 维（加入 `part_charge` 与标准 PID one-hot）。

### 6.4.3 `qg_kinpidplus.yaml`

- 13 维，但两项 hadron 特征改为“加权编码”：
- `part_isCHad`: π/K/p 加权 (1,0.5,0.2)
- `part_isNHad`: K0/n 加权 (1,0.2)
- 这依赖 `part_pid`（由 qg 转换脚本提供）。

## 6.5 TopLandscape

`top_kin.yaml`：

- 二分类标签：`jet_isTop`, `jet_isQCD`。
- 仅 7 维运动学特征。
- 不包含 PID 相关字段（与源数据结构一致）。

## 7. 模型脚本层解构（`networks/`）

## 7.1 统一接口约定

每个 network 文件都实现：

- `get_model(data_config, **kwargs)`
- `get_loss(data_config, **kwargs)`

`get_model` 返回 `(model, model_info)`，其中 `model_info` 定义：

- 输入名与输入形状。
- 输出名（`softmax`）。
- ONNX 动态轴描述（batch 维与粒子长度维）。

## 7.2 Particle Transformer（非 fine-tune）

文件：`networks/example_ParticleTransformer.py`

关键配置：

- `embed_dims=[128,512,128]`
- `pair_embed_dims=[64,64,64]`
- `pair_input_dim=4`
- `num_heads=8`
- `num_layers=8`
- `num_cls_layers=2`
- `activation='gelu'`
- `trim=True`
- `fc_params=[]`

前向：

- wrapper 的 `forward(points, features, lorentz_vectors, mask)` 中，实际只把 `features`、`lorentz_vectors`、`mask` 传给 `ParticleTransformer`。

额外：

- `no_weight_decay()` 返回 `{ 'mod.cls_token' }`。

## 7.3 Particle Transformer（fine-tune）

文件：`networks/example_ParticleTransformer_finetune.py`

机制：

- 先构建自己的 `self.fc`（基于 `fc_params` 逐层 MLP，最后线性到新类别）。
- 再把传给主干的 `num_classes` 和 `fc_params` 置为 `None`，让 backbone 输出 cls embedding。
- `forward`: `x_cls = self.mod(...)` 后接 `self.fc(x_cls)`。

这意味着：

- fine-tune 不是“冻结骨干 + 只训头”硬编码。
- 是否主要训练头由训练参数控制（本仓库通过 `lr_mult` 提高头部学习率）。

## 7.4 ParticleNet（非 fine-tune）

文件：`networks/example_ParticleNet.py`

默认结构参数：

- `conv_params = [(16,(64,64,64)), (16,(128,128,128)), (16,(256,256,256))]`
- `fc_params = [(256,0.1)]`
- `use_fusion=False`
- `use_fts_bn=True`
- `use_counts=True`

前向只用 `points/features/mask`，不使用 `lorentz_vectors`。

## 7.5 ParticleNet（fine-tune）

文件：`networks/example_ParticleNet_finetune.py`

机制：

- 从 `fc_params[-1][0]` 读入维度，新建 `fc_out = Linear(in_dim, num_classes)`。
- backbone 用 `ParticleNet(**kwargs)` 构建后，`self.mod.fc = self.mod.fc[:-1]` 去掉原最后分类层。
- 前向：`x_cls = self.mod(...)` 再过 `fc_out`。

同样：未显式冻结 backbone 参数。

## 7.6 PFN

文件：`networks/example_PFN.py`

结构：

- `phi`：3 层 `Conv1d(kernel=1)+ReLU`（默认 128,128,128）。
- 逐粒子表示经过 mask 后沿粒子维求和（Deep Sets 聚合）。
- `F`：3 层全连接 ReLU + 输出层。

默认 `get_model` 中 `use_bn=False`（即输入和 phi BN 关闭）。

## 7.7 PCNN

文件：`networks/example_PCNN.py`

实现是 1D ResNet 风格：

- 初始特征卷积 `BN -> Conv1d(k=3) -> BN -> ReLU`。
- 多 stage `ResNetUnit`，stage 首单元（除第0 stage）有步长下采样 `(2,1)`。
- 全局平均池化后接 FC（默认 `(512,0.2)` + 输出层）。

输入中仅 `features` 与 `mask` 有效，`points/lorentz_vectors` 未使用。

## 8. 训练脚本逐个剖析

## 8.1 `train_JetClass.sh`

关键特性：

- 支持 DataParallel 和 DDP。
- DDP 触发条件：环境变量 `DDP_NGPUS>1`。
- DDP 命令：`torchrun ... $(which weaver) --backend nccl`。

默认训练参数：

- `epochs=50`
- `samples_per_epoch = 10000*1024/NGPUS`
- `samples_per_epoch_val = 10000*128`
- `--num-workers 2 --fetch-step 0.01`

模型对应默认超参：

- `ParT`: `batch 512`, `lr 1e-3`, `--use-amp`
- `PN`: `batch 512`, `lr 1e-2`
- `PFN/PCNN`: `batch 4096`, `lr 2e-2`

数据模式：

- train：10 类分别给 pattern（带类别前缀）。
- val：统一 `val_5M/*.root`。
- test：10 类分别 pattern。

其它：

- 优化器固定 `ranger`。
- `--gpus 0` 默认写死，可通过附加参数覆盖。
- 脚本第 3 个参数开始直接透传给 `weaver`。

## 8.2 `train_QuarkGluon.sh`

关键特性：

- 支持 6 模式：`ParT/ParT-FineTune/PN/PN-FineTune/PFN/PCNN`。
- 默认特征类型 `kinpid`。
- `ParT` 系列加 `weight_decay 0.01`。

数据读取：

- train：`train_file_*.parquet`
- test：`test_file_*.parquet`
- validation 通过 `--train-val-split 0.8889` 从 train 切分。

固定训练参数：

- `num-epochs=20`
- `batch-size=512`（PFN/PCNN override 为 4096）
- `samples-per-epoch=1,600,000`
- `samples-per-epoch-val=200,000`
- `--in-memory`

微调策略：

- `ParT-FineTune`: `lr=1e-4` + `lr_mult ("fc.*",50)` + `--lr-scheduler none`
- `PN-FineTune`: `lr=1e-3` + `lr_mult ("fc_out.*",50)` + `--lr-scheduler none`
- 预训练权重按 `FEATURE_TYPE` 映射：
- `kin -> *_kin.pt`
- `kinpid/kinpidplus -> *_kinpid.pt`

## 8.3 `train_TopLandscape.sh`

与 qg 类似，差异：

- 只允许 `FEATURE_TYPE=kin`。
- 显式给 `--data-val val_file.parquet`（不是 train-val split）。
- `samples-per-epoch = 2400*512`
- `samples-per-epoch-val = 800*512`
- fine-tune 固定加载 `*_kin.pt`。

## 9. 预训练模型文件

`models/` 下共 6 个权重：

- ParT: `ParT_kin.pt`, `ParT_kinpid.pt`, `ParT_full.pt`
- ParticleNet: `ParticleNet_kin.pt`, `ParticleNet_kinpid.pt`, `ParticleNet_full.pt`

体量对比：

- ParT 单个约 8.2-8.3M
- ParticleNet 单个约 1.4-1.5M

用途：

- 主要在 qg/top 的 fine-tune 模式中通过 `--load-model-weights` 使用。

## 10. Notebook 角色

`notebooks/JetClass101.ipynb` 是“数据理解教程”，核心价值：

- 展示 ROOT tree 的列结构。
- 演示 jagged array 到定长张量的 `_pad`。
- 演示如何从原始列构造训练特征和标签。
- 其中 `build_features_and_labels` 与 `JetClass_full.yaml` 基本一致，是理解 YAML 语义的最佳对照样例。

## 11. 依赖与可复现性

## 11.1 直接可见依赖

- 训练/模型：`torch`, `weaver-core>=0.4`
- HEP 数据：`uproot`, `awkward`, `vector`
- 下载：`requests`, `tqdm`
- 转换：`pandas`, `tables`, `numpy`

## 11.2 可复现性现状

优点：

- 特征标准化参数写死在 YAML。
- 训练入口脚本集中，命令模板明确。
- 预训练权重随仓库提供。

不足：

- 无 `requirements.txt` / `environment.yml` / 锁版本文件。
- 无随机种子统一管理脚本。
- 无 CI 或最小回归测试。

## 12. 关键边界与“误解高发点”

1. “这个仓库里有完整 ParT 代码”是误解。  
真正主干在 `weaver-core`；这里是 wrapper + config。

2. `dataloader.py` 不是训练同款数据管线。  
它只是通用读取器，默认特征很简，不含 YAML 的完整变换。

3. fine-tune 不等于冻结 backbone。  
当前仅通过 `lr_mult` 强调新头；骨干并未硬冻结。

4. 三个数据集的验证集来源并不一致。  
JetClass 有独立 val，qg 走 train-val split，top 有独立 val。

5. `weights:` 在 YAML 里为空。  
即默认不做类别重加权，loss 为标准交叉熵。

## 13. 逐文件速查（你做改动前可直接定位）

- `README.md`：项目说明、命令入口、引用规范。
- `env.sh`：数据路径环境变量。
- `get_datasets.py`：下载与 env 回写。
- `dataloader.py`：ROOT 单文件到 numpy。
- `train_JetClass.sh`：JetClass 训练总入口（含 DDP）。
- `train_QuarkGluon.sh`：qg 训练与微调。
- `train_TopLandscape.sh`：top 训练与微调。
- `networks/example_ParticleTransformer.py`：ParT 包装器。
- `networks/example_ParticleTransformer_finetune.py`：ParT 微调头。
- `networks/example_ParticleNet.py`：ParticleNet 包装器。
- `networks/example_ParticleNet_finetune.py`：ParticleNet 微调头。
- `networks/example_PFN.py`：PFN 基线。
- `networks/example_PCNN.py`：PCNN/ResNet 基线。
- `data/JetClass/*.yaml`：JetClass 三档特征。
- `data/QuarkGluon/*.yaml`：qg 三档特征。
- `data/TopLandscape/top_kin.yaml`：top 特征。
- `utils/dataset_utils.py`：下载/校验/解压底层。
- `utils/convert_qg_datasets.py`：qg npz -> parquet。
- `utils/convert_top_datasets.py`：top h5 -> parquet。
- `models/*.pt`：预训练权重。
- `notebooks/JetClass101.ipynb`：数据教程。

## 14. 改进切入点地图（供你后续论文选题）

下面不是“结论”，而是可选改造方向清单：

1. 数据层：
- 重新设计 `new_variables`（物理先验增强、鲁棒变换、可解释派生量）。
- 比较 `pad_mode` 与最大粒子数对性能/速度影响。

2. 训练层：
- 统一/对比 JetClass 与 qg/top 的验证策略。
- 系统化学习率计划（当前 fine-tune 常用 `--lr-scheduler none`）。
- 更细粒度冻结策略、layer-wise lr decay。

3. 模型层：
- 在 wrapper 层加轻量模块（额外 token、投影头、蒸馏头）。
- 深改则进入 `weaver-core` 的 ParT/ParticleNet 主干。

4. 工程层：
- 完整环境锁定和实验追踪模板。
- 下载与解压的健壮性增强（重试、timeout、path safe extract）。
- 最小可运行测试（smoke test）和结果回归。

## 15. 你下一步最省时间的实践建议

如果目标是“论文里做有说服力的改进”，建议先用这条路线：

1. 先固定一个 baseline 组合（如 `JetClass + ParT + full`）跑通一次。  
2. 只改一个变量（特征 or 微调策略 or 结构）做可控对比。  
3. 把改动约束在 YAML 或 wrapper，先验证趋势，再决定是否下沉到 `weaver-core` 深改。  

---

## 附录 A：关键代码定位（行号）

- `dataloader.py`：
- `read_file` 入口：第 8 行。
- `_pad`：第 85 行。
- p4 衍生 `pt/eta/phi`：第 104-110 行。

- `get_datasets.py`：
- 数据源字典：第 9-40 行。
- 下载与解压循环：第 50-54 行。
- 回写 `env.sh`：第 56-64 行。

- `train_JetClass.sh`：
- DDP 分支：第 16-23 行。
- 模型选择：第 30-47 行。
- 训练命令：第 61-91 行。

- `train_QuarkGluon.sh`：
- 微调学习率倍率：第 25、32 行。
- 预训练权重注入：第 59-64 行。

- `train_TopLandscape.sh`：
- fine-tune 载入权重：第 24、31 行。

- `networks/example_ParticleTransformer.py`：
- 默认 cfg：第 26-44 行。
- `no_weight_decay`：第 16-18 行。

- `networks/example_ParticleTransformer_finetune.py`：
- 自定义头构建：第 21-26 行。
- backbone 关分类头：第 28-30 行。

- `networks/example_ParticleNet_finetune.py`：
- 删除原最后 FC：第 24 行。
- 新头 `fc_out`：第 20 行。

- `utils/convert_qg_datasets.py`：
- 按 pt 排序：第 37-43 行。
- PID 派生特征：第 83-98 行。

- `utils/convert_top_datasets.py`：
- HDF 列展开：第 34-40 行。
- 标签来源：第 56 行。

- `data/JetClass/JetClass_full.yaml`：
- `part_d0/part_dz`：第 15-16 行。
- `part_d0err/part_dzerr` clip 参数：第 54、56 行。
