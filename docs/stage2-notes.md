# VQ-VAE Stage 2 全过程：条件 PixelCNN 先验

> 相关脚本：`models.py`（模型定义）/ `stage2.py`（训练）/ `generate.py`（生成）
> 训练日志：`stage2_train.log`
> 最后更新：2026-08-03

本文档记录 Stage 2 的完整运行过程与设计原理。`stage2.py` 中的注释已精简为与
`stage1.py` 一致的密度（只保留必要的结构与参数注释），原先写在代码里的原理性说明
全部转移到本文档。

模型定义统一放在 `models.py`，由 `stage1.py` / `stage2.py` / `generate.py` 三个脚本
共同 import——改架构只需改这一处。

---

## 1. Stage 2 要解决什么问题

Stage 1 做的是**压缩**：encoder 把 28×28×1 的图变成 7×7 的整数索引网格（每格取值
0~127），decoder 再还原回去。但它**不会"生成"**——没法凭空造出一张新图，因为它不知道
什么样的索引组合才是合理的。随机抽 49 个 0~127 的整数解码出来只会是噪声。

Stage 2 补上这一块，并且做成**条件生成**：训练一个自回归模型 `p(z | label)`，其中
label 是 0-9 的数字类别。训练完成后，给定一个类别，按光栅顺序逐格采样出一张全新的索引图，
查码本、送进 Stage 1 的 decoder，就得到一张该类别的手写数字图像。

```
stage1        = 压缩器（重构已有的图）
stage1+stage2 = 条件生成器（输入 0-9 → 输出对应数字的新图）
```

**为什么先验必须单独训练**：VQ-VAE 原论文的做法是先固定 encoder/decoder/codebook，
再在离散潜变量上拟合先验。如果两者一起训，码本会持续漂移，先验刚学到的索引分布立刻失效。

---

## 2. 运行全过程

脚本是顺序执行的，从上到下分为 6 个阶段。

### 阶段 0：环境与配置（`stage2.py:1-53`）

- `torch.manual_seed(415)`，选择 device（有 CUDA 用 CUDA）
- `matplotlib.use('Agg')`——WSL 无显示环境，必须用非交互后端才能存图
- 读取配置：Stage 1 权重路径、输出文件名、超参数
- `os.makedirs(FIG_DIR)` 建立 `训练效果/` 目录

### 阶段 1：载入并冻结 Stage 1（`stage2.py:59-72`）

```
检查 best_vqvae_mnist_lr1e-3_10k_ne128.pth 是否存在 → 不存在直接报错退出
VQVAE(**VQVAE_CONFIG) → load_state_dict → .eval() → requires_grad_(False)
```

Stage 1 存的是**裸 state_dict**（不含 config），所以模型超参没法从权重文件恢复，
由 `models.py` 里的 `VQVAE_CONFIG` 常量提供（256/2/32/128/64/0.25），三个脚本共用同一份。
改动架构时只需改 `models.py`，否则 `load_state_dict` 会 shape mismatch。

> 权重文件名里的 `ne128` 指 num_embeddings=128。若该文件是旧的 512 版本，会在这一步报错。

### 阶段 2：把整个数据集编码成索引（`stage2.py:74-104`）

```python
encode_dataset(dataset):
    for data, label in DataLoader(batch_size=256):
        idx_list.append(vqvae.encode_indices(data))    # (B,7,7) 取值 0~127
```

`encode_indices` 定义在 `models.py` 里，内部就是 `_pre_vq_conv(_encoder(x))` 再过 VQ 取索引。

关键点：Stage 1 已冻结，**索引在整个训练过程中固定不变**，所以没必要每步重跑 encoder。
60000 张图的索引只占 `60000×7×7×8B ≈ 2.6MB`，直接常驻显存，后续训练变成纯粹的索引采样，
省掉了全部数据加载与 encoder 前向的开销。

实际输出：
```
train (60000, 7, 7)  test (10000, 7, 7)  取值 [0,127]  类别数 10
```

### 阶段 3：构造条件 PixelCNN（定义在 `models.py`，构造在 `stage2.py:110`）

```
indices (B,7,7) + label (B,)
  → idx_emb: Embedding(128 → 128)         索引无序，须过 embedding 变向量
  → input_conv: MaskedConv2d('A', 7×7)     第一层因果，中心也遮
  → + class_emb(label).view(B,C,1,1)       条件注入，broadcast 到所有空间位置
  → PixelCNNResidualBlock × 6              1×1降维 → 3×3掩码B → dropout → 1×1升维 + 残差
  → output_proj: ReLU→1×1→ReLU→1×1 → 128   每格 128 类 logits
输出 (B,128,7,7)
```

参数量 **1.17M**。

### 阶段 4：训练先验（`stage2.py:107-195`）

15000 步，每步：

```python
pick   = torch.randint(0, 60000, (128,))    # 有放回随机采样，非 epoch 式遍历
batch  = train_indices[pick]                # (128,7,7) 直接从显存取
logits = prior(batch, labels)               # (128,128,7,7)
loss   = F.cross_entropy(logits.float(), batch)   # 目标就是输入自己
```

- **loss 就是把输入当标签**：自回归模型预测每一格的索引，因果掩码保证它看不到答案
- `cross_entropy` 对 `(B,128,7,7)` vs `(B,7,7)` 原生支持，等价于 49 个位置各做一次
  128 类分类再取平均，单位是 **nats/token**
- 开启 **AMP**（`autocast` + `GradScaler`），logits 在算 loss 前 `.float()` 回全精度避免数值问题
- 每 1000 步在测试集前 20 个 batch 上评估，val loss 创新低就保存 checkpoint
  （含 iteration / state_dict / val_loss / config / vqvae_ckpt 来源）

**bits/dim 换算**（`BITS_PER_DIM_SCALE`）：latent 7×7=49 个 token 摊到 28×28×1=784 个
像素分量上，`nats → bits` 再除以 784：

```
bits/dim = nats/token × 49 / (ln2 × 784) = nats/token × 0.09017
```

注意这个数值**只统计先验，不包含 Stage 1 的重建误差**，不能直接和其他生成模型的
bits/dim 横向比较。

### 阶段 5：条件采样（`stage2.py:198-244`）

载入 best checkpoint，然后：

```python
indices = zeros(B,7,7)
for r in range(7):
    for c in range(7):
        logits = prior(indices, labels)[:, :, r, c] / temperature
        indices[:, r, c] = multinomial(softmax(logits), 1)
```

- **必须串行跑满 49 次前向**：第 (r,c) 格的分布依赖它前面所有已采样的格子。这是自回归
  模型采样慢的根本原因——latent 边长翻倍，采样步数变 4 倍
- 每次前向都算了整张 7×7 图但只用其中一格，是 O(n²) 的朴素实现。可以用缓存优化，
  但 49 步规模下没有必要
- **按概率采样而不是 argmax**：argmax 是确定性的，会让同一类别的 8 张样本全部塌成同一张图
- `temperature` 控制多样性，默认 1.0

采样完成后：索引 → 查码本 `_embedding(indices)` 得 (B,7,7,64) → permute 成 (B,64,7,7)
→ Stage 1 decoder → (B,1,28,28)。

反归一化：Stage 1 用 `Normalize(0.5, 0.5)` 把图映射到 [-1,1]，逆操作是 `x*0.5+0.5`
再 `clamp(0,1)`。

生成 0-9 每类 8 张共 80 张，排成 10 行 × 8 列存为 `训练效果/stage2_samples_mnist_cond.png`。

### 阶段 6：绘制训练曲线（`stage2.py:247-269`）

train loss 用 201 点滑动平均（`np.convolve`）平滑后画左图，val loss 画右图并在标题
标出最优值与对应步数，存为 `训练效果/stage2_training_curves_mnist_cond.png`。

---

## 3. 核心设计原理

### 3.1 因果掩码 MaskedConv2d

自回归的核心约束：预测第 i 格时只能看到第 0..i-1 格，不能看到自己和后面的格子。
用带掩码的卷积把"未来"的权重置零：

```
    mask A                    mask B
    1 1 1 1 1                 1 1 1 1 1
    1 1 1 1 1                 1 1 1 1 1
    1 1 0 0 0   ← 中心也遮      1 1 1 0 0   ← 中心可见
    0 0 0 0 0                 0 0 0 0 0
    0 0 0 0 0                 0 0 0 0 0
```

- **A 型只能用在第一层**：那时"当前格"就是要预测的目标本身，看到它就是信息泄漏
- **B 型用在之后所有层**：中心位置的特征已经是"由前面格子算出来的表示"，可以放行
- **1×1 输出投影不跨空间，不破坏因果性，不需要掩码**

实现细节（`models.py:164-165`）：用 `weight * mask` 而**不是**原地修改 `weight.data`。
后者会让被遮的权重虽然被置零、却仍在优化器状态里累积动量，下一步又被"复活"再置零，
造成隐蔽的错误。乘法版本的梯度天然为零。

**因果性验证**：对 `output[:, :, 3, 4]` 求和后反传到输入 embedding，统计有非零梯度的格子。
(3,4) 在光栅序中是第 25 格，检查结果为依赖恰好 25 格（即第 0~24 格）、违规 0 格。

### 3.2 类别条件注入

`class_emb(label)` 得到一个 (B, C) 的向量，reshape 成 (B, C, 1, 1) 后加到第一层掩码
卷积的输出上，**broadcast 到全部 7×7 个空间位置**。之后经过 6 个残差块的掩码卷积传播，
每个位置的预测都能感知类别信息。

这是最简单的条件注入方式。原论文用的是更复杂的门控结构（gated PixelCNN），这里因为
MNIST 足够简单没有必要。

### 3.3 为什么索引要过 Embedding

码字编号是**无序的类别标签**：码字 5 和码字 6 之间没有大小关系，也不代表相似。
如果直接把整数索引当数值喂给卷积，网络会错误地假设 5 和 6 比 5 和 100 更接近。
过一层 `nn.Embedding` 让每个码字学到自己的向量表示，消除这个伪序关系。

---

## 4. 超参数

| 参数 | 值 | 说明 |
|------|---|------|
| `prior_hiddens` | 128 | PixelCNN 通道数 |
| `prior_res_layers` | 6 | 残差块层数 |
| `prior_lr` | 3e-4 | Adam |
| `prior_batch_size` | 128 | — |
| `prior_dropout` | 0.1 | 仅在残差块内 |
| `num_iterations` | 15000 | — |
| `eval_interval` | 1000 | 评估 + 保存 best |
| `num_classes` | 10 | MNIST 数字 0-9 |
| AMP | 开（CUDA 时） | `GradScaler` |
| 随机种子 | 415 | 仅 `torch.manual_seed` |

继承自 Stage 1 且不可改动的量：`num_embeddings=128`、`embedding_dim=64`、latent 7×7。

MNIST 的 latent 是 7×7（比 CIFAR 的 8×8 略小）且数据简单，所以网络规模比 CIFAR 配置小一档。

---

## 5. 训练结果

| 步数 | train loss | val loss | bits/dim |
|------|-----------|----------|----------|
| 1000 | 2.8546 | 2.5407 | 0.2291 |
| 2000 | 2.4516 | 2.4214 | 0.2183 |
| 3000 | — | 2.3662 | 0.2134 |
| 4000 | — | 2.3281 | 0.2099 |
| 5000 | — | 2.3017 | 0.2075 |
| 14000 | 2.1167 | 2.2402 | 0.2020 |
| **15000** | **2.1057** | **2.2377** | **0.2018** |

单位为 nats/token（每格 128 类分类的交叉熵）。

**最佳 val loss 出现在最后一步（15000）**，说明训练尚未收敛到过拟合——train/val gap
只有 0.13，继续训练大概率还能小幅改善。当前的 15000 步是主动截断而非早停。

参考量级：均匀分布的 baseline 是 `ln(128) = 4.852` nats/token。先验把它降到 2.2377，
说明确实学到了索引之间的空间结构与类别相关性。

---

## 6. 产出文件

| 文件 | 内容 |
|------|------|
| `best_pixelcnn_prior_mnist_cond.pth` | 最佳先验权重（含 config 与来源 ckpt 记录），8.8MB |
| `训练效果/stage2_samples_mnist_cond.png` | 条件生成 grid，10 类 × 8 张 |
| `训练效果/stage2_training_curves_mnist_cond.png` | train/val loss 曲线 |
| `stage2_train.log` | 完整训练日志 |

---

## 7. 日常生成：generate.py

`stage2.py` 是**训练**脚本，跑完即退出，结尾那张 10×8 的 grid 只是训练效果验证。
日常生成图像用 `generate.py`——纯推理，载入两份现成权重后进入交互循环，不需要重新训练。

```bash
/usr/bin/python3.12 generate.py              # 交互模式
/usr/bin/python3.12 generate.py 2026 --n 4   # 生成一次即退出
```

```
> 3                 生成 8 张数字 3
> 3 --n 16          生成 16 张
> 2026              4 行分别是 2/0/2/6，每行 8 张
> 7 --temp 0.8      调温度
> q                 退出
```

结果存入 `生成结果/gen_<数字>_<序号>.png`，序号自动递增不覆盖历史。
GPU 上生成 16 张约 1 秒（49 步自回归，与张数基本无关，因为一批并行）。

先验超参不在脚本里硬编码，而是从 `best_pixelcnn_prior_mnist_cond.pth` 的 `config`
字段读取，所以换 checkpoint 不用改代码。

### 温度的实际影响

同样输入 `2026 --n 4` 的实测对比：

| temperature | 效果 |
|---|---|
| 1.0（默认） | 16 张里约 3–4 张笔画畸形，多样性高 |
| **0.8** | 16 张全部可辨认，字形更规整，代价是同类样本更接近 |

**推荐日常用 0.8。** 温度越低越接近 argmax，规整但趋于雷同；越高越多样但畸形率上升。

## 8. 复现命令

```bash
# 注意：PATH 中 conda 的 python3 优先但未装 torch，必须显式用 /usr/bin/python3.12
/usr/bin/python3.12 stage1.py 2>&1 | tee stage1_train.log   # 先跑完 Stage 1
/usr/bin/python3.12 stage2.py 2>&1 | tee stage2_train.log
```

前置条件：`best_vqvae_mnist_lr1e-3_10k_ne128.pth` 存在且为 `num_embeddings=128` 版本。

---

## 9. 与原论文的差异

| 维度 | 原论文 | 本项目 |
|------|--------|--------|
| 数据集 | CIFAR-10 / ImageNet | MNIST |
| 先验条件 | 无条件（ImageNet 上有类别条件） | 类别条件，label embedding 注入 |
| PixelCNN 结构 | Gated PixelCNN（垂直/水平双栈） | 朴素掩码卷积 + 残差块 |
| Codebook 更新 | EMA | 梯度反向传播 |
| 训练步数 | 250K–500K | Stage1 10K + Stage2 15K |

---

## 10. 已知问题

- 随机种子只设了 `torch.manual_seed(415)`，缺 `cuda` / `numpy` / `random`，复现性不完整
- 采样是朴素 O(n²) 实现，每步重算整图，未做缓存优化
- `evaluate()` 固定只用测试集**前** 20 个 batch（2560 张），不是全量测试集
- 默认温度 1.0 下畸形率偏高（约 20~25%），需手动降到 0.8

## 11. 可能的后续工作

- [ ] 补全随机种子（cuda/numpy/random）
- [x] ~~把模型定义抽到 `models.py`，三个脚本共同 import~~
- [ ] 继续训练到 val loss 真正平台期
- [ ] 实现 Gated PixelCNN，对比朴素掩码卷积
- [ ] 无条件先验版本，与条件版对比生成质量
- [ ] 添加 FID / IS 等生成质量定量指标
