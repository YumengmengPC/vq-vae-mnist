# VQ-VAE MNIST 复现

基于论文 *Neural Discrete Representation Learning*（van den Oord et al., NIPS 2017, [arXiv:1711.00937](https://arxiv.org/abs/1711.00937)）在 MNIST 上复现 VQ-VAE 的两阶段训练与条件生成流程：先用 VQ-VAE 把 28×28 的手写数字压缩成 7×7 的离散索引网格，再训练条件 Gated PixelCNN 先验 `p(z | label)`，实现输入数字 0-9、生成对应类别全新手写数字图像。

![条件生成结果](assets/grid_50tests.png)

## 复现结果

| 阶段 | 指标 | 数值 |
|------|------|------|
| Stage 1（VQ-VAE，3.06M 参数） | 测试集重建 MSE | 0.0071 |
| | 活码数 | 64 / 64（无死码） |
| | Perplexity / 利用率 | 49.1 / 76.7% |
| Stage 2（Gated PixelCNN，8.17M 参数） | 验证集 loss（best） | 2.0958 nats/token @ 6K 步 |
| | bits/dim（仅先验） | 0.1890 |
| 生成 | CNN 辨识度（主指标） | 98%（100 张） |
| | NMC 辨识度（次指标） | 88%（真实 MNIST 在该指标下为 82%） |
| | 50 张人眼验收 | 50 / 50 全部可辨识 |

上图为 0-9 每类 5 次独立采样的验收网格（T=0.7, top-k=16）。原始 28×28 像素拼接的干净版本见 `assets/grid_50tests_clean.png`。

## 方法

### Stage 1：VQ-VAE 自编码器

```
28×28 图像 -> Encoder（两次 stride-2 下采样）-> 7×7×16 连续特征
           -> 向量量化（K=64 码本，查最近邻）-> 7×7 离散索引
           -> 查码本 -> Decoder（两次上采样）-> 28×28 重构
```

向量量化层做了三处工程修正，均针对复现中实测出现的问题：

1. **数据驱动初始化**：码本初值直接取首个 batch 的 encoder 输出样本。论文惯例的 `uniform(-1/K, 1/K)` 初始化与 encoder 输出量级相差约 1600 倍，导致几乎所有样本 argmin 到同一小撮码字、其余码字梯度恒为零而永久死亡。改用数据驱动初始化后，首步 vq_loss 从 611 降到 0.023，死码彻底消失。
2. **EMA 码本更新**（decay=0.99，Laplace 平滑），码本不参与梯度，损失只剩承诺项。
3. **死码重置**：每 200 步把 EMA 计数低于 1.0 的码字重置为当前 batch 的随机样本加噪声。

### Stage 2：条件 Gated PixelCNN 先验

```
7×7 索引网格 + 类别标签
  -> 索引 Embedding（64 -> 128）
  -> GatedMaskedConv2d × 12（首层 mask A，其余 mask B；垂直栈 + 水平栈 + 门控激活）
     每层独立注入类别条件（类别 -> 门控偏置）
  -> 输出投影 -> 每格 64 个码字的 logits
```

自回归采样 49 步得到索引网格，查码本后经 Stage 1 decoder 解码成图像。类别条件在每一层注入而非只在第一层注入一次，是修掉「让它画 2 却画出 Q」这类类别混淆的关键。因果性已用梯度依赖图验证：7×7 隐空间上无泄漏、无盲点。

两阶段流程图见 `assets/stage1_flowchart.png` 与 `assets/stage2_flowchart.png`，架构总览见 `assets/vqvae_architecture.png`。

## 快速开始

### 环境

- Python 3.10+
- PyTorch 2.x（CUDA 可选，CPU 也能跑，采样会慢一些）

```bash
pip install -r requirements.txt
```

MNIST 数据集在首次运行时自动下载到 `./data`。

### 训练

两个阶段各 20K 步，GPU（RTX 级别）上 Stage 1 约 34 分钟，Stage 2 见 `logs/`：

```bash
python stage1.py    # 训练 VQ-VAE，保存 vqvae_7x7_ema_v2.pth
python stage2.py    # 载入 Stage 1，训练先验，保存 prior_7x7_gated.pth
```

### 生成与评估

```bash
python generate.py         # 交互式：输入 0-9 出一张对应数字的图，q 退出
python run_50_tests.py     # 0-9 各 5 张共 50 张，拼网格图（验收用）
python assess_quality.py   # CNN + NMC 双指标辨识度评估（生成 100 张 + 混淆矩阵）
python sweep_sampling.py   # 温度 × top-k 网格搜索，确定采样默认值
```

首次运行评估类脚本时，若 `mnist_classifier.pth` 不存在，`classifier.py` 会自动训练一个评估用 CNN（1-2 分钟）。

### 采样超参

默认 `T=0.7 / top-k=16`，由 `sweep_sampling.py` 网格搜索确定。温度越低字形越规整、辨识度越高，代价是同类样本趋同。搜索目标不是最高准确率，而是辨识度达标前提下的最高温度，把多样性损失压到最小。

### 评估口径

- **CNN 准确率**（主指标）：用 99.11% 的 MNIST CNN 判断生成图是否是它该是的数字。
- **NMC 准确率**（次指标）：最近均值分类器，在真实 MNIST 测试集上也只有 82.0%，到 82% 即为指标饱和。
- 分类器只用于评分，不参与生成回路：本项目不使用拒绝采样，50/50 的验收由模型本身与采样超参达成。

## 目录结构

```
├── models.py            # 模型定义（VQ-VAE + Gated PixelCNN）+ 权重文件名常量
├── stage1.py            # Stage 1 训练（VQ-VAE，20K 步）
├── stage2.py            # Stage 2 训练（条件 Gated PixelCNN 先验，20K 步）
├── sampling.py          # 批量采样公共实现（温度 + top-k）
├── generate.py          # 交互式条件生成
├── run_50_tests.py      # 50 张批量生成验收
├── sweep_sampling.py    # 采样超参网格搜索
├── classifier.py        # 评估专用 MNIST CNN（只评分，不参与生成）
├── assess_quality.py    # CNN + NMC 双指标评估
├── docs/
│   ├── REPORT.md                # 完整技术复现报告
│   ├── stage2-notes.md          # Stage 2 原理与全过程文档
│   └── vqvae-section3-notes.md  # 论文第 3 节（向量量化）深度笔记
├── logs/                # 两阶段训练日志
└── assets/              # 结果图与流程图
```

## 关键实验结论

复现过程中通过六轮架构迭代得到的结论，完整记录见 `docs/REPORT.md` 第 8 章：

1. **重建质量不等于生成质量**，两者由不同因素主导。
2. **重建质量取决于信息容量 = token 数 × log₂K，与码字维度 D 无关**。D 从 64 降到 16，同数重建持平，利用率反而更高。
3. **生成质量额外取决于离散表示的规整性**：stride-3 下采样窗口无重叠会产生混叠（数字 3 的弧线被拍成 8）；死码越多，离散表示区分度越差。
4. **死码的根因是初始化尺度失配**，不是梯度版码本天生易塌缩。EMA 之所以幸存，是因为它把码字直接赋值为 z_e 均值，一步跨过尺度鸿沟。
5. **活码数与 perplexity 利用率是两回事**：前者衡量有没有码字死掉，后者衡量使用分布是否均匀。MNIST 笔画模式出现频率本就不均等，追求 90%+ 利用率等于要求真实数据服从均匀分布。
6. **PixelCNN 盲点严重程度随隐空间尺寸急剧放大**：7×7 上朴素掩码卷积盲点为 0，28×28 上 783 个上文位置里 704 个是盲区。在 7×7 上换 Gated 的实际收益来自每层注入类别条件。

## 致谢

- 原论文：van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). *Neural Discrete Representation Learning*. NIPS 2017. [arXiv:1711.00937](https://arxiv.org/abs/1711.00937)
- Gated PixelCNN：van den Oord, A., et al. (2016). *Conditional Image Generation with PixelCNN Decoders*. [arXiv:1606.05328](https://arxiv.org/abs/1606.05328)

## License

MIT License，见 [LICENSE](LICENSE)。
