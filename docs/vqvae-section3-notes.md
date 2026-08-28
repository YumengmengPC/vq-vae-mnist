# VQ-VAE 第三节深度解读：向量量化的数学原理与直觉

> 基于论文 *Neural Discrete Representation Learning* (van den Oord et al., NIPS 2017) 第三节及附录 A.1

---

## 1. 为什么要"离散化"？—— 从连续到离散的动机

### 直觉

想象你在描述一张人脸照片。你不会说"第 37 个像素是 0.423 灰度"，而是会说"这是一张脸，有眼睛、鼻子、嘴巴"。**人类认知天然是离散的** —— 我们用有限的概念（"眼睛""红色""圆形"）去描述无限丰富的世界。

传统 VAE 学到的隐变量是**连续**的（比如一个 64 维的高斯向量），这在生成时有优势，但也带来了两个问题：

1. **后验坍缩 (Posterior Collapse)**：当解码器太强大时，它会"绕过"隐变量直接建模数据，导致隐变量被完全忽略，变成无用的噪声。
2. **缺乏可解释性**：连续向量的每个维度很难对应到语义上有意义的概念。

VQ-VAE 的核心思想是：**把隐变量变成离散的"码字" (codeword)**，就像人类用有限的词汇去描述世界一样。

---

## 2. 离散隐变量空间 (Discrete Latent Space)

### 2.1 定义嵌入空间

> 论文原文：*"We define a latent embedding space $\mathbf{e} \in \mathbb{R}^{K \times D}$ where $K$ is the size of the discrete latent space (i.e., a K-way categorical), and $D$ is the dimensionality of each latent embedding vector $\mathbf{e}_i$."*

**嵌入空间 (embedding space)** 是一个"码本" (codebook)，可以类比为一本词典：

$$
\mathbf{e} = \begin{bmatrix} \mathbf{e}_1 \\ \mathbf{e}_2 \\ \vdots \\ \mathbf{e}_K \end{bmatrix} \in \mathbb{R}^{K \times D}
$$

- $K$：码本大小 (codebook size)，即"词典中有多少个词"。论文中 CIFAR-10 用 $K=512$，本复现项目用 $K=128$。
- $D$：每个码字的维度 (embedding dimension)，论文和本项目均用 $D=64$。
- $\mathbf{e}_k \in \mathbb{R}^D$：第 $k$ 个码字，是码本中的第 $k$ 个"标准模板向量"。

**通俗理解**：码本就像一张"调色板"，上面有 $K$ 种颜色（每个颜色是 $D$ 维向量）。模型的任务就是为编码器输出的每个位置选一个最接近的颜色。

### 2.2 后验分布：确定性的 one-hot

> 论文原文 (公式 1)：*"The posterior categorical distribution $q(z|x)$ probabilities are defined as one-hot."*

对于编码器输出的一个向量 $\mathbf{z}_e(x)$，我们定义它属于第 $k$ 个码字的后验概率为：

$$
q(z = k \mid x) = \begin{cases} 1 & \text{if } k = \underset{j}{\arg\min} \; \|\mathbf{z}_e(x) - \mathbf{e}_j\|_2 \\ 0 & \text{otherwise} \end{cases}
$$

**关键点**：

- 这是一个 **确定性 (deterministic)** 分布，不是随机采样 —— 给定输入 $x$，编码结果 $z$ 是唯一确定的。
- 这和普通 VAE 有本质区别：普通 VAE 的后验 $q(z|x)$ 是高斯分布，需要重参数化技巧 (reparameterization trick) 来采样；VQ-VAE 的后验是 **one-hot 向量**，不需要采样。
- 由于后验是确定性的、先验是均匀分布，KL 散度项退化为常数 $\log K$，训练时可以忽略。

---

## 3. 量化过程：最近邻查找 (Nearest Neighbour Look-up)

### 3.1 前向计算

> 论文原文 (公式 2)：*"$z_q(x) = e_k$, where $k = \arg\min_j \|z_e(x) - e_j\|_2$"*

编码器的输出 $\mathbf{z}_e(x)$ 经过"量化瓶颈" (discretisation bottleneck)，被映射到码本中距离最近的码字：

$$
\mathbf{z}_q(x) = \mathbf{e}_k, \quad k = \underset{j}{\arg\min} \; \|\mathbf{z}_e(x) - \mathbf{e}_j\|_2
$$

**直觉**：这就像把一张照片中的每个像素都"四舍五入"到调色板上最接近的颜色。原始颜色可能有上百万种，但量化后只剩下 $K$ 种。

### 3.2 多维特征图的情况

论文指出，对于图像，编码器实际上输出的是一个 **2D 特征图** $\mathbf{z}_e(x) \in \mathbb{R}^{H' \times W' \times D}$。量化是在特征图的每个空间位置上**独立进行**的：

$$
\text{对每个位置 } (h, w): \quad \mathbf{z}_q^{(h,w)} = \mathbf{e}_{k^{(h,w)}}, \quad k^{(h,w)} = \underset{j}{\arg\min} \; \|\mathbf{z}_e^{(h,w)} - \mathbf{e}_j\|_2
$$

在本项目中，编码器输出形状为 $[B, 256, 7, 7]$，经过 $1 \times 1$ 卷积降维后变为 $[B, 64, 7, 7]$。量化在 $7 \times 7 = 49$ 个位置上分别进行，每个位置从 128 个码字中选一个。

### 3.3 代码实现中的 L2 距离计算

```python
# stage1.py 第 63 行
distances = (torch.sum(flat_x ** 2, dim=1, keepdim=True)     # ||z||^2
           + torch.sum(self._embedding.weight ** 2, dim=1)    # ||e||^2
           - 2 * torch.matmul(flat_x, self._embedding.weight.t()))  # -2·z·e^T
encoding_indices = torch.argmin(distances, dim=1)
```

这里用了一个巧妙的数学展开来高效计算 L2 距离：

$$
\|\mathbf{z} - \mathbf{e}\|_2^2 = \|\mathbf{z}\|_2^2 + \|\mathbf{e}\|_2^2 - 2 \mathbf{z}^\top \mathbf{e}
$$

**推导**：
$$
\|\mathbf{z} - \mathbf{e}\|_2^2 = (\mathbf{z} - \mathbf{e})^\top (\mathbf{z} - \mathbf{e}) = \mathbf{z}^\top\mathbf{z} - 2\mathbf{z}^\top\mathbf{e} + \mathbf{e}^\top\mathbf{e}
$$

**为什么这么做？** 直接计算所有 $(z, e)$ 对的距离需要 $O(N \times K \times D)$ 次减法 + 乘法。而展开后：
- $\|\mathbf{z}\|^2$ 可以预先算好 ($O(N \times D)$)
- $\|\mathbf{e}\|^2$ 可以预先算好 ($O(K \times D)$)
- $\mathbf{z}^\top\mathbf{e}$ 用矩阵乘法一次算完 ($O(N \times K \times D)$ 但高度并行化)

---

## 4. 梯度流：直通估计器 (Straight-Through Estimator)

### 4.1 问题：量化操作不可微

$\arg\min$ 操作是一个**离散选择**，没有有意义的梯度。具体来说：

$$
\frac{\partial \mathbf{z}_q}{\partial \mathbf{z}_e} = \frac{\partial \mathbf{e}_k}{\partial \mathbf{z}_e}
$$

由于 $\mathbf{e}_k$ 是码本中的固定向量，它并不直接依赖于 $\mathbf{z}_e$（只是**选择**依赖于 $\mathbf{z}_e$），所以这个梯度要么是零，要么未定义。这意味着梯度无法从解码器传回编码器 —— 整个网络无法端到端训练。

### 4.2 解决方案：直通估计器 (STE)

> 论文原文：*"Note that there is no real gradient defined for equation 2, however we approximate the gradient similar to the straight-through estimator and just copy gradients from decoder input $z_q(x)$ to encoder output $z_e(x)$."*

STE 的思想极其简单：**前向传播用离散值，反向传播假装量化没发生**。

$$
\hat{\mathbf{z}}_q = \mathbf{z}_e + \text{sg}(\mathbf{z}_q - \mathbf{z}_e)
$$

其中 $\text{sg}(\cdot)$ 是 **stop-gradient 算子**：

$$
\text{sg}(x) = \begin{cases} x & \text{(前向传播时)} \\ 0 & \text{(反向传播时，即 } \frac{\partial \text{sg}(x)}{\partial x} = 0 \text{)} \end{cases}
$$

**推导 STE 的梯度**：

前向传播时：
$$
\hat{\mathbf{z}}_q = \mathbf{z}_e + (\mathbf{z}_q - \mathbf{z}_e) = \mathbf{z}_q \quad \text{(结果就是离散的量化值)}
$$

反向传播时（对 $\mathbf{z}_e$ 求导）：
$$
\frac{\partial \hat{\mathbf{z}}_q}{\partial \mathbf{z}_e} = \frac{\partial}{\partial \mathbf{z}_e} [\mathbf{z}_e + \text{sg}(\mathbf{z}_q - \mathbf{z}_e)] = \mathbf{I} + 0 = \mathbf{I}
$$

也就是说：**解码器的梯度 $\nabla_{\mathbf{z}_q} L$ 被原封不动地复制给了编码器输出 $\mathbf{z}_e$**。

**直觉**：想象你在超市找最近的收银台。前向传播时，你走到最近的 3 号收银台（离散选择）。反向传播时，假设有人告诉你"往左走两步会更好"，你不会站在 3 号台原地不动，而是会把这个方向信息传递给"你在超市里的位置"—— 下次你可能就会走到更近的 2 号台。

### 4.3 代码实现

```python
# stage1.py 第 72 行
quantized = x + (quantized - x).detach()
```

- `x`：编码器输出 $\mathbf{z}_e$（需要梯度）
- `quantized`：量化后的 $\mathbf{z}_q$（查表得到的固定值）
- `.detach()`：PyTorch 的 stop-gradient 操作，等价于论文中的 $\text{sg}(\cdot)$

---

## 5. 三部分损失函数

### 5.1 总损失

> 论文原文 (公式 3)：*"Thus, the total training objective becomes..."*

$$
\mathcal{L} = \underbrace{-\log p(x \mid \mathbf{z}_q(x))}_{\text{重建损失}} + \underbrace{\|\text{sg}[\mathbf{z}_e(x)] - \mathbf{e}\|_2^2}_{\text{码本损失 (VQ loss)}} + \underbrace{\beta \|\mathbf{z}_e(x) - \text{sg}[\mathbf{e}]\|_2^2}_{\text{承诺损失 (commitment loss)}}
$$

### 5.2 逐项分析

#### 损失 1：重建损失 (Reconstruction Loss)

$$
\mathcal{L}_{\text{recon}} = -\log p(x \mid \mathbf{z}_q(x))
$$

在实践中通常用 MSE 近似：

$$
\mathcal{L}_{\text{recon}} = \text{MSE}(x, \hat{x}) = \frac{1}{N} \sum_{i} (x_i - \hat{x}_i)^2
$$

- **优化对象**：编码器 + 解码器（通过 STE 传梯度）
- **作用**：让解码器学会从离散码字重建输入，同时引导编码器输出"有用"的特征

#### 损失 2：码本损失 / VQ 损失 (Embedding Loss)

$$
\mathcal{L}_{\text{vq}} = \|\text{sg}[\mathbf{z}_e(x)] - \mathbf{e}\|_2^2
$$

- **$\text{sg}$ 加在 $\mathbf{z}_e$ 上**：编码器输出被视为常量，只有码本参数 $\mathbf{e}$ 接收梯度
- **优化对象**：仅码本 embedding
- **作用**：把码字"拉向"编码器输出，让码本学会编码数据的真实分布
- **本质**：这是一种最简单的**字典学习** (dictionary learning) 算法

**数学展开**：

$$
\frac{\partial \mathcal{L}_{\text{vq}}}{\partial \mathbf{e}_k} = 2(\mathbf{e}_k - \mathbf{z}_e) \quad \text{(当 } k = \text{选中的码字索引)}
$$

这就是让 $\mathbf{e}_k$ 向 $\mathbf{z}_e$ 移动，类似于 K-Means 中把聚类中心移向簇内点的均值。

#### 损失 3：承诺损失 (Commitment Loss)

$$
\mathcal{L}_{\text{commit}} = \beta \|\mathbf{z}_e(x) - \text{sg}[\mathbf{e}]\|_2^2
$$

- **$\text{sg}$ 加在 $\mathbf{e}$ 上**：码本被视为常量，只有编码器输出 $\mathbf{z}_e$ 接收梯度
- **优化对象**：仅编码器
- **作用**：约束编码器输出不要"跑太远"，必须承诺 (commit) 到某个码字附近
- **$\beta$ 的作用**：控制承诺强度。论文发现 $\beta \in [0.1, 2.0]$ 都很稳健，实验中统一用 $\beta = 0.25$

**为什么需要承诺损失？** 论文原文解释得很清楚：

> *"Since the volume of the embedding space is dimensionless, it can grow arbitrarily if the embeddings $\mathbf{e}_i$ do not train as fast as the encoder parameters."*

没有承诺损失的话，编码器输出的范数可以无限增长，码本来不及跟上，导致量化完全失效。承诺损失就像一个"弹性绳"，把编码器输出拴在码本附近。

### 5.3 梯度流总结

| 损失项 | 编码器梯度 | 码本梯度 | 解码器梯度 |
|--------|-----------|---------|-----------|
| $\mathcal{L}_{\text{recon}}$ | ✅ (通过 STE) | ❌ | ✅ |
| $\mathcal{L}_{\text{vq}}$ | ❌ (sg 阻断) | ✅ | ❌ |
| $\mathcal{L}_{\text{commit}}$ | ✅ | ❌ (sg 阻断) | ❌ |

### 5.4 代码实现

```python
# stage1.py 第 68-70 行
e_latent_loss = F.mse_loss(quantized, x.detach())         # L_vq: sg 在 x 上
q_latent_loss = F.mse_loss(quantized.detach(), x)         # L_commit: sg 在 quantized 上
loss = e_latent_loss + self._commitment_cost * q_latent_loss  # β = 0.25
```

注意 `x.detach()` 和 `quantized.detach()` 的位置完美对应了论文公式中 $\text{sg}$ 的位置。

---

## 6. 码本更新的替代方案：指数移动平均 (EMA)

> 附录 A.1：*"One can also use exponential moving averages (EMA) to update the dictionary items instead of the loss term."*

### 6.1 从 K-Means 视角理解

码本损失 $\|\text{sg}[\mathbf{z}_e] - \mathbf{e}\|_2^2$ 的闭式解（对一个完整 batch）是：

$$
\mathbf{e}_i = \frac{1}{n_i} \sum_{j=1}^{n_i} \mathbf{z}_{i,j}
$$

其中 $\{\mathbf{z}_{i,1}, \mathbf{z}_{i,2}, \dots, \mathbf{z}_{i,n_i}\}$ 是被分配到码字 $\mathbf{e}_i$ 的所有编码器输出。这正是 **K-Means 聚类** 中更新聚类中心的方式。

### 6.2 EMA 在线更新

在小 batch 训练中，直接用均值更新不稳定。论文提出用指数移动平均 (EMA) 来做在线版本：

$$
N_i^{(t)} = \gamma \cdot N_i^{(t-1)} + (1 - \gamma) \cdot n_i^{(t)} \tag{6}
$$

$$
\mathbf{m}_i^{(t)} = \gamma \cdot \mathbf{m}_i^{(t-1)} + (1 - \gamma) \cdot \sum_j \mathbf{z}_{i,j}^{(t)} \tag{7}
$$

$$
\mathbf{e}_i^{(t)} = \frac{\mathbf{m}_i^{(t)}}{N_i^{(t)}} \tag{8}
$$

其中：
- $N_i^{(t)}$：第 $i$ 个码字的 EMA 计数（加权历史使用次数）
- $\mathbf{m}_i^{(t)}$：第 $i$ 个码字的 EMA 累加和
- $\gamma = 0.99$：衰减系数，控制历史权重
- 最终码字 = EMA 累加和 / EMA 计数

**直觉**：EMA 就像"加权平均"—— 最近 batch 的数据权重高，历史数据权重按 $\gamma^t$ 指数衰减。这比梯度更新更稳定，因为它是直接"统计"数据分布，而不是通过梯度下降"摸索"。

### 6.3 本项目的实现差异

本项目 **没有使用 EMA**，而是使用论文公式 (3) 中的梯度更新方式：

$$
\mathcal{L}_{\text{vq}} = \|\text{sg}[\mathbf{z}_e] - \mathbf{e}\|_2^2
$$

码本参数 $\mathbf{e}$ 作为 `nn.Embedding` 的一部分，由 Adam 优化器通过梯度下降更新。这是更简单的方式，但在某些情况下可能导致 **codebook collapse**（部分码字永远不被使用）。

---

## 7. 先验分布与生成

### 7.1 训练期间：均匀先验

> 论文原文：*"Whilst training the VQ-VAE, the prior is kept constant and uniform."*

$$
p(z = k) = \frac{1}{K}, \quad \forall k \in \{1, 2, \dots, K\}
$$

由于后验 $q(z|x)$ 是确定性的 one-hot，先验是均匀分布，KL 散度为：

$$
\text{KL}(q(z|x) \| p(z)) = \sum_k q(z=k|x) \log \frac{q(z=k|x)}{p(z=k)} = 1 \cdot \log \frac{1}{1/K} = \log K
$$

这是一个**常数**，对编码器参数没有梯度，因此训练时可以安全地忽略。

### 7.2 训练后：自回归先验

> 论文原文：*"After training, we fit an autoregressive distribution over $z$, $p(z)$, so that we can generate $x$ via ancestral sampling."*

训练完 VQ-VAE 后，码本已固定。接下来训练一个**自回归模型**来学习离散码字的联合分布：

- 图像：使用 **PixelCNN** 建模 $p(z_1, z_2, \dots, z_N) = \prod_i p(z_i \mid z_{<i})$
- 音频：使用 **WaveNet**

生成新样本的流程：
1. 从 PixelCNN 逐位置采样离散码字 $z_1, z_2, \dots, z_N$
2. 查码本得到量化向量 $\mathbf{z}_q = [\mathbf{e}_{z_1}, \mathbf{e}_{z_2}, \dots, \mathbf{e}_{z_N}]$
3. 送入解码器得到生成样本 $\hat{x}$

### 7.3 对数似然的近似

论文给出了完整模型对数似然的推导：

$$
\log p(x) = \log \sum_k p(x \mid z_k) p(z_k)
$$

由于解码器是用 MAP 推理（最近邻查找）训练的，收敛后对于 $z \neq z_q(x)$，$p(x|z)$ 应该接近 0。因此：

$$
\log p(x) \approx \log p(x \mid z_q(x)) + \log p(z_q(x))
$$

由 Jensen 不等式还可以得到下界：

$$
\log p(x) \geq \log p(x \mid z_q(x)) + \log p(z_q(x))
$$

---

## 8. Codebook 利用率：困惑度 (Perplexity)

### 8.1 定义

困惑度用于衡量码本中码字的使用均匀度：

$$
\text{Perplexity} = \exp\left(-\sum_{k=1}^{K} p_k \log p_k\right)
$$

其中 $p_k$ 是第 $k$ 个码字在一个 batch 中被选中的频率：

$$
p_k = \frac{\text{码字 } k \text{ 被选中的次数}}{\text{总查找次数}}
$$

### 8.2 取值范围与意义

| 情况 | Perplexity | 含义 |
|------|-----------|------|
| 只用了 1 个码字 | 1 | Codebook 完全坍缩 |
| 均匀使用所有 $K$ 个码字 | $K$ | 最佳利用率 |
| 本项目 ($K=128$) | 越接近 128 越好 | — |

**直觉**：Perplexity 可以理解为"**有效使用的码字数量**"。如果 128 个码字中只有 30 个被频繁使用，其余几乎不用，那 Perplexity 大约就是 30 左右。

### 8.3 与熵的关系

$$
H(Z) = -\sum_{k=1}^{K} p_k \log p_k \quad \text{(信息熵)}
$$

$$
\text{Perplexity} = e^{H(Z)} = 2^{H(Z)/\ln 2}
$$

Perplexity 就是**指数化的熵**。当分布最均匀时，$H(Z) = \log K$，$\text{Perplexity} = K$；当分布最集中时，$H(Z) = 0$，$\text{Perplexity} = 1$。

### 8.4 代码实现

```python
# stage1.py 第 75-78 行
avg_probs = torch.bincount(
    encoding_indices, minlength=self._num_embeddings
).float() / encoding_indices.numel()
perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
```

`1e-10` 是为了避免 $\log(0)$ 的数值问题。

---

## 9. 与 ELBO 的关系

VQ-VAE 可以被理解为一个特殊的 VAE，其 ELBO (Evidence Lower Bound) 为：

$$
\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - \text{KL}(q(z|x) \| p(z))
$$

在 VQ-VAE 中：
- $q(z|x)$ 是确定性的 one-hot → 期望退化为单点取值
- $p(z)$ 是均匀分布 → $\text{KL} = \log K$ (常数)

因此 ELBO 简化为：

$$
\mathcal{L}_{\text{ELBO}} = \log p(x \mid z_q(x)) - \log K
$$

由于 $\log K$ 是常数，优化 ELBO 等价于优化重建似然 $\log p(x \mid z_q(x))$。论文中的码本损失和承诺损失**不是来自 ELBO**，而是为了实现端到端训练而额外引入的辅助损失。

---

## 10. 总结：VQ-VAE 的完整训练流程

```
输入 x
  │
  ▼
编码器 → z_e(x) ∈ R^D         (连续向量)
  │
  ▼
最近邻查找 → z_q(x) = e_k      (离散码字，不可微)
  │                                │
  │  STE: 前向用 z_q              │  码本损失: sg[z_e] → e
  │       反向梯度直通 z_e        │  承诺损失: z_e → sg[e]
  │                                │
  ▼
解码器 → x_hat                   (重建)
  │
  ▼
重建损失: MSE(x_hat, x)
```

**三个损失各司其职**：
1. **重建损失**：训练编码器（通过 STE）和解码器 —— "重建要像"
2. **码本损失**：训练码本 —— "码字要贴近编码器输出"
3. **承诺损失**：训练编码器 —— "编码器输出别跑太远"

**与传统 VAE 的三个核心区别**：
1. 隐变量是**离散的** (one-hot)，而非连续高斯
2. 后验是**确定性的** (argmin)，而非随机采样
3. KL 项是**常数**，不约束编码器；取而代之的是码本损失 + 承诺损失
