"""VQ-VAE 与条件 PixelCNN 的模型定义，由 stage1.py / stage2.py / generate.py 共用。

改动这里的架构会让已有的 .pth 权重失效（load_state_dict 报 shape mismatch），
需要重新训练。原理说明见 stage2_全过程.md。

当前版本（v2）相对早期版本的三处关键修正，见各处注释：
  1. 码本改为数据驱动初始化 —— 修掉死码的根因（尺度失配）
  2. 码本改回 EMA 更新 + 死码重置 —— 维持利用率
  3. 先验改为 Gated PixelCNN —— 消除盲点，并让类别条件注入每一层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 权重文件名集中在这里，四个下游脚本（stage1/stage2/generate/assess_quality/run_50_tests）
# 都从这里 import，避免改架构时要同步改五处硬编码路径
VQVAE_CKPT = 'vqvae_7x7_ema_v2.pth'
PRIOR_CKPT = 'prior_7x7_gated.pth'

# stage1 存的是裸 state_dict（无 config），这里给出与已有权重匹配的默认配置
VQVAE_CONFIG = dict(
    num_hiddens=256,
    num_residual_layers=2,
    num_residual_hiddens=32,
    num_embeddings=64,        # K=64：匹配 MNIST 数据复杂度
    embedding_dim=16,         # D=16：见 VectorQuantize 文档字符串「为什么降维」
    commitment_cost=0.25,     # β=0.25：论文原值。早期为救利用率降到 0.02，根因修好后不再需要
)


# 第一阶段：VQ-VAE

class VectorQuantize(nn.Module):
    """
    向量量化层（EMA 版 + 数据驱动初始化 + 死码重置）。
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99, eps=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._decay = decay
        self._eps = eps

        # 码本与 EMA 统计量都是 buffer（不参与梯度，但要存进 checkpoint）
        self.register_buffer('_embedding', torch.randn(num_embeddings, embedding_dim))
        self.register_buffer('ema_count', torch.ones(num_embeddings))
        self.register_buffer('ema_weight', torch.randn(num_embeddings, embedding_dim))
        # 标记码本是否已用数据初始化过；存进 ckpt 以免推理时被重新初始化
        self.register_buffer('_initialized', torch.zeros(1, dtype=torch.bool))

        # 最近一批 encoder 输出，供死码重置取样。不注册为 buffer：属于临时状态，不该进 ckpt
        self._last_flat = None

    @torch.no_grad()
    def _init_codebook_from_data(self, flat_x):
        """用当前 batch 的 encoder 输出初始化码本，消除初始化尺度失配。"""
        n = flat_x.size(0)
        if n >= self._num_embeddings:
            pick = torch.randperm(n, device=flat_x.device)[:self._num_embeddings]
        else:  # batch 行数不足 K 时带重复抽样
            pick = torch.randint(0, n, (self._num_embeddings,), device=flat_x.device)
        self._embedding.copy_(flat_x[pick])
        self.ema_weight.copy_(self._embedding)
        self.ema_count.fill_(1.0)
        self._initialized.fill_(True)

    @torch.no_grad()
    def reset_dead_codes(self, threshold=1.0):
        """把长期未被选中的码字重置为最近一批 z_e 中的随机向量，返回重置个数。

        由训练循环定期调用（见 stage1.py）。EMA 计数低于 threshold 即视为死码。
        """
        if not bool(self._initialized) or self._last_flat is None:
            return 0
        dead = self.ema_count < threshold
        n_dead = int(dead.sum())
        if n_dead == 0:
            return 0
        src = self._last_flat
        pick = torch.randint(0, src.size(0), (n_dead,), device=src.device)
        # 加噪避免多个死码被重置到同一点后再次退化成一个
        new = src[pick] + 0.01 * torch.randn(n_dead, self.embedding_dim, device=src.device)
        self._embedding[dead] = new
        self.ema_weight[dead] = new
        self.ema_count[dead] = 1.0
        return n_dead

    def forward(self, x):
        x = x.float()
        x = x.permute(0, 2, 3, 1).contiguous()
        x_shape = x.shape
        flat_x = x.view(-1, self.embedding_dim)

        if self.training and not bool(self._initialized):
            self._init_codebook_from_data(flat_x.detach())

        # L2 距离展开式，找最近码字
        distances = (torch.sum(flat_x ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding ** 2, dim=1)
                     - 2 * torch.matmul(flat_x, self._embedding.t()))
        encoding_indices = torch.argmin(distances, dim=1)

        quantized = F.embedding(encoding_indices, self._embedding).view(x_shape)

        if self.training:
            self._last_flat = flat_x.detach()
            self._ema_update(flat_x.detach(), encoding_indices)

        # EMA 版码本不走梯度，损失里只剩承诺项（stop-grad 在码本，只更新 encoder）
        loss = self._commitment_cost * F.mse_loss(quantized.detach(), x)

        # Straight-Through Estimator
        quantized = x + (quantized - x).detach()

        # 困惑度：codebook 实际被用到的平均码字数，越接近 num_embeddings 说明利用率越高
        avg_probs = torch.bincount(
            encoding_indices, minlength=self._num_embeddings
        ).float() / encoding_indices.numel()
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        encoding_indices = encoding_indices.view(x_shape[0], x_shape[1], x_shape[2])
        return loss, quantized, perplexity, encoding_indices

    @torch.no_grad()
    def _ema_update(self, flat_x, encoding_indices):
        """EMA 更新码本：码字 ← 分配到它的 z_e 的滑动平均，配 Laplace 平滑防除零。"""
        onehot = F.one_hot(encoding_indices, self._num_embeddings).to(flat_x.dtype)  # (N,K)
        batch_count = onehot.sum(dim=0)                                              # (K,)
        batch_sum = onehot.t() @ flat_x                                              # (K,D)

        self.ema_count.mul_(self._decay).add_(batch_count, alpha=1 - self._decay)
        self.ema_weight.mul_(self._decay).add_(batch_sum, alpha=1 - self._decay)

        # Laplace 平滑：把计数抬离 0，避免刚复活的码字被除爆
        n = self.ema_count.sum()
        smoothed = (self.ema_count + self._eps) / (n + self._num_embeddings * self._eps) * n
        self._embedding.copy_(self.ema_weight / smoothed.unsqueeze(1))


# 定义残差连接块类
class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super().__init__()
        self._block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, num_residual_hiddens, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(num_residual_hiddens, num_hiddens, 1, 1),
        )

    def forward(self, x):
        return x + self._block(x)


# 定义编码器（两次下采样：28 → 14 → 7）
class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, num_hiddens // 2, 4, 2, 1),   # 28→14
            nn.ReLU(),
            nn.Conv2d(num_hiddens // 2, num_hiddens, 4, 2, 1),   # 14→7
            nn.ReLU(),
            nn.Conv2d(num_hiddens, num_hiddens, 3, 1, 1),        # 7→7 保持
            *[Residual(num_hiddens, num_hiddens, num_residual_hiddens)
              for _ in range(num_residual_layers)],
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# 定义解码器（两次上采样：7 → 14 → 28）
class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, num_hiddens, 3, 1, 1),        # 7→7 保持
            *[Residual(num_hiddens, num_hiddens, num_residual_hiddens)
              for _ in range(num_residual_layers)],
            nn.ReLU(),
            nn.ConvTranspose2d(num_hiddens, num_hiddens, 4, 2, 1),        # 7→14
            nn.ReLU(),
            nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, 4, 2, 1),   # 14→28
            nn.ReLU(),
            nn.Conv2d(num_hiddens // 2, 1, 3, 1, 1),              # 28→28 保持，输出 1 通道
        )

    def forward(self, x):
        return self.net(x)


# 定义VQ-VAE模型
class VQVAE(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self._encoder = Encoder(1, num_hiddens, num_residual_layers, num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(num_hiddens, embedding_dim, 1, 1)
        self._vq_vae = VectorQuantize(num_embeddings, embedding_dim, commitment_cost)
        self._decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)

    def forward(self, x):
        z = self._pre_vq_conv(self._encoder(x))
        loss, quantized, perplexity, _ = self._vq_vae(z)
        return loss, self._decoder(quantized), perplexity

    @torch.no_grad()
    def encode_indices(self, x):
        """图像 -> 离散索引网格 (B,H,W)。"""
        z = self._pre_vq_conv(self._encoder(x))
        _, _, _, indices = self._vq_vae(z)
        return indices

    @torch.no_grad()
    def decode_indices(self, indices):
        """离散索引网格 (B,H,W) -> 图像。"""
        # 码本是 buffer 而非 nn.Embedding，用函数式 embedding 查表
        quantized = F.embedding(indices, self._vq_vae._embedding).permute(0, 3, 1, 2).contiguous()
        return self._decoder(quantized)


# 第二阶段：Gated PixelCNN 先验

class GatedMaskedConv2d(nn.Module):
    """
    Gated PixelCNN 的一层（van den Oord et al. 2016b, arXiv:1606.05328）。
    """

    def __init__(self, mask_type, channels, kernel_size, num_classes, dropout=0.0):
        super().__init__()
        assert mask_type in ('A', 'B')
        assert kernel_size % 2 == 1, 'kernel_size 必须为奇数'
        self.mask_type = mask_type
        self.half_k = kernel_size // 2

        # 输出 2*channels：前一半走 tanh，后一半走 sigmoid，构成门控激活
        self.vert_conv = nn.Conv2d(channels, 2 * channels,
                                   (self.half_k + 1, kernel_size),
                                   padding=(self.half_k, self.half_k))
        self.horiz_conv = nn.Conv2d(channels, 2 * channels,
                                    (1, self.half_k + 1),
                                    padding=(0, self.half_k))
        # 垂直栈 → 水平栈的单向连接（1×1，作用在门控之前的特征上）
        self.vert_to_horiz = nn.Conv2d(2 * channels, 2 * channels, 1)

        # 每层独立的条件投影：类别 → 门控偏置。这是修掉类别混淆的关键
        self.class_emb_v = nn.Embedding(num_classes, 2 * channels)
        self.class_emb_h = nn.Embedding(num_classes, 2 * channels)

        self.res_conv = nn.Conv2d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @staticmethod
    def _gate(x):
        a, b = x.chunk(2, dim=1)
        return torch.tanh(a) * torch.sigmoid(b)

    def forward(self, v, h, label):
        H, W = v.shape[2], v.shape[3]
        cond_v = self.class_emb_v(label)[:, :, None, None]
        cond_h = self.class_emb_h(label)[:, :, None, None]

        # ---- 垂直栈 ----
        v_pre = self.vert_conv(v)[:, :, :H, :]              # 裁掉底部多出的 half_k 行
        if self.mask_type == 'A':
            # 向下移一行：切断「看到当前行」，补零填第一行
            v_pre = F.pad(v_pre, (0, 0, 1, 0))[:, :, :H, :]
        v_out = self._gate(v_pre + cond_v)

        # ---- 水平栈 ----
        h_pre = self.horiz_conv(h)[:, :, :, :W]             # 裁掉右侧多出的 half_k 列
        if self.mask_type == 'A':
            # 向右移一列：切断「看到自己」，补零填第一列
            h_pre = F.pad(h_pre, (1, 0, 0, 0))[:, :, :, :W]
        h_pre = h_pre + self.vert_to_horiz(v_pre)
        h_out = self._gate(h_pre + cond_h)
        h_out = self.res_conv(self.dropout(h_out))

        # mask A 层是第一层，加残差会把「当前位置自身」直接漏给输出，必须跳过
        if self.mask_type == 'B':
            h_out = h_out + h

        return v_out, h_out


class GatedPixelCNN(nn.Module):
    """条件 Gated PixelCNN 先验。

    输入索引 (B,H,W) 与类别 (B,)，输出每格 num_embeddings 个码字的 logits
    (B,num_embeddings,H,W)。接口与旧的 `PixelCNN` 保持一致，可直接替换。
    """

    def __init__(self, num_embeddings, num_classes, channels, num_layers,
                 dropout=0.0, kernel_size=5):
        super().__init__()
        self.idx_emb = nn.Embedding(num_embeddings, channels)   # 索引无序，需过 embedding

        self.layers = nn.ModuleList([
            GatedMaskedConv2d('A' if i == 0 else 'B', channels, kernel_size,
                              num_classes, dropout if i > 0 else 0.0)
            for i in range(num_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, num_embeddings, 1),
        )

    def forward(self, indices, label):
        x = self.idx_emb(indices).permute(0, 3, 1, 2).contiguous()
        v, h = x, x
        for layer in self.layers:
            v, h = layer(v, h, label)
        return self.output_proj(h)


# ============================================================
# 旧版朴素掩码卷积先验（保留以便与历史 NMC 82% 基线对照，新训练请用 GatedPixelCNN）
# ============================================================

class MaskedConv2d(nn.Conv2d):
    """因果掩码卷积：A 型遮住中心（仅第一层用），B 型中心可见（其余层用）。

    已知缺陷：叠多层后当前位置右上方存在盲区。改用 `GatedMaskedConv2d` 可消除。
    """

    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in ('A', 'B')
        self.register_buffer('mask', torch.ones_like(self.weight))
        _, _, kh, kw = self.weight.shape
        self.mask[:, :, kh // 2, kw // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kh // 2 + 1:] = 0

    def forward(self, x):
        # 乘 mask 而非原地改 weight.data：后者会让被遮权重在优化器动量里累积
        return F.conv2d(x, self.weight * self.mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


class PixelCNNResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels // 2, 1),
            nn.ReLU(),
            MaskedConv2d('B', channels // 2, channels // 2, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(channels // 2, channels, 1),
        )

    def forward(self, x):
        return x + self.net(x)


class PixelCNN(nn.Module):
    """输入索引 (B,H,W) 与类别 (B,)，输出每格 num_embeddings 个码字的 logits (B,num_embeddings,H,W)。

    已知缺陷：掩码卷积有盲区，且类别条件只在第一层注入一次，深层会被稀释。
    """

    def __init__(self, num_embeddings, num_classes, channels, num_res_layers, dropout=0.0):
        super().__init__()
        self.idx_emb = nn.Embedding(num_embeddings, channels)      # 索引无序，需过 embedding
        self.class_emb = nn.Embedding(num_classes, channels)       # 类别条件向量

        self.input_conv = MaskedConv2d('A', channels, channels, 7, padding=3)
        self.res_blocks = nn.ModuleList([
            PixelCNNResidualBlock(channels, dropout) for _ in range(num_res_layers)
        ])
        self.output_proj = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, num_embeddings, 1),
        )

    def forward(self, indices, label):
        x = self.idx_emb(indices).permute(0, 3, 1, 2).contiguous()
        x = self.input_conv(x)
        x = x + self.class_emb(label).view(label.size(0), -1, 1, 1)  # 条件 broadcast 到所有空间位置
        for block in self.res_blocks:
            x = block(x)
        return self.output_proj(x)


def build_prior(cfg):
    """按 checkpoint 里的 config 实例化对应的先验模型。

    `prior_type` 是 v2 才加的字段，缺失时按旧的朴素 PixelCNN 处理，
    这样历史 checkpoint 仍能直接载入做对照。
    """
    if cfg.get('prior_type') == 'gated':
        return GatedPixelCNN(cfg['num_embeddings'], cfg['num_classes'], cfg['prior_hiddens'],
                             cfg['prior_res_layers'], cfg['prior_dropout'])
    return PixelCNN(cfg['num_embeddings'], cfg['num_classes'], cfg['prior_hiddens'],
                    cfg['prior_res_layers'], cfg['prior_dropout'])
