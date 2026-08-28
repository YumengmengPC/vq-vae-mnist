"""条件采样的公共实现。

被 generate.py / run_50_tests.py / assess_quality.py / sweep_sampling.py 共用，
避免各脚本各写一份采样循环（早期 run_50_tests.py 甚至是起 50 个子进程、
每个子进程重新加载一遍模型）。
"""

import torch
import torch.nn.functional as F

from models import VQVAE, VQVAE_CONFIG, VQVAE_CKPT, PRIOR_CKPT, build_prior

# 采样默认值。由 sweep_sampling.py 网格搜索确定，改动请附上扫描结果
# 2026-08-14 扫描：T=0.7/top-k=16 → CNN 99.5%、多样性 8.14（T=1.0 虽多样性略高但
# 准确率掉到 97%，50 张里易混入判错样本，故取此更稳健的折中）
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 16       # None = 不截断；正整数 = 只在概率最高的 k 个码字里采样

MNIST_NORM_MEAN = 0.5     # stage1 归一化到 [-1,1]，反归一化时 *0.5+0.5
MNIST_NORM_STD = 0.5


def load_models(device, vqvae_ckpt=VQVAE_CKPT, prior_ckpt=PRIOR_CKPT, verbose=True):
    """载入并冻结 stage1 VQ-VAE 与 stage2 条件先验，返回 (vqvae, prior, latent_shape)。"""
    import os
    for path in (vqvae_ckpt, prior_ckpt):
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到权重 {path}。先跑完 stage1.py 和 stage2.py。")

    vqvae = VQVAE(**VQVAE_CONFIG).to(device)
    vqvae.load_state_dict(torch.load(vqvae_ckpt, map_location=device))
    vqvae.eval()

    ckpt = torch.load(prior_ckpt, map_location=device)
    cfg = ckpt['config']
    prior = build_prior(cfg).to(device)     # 按 config 里的 prior_type 选类，兼容旧 checkpoint
    prior.load_state_dict(ckpt['model_state_dict'])
    prior.eval()                            # 关掉 dropout

    if verbose:
        print(f"载入 {vqvae_ckpt}")
        print(f"载入 {prior_ckpt}  (iteration {ckpt['iteration']}, "
              f"val_loss {ckpt['val_loss']:.4f}, type {cfg.get('prior_type', 'plain')})")

    return vqvae, prior, tuple(cfg['latent_shape'])


@torch.no_grad()
def sample_batch(vqvae, prior, latent_shape, labels, device,
                 temperature=DEFAULT_TEMPERATURE, top_k=DEFAULT_TOP_K):
    """一次采样一批图像。

    参数
        labels: 长度 B 的整数序列或 (B,) LongTensor，指定每张图要生成的数字
        temperature: 越低字形越规整、多样性越低；<=0 时退化为逐格取 argmax
        top_k: 只在概率最高的 k 个码字里采样，切掉长尾码字对结构的破坏

    返回 (B,1,28,28) 的张量，取值已反归一化到 [0,1]。

    整批共享同一个自回归循环：H*W 步前向而非 B*H*W 步，比逐张采样快约 B 倍。
    """
    labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    batch = labels.size(0)
    lh, lw = latent_shape

    indices = torch.zeros(batch, lh, lw, dtype=torch.long, device=device)
    for r in range(lh):
        for c in range(lw):
            logits = prior(indices, labels)[:, :, r, c].float()
            if temperature <= 0:
                indices[:, r, c] = logits.argmax(dim=-1)
                continue
            logits = logits / temperature
            if top_k is not None and 0 < top_k < logits.size(-1):
                kth = logits.topk(top_k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float('-inf'))
            probs = F.softmax(logits, dim=-1)
            indices[:, r, c] = torch.multinomial(probs, 1).squeeze(-1)

    img = vqvae.decode_indices(indices)
    return (img * MNIST_NORM_STD + MNIST_NORM_MEAN).clamp(0, 1)
