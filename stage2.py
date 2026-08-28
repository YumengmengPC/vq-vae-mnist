"""VQ-VAE 第二阶段：在 stage1 的离散索引上训练条件 PixelCNN 先验 p(z|label)，
实现按数字类别 0-9 生成全新手写数字图像。详细原理见 stage2_全过程.md。

本脚本只负责训练。训练完成后用 generate.py 交互式生成图像，无需重新训练。
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # 无显示环境时也能保存图片
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import VQVAE, GatedPixelCNN, VQVAE_CONFIG, VQVAE_CKPT, PRIOR_CKPT

# 设置随机种子
torch.manual_seed(415)

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 配置


# 第一阶段权重与本阶段输出权重的文件名都定义在 models.py，避免多处硬编码不同步
RUN_NAME = 'mnist_cond'
FIG_DIR = '训练效果'
os.makedirs(FIG_DIR, exist_ok=True)
CURVES_PNG = os.path.join(FIG_DIR, f'stage2_training_curves_{RUN_NAME}.png')

# （可选）条件采样脚本见 generate.py

MNIST_NORM_MEAN = 0.5   # stage1 归一化到 [-1,1]，反归一化时 *0.5+0.5
MNIST_NORM_STD = 0.5

num_classes = 10

# 定义超参数
prior_hiddens = 128
prior_res_layers = 12
prior_lr = 3e-4
prior_batch_size = 128
prior_dropout = 0.2
num_iterations = 20000
eval_interval = 1000    # 每多少步在测试集上评估一次并保存 best


# 载入第一阶段模型并把数据集编码成索引

if not os.path.exists(VQVAE_CKPT):
    raise FileNotFoundError(
        f"找不到第一阶段权重 {VQVAE_CKPT}。先跑完 stage1.py。")

num_embeddings = VQVAE_CONFIG['num_embeddings']

vqvae = VQVAE(**VQVAE_CONFIG).to(device)
vqvae.load_state_dict(torch.load(VQVAE_CKPT, map_location=device))
print(f"载入第一阶段权重 {VQVAE_CKPT} (num_embeddings={num_embeddings})")

# 冻结第一阶段：码本一变，已编码的索引就全部失效
vqvae.eval()
for p in vqvae.parameters():
    p.requires_grad_(False)

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((MNIST_NORM_MEAN,), (MNIST_NORM_STD,)),
])

# 加载数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)


@torch.no_grad()
def encode_dataset(dataset):
    """把整个数据集编码成索引网格并收集类别标签，结果常驻显存（约 2.6MB）。"""
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    idx_list, lbl_list = [], []
    for data, label in loader:
        data = data.to(device, non_blocking=True)
        idx_list.append(vqvae.encode_indices(data))
        lbl_list.append(label)
    return torch.cat(idx_list, dim=0), torch.cat(lbl_list, dim=0)


print("正在把数据集编码成离散索引……")
train_indices, train_labels = encode_dataset(train_dataset)
test_indices, test_labels = encode_dataset(test_dataset)
train_labels = train_labels.to(device)
test_labels = test_labels.to(device)
latent_h, latent_w = train_indices.shape[1], train_indices.shape[2]
print(f"  train {tuple(train_indices.shape)}  test {tuple(test_indices.shape)}"
      f"  取值 [{int(train_indices.min())},{int(train_indices.max())}]  类别数 {num_classes}")

# 训练循环与评估

prior = GatedPixelCNN(num_embeddings, num_classes, prior_hiddens, prior_res_layers, prior_dropout).to(device)
print(f"GatedPixelCNN 参数量 {sum(p.numel() for p in prior.parameters()) / 1e6:.2f}M"
      f"  {prior_res_layers} 层 × {prior_hiddens} 通道  dropout {prior_dropout}  run '{RUN_NAME}'")

optimizer = optim.AdamW(prior.parameters(), lr=prior_lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations)
use_amp = (device.type == 'cuda')
scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

# nats/token -> bits/dim：latent 49 格摊到 28*28 个像素分量，只统计先验不含重建误差
BITS_PER_DIM_SCALE = latent_h * latent_w / (np.log(2) * 28 * 28 * 1)


@torch.no_grad()
def evaluate(num_batches=20):
    """在测试集前若干个 batch 上计算平均交叉熵。"""
    prior.eval()
    losses = []
    n = test_indices.size(0)
    for j in range(num_batches):
        s = j * prior_batch_size
        e = min(s + prior_batch_size, n)
        if s >= e:
            break
        batch = test_indices[s:e]
        labels = test_labels[s:e]
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = prior(batch, labels)
            loss = F.cross_entropy(logits.float(), batch)
        losses.append(loss.item())
    prior.train()
    return float(np.mean(losses)) if losses else float('nan')


prior.train()

# 记录曲线数据
train_losses = []
val_losses = []
val_iters = []
best_val_loss = float('inf')

N = train_indices.size(0)
for i in range(num_iterations):
    pick = torch.randint(0, N, (prior_batch_size,), device=device)
    batch = train_indices[pick]
    labels = train_labels[pick]

    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda', enabled=use_amp):
        logits = prior(batch, labels)
        # logits (B,128,7,7) 对 target (B,7,7)：等价于 49 个位置各做一次 128 类分类
        loss = F.cross_entropy(logits.float(), batch)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    train_losses.append(loss.item())

    if (i + 1) % eval_interval == 0:
        val_loss = evaluate()
        val_losses.append(val_loss)
        val_iters.append(i + 1)
        print(f"step {i+1}/{num_iterations}")
        print(f"  train loss {np.mean(train_losses[-eval_interval:]):.4f} nats/token")
        print(f"  val   loss {val_loss:.4f} nats/token"
              f"  ({val_loss * BITS_PER_DIM_SCALE:.4f} bits/dim, 仅先验)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'iteration': i + 1,
                'model_state_dict': prior.state_dict(),
                'val_loss': best_val_loss,
                'config': {
                    'prior_type': 'gated',   # 供 build_prior() 分辨该实例化哪个先验类
                    'num_embeddings': num_embeddings,
                    'num_classes': num_classes,
                    'prior_hiddens': prior_hiddens,
                    'prior_res_layers': prior_res_layers,
                    'prior_dropout': prior_dropout,
                    'latent_shape': (latent_h, latent_w),
                },
                'vqvae_ckpt': VQVAE_CKPT,
            }, PRIOR_CKPT)
            print(f"  saved best prior -> {PRIOR_CKPT}")
        print()

# 绘制训练曲线

_win = min(201, len(train_losses) // 4 * 2 + 1)
smooth = np.convolve(train_losses, np.ones(_win) / _win, mode='valid')

f = plt.figure(figsize=(11, 5))
ax = f.add_subplot(1, 2, 1)
ax.plot(smooth)
ax.set_title('Smoothed train loss')
ax.set_xlabel('iteration')
ax.set_ylabel('nats / token')

ax = f.add_subplot(1, 2, 2)
ax.plot(val_iters, val_losses, marker='o', markersize=3)
ax.set_title('Validation loss (best %.4f @ step %d)'
             % (best_val_loss, val_iters[int(np.argmin(val_losses))]))
ax.set_xlabel('iteration')
ax.set_ylabel('nats / token')

f.tight_layout()
f.savefig(CURVES_PNG, dpi=150, bbox_inches='tight')
print(f"saved {CURVES_PNG}")
