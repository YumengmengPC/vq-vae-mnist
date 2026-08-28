import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

from models import VQVAE, VQVAE_CKPT, VQVAE_CONFIG

# 设置随机种子
torch.manual_seed(415)

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device:{device}")

# 数据预处理
train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 加载数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# 定义超参数（架构超参统一从 models.VQVAE_CONFIG 取，避免两处不一致）
num_iterations = 20000

num_embeddings = VQVAE_CONFIG['num_embeddings']

lr = 0.001
dead_code_interval = 200   # 每多少步做一次死码重置

best_model_path = VQVAE_CKPT

# 模型定义见 models.py（stage1/stage2/generate 共用）
model = VQVAE(**VQVAE_CONFIG).to(device)
print(f"VQVAE 配置 {VQVAE_CONFIG}")
print(f"参数量 {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=False)
# cosine 退火：后期小 lr 让重建收敛得更干净
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations)


# 训练循环与评估

import time
from itertools import cycle

import matplotlib
matplotlib.use('Agg')  # 无显示环境时也能保存图片
import matplotlib.pyplot as plt

log_interval = 100      # 每多少步打印一次训练 loss
eval_interval = 500     # 每多少步在测试集上评估一次
num_show = 8            # 重构可视化时展示的样本数

# 记录曲线数据
train_steps = []
train_recon_hist = []
train_vq_hist = []
test_steps = []
test_recon_hist = []
test_perp_hist = []

best_test_recon = float('inf')
total_reset = 0          # 累计重置的死码数


@torch.no_grad()
def evaluate():
    """在测试集上计算平均重构 loss、困惑度，以及实际被用到的码字数。

    困惑度衡量的是使用分布的均匀度，而「用到几个码字」才直接反映有没有死码 ——
    本轮修复（数据驱动初始化 + 死码重置）要验收的正是后者。
    """
    model.eval()
    total_recon, total_perp, total_count = 0.0, 0.0, 0
    used = torch.zeros(num_embeddings, dtype=torch.bool, device=device)
    for x, _ in test_loader:
        x = x.to(device)
        # 手动展开一次前向，一趟同时拿到重建与索引，避免再跑一遍 encoder
        z = model._pre_vq_conv(model._encoder(x))
        _, quantized, perplexity, indices = model._vq_vae(z)
        x_recon = model._decoder(quantized)
        recon = F.mse_loss(x_recon, x)
        b = x.size(0)
        total_recon += recon.item() * b
        total_perp += perplexity.item() * b
        total_count += b
        used[indices.reshape(-1).unique()] = True
    model.train()
    return total_recon / total_count, total_perp / total_count, int(used.sum())


train_iter = cycle(train_loader)
print(f"开始训练：{num_iterations} iterations，lr={lr}")
start = time.time()
model.train()

for step in range(1, num_iterations + 1):
    x, _ = next(train_iter)
    x = x.to(device)

    optimizer.zero_grad()
    vq_loss, x_recon, perplexity = model(x)
    recon_loss = F.mse_loss(x_recon, x)
    loss = recon_loss + vq_loss
    loss.backward()
    optimizer.step()
    scheduler.step()

    # 死码重置：把 EMA 计数掉到 0 附近的码字拉回数据分布，防止永久死亡
    if step % dead_code_interval == 0:
        n_reset = model._vq_vae.reset_dead_codes()
        total_reset += n_reset
        if n_reset:
            print(f"  [step {step}] 重置 {n_reset} 个死码（累计 {total_reset}）")

    if step % log_interval == 0:
        train_steps.append(step)
        train_recon_hist.append(recon_loss.item())
        train_vq_hist.append(vq_loss.item())
        elapsed = time.time() - start
        print(f"[step {step:5d}/{num_iterations}] "
              f"recon={recon_loss.item():.4f} vq={vq_loss.item():.4f} "
              f"perp={perplexity.item():.1f} lr={scheduler.get_last_lr()[0]:.2e} "
              f"({elapsed:.0f}s)")

    if step % eval_interval == 0 or step == num_iterations:
        test_recon, test_perp, n_used = evaluate()
        test_steps.append(step)
        test_recon_hist.append(test_recon)
        test_perp_hist.append(test_perp)
        print(f"  -> test recon={test_recon:.4f} perplexity={test_perp:.1f} "
              f"(利用率 {test_perp / num_embeddings * 100:.1f}%, "
              f"活码 {n_used}/{num_embeddings})")

        if test_recon < best_test_recon:
            best_test_recon = test_recon
            torch.save(model.state_dict(), best_model_path)
            print(f"     保存 best model -> {best_model_path}")

total = time.time() - start
final_recon, final_perp, final_used = evaluate()
print(f"\n训练完成，总耗时 {total:.1f}s，最佳 test recon={best_test_recon:.4f}")
print(f"末步码本状态：活码 {final_used}/{num_embeddings}，"
      f"利用率 {final_perp / num_embeddings * 100:.1f}%，累计重置死码 {total_reset} 次")


# 保存 best model 与最终模型

torch.save(model.state_dict(), 'final_' + VQVAE_CKPT)


# 绘制损失曲线

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].plot(train_steps, train_recon_hist, label='train recon', alpha=0.8)
axes[0].plot(train_steps, train_vq_hist, label='train vq', alpha=0.8)
axes[0].plot(test_steps, test_recon_hist, label='test recon', marker='o', alpha=0.9)
axes[0].set_xlabel('iteration')
axes[0].set_ylabel('loss')
axes[0].set_title('Reconstruction & VQ loss')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(test_steps, test_perp_hist, marker='o', color='tab:green')
axes[1].axhline(num_embeddings, color='r', linestyle='--', alpha=0.5,
                label=f'max={num_embeddings}')
axes[1].set_xlabel('iteration')
axes[1].set_ylabel('perplexity')
axes[1].set_title('Codebook usage (perplexity)')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('loss_curve.png', dpi=120)
print("损失曲线已保存 -> loss_curve.png")


# 重构可视化

model.eval()
with torch.no_grad():
    samples, _ = next(iter(test_loader))
    samples = samples[:num_show].to(device)
    _, recon, _ = model(samples)

samples = samples.cpu().squeeze(1).numpy()
recon = recon.cpu().squeeze(1).numpy()

fig, axes = plt.subplots(2, num_show, figsize=(num_show * 1.5, 3))
for i in range(num_show):
    axes[0, i].imshow(samples[i], cmap='gray')
    axes[0, i].set_title('orig' if i == 0 else '')
    axes[0, i].axis('off')
    axes[1, i].imshow(recon[i], cmap='gray')
    axes[1, i].set_title('recon' if i == 0 else '')
    axes[1, i].axis('off')
plt.tight_layout()
plt.savefig('reconstruction.png', dpi=120)
print("重构对比已保存 -> reconstruction.png")
