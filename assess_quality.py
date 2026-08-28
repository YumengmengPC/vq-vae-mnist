"""客观评估条件生成的辨识度，并存一张干净网格图供人工核对。

两个指标：
  - **CNN 准确率**（主）：用一个 99%+ 的 MNIST CNN 判断生成图是否是它该是的数字。
  - **NMC 准确率**（次）：最近均值分类器，保留它只为和历史结果（76% / 82%）纵向对比。
    注意 NMC 在**真实** MNIST 测试集上也只有 82.0% —— 这就是它的天花板，
    生成结果一旦到 82% 就说明「在这个指标下已与真实数据无异」，不能再往上读。

⚠️ 分类器仅用于评分，不参与生成回路 —— 本项目不使用拒绝采样。
"""

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sampling import load_models, sample_batch, DEFAULT_TEMPERATURE, DEFAULT_TOP_K
from classifier import load_classifier, score_images

N_PER = 10          # 每类生成张数
TEMPERATURE = DEFAULT_TEMPERATURE
TOP_K = DEFAULT_TOP_K

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

vqvae, prior, latent_shape = load_models(device)
clf, clf_acc = load_classifier(device)

# ---- 批量生成 ----
labels = [d for d in range(10) for _ in range(N_PER)]
print(f'\n生成 {len(labels)} 张 (每类 {N_PER} 张)，T={TEMPERATURE}, top-k={TOP_K} ...')
imgs = sample_batch(vqvae, prior, latent_shape, labels, device,
                    temperature=TEMPERATURE, top_k=TOP_K)
imgs_np = imgs.squeeze(1).cpu().numpy()          # (N,28,28) in [0,1]


# ---- 指标 1：CNN 分类器 ----
cnn_acc, cnn_conf, cnn_pred = score_images(clf, imgs, labels, device)
cnn_pred = cnn_pred.numpy()


# ---- 指标 2：NMC（保留以便与历史结果对比）----
def mnist_class_means():
    ds = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor())
    sums = np.zeros((10, 28, 28), dtype=np.float64)
    cnt = np.zeros(10, dtype=np.int64)
    for img, lab in DataLoader(ds, batch_size=512):
        img = img.squeeze(1).numpy()
        lab = lab.numpy()
        for k in range(10):
            m = (lab == k)
            sums[k] += img[m].sum(axis=0)
            cnt[k] += m.sum()
    return sums / cnt[:, None, None]


means = mnist_class_means()
nmc_pred = np.array([
    int(np.argmin([np.mean((img - means[k]) ** 2) for k in range(10)]))
    for img in imgs_np
])
labels_np = np.array(labels)
nmc_acc = float((nmc_pred == labels_np).mean())

# ---- 报告 ----
print(f'\nCNN 辨识度准确率: {int(cnn_acc * len(labels))}/{len(labels)} = {cnn_acc * 100:.1f}%'
      f'   (平均置信度 {cnn_conf * 100:.1f}%)')
print(f'NMC 辨识度准确率: {int(nmc_acc * len(labels))}/{len(labels)} = {nmc_acc * 100:.1f}%'
      f'   (真实 MNIST 在 NMC 下也只有 82.0%，到 82% 即为饱和)')

confusion = np.zeros((10, 10), dtype=int)
for t, p in zip(labels_np, cnn_pred):
    confusion[t, p] += 1

print('\nCNN 混淆矩阵 (行=目标标签, 列=预测标签):')
print('     ' + ' '.join(f'{k:3d}' for k in range(10)))
for r in range(10):
    print(f'  {r}: ' + ' '.join(f'{confusion[r, c]:3d}' for c in range(10)))

print('\n每类准确率 (CNN):')
bad = []
for r in range(10):
    print(f'  {r}: {confusion[r, r]}/{N_PER}')
    if confusion[r, r] < N_PER:
        bad.append(r)
if bad:
    print(f'\n仍有失误的类别: {bad}')
else:
    print('\n全部类别 10/10 ✓')

# ---- 干净网格图 ----
grid = np.zeros((10 * 28, N_PER * 28), dtype=np.float32)
for d in range(10):
    for t in range(N_PER):
        grid[d * 28:(d + 1) * 28, t * 28:(t + 1) * 28] = imgs_np[d * N_PER + t]
Image.fromarray((grid * 255).astype(np.uint8)).save('生成结果/grid_clean_10x10.png')
print('\n已保存 生成结果/grid_clean_10x10.png')
