"""条件生成的 50 张验收测试：对 0-9 各生成 5 张，拼成 10 行 × 5 列网格图。

验收标准是这 50 张全部人眼可辨识，所以另外存一份无标题、无缩放的干净网格
（grid_50tests_clean.png），按原始 28×28 像素拼接，避免插值把字形修饰得更好看。
"""

import os
import time
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sampling import load_models, sample_batch, DEFAULT_TEMPERATURE, DEFAULT_TOP_K

OUT_DIR = '生成结果'
N_TRIALS = 5
TEMPERATURE = DEFAULT_TEMPERATURE
TOP_K = DEFAULT_TOP_K

os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vqvae, prior, latent_shape = load_models(device)

print(f"\n开始 50 次生成 (0-9 各 {N_TRIALS} 次)，T={TEMPERATURE}, top-k={TOP_K} ...")
start = time.time()

# 一次性批量采样 50 张：整批共享同一个自回归循环，只需 latent_h*latent_w 步前向
labels = [d for d in range(10) for _ in range(N_TRIALS)]
imgs = sample_batch(vqvae, prior, latent_shape, labels, device,
                    temperature=TEMPERATURE, top_k=TOP_K)
imgs = imgs.squeeze(1).cpu().numpy()          # (50,28,28) in [0,1]

print(f"50 张生成完成，耗时 {time.time() - start:.1f}s")

# ---- 干净网格：原始 28×28 像素直接拼接，不缩放不插值 ----
clean = np.zeros((10 * 28, N_TRIALS * 28), dtype=np.float32)
for d in range(10):
    for t in range(N_TRIALS):
        clean[d * 28:(d + 1) * 28, t * 28:(t + 1) * 28] = imgs[d * N_TRIALS + t]
clean_path = os.path.join(OUT_DIR, 'grid_50tests_clean.png')
Image.fromarray((clean * 255).astype(np.uint8)).save(clean_path)
print(f"干净网格已保存 -> {clean_path}")

# ---- 带标注的网格，便于逐张核对 ----
f, axes = plt.subplots(10, N_TRIALS, figsize=(N_TRIALS * 1.4, 10 * 1.4))
for d in range(10):
    for t in range(N_TRIALS):
        ax = axes[d, t]
        ax.imshow(imgs[d * N_TRIALS + t], cmap='gray', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        if t == 0:
            ax.set_ylabel(str(d), fontsize=13, fontweight='bold', rotation=0,
                          labelpad=12, va='center')
f.suptitle(f'Stage 2 conditional generation: 10 digits x {N_TRIALS} trials '
           f'(T={TEMPERATURE}, top-k={TOP_K})', fontsize=11)
f.tight_layout(rect=[0, 0, 1, 0.98])
grid_path = os.path.join(OUT_DIR, 'grid_50tests.png')
f.savefig(grid_path, dpi=150, bbox_inches='tight')
plt.close(f)
print(f"标注网格已保存 -> {grid_path}")
