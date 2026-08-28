"""扫描采样超参（温度 × top-k），为 sampling.py 的默认值找依据。

验收目标是 50 张全部人眼可辨识。压低温度能提升辨识度，代价是字形多样性下降，
所以这里的目的不是「找最高准确率」，而是**找到仍能满足辨识度要求的最高温度**，
把多样性损失压到最小。

用法：
    python sweep_sampling.py
"""

import numpy as np
import torch

from sampling import load_models, sample_batch
from classifier import load_classifier, score_images

N_PER_CLASS = 20                      # 每类采样张数，共 10*N 张
TEMPERATURES = [0.5, 0.6, 0.7, 0.8, 1.0]
TOP_KS = [None, 32, 16, 8]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vqvae, prior, latent_shape = load_models(device)
clf, clf_acc = load_classifier(device)

labels = [d for d in range(10) for _ in range(N_PER_CLASS)]
n_total = len(labels)


def distinct_ratio(imgs):
    """字形多样性代理：同类样本两两像素距离的均值，越大说明变化越丰富。"""
    flat = imgs.reshape(10, N_PER_CLASS, -1)
    dists = []
    for k in range(10):
        a = flat[k]
        d = torch.cdist(a, a)
        iu = torch.triu_indices(N_PER_CLASS, N_PER_CLASS, offset=1)
        dists.append(d[iu[0], iu[1]].mean().item())
    return float(np.mean(dists))


print(f"\n每组合采样 {n_total} 张（每类 {N_PER_CLASS} 张）")
print(f"{'温度':>6} {'top-k':>7} {'CNN准确率':>10} {'平均置信度':>11} {'多样性':>9}")
print('-' * 50)

results = []
for T in TEMPERATURES:
    for k in TOP_KS:
        torch.manual_seed(415)                    # 同一随机种子，组合间可比
        imgs = sample_batch(vqvae, prior, latent_shape, labels, device,
                            temperature=T, top_k=k)
        acc, conf, _ = score_images(clf, imgs, labels, device)
        div = distinct_ratio(imgs.squeeze(1).cpu())
        results.append({'T': T, 'top_k': k, 'acc': acc, 'conf': conf, 'div': div})
        print(f"{T:>6.1f} {str(k):>7} {acc * 100:>9.1f}% {conf * 100:>10.1f}% {div:>9.2f}")

print('-' * 50)
print(f"（评估用分类器自身在 MNIST 测试集上的准确率：{clf_acc * 100:.2f}%）")

# 在达标的组合里挑温度最高的：辨识度达标的前提下尽量保住多样性
THRESHOLD = 0.96
ok = [r for r in results if r['acc'] >= THRESHOLD]
if ok:
    best = max(ok, key=lambda r: (r['T'], r['div']))
    print(f"\n推荐组合（准确率 ≥ {THRESHOLD * 100:.0f}% 中温度最高者）：")
    print(f"  DEFAULT_TEMPERATURE = {best['T']}")
    print(f"  DEFAULT_TOP_K = {best['top_k']}")
    print(f"  → 准确率 {best['acc'] * 100:.1f}%，置信度 {best['conf'] * 100:.1f}%，多样性 {best['div']:.2f}")
else:
    best = max(results, key=lambda r: r['acc'])
    print(f"\n⚠️ 没有组合达到 {THRESHOLD * 100:.0f}%。当前最好的是：")
    print(f"  T={best['T']}, top_k={best['top_k']} → 准确率 {best['acc'] * 100:.1f}%")
    print("  说明瓶颈在模型而非采样超参，应回头加强先验或检查码本利用率。")

print("\n把推荐值写入 sampling.py 的 DEFAULT_TEMPERATURE / DEFAULT_TOP_K 后，"
      "再跑 assess_quality.py 与 run_50_tests.py。")
