"""VQ-VAE 交互式条件生成：输入一个数字（0-9），生成一张对应手写数字图像，
循环直到手动退出。

用法：
    python generate.py

采样实现见 sampling.py，原理见 stage2_全过程.md。
"""

import os
import time
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sampling import load_models, sample_batch, DEFAULT_TEMPERATURE, DEFAULT_TOP_K

OUT_DIR = '生成结果'

TEMPERATURE = DEFAULT_TEMPERATURE  # 越低字形越规整，越高越多样但易畸形
TOP_K = DEFAULT_TOP_K

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def next_seq():
    mx = 0
    for name in os.listdir(OUT_DIR):
        if name.startswith('gen_') and name.endswith('.png'):
            m = name.split('_')[-1].replace('.png', '')
            try:
                mx = max(mx, int(m))
            except ValueError:
                pass
    return mx + 1


def generate_one(digit):
    """生成一张数字图像并保存，返回路径和耗时。"""
    start = time.time()
    img = sample_batch(vqvae, prior, latent_shape, [digit], device,
                       temperature=TEMPERATURE, top_k=TOP_K)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f'gen_{digit}_{next_seq():03d}.png')

    f = plt.figure(figsize=(3, 3))
    ax = f.add_subplot(1, 1, 1)
    ax.imshow(img[0][0].cpu().numpy(), cmap='gray', interpolation='nearest')
    ax.set_title(f'digit {digit}  T={TEMPERATURE}'
                 + (f'  top-k={TOP_K}' if TOP_K else ''))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    f.tight_layout()
    f.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(f)

    return path, time.time() - start


def main():
    print("VQ-VAE 条件生成 | 输入数字 0-9 回车即出图，q 退出\n")
    if device.type != 'cuda':
        print("注意：当前在 CPU 上运行，采样会明显慢于 GPU。\n")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line.lower() in ('q', 'quit', 'exit'):
            break

        # 只接受 0-9 单个数字
        if len(line) != 1 or not line.isdigit():
            print("  输入有误：请输入单个数字 0-9，或 q 退出")
            continue

        digit = int(line)
        try:
            path, elapsed = generate_one(digit)
        except KeyboardInterrupt:
            print("\n  已中断本次生成")
            continue
        print(f"  saved {path}  [{elapsed:.1f}s]")

    print("退出。")


if __name__ == '__main__':
    vqvae, prior, latent_shape = load_models(device)
    main()
