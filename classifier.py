"""用于客观评估生成质量的 MNIST 分类器。

⚠️ 仅用于**评估**，不参与生成回路 —— 生成过程不会因分类器的判断而重采样或筛选。
本项目明确不使用分类器拒绝采样，50/50 的验收要靠模型本身与采样超参达成。

之所以引入它：assess_quality.py 原有的 NMC（最近均值分类器）用每类平均图作模板，
对笔画粗细、位移、倾斜过于敏感，本身就是个粗糙代理。

实测校准：**NMC 在真实 MNIST 测试集上的准确率只有 82.0%**，而本 CNN 是 99.1%。
这意味着 NMC 的天花板就是 82% —— 历史报告里 7×7 EMA 版「NMC 82%」其实已经与
真实数据持平、指标完全饱和，把它读成「18% 的生成图不可辨识」是误读。
用 NMC 无法区分「和真实 MNIST 一样好」与「还差一点」，所以主指标必须换成 CNN。
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CLASSIFIER_CKPT = 'mnist_classifier.pth'

# 与 stage1 相同的归一化，这样 [0,1] 的生成图只需 *2-1 就能喂进来
_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),   # 28→14
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),   # 14→7
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.head(self.features(x))


def train_classifier(device, epochs=3, verbose=True):
    """训练分类器并存盘。返回 (模型, 测试集准确率)。"""
    train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=_TRANSFORM)
    test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=_TRANSFORM)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=512, num_workers=2, pin_memory=True)

    model = MnistCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                pred = model(x.to(device)).argmax(1).cpu()
                correct += (pred == y).sum().item()
        acc = correct / len(test_ds)
        if verbose:
            print(f"  分类器 epoch {ep + 1}/{epochs}  测试集准确率 {acc * 100:.2f}%")

    torch.save({'state_dict': model.state_dict(), 'test_acc': acc}, CLASSIFIER_CKPT)
    return model, acc


def load_classifier(device, verbose=True):
    """载入分类器；没有缓存权重就先训一个（约 1-2 分钟）。"""
    if os.path.exists(CLASSIFIER_CKPT):
        ckpt = torch.load(CLASSIFIER_CKPT, map_location=device)
        model = MnistCNN().to(device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        if verbose:
            print(f"载入评估用分类器 {CLASSIFIER_CKPT} (测试集准确率 {ckpt['test_acc'] * 100:.2f}%)")
        return model, ckpt['test_acc']

    if verbose:
        print("未找到评估用分类器，开始训练……")
    model, acc = train_classifier(device, verbose=verbose)
    model.eval()
    return model, acc


@torch.no_grad()
def score_images(model, imgs01, labels, device):
    """给一批 [0,1] 的生成图打分。

    参数
        imgs01: (B,1,28,28) 或 (B,28,28)，取值 [0,1]
        labels: 期望的数字标签，长度 B

    返回 (准确率, 预测正确样本上的平均置信度, 预测标签 LongTensor)
    """
    if imgs01.dim() == 3:
        imgs01 = imgs01.unsqueeze(1)
    x = (imgs01.to(device) * 2 - 1)          # [0,1] → [-1,1]，与训练归一化一致
    labels = torch.as_tensor(labels, dtype=torch.long, device=device)

    probs = F.softmax(model(x), dim=1)
    conf, pred = probs.max(dim=1)
    hit = (pred == labels)
    acc = hit.float().mean().item()
    mean_conf = conf.mean().item()
    return acc, mean_conf, pred.cpu()
