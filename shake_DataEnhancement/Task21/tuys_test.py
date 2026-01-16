import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms

from shake_aug import ShakeRecipe, ShakeAugDataset, ShakeAugBatchCollator


def main():
    # -----------------------------
    # 1. 加载 MNIST 原始数据
    # -----------------------------
    transform = transforms.ToTensor()  # MNIST 默认是 [0,1] float32
    mnist = MNIST(root="./data", train=True, download=True, transform=transform)

    # -----------------------------
    # 2. 定义 Shake 配方
    # -----------------------------
    recipe = ShakeRecipe(
        N=2,
        beta=0.3,
        alpha=0.2,
        S=(4, 4),
        offset="random",
        p_apply=1.0,
    )

    # -----------------------------
    # 3. 构建 ShakeAugDataset
    #    include_original=True → 每个样本变成 2 份：原图 + shake 图
    # -----------------------------
    shake_ds = ShakeAugDataset(
        base=mnist,
        recipes=[recipe],
        include_original=True,
        dynamic=True,
        base_seed=0,
        apply_in_getitem=False,   # 使用 batch-collator 方式
    )

    # -----------------------------
    # 4. 划分训练集 / 测试集
    #    MNIST train=60000 → shake 后变成 120000
    # -----------------------------
    total_len = len(shake_ds)
    train_len = int(total_len * 0.8)
    test_len = total_len - train_len

    train_ds, test_ds = random_split(shake_ds, [train_len, test_len])

    # -----------------------------
    # 5. DataLoader + ShakeAugBatchCollator
    # -----------------------------
    collator = ShakeAugBatchCollator([recipe])

    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        collate_fn=collator,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        collate_fn=collator,
    )

    # -----------------------------
    # 6. 取一个 batch 测试
    # -----------------------------
    for xb, yb in train_loader:
        print("Batch x:", xb.shape)  # (B,1,28,28)
        print("Batch y:", yb.shape)  # (B,)
        print("x range:", xb.min().item(), xb.max().item())
        break


if __name__ == "__main__":
    main()
