import os
import sys
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from shake_aug import (
    ShakeRecipe,
    ShakeAugDataset,
    ShakeAugBatchCollator,
)

print("Args:", sys.argv)

# =========================
# CNN 网络
# =========================
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# =========================
# 测试准确率
# =========================
def accuracy(model, device, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


# =========================
# ShakeAug 参数打印
# =========================
def print_shakeaug_config(recipes, train_aug, collate_fn, train_loader_aug):
    print("\n================ ShakeAug Configuration ================")

    print("\n[ShakeRecipe]")
    for i, r in enumerate(recipes):
        print(f"Recipe {i}:")
        for field in r.__dataclass_fields__:
            print(f"  {field}: {getattr(r, field)}")

    print("\n[ShakeAugDataset]")
    print(f"  include_original: {train_aug.include_original}")
    print(f"  dynamic: {train_aug.dynamic}")
    print(f"  base_seed: {train_aug.base_seed}")
    print(f"  apply_in_getitem: {train_aug.apply_in_getitem}")
    print(f"  base_dataset: {train_aug.base.__class__.__name__}")
    print(f"  dataset_length: {len(train_aug)}")

    print("\n[ShakeAugBatchCollator]")
    print(f"  num_recipes: {len(collate_fn.recipes)}")

    print("\n[DataLoader]")
    print(f"  batch_size: {train_loader_aug.batch_size}")
    print(f"  num_workers: {train_loader_aug.num_workers}")
    print(f"  pin_memory: {train_loader_aug.pin_memory}")

    print("========================================================\n")


# =========================
# 主程序
# =========================
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    root = "/home/ystu/research_code/Machine_Learning/shake_DataEnhancement/Handwriting_recognition"

    # =========================
    # 数据集
    # =========================
    train_base = MNIST(root=root, train=True, download=True, transform=ToTensor())
    test_base = MNIST(root=root, train=False, download=True, transform=ToTensor())

    # ShakeAug 配方
    recipes = [
        ShakeRecipe(
            N=2,
            beta=0.5,
            alpha=0.3,
            S=(4, 4),
        )
    ]

    # =========================
    # Plain DataLoader
    # =========================
    train_loader_plain = DataLoader(
        train_base,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # =========================
    # Aug DataLoader（batch 级增强）
    # =========================
    train_aug = ShakeAugDataset(
        base=train_base,
        recipes=recipes,
        include_original=True,
        dynamic=True,
        base_seed=0,
        apply_in_getitem=False,
    )

    collate_fn = ShakeAugBatchCollator(recipes)

    train_loader_aug = DataLoader(
        train_aug,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_base,
        batch_size=512,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # =========================
    # 两个模型 & 优化器
    # =========================
    model_plain = ConvNet().to(device)
    model_aug = ConvNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_plain = optim.Adam(model_plain.parameters(), lr=1e-4)
    optimizer_aug = optim.Adam(model_aug.parameters(), lr=1e-4)

    # =========================
    # 训练
    # =========================
    num_epochs = 5

    print("Starting training plain model...")
    for epoch in range(num_epochs):
        model_plain.train()
        for x, y in train_loader_plain:
            x, y = x.to(device), y.to(device)
            optimizer_plain.zero_grad()
            loss = criterion(model_plain(x), y)
            loss.backward()
            optimizer_plain.step()

        acc_plain = accuracy(model_plain, device, test_loader)
        print(f"Epoch {epoch + 1:2d} | Test(Plain): {acc_plain * 100:.2f}%")

    print("Starting training aug model...")
    for epoch in range(num_epochs):
        train_aug.set_epoch(epoch)
        model_aug.train()

        for x, y in train_loader_aug:
            x, y = x.to(device), y.to(device)
            optimizer_aug.zero_grad()
            loss = criterion(model_aug(x), y)
            loss.backward()
            optimizer_aug.step()

        acc_aug = accuracy(model_aug, device, test_loader)
        print(f"Epoch {epoch + 1:2d} | Test(Aug): {acc_aug * 100:.2f}%")

    # =========================
    # 打印 ShakeAug 相关参数
    # =========================
    print_shakeaug_config(
        recipes=recipes,
        train_aug=train_aug,
        collate_fn=collate_fn,
        train_loader_aug=train_loader_aug,
    )

    # =========================
    # 保存模型
    # =========================
    os.makedirs("checkpoint", exist_ok=True)
    torch.save(model_plain.state_dict(), "checkpoint/plain.pth")
    torch.save(model_aug.state_dict(), "checkpoint/aug.pth")
