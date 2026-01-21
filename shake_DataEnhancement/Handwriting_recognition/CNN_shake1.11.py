import os
import sys
import random
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

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
# Batch 级旋转增强 Collator
# =========================
class RotationBatchCollator:
    def __init__(self, degrees=(0, 15)):
        self.degrees = degrees

    def __call__(self, batch):
        xs, ys = zip(*batch)
        x = torch.stack(xs, dim=0)
        y = torch.as_tensor(ys)

        angle = random.uniform(*self.degrees)
        x = TF.rotate(x, angle)

        return x, y


# =========================
# 自动配置输出工具
# =========================
def print_model_info(model):
    print("\n[Model]")
    print(model)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total}")
    print(f"Trainable params: {trainable}")


def print_dataset_info(dataset):
    print("\n[Dataset]")
    print(f"Class: {dataset.__class__.__name__}")
    print(f"Length: {len(dataset)}")
    print("Library: torchvision.datasets")


def print_shakeaug_info(recipes):
    print("\n[ShakeAug]")
    for i, r in enumerate(recipes):
        print(f"Recipe {i}:")
        for field in r.__dataclass_fields__:
            print(f"  {field}: {getattr(r, field)}")


def print_rotation_info(rotation_collator):
    print("\n[Rotation Augmentation]")
    if hasattr(rotation_collator, "degrees"):
        print(f"Degrees range: {rotation_collator.degrees}")
        print(f"Max rotation angle: {max(rotation_collator.degrees)}")


def print_training_info(loader, optimizer, epochs):
    print("\n[Training]")
    print(f"Batch size: {loader.batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    for k, v in optimizer.defaults.items():
        print(f"  {k}: {v}")


# =========================
# 主程序
# =========================
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    root = "/home/ystu/research_code/Machine_Learning/shake_DataEnhancement/Handwriting_recognition"

    # 数据集
    train_base = MNIST(root=root, train=True, download=True, transform=ToTensor())
    test_base = MNIST(root=root, train=False, download=True, transform=ToTensor())

    # ShakeAug 配方
    recipes = [
        ShakeRecipe(N=2, beta=0.5, alpha=0.3, S=(4, 4))
    ]

    # DataLoader
    train_loader_plain = DataLoader(train_base, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

    train_aug = ShakeAugDataset(
        base=train_base,
        recipes=recipes,
        include_original=True,
        dynamic=True,
        base_seed=0,
        apply_in_getitem=False,
    )

    shake_collate_fn = ShakeAugBatchCollator(recipes)

    train_loader_aug = DataLoader(
        train_aug,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=shake_collate_fn,
    )

    rotation_collate_fn = RotationBatchCollator(degrees=(0, 5))

    train_loader_rotation = DataLoader(
        train_base,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=rotation_collate_fn,
    )

    test_loader = DataLoader(test_base, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    # 三个模型 & 优化器
    model_plain = ConvNet().to(device)
    model_aug = ConvNet().to(device)
    model_rotation = ConvNet().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_plain = optim.Adam(model_plain.parameters(), lr=1e-4)
    optimizer_aug = optim.Adam(model_aug.parameters(), lr=1e-4)
    optimizer_rotation = optim.Adam(model_rotation.parameters(), lr=1e-4)

    num_epochs = 5

    # =========================
    # 自动输出实验配置
    # =========================
    print("\n================ Auto Experiment Summary ================")
    print_model_info(model_plain)
    print_dataset_info(train_base)
    print_shakeaug_info(recipes)
    print_rotation_info(rotation_collate_fn)
    print_training_info(train_loader_plain, optimizer_plain, num_epochs)
    print("=========================================================\n")

    # =========================
    # 训练
    # =========================
    print("Training Plain model...")
    for epoch in range(num_epochs):
        model_plain.train()
        for x, y in train_loader_plain:
            x, y = x.to(device), y.to(device)
            optimizer_plain.zero_grad()
            loss = criterion(model_plain(x), y)
            loss.backward()
            optimizer_plain.step()
        print(f"Epoch {epoch+1} | Plain Acc: {accuracy(model_plain, device, test_loader)*100:.2f}%")

    print("Training ShakeAug model...")
    for epoch in range(num_epochs):
        train_aug.set_epoch(epoch)
        model_aug.train()
        for x, y in train_loader_aug:
            x, y = x.to(device), y.to(device)
            optimizer_aug.zero_grad()
            loss = criterion(model_aug(x), y)
            loss.backward()
            optimizer_aug.step()
        print(f"Epoch {epoch+1} | ShakeAug Acc: {accuracy(model_aug, device, test_loader)*100:.2f}%")

    print("Training Rotation model...")
    for epoch in range(num_epochs):
        model_rotation.train()
        for x, y in train_loader_rotation:
            x, y = x.to(device), y.to(device)
            optimizer_rotation.zero_grad()
            loss = criterion(model_rotation(x), y)
            loss.backward()
            optimizer_rotation.step()
        print(f"Epoch {epoch+1} | Rotation Acc: {accuracy(model_rotation, device, test_loader)*100:.2f}%")

    # 保存模型
    os.makedirs("checkpoint", exist_ok=True)
    torch.save(model_plain.state_dict(), "checkpoint/plain.pth")
    torch.save(model_aug.state_dict(), "checkpoint/aug.pth")
    torch.save(model_rotation.state_dict(), "checkpoint/rotation.pth")
