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
# Batch 级增强 Collators
# =========================
class RandomRotationBatchCollator:
    def __init__(self, degrees=(0, 5)):
        self.degrees = degrees

    def __call__(self, batch):
        xs, ys = zip(*batch)
        x = torch.stack(xs, dim=0)
        y = torch.as_tensor(ys)
        angle = random.uniform(*self.degrees)
        x = TF.rotate(x, angle)
        return x, y


class RandomHorizontalFlipBatchCollator:
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, batch):
        xs, ys = zip(*batch)
        x = torch.stack(xs, dim=0)
        y = torch.as_tensor(ys)
        if random.random() < self.p:
            x = TF.hflip(x)
        return x, y


class RandomResizedCropBatchCollator:
    def __init__(self, size=28, scale=(0.08, 1.0), ratio=(3/4, 4/3)):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, batch):
        xs, ys = zip(*batch)
        x = torch.stack(xs, dim=0)
        y = torch.as_tensor(ys)

        _, _, H, W = x.shape
        scale = random.uniform(*self.scale)
        ratio = random.uniform(*self.ratio)

        crop_h = int(H * scale)
        crop_w = int(crop_h * ratio)
        crop_h = min(crop_h, H)
        crop_w = min(crop_w, W)

        top = random.randint(0, H - crop_h)
        left = random.randint(0, W - crop_w)

        x = TF.resized_crop(x, top, left, crop_h, crop_w, size=(self.size, self.size))
        return x, y


# =========================
# 模组参数输出（核心新增）
# =========================
def print_module_configs(
    recipes,
    rotation_collate_fn,
    flip_collate_fn,
    rrc_collate_fn,
    train_loader_plain,
):
    print("\n================ Module Configuration Summary ================")

    print("\n[ShakeAug]")
    for i, r in enumerate(recipes):
        print(f"Recipe {i}:")
        for field in r.__dataclass_fields__:
            print(f"  {field}: {getattr(r, field)}")

    print("\n[RandomRotation]")
    print(f"  degrees: {rotation_collate_fn.degrees}")

    print("\n[RandomHorizontalFlip]")
    print(f"  p: {flip_collate_fn.p}")

    print("\n[RandomResizedCrop]")
    print(f"  size: {rrc_collate_fn.size}")
    print(f"  scale: {rrc_collate_fn.scale}")
    print(f"  ratio: {rrc_collate_fn.ratio}")

    print("\n[DataLoader]")
    print(f"  batch_size: {train_loader_plain.batch_size}")
    print(f"  num_workers: {train_loader_plain.num_workers}")
    print(f"  pin_memory: {train_loader_plain.pin_memory}")

    print("==============================================================\n")


# =========================
# 主程序
# =========================
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    root = "/home/ystu/research_code/Machine_Learning/shake_DataEnhancement/Handwriting_recognition"

    train_base = MNIST(root=root, train=True, download=True, transform=ToTensor())
    test_base = MNIST(root=root, train=False, download=True, transform=ToTensor())

    recipes = [ShakeRecipe(N=2, beta=0.5, alpha=0.3, S=(4, 4))]

    train_loader_plain = DataLoader(train_base, 256, True, num_workers=4, pin_memory=True)

    train_aug = ShakeAugDataset(train_base, recipes)
    train_loader_aug = DataLoader(
        train_aug, 256, True, num_workers=4, pin_memory=True,
        collate_fn=ShakeAugBatchCollator(recipes)
    )

    rotation_collate_fn = RandomRotationBatchCollator()
    flip_collate_fn = RandomHorizontalFlipBatchCollator()
    rrc_collate_fn = RandomResizedCropBatchCollator()

    train_loader_random_rotation = DataLoader(
        train_base, 256, True, num_workers=4, pin_memory=True,
        collate_fn=rotation_collate_fn
    )
    train_loader_random_horizontal_flip = DataLoader(
        train_base, 256, True, num_workers=4, pin_memory=True,
        collate_fn=flip_collate_fn
    )
    train_loader_random_resized_crop = DataLoader(
        train_base, 256, True, num_workers=4, pin_memory=True,
        collate_fn=rrc_collate_fn
    )

    test_loader = DataLoader(test_base, 512, False, num_workers=4, pin_memory=True)

    models = {
        "Plain": ConvNet().to(device),
        "ShakeAug": ConvNet().to(device),
        "RandomRotation": ConvNet().to(device),
        "RandomHorizontalFlip": ConvNet().to(device),
        "RandomResizedCrop": ConvNet().to(device),
    }

    loaders = {
        "Plain": train_loader_plain,
        "ShakeAug": train_loader_aug,
        "RandomRotation": train_loader_random_rotation,
        "RandomHorizontalFlip": train_loader_random_horizontal_flip,
        "RandomResizedCrop": train_loader_random_resized_crop,
    }

    optimizers = {k: optim.Adam(v.parameters(), lr=1e-4) for k, v in models.items()}

    criterion = nn.CrossEntropyLoss()
    num_epochs = 5

    for name in models:
        print(f"Training {name} model...")
        for epoch in range(num_epochs):
            models[name].train()
            for x, y in loaders[name]:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizers[name].zero_grad()
                loss = criterion(models[name](x), y)
                loss.backward()
                optimizers[name].step()
            acc = accuracy(models[name], device, test_loader)
            print(f"Epoch {epoch+1} | {name} Acc: {acc*100:.2f}%")

    # ===== 输出模组参数配置 =====
    print_module_configs(
        recipes=recipes,
        rotation_collate_fn=rotation_collate_fn,
        flip_collate_fn=flip_collate_fn,
        rrc_collate_fn=rrc_collate_fn,
        train_loader_plain=train_loader_plain,
    )

    os.makedirs("checkpoint", exist_ok=True)
    for name, model in models.items():
        torch.save(model.state_dict(), f"checkpoint/{name}.pth")
