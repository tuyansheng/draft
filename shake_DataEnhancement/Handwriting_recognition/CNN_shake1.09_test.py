import os
import sys
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from shake_aug import ShakeRecipe, ShakeAugDataset, ShakeAugBatchCollator


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
        return self.fc2(x)


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


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    root = "/home/ystu/research_code/Machine_Learning/shake_DataEnhancement/Handwriting_recognition"

    train_base = MNIST(root=root, train=True, download=True, transform=ToTensor())
    test_base = MNIST(root=root, train=False, download=True, transform=ToTensor())

    # ============================
    # 从命令行读取 ShakeRecipe 参数
    # ============================
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    beta = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
    Sx = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    Sy = int(sys.argv[5]) if len(sys.argv) > 5 else 4

    recipes = [ShakeRecipe(N=N, beta=beta, alpha=alpha, S=(Sx, Sy))]

    train_aug = ShakeAugDataset(
        base=train_base,
        recipes=recipes,
        include_original=True,
        dynamic=True,
        base_seed=0,
        apply_in_getitem=False,
    )

    train_loader_aug = DataLoader(
        train_aug,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=ShakeAugBatchCollator(recipes),
    )

    test_loader = DataLoader(
        test_base,
        batch_size=512,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = ConvNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # ======= 只执行第一个 epoch =======
    epoch = 0
    train_aug.set_epoch(epoch)
    model.train()

    for x, y in train_loader_aug:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    acc = accuracy(model, device, test_loader)

    # ======= 输出 recipes 参数 + 正确率 =======
    print(acc)

    os.makedirs("checkpoint", exist_ok=True)
    torch.save(model.state_dict(), "checkpoint/aug.pth")
