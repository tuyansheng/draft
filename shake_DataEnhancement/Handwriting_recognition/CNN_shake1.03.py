import os
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from shake_aug import ShakeRecipe, ShakeAugDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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
        return F.log_softmax(x, dim=1)


# =========================
# 训练
# =========================
def train(model, device, loader, optimizer, criterion, epoch):
    model.train()
    for i, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f"Epoch [{epoch}] Iter [{i}/{len(loader)}] Loss: {loss.item():.4f}")


# =========================
# 测试
# =========================
def test(model, device, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    print(f"Test Accuracy: {acc * 100:.2f}%")
    return acc


# =========================
# 主程序
# =========================
if __name__ == "__main__":

    # 设备
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("Using device:", device)

    root = "/home/ystu/research_code/Machine_Learning/shake_DataEnhancement/Handwriting_recognition"

    # =========================
    # 数据集
    # =========================
    train_BaseDataset = MNIST(
        root=root,
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_Dataset = MNIST(
        root=root,
        train=False,
        download=True,
        transform=ToTensor()
    )

    # =========================
    # 训练集（增强）
    # =========================
    recipe = [ShakeRecipe(N=2, beta=0.5, alpha=0.3, S=(4, 4))]
    train_dataset = ShakeAugDataset(
        base=train_BaseDataset,
        recipes=recipe,
        include_original=True,
        dynamic=False,
        base_seed=0,
        apply_in_getitem=True
    )

    train_loader1 = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=False
    )

    # =========================
    # 测试集（不增强）
    # =========================
    test_loader = DataLoader(
        base=train_BaseDataset,
        batch_size=256,
        shuffle=False
    )

    # =========================
    # 模型 & 优化器
    # =========================
    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # =========================
    # 训练循环
    # =========================


    # =========================
    # 保存模型
    # =========================
    os.makedirs("checkpoint", exist_ok=True)
    torch.save(model.state_dict(), "checkpoint/latest_checkpoint.pth")
