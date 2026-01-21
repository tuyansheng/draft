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
    return acc


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
    test_dataset = MNIST(root=root, train=False, download=True, transform=ToTensor())

    # 增强训练集
    recipe = [ShakeRecipe(N=2, beta=0.5, alpha=0.3, S=(4, 4))]
    train_aug = ShakeAugDataset(
        base=train_base,
        recipes=recipe,
        include_original=True,
        dynamic=False,
        base_seed=0,
        apply_in_getitem=True
    )

    # 原始训练集
    train_plain = train_base

    train_loader_aug = DataLoader(train_aug, batch_size=256*2, shuffle=True)
    train_loader_plain = DataLoader(train_plain, batch_size=256, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # =========================
    # 模型 & 优化器
    # =========================
    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # =========================
    # 并行训练循环
    # =========================
    global_step = 0
    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()
        print(f"\nEpoch {epoch + 1}")

        for (x_aug, y_aug), (x_plain, y_plain) in zip(train_loader_aug, train_loader_plain):

            # ---- 增强 batch ----
            x_aug, y_aug = x_aug.to(device), y_aug.to(device)
            optimizer.zero_grad()
            loss_aug = criterion(model(x_aug), y_aug)
            loss_aug.backward()
            optimizer.step()

            global_step += 1

            # ---- 原始 batch ----
            x_plain, y_plain = x_plain.to(device), y_plain.to(device)
            optimizer.zero_grad()
            loss_plain = criterion(model(x_plain), y_plain)
            loss_plain.backward()
            optimizer.step()

            global_step += 1

            # ---- 每 2 个 batch 测试一次 ----
            if global_step % 2 == 0:
                acc = test(model, device, test_loader)
                print(
                    f"Step {global_step:5d} | "
                    f"Loss_aug: {loss_aug.item():.4f} | "
                    f"Loss_plain: {loss_plain.item():.4f} | "
                    f"Test Acc: {acc * 100:.2f}%"
                )

    # =========================
    # 保存模型
    # =========================
    os.makedirs("checkpoint", exist_ok=True)
    torch.save(model.state_dict(), "checkpoint/latest_checkpoint.pth")
