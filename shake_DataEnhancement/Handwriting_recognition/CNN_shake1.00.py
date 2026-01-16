import os
import torch
from torch import nn, optim, utils
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from shake_aug import ShakeRecipe, ShakeAugDataset, ShakeAugBatchCollator
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# CNN网络构建
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        in_size = x.size(0)
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2, 2)
        out = F.relu(self.conv2(out))
        out = out.view(in_size, -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)


def train(model, device, train_dataloader, optimizer, criterion):
    model.train()
    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def test(model, device, val_dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":

    # 设备选择
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # 超参数
    batch_size = 256
    num_epochs = 10
    lr = 1e-4

    # MNIST + ShakeAugDataset
    mnist = MNIST(
        root="/home/yansheng/shake_DataEnhancement/data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    recipe = [ShakeRecipe(N=2, beta=0.5, alpha=0.3, S=(4, 4))]
    dataset = ShakeAugDataset(
        base=mnist,
        recipes=recipe,
        include_original=True,
        dynamic=True,
        base_seed=0,
        apply_in_getitem=True
    )

    N_data = len(dataset)
    train_dataset, val_dataset = utils.data.random_split(
        dataset,
        [int(N_data * 0.8), N_data - int(N_data * 0.8)]
    )

    train_dataloader = utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_dataloader = utils.data.DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False
    )

    # 模型
    model = ConvNet().to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练循环
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train(model, device, train_dataloader, optimizer, criterion)
        if epoch % 2 == 0:
            test(model, device, val_dataloader)

    # 保存模型
    os.makedirs("checkpoint", exist_ok=True)
    torch.save(model.state_dict(), "checkpoint/latest_checkpoint.pth")
