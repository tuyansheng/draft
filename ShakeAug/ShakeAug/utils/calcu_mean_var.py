import torch
from torchvision import datasets, transforms

# 先不归一化，加载数据
dataset = datasets.ImageFolder('./data', transform=transforms.ToTensor())

loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

mean = 0.
std = 0.
total_images = 0

for images, _ in loader:
    batch_samples = images.size(0)  # batch size
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    total_images += batch_samples

mean /= total_images
std /= total_images

print("Mean:", mean)
print("Std:", std)