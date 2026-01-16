import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

dataset = MNIST(root="/home/yansheng/shake_DataEnhancement/data", train=True, download=True, transform=ToTensor())

print("number of picture",len(dataset))
for i in range(10):
    img, label = dataset[i]
    print(type(img), img.shape, label)

    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(f"Label: {label}")
    plt.show()


print("data:", dataset.data.shape)
print("labels:", dataset.labels.shape)