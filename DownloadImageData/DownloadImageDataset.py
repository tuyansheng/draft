import os
from torchvision.datasets import MNIST, CIFAR10, CIFAR100

def download_datasets(root="~/ImageData"):
    # 展开 ~ 为绝对路径
    root = os.path.expanduser(root)
    os.makedirs(root, exist_ok=True)

    print("Downloading MNIST...")
    MNIST(
        root=root,
        train=True,
        download=True
    )
    MNIST(
        root=root,
        train=False,
        download=True
    )

    print("Downloading CIFAR-10...")
    CIFAR10(
        root=root,
        train=True,
        download=True
    )
    CIFAR10(
        root=root,
        train=False,
        download=True
    )

    print("Downloading CIFAR-100...")
    CIFAR100(
        root=root,
        train=True,
        download=True
    )
    CIFAR100(
        root=root,
        train=False,
        download=True
    )

    print("All datasets downloaded successfully.")
    print(f"Datasets are stored in: {root}")

if __name__ == "__main__":
    download_datasets()
