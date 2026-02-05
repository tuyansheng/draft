import torch
from torch.utils.data import Dataset
from PIL import Image

import torchvision.datasets as tv_datasets



class MNISTDataset(Dataset):
    def __init__(self, data_path: str, train: bool = True, transform = None):
        self.root = data_path
        self.train = train
        self.transform = transform

        base = tv_datasets.MNIST(root=self.root, train=self.train, transform=None, download=False)


        # base.data: torch.Tensor [N, 28, 28] (uint8)
        # base.targets: torch.Tensor [N]
        self.data = base.data
        self.targets = base.targets

    def __len__(self):
        return int(self.targets.shape[0])

    def __getitem__(self, idx: int):
        img = self.data[idx]  # uint8 tensor [28,28]
        label = int(self.targets[idx])

        # to PIL (mode "L" for grayscale)
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)