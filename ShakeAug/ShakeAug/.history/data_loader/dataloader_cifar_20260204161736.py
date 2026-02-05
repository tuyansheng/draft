import os
import torch
from torch.utils.data import Dataset
from PIL import Image

import torchvision.datasets as tv_datasets




class CIFARDataset(Dataset):
    def __init__(self, data_path: str, dataset: str = "cifar10", train: bool = True, transform=None):
        self.root = data_path
        self.dataset = dataset.lower()
        self.train = train
        self.transform = transform

        if self.dataset == "cifar10":
            base = tv_datasets.CIFAR10(root=self.root, train=self.train, transform=None, download=False)
        elif self.dataset == "cifar100":
            base = tv_datasets.CIFAR100(root=self.root, train=self.train, transform=None, download=False)
        
        else:
            raise ValueError("dataset must be 'cifar10' or 'cifar100'")
        

        # torchvision CIFAR stores:
        # base.data: numpy array (N, 32, 32, 3)
        # base.targets: list[int] or numpy

        