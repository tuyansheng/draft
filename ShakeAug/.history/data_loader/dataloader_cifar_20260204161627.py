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

        