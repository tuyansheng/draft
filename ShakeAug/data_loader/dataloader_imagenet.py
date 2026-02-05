import os
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".JPEG", ".JPG"}


def _is_image_file(fname: str) -> bool:
    return os.path.splitext(fname)[1] in IMG_EXTS

class ImageNetDataset(Dataset):
    def __init__(self, data_path: str, transform=None, return_path: bool = False):
        self.root = data_path
        self.transform = transform
        self.return_path = return_path

        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Split folder not found: {self.root}")
        
        self.ds = ImageFolder(root=self.root, transform=self.transform)

        self.classes = self.ds.classes
        self.class_to_idx = self.ds.class_to_idx
        self.samples = self.ds.samples  # list[(path, class_idx)]

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx: int):
        img, label = self.ds[idx]
        label = torch.tensor(label, dtype=torch.long)

        if self.return_path:
            path, _ = self.samples[idx]
            return img, label, path
        return img, label