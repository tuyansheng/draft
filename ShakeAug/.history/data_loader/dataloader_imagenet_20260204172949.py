import os
from typing import Literal, Optional, Tuple, List

import torch
from torch.utils.data import Dataset
from PIL import Image


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