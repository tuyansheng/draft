from dataclasses import dataclass
from typing import Tuple, Optional, Literal
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

