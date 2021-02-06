import random
from torch.utils.data import DataLoader
import torchvision
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from data_utils.utils import *
from utils import *
from PIL import Image

class SimpleImageFolder(Dataset):
    def __init__(self, root, transforms_=None):
        self.root = root
        all_files = glob(root + "/*")
        self.data_list = [os.path.abspath(f) for f in all_files]
        self.data_len = len(self.data_list)
        self.transforms = transforms_

    def __getitem__(self, index):
        img_name = self.data_list[index]
        img = Image.open(self.data_list[index])
        if self.transforms:
            img = self.transforms(img)
        return img, img_name

    def __len__(self) -> int:
        return self.data_len
