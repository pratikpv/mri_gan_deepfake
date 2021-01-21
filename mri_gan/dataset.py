import random
from torch.utils.data import DataLoader
import torchvision
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from data_utils.utils import *
from utils import *
from PIL import Image


class MRIDataset(Dataset):
    def __init__(self, transforms=None, mode="train"):
        self.transforms = transforms

        if mode == "train":
            self.data_csv = ConfigParser.getInstance().get_mri_train_dataset_csv_path()
        elif mode == "test":
            self.data_csv = ConfigParser.getInstance().get_mri_test_dataset_csv_path()
        else:
            raise Exception("Unknown mode")

        self.df = pd.read_csv(self.data_csv)
        self.data_dict = self.df.to_dict(orient='records')
        self.df_len = len(self.df)

    def __getitem__(self, index):
        while True:
            try:
                item = self.data_dict[index].copy()
                img_A = Image.open(str(item['face_image']))
                img_B = Image.open(str(item['mri_image']))

                if np.random.random() < 0.5:
                    img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
                    img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

                if self.transforms:
                    img_A = self.transforms(img_A)
                    img_B = self.transforms(img_B)

                return {"A": img_A, "B": img_B}

            except Exception:
                index = random.randint(0, self.df_len)

    def __len__(self) -> int:
        return self.df_len
