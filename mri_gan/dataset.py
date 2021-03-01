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
    def __init__(self, transforms=None, mode="train", frac=1.0):
        self.transforms = transforms

        if mode == "train":
            self.real_data_csv = ConfigParser.getInstance().get_mri_train_real_dataset_csv_path()
            self.fake_data_csv = ConfigParser.getInstance().get_mri_train_fake_dataset_csv_path()
        elif mode == "test":
            self.real_data_csv = ConfigParser.getInstance().get_mri_test_real_dataset_csv_path()
            self.fake_data_csv = ConfigParser.getInstance().get_mri_test_fake_dataset_csv_path()
        else:
            raise Exception("Unknown mode")

        self.real_df = pd.read_csv(self.real_data_csv)
        if frac < 1.0:
            self.real_df = self.real_df.sample(frac=frac).reset_index(drop=True)
        self.real_df_len = len(self.real_df)
        self.fake_df = pd.read_csv(self.fake_data_csv)
        # our dataset if skewed. we have lesser real samples and more fake :)
        self.fake_df_trimmed = self.fake_df.sample(n=self.real_df_len)

        # merge both df
        self.df = pd.concat([self.real_df, self.fake_df_trimmed])
        # shuffle the df
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.data_dict = self.df.to_dict(orient='records')
        self.df_len = len(self.df)

        # print(mode)
        # print(f'real ={len(self.fake_df_trimmed)}')
        # print(f'fake ={len(self.real_df)}')
        # print(f'df_len ={self.df_len}')
        # self.df.to_csv('{}_epoch.csv'.format(mode))

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

            except Exception as e:
                index = random.randint(0, self.df_len)

    def __len__(self) -> int:
        return self.df_len
