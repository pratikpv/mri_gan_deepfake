import random
from torch.utils.data import DataLoader
import torchvision
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from data_utils.utils import *
from utils import *
from PIL import Image


class DFDCDatasetSimple(Dataset):
    def __init__(self, mode=None, transform=None, data_size=1, dataset=None, label_smoothing=0):
        super().__init__()
        self.mode = mode
        self.label_smoothing = 0  # use only in training, so update to param passed in train mode
        if mode == 'train':
            if dataset == 'plain':
                self.labels_csv = ConfigParser.getInstance().get_dfdc_train_frame_label_csv_path()
                self.crops_dir = ConfigParser.getInstance().get_dfdc_crops_train_path()
            elif dataset == 'mri':
                self.labels_csv = ConfigParser.getInstance().get_train_mriframe_label_csv_path()
                self.crops_dir = ConfigParser.getInstance().get_train_mrip2p_png_data_path()
            else:
                raise Exception('Bad dataset in DFDCDatasetSimple')

            self.label_smoothing = label_smoothing
        elif mode == 'valid':
            if dataset == 'plain':
                self.labels_csv = ConfigParser.getInstance().get_dfdc_valid_frame_label_csv_path()
                self.crops_dir = ConfigParser.getInstance().get_dfdc_crops_valid_path()
            elif dataset == 'mri':
                self.labels_csv = ConfigParser.getInstance().get_valid_mriframe_label_csv_path()
                self.crops_dir = ConfigParser.getInstance().get_valid_mrip2p_png_data_path()
            else:
                raise Exception('Bad dataset in DFDCDatasetSimple')

        elif mode == 'test':
            if dataset == 'plain':
                self.labels_csv = ConfigParser.getInstance().get_dfdc_test_frame_label_csv_path()
                self.crops_dir = ConfigParser.getInstance().get_dfdc_crops_test_path()
            elif dataset == 'mri':
                self.labels_csv = ConfigParser.getInstance().get_test_mriframe_label_csv_path()
                self.crops_dir = ConfigParser.getInstance().get_test_mrip2p_png_data_path()
            else:
                raise Exception('Bad dataset in DFDCDatasetSimple')
        else:
            raise Exception('Bad mode in DFDCDatasetSimple')

        self.data_df = pd.read_csv(self.labels_csv)
        if data_size < 1:
            total_data_len = int(len(self.data_df) * data_size)
            self.data_df = self.data_df.iloc[0:total_data_len]
        self.data_dict = self.data_df.to_dict(orient='records')
        self.data_len = len(self.data_df)
        self.transform = transform

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int):
        while True:
            try:
                item = self.data_dict[index].copy()
                frame = Image.open(os.path.join(self.crops_dir, str(item['video_id']), item['frame']))
                if self.transform is not None:
                    frame = self.transform(frame)
                item['frame_tensor'] = frame
                if self.label_smoothing != 0:
                    label = np.clip(item['label'], self.label_smoothing, 1 - self.label_smoothing)
                else:
                    label = item['label']
                item['label'] = torch.tensor(label)
                return item
            except Exception as e:
                # print(f"bad {os.path.join(self.crops_dir, str(item['video_id']), item['frame'])}")
                index = random.randint(0, self.data_len)


