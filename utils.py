import yaml
import torch
import sys
import cv2
import yaml
from datetime import datetime
import os
from pprint import pprint
import shutil


class ConfigParser:
    __instance = None

    @staticmethod
    def getInstance():
        if ConfigParser.__instance is None:
            ConfigParser()
        return ConfigParser.__instance

    def __init__(self):
        self.config_file = 'config.yml'
        self.config = dict()
        self.init_time_str = str(datetime.now().strftime("%d-%b-%Y_%H_%M_%S"))

        if ConfigParser.__instance is not None:
            raise Exception("ConfigParser class is a singleton!")
        else:
            ConfigParser.__instance = self

        with open(self.config_file, 'r') as c_file:
            self.config = yaml.safe_load(c_file)

    def print_config(self):
        pprint(self.config)

    def copy_config(self, dest=None):
        shutil.copy(self.config_file, dest)

    def get_assets_path(self):
        return self.config['assets']

    def get_dfdc_train_data_path(self):
        return self.config['data_path']['dfdc']['train']

    def get_dfdc_valid_data_path(self):
        return self.config['data_path']['dfdc']['valid']

    def get_dfdc_test_data_path(self):
        return self.config['data_path']['dfdc']['test']

    def get_dfdc_backup_train_data_path(self):
        return self.config['data_path']['dfdc']['train_backup']

    def get_train_frame_label_csv_path(self):
        return os.path.join(self.get_assets_path(), self.config['data_path']['train_frame_label'])

    def get_valid_frame_label_csv_path(self):
        return os.path.join(self.get_assets_path(), self.config['data_path']['valid_frame_label'])

    def get_test_frame_label_csv_path(self):
        return os.path.join(self.get_assets_path(), self.config['data_path']['test_frame_label'])

    def get_train_labels_csv_filepath(self):
        return os.path.join(self.get_assets_path(), self.config['data_path']['train_labels_csv_filename'])

    """
    def get_valid_labels_csv_filepath(self):
        return os.path.join(get_validation_data_path(), self.config['data_path']['valid_labels_csv_filename'])

    def get_test_labels_csv_filepath(self):
        return os.path.join(get_test_data_path(), self.config['data_path']['test_labels_csv_filename'])
    """

    def get_train_mriframe_label_csv_path(self):
        return os.path.join(self.get_assets_path(), self.config['data_path']['train_mriframe_label'])

    def get_valid_mriframe_label_csv_path(self):
        return os.path.join(self.get_assets_path(), self.config['data_path']['valid_mriframe_label'])

    def get_test_mriframe_label_csv_path(self):
        return os.path.join(self.get_assets_path(), self.config['data_path']['test_mriframe_label'])

    def get_processed_train_data_filepath(self):
        return os.path.join(self.get_assets_path(), self.config['data_path']['processed_train_filename'])

    def get_processed_validation_data_filepath(self):
        return os.path.join(self.get_assets_path(), self.config['data_path']['processed_valid_filename'])

    def get_processed_test_data_filepath(self):
        return os.path.join(self.get_assets_path(), self.config['data_path']['processed_test_filename'])

    def get_train_facecount_csv_filepath(self):
        return os.path.join(self.get_assets_path(), self.config['data_path']['train_faces_count'])

    def get_valid_facecount_csv_filepath(self):
        return os.path.join(self.get_assets_path(), self.config['data_path']['valid_faces_count'])

    def get_test_facecount_csv_filepath(self):
        return os.path.join(self.get_assets_path(), self.config['data_path']['test_faces_count'])

    def get_data_aug_plan_pkl_filename(self):
        return os.path.join(self.config['assets'], self.config['data_augmentation']['plan_pkl_filename'])

    def get_data_aug_plan_txt_filename(self):
        return os.path.join(self.config['assets'], self.config['data_augmentation']['plan_txt_filename'])

    def get_aug_metadata_path(self):
        return os.path.join(self.config['assets'], self.config['data_augmentation']['metadata'])

    def get_compression_csv_path(self):
        return os.path.join(self.get_assets_path(), self.config['data_augmentation']['compression_csv_filename'])

    def get_video_integrity_data_path(self):
        return os.path.join(self.get_assets_path(), self.config['data_augmentation']['integrity_csv_filename'])

    def get_mri_metadata_csv(self):
        return os.path.join(self.get_assets_path(), self.config['features']['mri_metadata_csv'])

    def get_mri_pairs_train_csv(self):
        return os.path.join(self.get_assets_path(), self.config['features']['mri_pairs_train_csv'])

    def get_mri_pairs_test_csv(self):
        return os.path.join(self.get_assets_path(), self.config['features']['mri_pairs_test_csv'])

    def get_blank_imagepath(self):
        return os.path.join(self.get_assets_path(), self.config['features']['blank_png'])

    def get_default_cnn_encoder_name(self):
        return self.config['cnn_encoder']['default']

    def get_train_faces_cnn_features_data_path(self):
        return os.path.join(self.config['features']['train_faces_cnn'], self.get_default_cnn_encoder_name())

    def get_valid_faces_cnn_features_data_path(self):
        return os.path.join(self.config['features']['valid_faces_cnn'], self.get_default_cnn_encoder_name())

    def get_test_faces_cnn_features_data_path(self):
        return os.path.join(self.config['features']['test_faces_cnn'], self.get_default_cnn_encoder_name())

    def get_faces_loc_video_path(self):
        return os.path.join(self.get_assets_path(), self.config['features']['face_location_video_data_path'])

    def get_log_dir_name(self, create_logdir=True):
        log_dir = os.path.join(self.config['logging']['root_log_dir'], self.init_time_str)
        if create_logdir:
            os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def get_training_sample_size(self):
        return float(self.config['training']['train_size'])

    def get_valid_sample_size(self):
        return float(self.config['training']['valid_size'])

    def get_test_sample_size(self):
        return float(self.config['training']['test_size'])

    def get_mri_path(self):
        return self.config['features']['train_mri_png']

    def get_checkpoint_root_path(self):
        return os.path.join(self.get_assets_path(), self.config['training']['checkpoint_path'])

    def get_training_params(self):
        return self.config['training']['params']

    def get_log_params(self):
        return self.config['logging']

    def create_assets_placeholder(self):
        os.makedirs(self.get_assets_path(), exist_ok=True)
        os.makedirs(self.get_aug_metadata_path(), exist_ok=True)

    def get_celeb_df_v2_real_path(self):
        return self.config['data_path']['celeb_df_v2']['real']

    def get_celeb_df_v2_fake_path(self):
        return self.config['data_path']['celeb_df_v2']['fake']

    def get_celeb_df_v2_landmarks_path(self):
        return self.config['features']['celeb_df_v2']['landmarks_path']['train']

    def get_dfdc_landmarks_train_path(self):
        return self.config['features']['dfdc']['landmarks_path']['train']

    def get_dfdc_landmarks_valid_path(self):
        return self.config['features']['dfdc']['landmarks_path']['valid']

    def get_dfdc_landmarks_test_path(self):
        return self.config['features']['dfdc']['landmarks_path']['test']

    def get_dfdc_crops_train_path(self):
        return self.config['features']['dfdc']['crop_faces']['train']

    def get_dfdc_crops_valid_path(self):
        return self.config['features']['dfdc']['crop_faces']['valid']

    def get_dfdc_crops_test_path(self):
        return self.config['features']['dfdc']['crop_faces']['test']

    def get_celeb_df_v2_crops_train_path(self):
        return self.config['features']['celeb_df_v2']['crop_faces']['train']


def print_line(print_len=None):
    if print_len is None:
        print('-' * ConfigParser.getInstance().config['logging']['line_len'])


def print_banner():
    print_line()

    log_dir = ConfigParser.getInstance().get_log_dir_name()
    print(f'LOG_DIR = {log_dir}')
    print(f'PyTorch version = {torch.__version__}')
    if torch.cuda.is_available():
        print(f'PyTorch GPU = {torch.cuda.get_device_name(torch.cuda.current_device())}')
    else:
        print('PyTorch No cuda-based GPU detected.')
    print(f'OpenCV version  = {cv2.__version__}')

    print_line()

    return log_dir
