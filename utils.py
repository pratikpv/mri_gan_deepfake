import torch
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

        self.create_placeholders()

    def print_config(self):
        pprint(self.config)

    def copy_config(self, dest):
        shutil.copy(self.config_file, dest)

    def get_log_dir_name(self, create_logdir=True):
        log_dir = os.path.join(self.config['logging']['root_log_dir'], self.init_time_str)
        if create_logdir:
            os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def get_assets_path(self):
        return self.config['assets']

    def get_dfdc_train_data_path(self):
        return self.config['data_path']['dfdc']['train']

    def get_dfdc_valid_data_path(self):
        return self.config['data_path']['dfdc']['valid']

    def get_dfdc_test_data_path(self):
        return self.config['data_path']['dfdc']['test']

    def get_dfdc_train_label_csv_path(self):
        return os.path.join(self.get_assets_path(), self.config['data_path']['dfdc']['train_labels_csv_filename'])

    def get_dfdc_valid_label_csv_path(self):
        return os.path.join(self.get_dfdc_valid_data_path(),
                            self.config['data_path']['dfdc']['valid_labels_csv_filename'])

    def get_dfdc_test_label_csv_path(self):
        return os.path.join(self.get_dfdc_test_data_path(),
                            self.config['data_path']['dfdc']['test_labels_csv_filename'])

    def get_dfdc_train_frame_label_csv_path(self):
        return os.path.join(self.get_assets_path(), self.config['data_path']['dfdc']['train_frame_labels_csv_filename'])

    def get_dfdc_valid_frame_label_csv_path(self):
        return os.path.join(self.get_assets_path(), self.config['data_path']['dfdc']['valid_frame_labels_csv_filename'])

    def get_dfdc_test_frame_label_csv_path(self):
        return os.path.join(self.get_assets_path(), self.config['data_path']['dfdc']['test_frame_labels_csv_filename'])

    def get_data_aug_plan_pkl_filename(self):
        return os.path.join(self.get_assets_path(),
                            self.config['data_path']['dfdc']['data_augmentation']['plan_pkl_filename'])

    def get_aug_metadata_path(self):
        return os.path.join(self.get_assets_path(),
                            self.config['data_path']['dfdc']['data_augmentation']['metadata'])

    def get_celeb_df_v2_real_path(self):
        return self.config['data_path']['celeb_df_v2']['real']

    def get_celeb_df_v2_fake_path(self):
        return self.config['data_path']['celeb_df_v2']['fake']

    def get_celeb_df_v2_real_fake_mapping_json(self):
        return os.path.join(self.get_assets_path(), self.config['data_path']['celeb_df_v2']['real_fake_mapping'])

    def get_fdf_data_path(self):
        return os.path.join(self.get_assets_path(), self.config['data_path']['fdf']['data_path'])

    def get_ffhq_data_path(self):
        return os.path.join(self.get_assets_path(), self.config['data_path']['ffhq']['data_path'])

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

    def get_train_mrip2p_png_data_path(self):
        return self.config['features']['dfdc']['train_mrip2p_faces']

    def get_valid_mrip2p_png_data_path(self):
        return self.config['features']['dfdc']['valid_mrip2p_faces']

    def get_test_mrip2p_png_data_path(self):
        return self.config['features']['dfdc']['test_mrip2p_faces']

    def get_train_mriframe_label_csv_path(self):
        return os.path.join(self.get_assets_path(), self.config['features']['dfdc']['train_mriframe_label'])

    def get_valid_mriframe_label_csv_path(self):
        return os.path.join(self.get_assets_path(), self.config['features']['dfdc']['valid_mriframe_label'])

    def get_test_mriframe_label_csv_path(self):
        return os.path.join(self.get_assets_path(), self.config['features']['dfdc']['test_mriframe_label'])

    def get_dfdc_mri_metadata_csv_path(self):
        return os.path.join(self.get_assets_path(),
                            self.config['features']['dfdc']['mri_metadata_csv'])

    def get_dfdc_mri_path(self):
        return self.config['features']['dfdc']['mri_path']

    def get_celeb_df_v2_landmarks_path(self):
        return self.config['features']['celeb_df_v2']['landmarks_path']['train']

    def get_celeb_df_v2_crops_train_path(self):
        return self.config['features']['celeb_df_v2']['crop_faces']['train']

    def get_celeb_df_v2_mri_metadata_csv_path(self):
        return os.path.join(self.get_assets_path(),
                            self.config['features']['celeb_df_v2']['mri_metadata_csv'])

    def get_celeb_df_v2_mri_path(self):
        return self.config['features']['celeb_df_v2']['mri_path']

    def get_fdf_json_path(self):
        return os.path.join(self.get_assets_path(), self.config['features']['fdf']['json_filename'])

    def get_fdf_crops_path(self):
        return os.path.join(self.get_assets_path(), self.config['features']['fdf']['crops_path'])

    def get_fdf_json_path(self):
        return os.path.join(self.get_assets_path(), self.config['features']['ffhq']['json_filename'])

    def get_fdf_crops_path(self):
        return os.path.join(self.get_assets_path(), self.config['features']['ffhq']['crops_path'])

    def get_mri_train_real_dataset_csv_path(self):
        return os.path.join(self.get_assets_path(),
                            self.config['features']['mri_dataset_real_train_csv'])

    def get_mri_train_fake_dataset_csv_path(self):
        return os.path.join(self.get_assets_path(),
                            self.config['features']['mri_dataset_fake_train_csv'])

    def get_mri_test_real_dataset_csv_path(self):
        return os.path.join(self.get_assets_path(),
                            self.config['features']['mri_dataset_real_test_csv'])

    def get_mri_test_fake_dataset_csv_path(self):
        return os.path.join(self.get_assets_path(),
                            self.config['features']['mri_dataset_fake_test_csv'])

    def get_mri_dataset_csv_path(self):
        return os.path.join(self.get_assets_path(),
                            self.config['features']['mri_dataset_csv'])

    def get_blank_imagepath(self):
        return os.path.join(self.get_assets_path(), self.config['features']['blank_png'])

    def get_mri_gan_weight_path(self):
        return os.path.join(self.get_assets_path(), self.config['MRI_GAN']['weights'])

    def get_mri_gan_model_params(self):
        return self.config['MRI_GAN']['model_params']

    def get_default_cnn_encoder_name(self):
        return self.config['cnn_encoder']['default']

    def get_training_sample_size(self):
        return float(self.config['deep_fake']['training']['train_size'])

    def get_valid_sample_size(self):
        return float(self.config['deep_fake']['training']['valid_size'])

    def get_test_sample_size(self):
        return float(self.config['deep_fake']['training']['test_size'])

    def get_deep_fake_training_params(self):
        return self.config['deep_fake']['training']['model_params']

    def get_log_params(self):
        return self.config['logging']

    def create_placeholders(self):
        os.makedirs(self.get_assets_path(), exist_ok=True)


def print_line():
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


def print_green(text):
    """
    print text in green color
    @param text: text to print
    """
    print('\033[32m', text, '\033[0m', sep='')


def print_red(text):
    """
    print text in green color
    @param text: text to print
    """
    print('\033[31m', text, '\033[0m', sep='')
