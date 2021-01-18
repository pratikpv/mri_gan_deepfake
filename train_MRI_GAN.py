import argparse
from mri_gan.training import *
from data_utils.face_mri import *


def main():
    if args.train_from_scratch:
        print(f'Training MRI-GAN from scratch')
        train_MRI_GAN_model(log_dir=ConfigParser.getInstance().get_log_dir_name())
    if args.train_resume_checkpoint_dir:
        print(f'Resume MRI-GAN training from checkpoint {args.train_resume_checkpoint_dir}')
        train_MRI_GAN_model(log_dir=ConfigParser.getInstance().get_log_dir_name(),
                            train_resume_dir=args.train_resume_checkpoint_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MRI-GAN')

    parser.add_argument('--train_from_scratch', action='store_true', default=False,
                        help='Train MRI-GAN from scratch')
    parser.add_argument('--train_resume', dest='train_resume_checkpoint_dir', default=False,
                        help='Resume MRI-GAN training')
    args = parser.parse_args()
    main()
