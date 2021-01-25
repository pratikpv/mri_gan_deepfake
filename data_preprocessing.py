import argparse
from data_utils.face_mri import *
from data_utils.face_detection import *
from deep_fake_detect.features import *


def main():
    if args.extract_landmarks:
        print(f'Extract Landmarks')
        extract_landmarks_for_datasets()

    if args.crop_faces:
        print(f'Crop Faces')
        crop_faces_for_datasets()

    if args.gen_mri_dataset:
        print(f'Generate MRI dataset')
        generate_MRI_dataset()

    if args.gen_dfdc_mri:
        print(f'Generate MRIs of DFDC dataset using trained MRI-GAN')
        generate_DFDC_MRIs()

    if args.gen_deepfake_metadata:
        print(f'Generate frame label csv files')
        generate_frame_label_csv_files()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preprocessing')

    parser.add_argument('--extract_landmarks', action='store_true', default=False,
                        help='Extract landmarks')
    parser.add_argument('--crop_faces', action='store_true', default=False,
                        help='Crop faces')
    parser.add_argument('--gen_mri_dataset', action='store_true', default=False,
                        help='Generate MRI dataset')
    parser.add_argument('--gen_dfdc_mri', action='store_true', default=False,
                        help='Generate MRIs of DFDC dataset using trained MRI-GAN')
    parser.add_argument('--gen_deepfake_metadata', action='store_true', default=False,
                        help='Generate metadata')
    args = parser.parse_args()
    main()
