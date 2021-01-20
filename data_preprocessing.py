import argparse
from data_utils.face_mri import *
from data_utils.face_detection import *


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preprocessing')

    parser.add_argument('--extract_landmarks', action='store_true', default=False,
                        help='Extract landmarks')
    parser.add_argument('--crop_faces', action='store_true', default=False,
                        help='Crop faces')
    parser.add_argument('--gen_mri_dataset', action='store_true', default=False,
                        help='Generate MRI dataset')
    args = parser.parse_args()
    main()
