import argparse
from utils import *
from data_utils.face_detection import *


def main():
    print(ConfigParser.getInstance().get_log_dir_name())
    extract_landmarks_from_video('/home/therock/data2/data_workset/dfdc/astfuznijm.mp4',
                                 '/home/therock/data2/data_workset/dfdc/')
    crop_faces_from_video('/home/therock/data2/data_workset/dfdc/astfuznijm.mp4',
                          '/home/therock/data2/data_workset/dfdc/',
                          '/home/therock/data2/data_workset/dfdc/astfuznijm', buf=0.)


if __name__ == '__main__':
    main()
