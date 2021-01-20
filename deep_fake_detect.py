import argparse
from utils import *
from data_utils.face_detection import *


def main():
    print(ConfigParser.getInstance().get_log_dir_name())


if __name__ == '__main__':
    main()
