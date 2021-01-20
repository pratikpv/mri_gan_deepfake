import json
from glob import glob
from pathlib import Path
from utils import *


def create_video_from_images(images, output_video_filename, fps=30, res=(1920, 1080)):
    # video = cv2.VideoWriter(output_video_filename, cv2.VideoWriter_fourcc('H', '2', '6', '4'), fps, res)
    video = cv2.VideoWriter(output_video_filename, cv2.VideoWriter_fourcc(*"mp4v"), fps, res)
    for image in images:
        video.write(image)
    video.release()


def extract_images_from_video(input_video_filename, output_folder, res=None):
    os.makedirs(output_folder, exist_ok=True)
    capture = cv2.VideoCapture(input_video_filename)
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(frames_num):
        capture.grab()
        success, frame = capture.retrieve()
        if not success:
            continue
        out_image_name = os.path.join(output_folder, "{}.jpg".format(i))
        if res is not None:
            frame = cv2.resize(frame, res, interpolation=cv2.INTER_AREA)
        cv2.imwrite(out_image_name, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])


"""
sample entries from metadata.json of DFDC

{"iqqejyggsm.mp4": {"label": "FAKE", "split": "train", "original": "gzesfubacw.mp4"}
{"ooafcxxfrs.mp4": {"label": "REAL", "split": "train"}

"""


def get_dfdc_training_real_fake_pairs(root_dir):
    pairs = []
    for json_path in glob(os.path.join(root_dir, "*/metadata.json")):
        with open(json_path, "r") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            original = v.get("original", None)
            if v["label"] == "FAKE":
                pairs.append((os.path.splitext(original)[0], os.path.splitext(k)[0]))

    return pairs


def get_dfdc_training_video_filepaths(root_dir):
    video_filepaths = []
    for json_path in glob(os.path.join(root_dir, "*/metadata.json")):
        pdir = Path(json_path).parent
        with open(json_path, "r") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            full_path = os.path.join(pdir, k)
            video_filepaths.append(full_path)

    return video_filepaths


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
