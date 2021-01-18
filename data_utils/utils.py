import cv2
import json
import os
from glob import glob
from pathlib import Path
import random
from subprocess import Popen, PIPE
from utils import *
import shutil
from tqdm import tqdm
import pandas as pd
from utils import *
import pickle
from utils import *

random_setting = '$RANDOM$'


def get_frame_count_from_video(video_file):
    cap = cv2.VideoCapture(video_file)
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def compress_video(input_videofile, output_videofile, lvl=None):
    if lvl is None:
        lvl = random.choice([27, 28, 29])
    command = ['ffmpeg', '-i', input_videofile, '-c:v', 'libx264', '-crf', str(lvl),
               '-threads', '1', '-loglevel', 'quiet', '-y', output_videofile]
    try:
        # print(command)
        process = Popen(command, stdout=PIPE, stderr=PIPE)
        process.wait()
        p_out, p_err = process.communicate()
    except Exception as e:
        print_line()
        print("Failed to compress video", str(e))


def in_range(val, min_v, max_v):
    if min_v <= val <= max_v:
        return True
    return False


def adaptive_video_compress(input_videofile, min_file_size, max_file_size, max_tries=4):
    output_videofile = os.path.join(os.path.dirname(input_videofile),
                                    os.path.splitext(os.path.basename(input_videofile))[0] + '_tmp.mp4',
                                    )
    file_size_org = os.path.getsize(input_videofile)
    result = {'input_file': input_videofile,
              'cmprsn_lvl': -1,
              'file_size_org': file_size_org,
              'file_size_comprsd': -1}
    if file_size_org <= max_file_size:
        return result

    cmprsn_lvl = 30
    already_tried_lvls = list()
    for i in range(max_tries):
        if cmprsn_lvl in already_tried_lvls:
            break
        compress_video(input_videofile, output_videofile, lvl=cmprsn_lvl)
        already_tried_lvls.append(cmprsn_lvl)
        f_size = os.path.getsize(output_videofile)
        if f_size > max_file_size:
            # increase compression
            cmprsn_lvl += 1
        elif f_size < min_file_size:
            # decrease compression
            cmprsn_lvl -= 1
        else:
            break

    shutil.move(output_videofile, input_videofile)
    file_size_comprsd = os.path.getsize(input_videofile)
    result['cmprsn_lvl'] = cmprsn_lvl
    result['file_size_comprsd'] = file_size_comprsd
    return result


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
sample entries from metadata.json

{"iqqejyggsm.mp4": {"label": "FAKE", "split": "train", "original": "gzesfubacw.mp4"}
{"ooafcxxfrs.mp4": {"label": "REAL", "split": "train"}

"""


def get_original_training_video_paths(root_dir, basename=False):
    originals = set()
    originals_v = set()
    for json_path in glob(os.path.join(root_dir, "*/metadata.json")):
        dir = Path(json_path).parent
        with open(json_path, "r") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            original = v.get("original", None)
            if v["label"] == "REAL":
                original = k
                originals_v.add(original)
                originals.add(os.path.join(dir, original))
    originals = list(originals)
    originals_v = list(originals_v)
    print(len(originals))
    return originals_v if basename else originals


def get_training_original_with_fakes(root_dir):
    pairs = []
    for json_path in glob(os.path.join(root_dir, "*/metadata.json")):
        with open(json_path, "r") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            original = v.get("original", None)
            if v["label"] == "FAKE":
                pairs.append((original[:-4], k[:-4]))

    return pairs


def get_training_reals_and_fakes(root_dir):
    originals = []
    fakes = []
    for json_path in glob(os.path.join(root_dir, "*/metadata.json")):
        with open(json_path, "r") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            if v["label"] == "FAKE":
                fakes.append(k)
            else:
                originals.append(k)

    return originals, fakes


def get_valid_reals_and_fakes():
    labels_csv = ConfigParser.getInstance().get_valid_labels_csv_filepath()
    df = pd.read_csv(labels_csv, index_col=0)
    originals = list(df[df['label'] == 0].index.values)
    fakes = list(df[df['label'] == 1].index.values)

    return originals, fakes


def get_test_reals_and_fakes():
    labels_csv = ConfigParser.getInstance().get_test_labels_csv_filepath()
    df = pd.read_csv(labels_csv, index_col=0)
    originals = list(df[df['label'] == 0].index.values)
    fakes = list(df[df['label'] == 1].index.values)

    return originals, fakes


def get_dfdc_training_video_filepaths(root_dir):
    video_filepaths = []
    for json_path in glob(os.path.join(root_dir, "*/metadata.json")):
        dir = Path(json_path).parent
        with open(json_path, "r") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            full_path = os.path.join(dir, k)
            video_filepaths.append(full_path)

    return video_filepaths



def get_all_validation_video_filepaths(root_dir, match_proc=False, min_num_frames=10):
    video_filepaths = []
    for f in glob(os.path.join(root_dir, "*.mp4")):
        video_filepaths.append(os.path.join(root_dir, f))

    if match_proc:
        video_filepaths_ = video_filepaths.copy()
        crops_path = ConfigParser.getInstance().get_valid_crop_faces_data_path()
        for v in tqdm(video_filepaths_):
            crops_id_path = os.path.join(crops_path, os.path.splitext(os.path.basename(v))[0])
            if not os.path.isdir(crops_id_path):
                video_filepaths.remove(v)
                continue
            # print(crops_id_path)
            frame_names = glob(crops_id_path + '/*_0.png')
            # print(len(frame_names))
            if len(frame_names) < min_num_frames:
                video_filepaths.remove(v)
    return video_filepaths


def get_all_test_video_filepaths(root_dir, match_proc=False, min_num_frames=10):
    video_filepaths = []
    for f in glob(os.path.join(root_dir, "*.mp4")):
        video_filepaths.append(os.path.join(root_dir, f))

    if match_proc:
        video_filepaths_ = video_filepaths.copy()
        crops_path = ConfigParser.getInstance().get_test_crop_faces_data_path()
        for v in tqdm(video_filepaths_):
            crops_id_path = os.path.join(crops_path, os.path.splitext(os.path.basename(v))[0])
            if not os.path.isdir(crops_id_path):
                video_filepaths.remove(v)
                continue
            # print(crops_id_path)
            frame_names = glob(crops_id_path + '/*_0.png')
            # print(len(frame_names))
            if len(frame_names) < min_num_frames:
                video_filepaths.remove(v)
    return video_filepaths


def generate_processed_validation_video_filepaths(root_dir, min_num_frames=10):
    files = get_all_validation_video_filepaths(root_dir, match_proc=True, min_num_frames=min_num_frames)
    print(f'num of files: {len(files)}')
    filename = ConfigParser.getInstance().get_processed_validation_data_filepath()
    with open(filename, 'wb') as f:
        pickle.dump(files, f)


def generate_processed_test_video_filepaths(root_dir, min_num_frames=10):
    print(root_dir)
    files = get_all_test_video_filepaths(root_dir, match_proc=True, min_num_frames=min_num_frames)
    print(f'num of files: {len(files)}')
    filename = ConfigParser.getInstance().get_processed_test_data_filepath()
    with open(filename, 'wb') as f:
        pickle.dump(files, f)


def get_processed_validation_video_filepaths():
    filename = ConfigParser.getInstance().get_processed_validation_data_filepath()
    with open(filename, 'rb') as f:
        files = pickle.load(f)

    return files


def get_processed_test_video_filepaths():
    filename = ConfigParser.getInstance().get_processed_test_data_filepath()
    with open(filename, 'rb') as f:
        files = pickle.load(f)

    return files


def restore_augmented_files(aug_metadata, src_root, dest_root):
    vdo_files = glob(aug_metadata + '/*')
    vdo_set = set()
    for f in tqdm(vdo_files, desc='Restoring augmented files'):
        df = pd.read_csv(f, index_col=0)
        input_file = df.loc['input_file'].values[0]
        input_file = '/'.join(input_file.split('/')[-2:])
        if input_file in vdo_set:
            continue
        vdo_set.add(input_file)
        src_path = os.path.join(src_root, input_file)
        dest_path = os.path.join(dest_root, input_file)
        shutil.copyfile(src_path, dest_path)


def get_files_size(train_data_path, in_MB=False):
    v_paths = get_all_training_video_filepaths(train_data_path)
    file_size_map = list()
    for v in v_paths:
        try:
            f_size = os.path.getsize(v)
            if in_MB:
                f_size = round(f_size / (1024 * 1024), 2)
            file_size_map.append((v, f_size))
        except FileNotFoundError as e:
            # print(f'not found {v}')
            pass

    return file_size_map


def get_video_integrity(input_videofile):
    command = ['ffmpeg', '-v', 'error', '-i', input_videofile, '-f', 'null', '-']
    result = {'filename': input_videofile,
              'status': 'valid'}
    try:
        # print(command)
        process = Popen(command, stdout=PIPE, stderr=PIPE)
        process.wait()
        p_out, p_err = process.communicate()
        p_err = str(p_err)[2:-1]
        p_out = str(p_out)[2:-1]
        # print(f'err: {p_err}, len = {len(p_err)}')
        # print(f'out: {p_out}, len = {len(p_err)}')

        if len(p_err) > 0:
            result['status'] = 'invalid'
            return result
        else:
            result['status'] = 'valid'
            return result
    except Exception as e:
        print_line()
        print("Failed to get_video_integrity", str(e))
        result['status'] = 'failed_to_check'
        return result


def consolidate_training_labels(data_root_dir, csv_file):
    reals, fakes = get_training_reals_and_fakes(data_root_dir)
    real_label = 0
    fake_label = 1
    df_real = pd.DataFrame(list(zip(reals, [real_label] * len(reals))),
                           columns=['filename', 'label']).set_index('filename')
    df_fake = pd.DataFrame(list(zip(fakes, [fake_label] * len(fakes))),
                           columns=['filename', 'label']).set_index('filename')
    df = df_real.append(df_fake)
    df.to_csv(csv_file)


def get_video_frame_labels_mapping(cid, originals, fakes):
    cid_ = os.path.basename(cid)
    if cid_ in originals:
        crop_label = 0
    elif cid_ in fakes:
        crop_label = 1
    else:
        raise Exception('Unknown label')
    crop_items = glob(cid + '/*')
    df = pd.DataFrame(columns=['video_id', 'frame', 'label'])
    for crp_itm in crop_items:
        crp_itm_ = os.path.basename(crp_itm)
        new_row = {'video_id': cid_, 'frame': crp_itm_, 'label': crop_label}
        df = df.append(new_row, ignore_index=True)

    return df


def get_number_of_faces_detected(crops_id_path):
    cid = os.path.basename(crops_id_path)
    frame_names = glob(crops_id_path + '/*_*.png')
    face_id_list = list(
        set([int(i.replace(crops_id_path + '/', '').replace('.png', '').split('_')[1]) + 1 for i in frame_names]))
    face_id_list.append(0)
    num_faces = max(face_id_list)
    return {'video_id': cid, 'num_faces': num_faces}


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
