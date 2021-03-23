import operator
from functools import partial
from facenet_pytorch.models.mtcnn import MTCNN
import cv2
from PIL import Image
import os
import numpy as np
from skimage.util import random_noise
import time
import torch
from data_utils.utils import *
from random_word import RandomWords
import random
import string

random_setting = '$RANDOM$'


def get_supported_rolling_dir():
    return ['l_to_r', 'r_to_l', 't_to_b', 'b_to_t']


def get_supported_shapes():
    return ['circle', 'rectangle']


def get_supported_colors():
    return [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255), (0, 0, 0), ]


def get_supported_shape_sizes():
    return ['xsmall', 'small', 'medium', 'large']


size_per = {
    'xsmall': 0.005,
    'small': 0.05,
    'medium': 0.08,
    'large': 0.12,
}


def get_random_alphanumeric_string(length=8):
    letters_and_digits = string.ascii_letters + string.digits
    result_str = ''.join((random.choice(letters_and_digits) for i in range(length)))
    return result_str


def apply_static_text_to_image(image, distraction_param=None):
    text = distraction_param['text']
    if distraction_param['text'] == '':
        return image

    loc = distraction_param['loc']
    color = distraction_param['color']
    thickness = distraction_param['thickness']
    font_scale = distraction_param['fontScale']

    image = cv2.putText(image, text, loc, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        color, thickness, cv2.LINE_AA, False)
    return image


def apply_rolling_text_to_image(image, distraction_param):
    if distraction_param['text'] == '':
        return image
    return apply_static_text_to_image(image, distraction_param)


def apply_spontaneous_text(image, distraction_param):
    if distraction_param['text'] == '':
        return image
    return apply_static_text_to_image(image, distraction_param)


def apply_static_shape_to_image(image, distraction_param):
    shape = distraction_param['shape']
    if distraction_param['shape'] == '':
        return image

    loc = distraction_param['loc']
    color = distraction_param['color']
    size = distraction_param['size']
    w = distraction_param['image_width']
    h = distraction_param['image_height']

    if shape == 'circle':
        rad = int(w * size_per[size])
        cv2.circle(image, loc, rad, color, -1)
    elif shape == 'rectangle':
        p2 = (int(w * size_per[size]), int(h * size_per[size]))
        loc2 = tuple(map(operator.add, loc, p2))
        cv2.rectangle(image, loc, loc2, color, -1)

    return image


def apply_rolling_shape_to_image(image, distraction_param):
    if distraction_param['shape'] == '':
        return image

    return apply_static_shape_to_image(image, distraction_param)


def apply_spontaneous_shape(image, distraction_param):
    if distraction_param['shape'] == '':
        return image
    return apply_static_shape_to_image(image, distraction_param)


def get_random_loc(image_width, image_height):
    return random.randint(1, image_width), random.randint(1, image_height)


def get_random_font_thickness():
    return random.randint(1, 3)


def get_random_font_scale():
    return random.randint(1, 6)


def get_updated_loc(rolling_dir, x, y, image_width, image_height, pixel_delta=10):
    if rolling_dir == 'l_to_r':
        x_diff = pixel_delta
        y_diff = 0
    elif rolling_dir == 'r_to_l':
        x_diff = -pixel_delta
        y_diff = 0
    elif rolling_dir == 't_to_b':
        x_diff = 0
        y_diff = pixel_delta
    elif rolling_dir == 'b_to_t':
        x_diff = 0
        y_diff = -pixel_delta

    x += x_diff
    y += y_diff
    if x < 0:
        x = image_width
    if x > image_width:
        x = 0
    if y < 0:
        y = image_height
    if y > image_height:
        y = 0

    return x, y


def get_random_spontaneous_rate():
    return random.randint(10, 50)


def prepare_distraction_param(distraction, distraction_param, frame_num, res):
    # print(distraction)
    # print(distraction_param)

    image_width = distraction_param['image_width']
    image_height = distraction_param['image_height']

    if distraction == 'static_text':
        # Set random values at first frame
        if frame_num == 0:
            if distraction_param['text'] == random_setting:
                distraction_param['text'] = get_random_alphanumeric_string()
            if distraction_param['loc'] == random_setting:
                distraction_param['loc'] = get_random_loc(image_width, image_height)
            if distraction_param['color'] == random_setting:
                distraction_param['color'] = random.choice(get_supported_colors())
            if distraction_param['thickness'] == random_setting:
                distraction_param['thickness'] = get_random_font_thickness()
            if distraction_param['fontScale'] == random_setting:
                distraction_param['fontScale'] = get_random_font_scale()

    if distraction == 'rolling_text':
        if frame_num == 0:
            if distraction_param['text'] == random_setting:
                distraction_param['text'] = get_random_alphanumeric_string()
                if distraction_param['loc'] == random_setting:
                    distraction_param['loc'] = get_random_loc(image_width, image_height)
                if distraction_param['color'] == random_setting:
                    distraction_param['color'] = random.choice(get_supported_colors())
                if distraction_param['thickness'] == random_setting:
                    distraction_param['thickness'] = get_random_font_thickness()
                if distraction_param['fontScale'] == random_setting:
                    distraction_param['fontScale'] = get_random_font_scale()
                if distraction_param['rolling_dir'] == random_setting:
                    distraction_param['rolling_dir'] = random.choice(get_supported_rolling_dir())
                if distraction_param['rolling_dir'] not in get_supported_rolling_dir():
                    raise Exception("Unsupported rolling_dir given for rolling text distractor")
        else:
            # get new loc for each frames
            rolling_dir = distraction_param['rolling_dir']
            x, y = distraction_param['loc']
            x, y = get_updated_loc(rolling_dir, x, y, image_width, image_height)
            distraction_param['loc'] = (x, y)

    if distraction == 'static_shape':
        # Set random values at first frame
        if frame_num == 0:
            if distraction_param['shape'] == random_setting:
                distraction_param['shape'] = random.choice(get_supported_shapes())
            if distraction_param['loc'] == random_setting:
                distraction_param['loc'] = get_random_loc(image_width, image_height)
            if distraction_param['color'] == random_setting:
                distraction_param['color'] = random.choice(get_supported_colors())
            if distraction_param['size'] == random_setting:
                distraction_param['size'] = random.choice(get_supported_shape_sizes())

    if distraction == 'rolling_shape':
        # Set random values at first frame
        if frame_num == 0:
            if distraction_param['shape'] == random_setting:
                distraction_param['shape'] = random.choice(get_supported_shapes())
            if distraction_param['loc'] == random_setting:
                distraction_param['loc'] = get_random_loc(image_width, image_height)
            if distraction_param['color'] == random_setting:
                distraction_param['color'] = random.choice(get_supported_colors())
            if distraction_param['size'] == random_setting:
                distraction_param['size'] = random.choice(get_supported_shape_sizes())
            if distraction_param['rolling_dir'] == random_setting:
                distraction_param['rolling_dir'] = random.choice(get_supported_rolling_dir())
        else:
            rolling_dir = distraction_param['rolling_dir']
            x, y = distraction_param['loc']
            x, y = get_updated_loc(rolling_dir, x, y, image_width, image_height)
            distraction_param['loc'] = (x, y)

    if distraction == 'spontaneous_text':
        if frame_num == 0:
            if distraction_param['rate'] == random_setting:
                distraction_param['rate'] = get_random_spontaneous_rate()
        if random.random() <= distraction_param['rate']:
            distraction_param['text'] = get_random_alphanumeric_string()
            distraction_param['loc'] = get_random_loc(image_width, image_height)
            distraction_param['color'] = random.choice(get_supported_colors())
            distraction_param['thickness'] = get_random_font_thickness()
            distraction_param['fontScale'] = get_random_font_scale()
        else:
            distraction_param['text'] = ''

    if distraction == 'spontaneous_shape':
        if frame_num == 0:
            if distraction_param['rate'] == random_setting:
                distraction_param['rate'] = get_random_spontaneous_rate()
        if random.random() <= distraction_param['rate']:
            distraction_param['shape'] = random.choice(get_supported_shapes())
            distraction_param['loc'] = get_random_loc(image_width, image_height)
            distraction_param['color'] = random.choice(get_supported_colors())
            distraction_param['size'] = random.choice(get_supported_shape_sizes())
        else:
            distraction_param['shape'] = ''

    # print(distraction_param)
    return distraction_param


distraction_mapping = {
    'static_text': apply_static_text_to_image,
    'rolling_text': apply_rolling_text_to_image,
    'spontaneous_text': apply_spontaneous_text,
    'static_shape': apply_static_shape_to_image,
    'rolling_shape': apply_rolling_shape_to_image,
    'spontaneous_shape': apply_spontaneous_shape,
}


def get_supported_distraction_methods():
    return list(distraction_mapping.keys())


def get_random_static_text_param():
    return {
        'text': random_setting,
        'loc': random_setting,
        'color': random_setting,
        'fontScale': random_setting,
        'thickness': random_setting
    }


def get_random_rolling_text_param():
    return {
        'text': random_setting,
        'loc': random_setting,
        'color': random_setting,
        'fontScale': random_setting,
        'thickness': random_setting,
        'rolling_dir': random_setting
    }


def get_random_spontaneous_text_param():
    return {
        'text': random_setting,
        'loc': random_setting,
        'color': random_setting,
        'fontScale': random_setting,
        'thickness': random_setting,
        'rate': random_setting
    }


def get_random_static_shape_param():
    return {
        'shape': random_setting,
        'loc': random_setting,
        'color': random_setting,
        'size': random_setting
    }


def get_random_rolling_shape_param():
    return {
        'shape': random_setting,
        'loc': random_setting,
        'color': random_setting,
        'size': random_setting,
        'rolling_dir': random_setting
    }


def get_random_spontaneous_shape_param():
    return {
        'shape': random_setting,
        'loc': random_setting,
        'color': random_setting,
        'size': random_setting,
        'rate': random_setting
    }


def get_distractor_setting_by_type(type=None):
    if type == 'static_text':
        return get_random_static_text_param()
    if type == 'rolling_text':
        return get_random_rolling_text_param()
    if type == 'spontaneous_text':
        return get_random_spontaneous_text_param()
    if type == 'static_shape':
        return get_random_static_shape_param()
    if type == 'rolling_shape':
        return get_random_rolling_shape_param()
    if type == 'spontaneous_shape':
        return get_random_spontaneous_shape_param()

    raise Exception("Unknown type of distractor given")


def get_random_distractor():
    distractor_type = random.choice(get_supported_distraction_methods())
    return distractor_type, get_distractor_setting_by_type(distractor_type)


def apply_distraction_to_videofile(input_video_filename, output_video_filename, distraction=None,
                                   distraction_param=None, save_intermdt_files=False, test_mode=False):
    # t = time.time()

    if distraction in get_supported_distraction_methods():
        distraction_func = distraction_mapping[distraction]
    else:
        raise Exception("Unknown distraction supplied")

    capture = cv2.VideoCapture(input_video_filename)
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    org_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    org_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    res = (org_width, org_height)
    org_fps = int(capture.get(cv2.CAP_PROP_FPS))

    if distraction_param is None:
        distraction_param = dict()
    distraction_param['image_width'] = res[0]
    distraction_param['image_height'] = res[1]

    out_images_path = os.path.join(os.path.dirname(output_video_filename),
                                   os.path.splitext(os.path.basename(output_video_filename))[0],
                                   )
    os.makedirs(os.path.dirname(output_video_filename), exist_ok=True)
    if save_intermdt_files:
        os.makedirs(out_images_path, exist_ok=True)

    frames = list()
    if not test_mode:
        for i in range(frames_num):
            capture.grab()
            success, frame = capture.retrieve()
            if not success:
                continue

            distraction_param = prepare_distraction_param(distraction, distraction_param, i, res)
            frame = distraction_func(image=frame, distraction_param=distraction_param)

            if save_intermdt_files:
                out_image_name = os.path.join(out_images_path, "{}.jpg".format(i))
                # print(f'saving {out_image_name}')
                cv2.imwrite(out_image_name, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])

            frames.append(frame)

        create_video_from_images(frames, output_video_filename, fps=org_fps, res=res)
    # print('Done in', (time.time() - t))
    # print(output_video_filename)
    distraction_param['input_file'] = input_video_filename
    distraction_param['out_file'] = output_video_filename
    distraction_param['distraction'] = distraction
    return distraction_param
