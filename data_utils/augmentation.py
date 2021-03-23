from functools import partial
from facenet_pytorch.models.mtcnn import MTCNN
import cv2
from PIL import Image
import os
import numpy as np
from skimage.util import random_noise
from skimage import exposure
import time
import torch
from data_utils.utils import *
import random

random_setting = '$RANDOM$'


def apply_blur_to_image(image, augmentation_param):
    return cv2.blur(image, augmentation_param['ksize'])


def apply_noise_to_image(image=None, augmentation_param=None):
    mode = augmentation_param['noise_type']
    image = random_noise(image, mode=mode)
    image = np.array(255 * image, dtype=np.uint8)
    return image


def apply_contrast_to_image(image, augmentation_param):
    contrast = augmentation_param['contrast_value']
    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)
    return image


def apply_brightness_to_image(image, augmentation_param):
    brightness = augmentation_param['brightness_value']
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
    return image


def apply_graysclae_to_image(image, augmentation_param):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def get_supported_augmentation_methods():
    return list(augmentation_mapping.keys())


def get_random_blur_value():
    val = random.randint(7, 12)
    return val, val


def get_random_contrast_value():
    return random.randint(-20, 20)


def get_random_brightness_value():
    return random.randint(-20, 20)


def get_random_angle_value():
    return random.randint(-15, 15)


def get_supported_res_value():
    return [(800, 600), (1280, 720)]


def get_random_res_value():
    return random.choice(get_supported_res_value())


def apply_rotation_to_image(image, augmentation_param):
    angle = augmentation_param['angle']
    row, col, c = np.asarray(image).shape
    center = tuple(np.array([row, col]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col, row))
    return new_image


def apply_flip_horizontal(image, augmentation_param):
    return cv2.flip(image, 1)


def apply_rescale_to_image(image, augmentation_param):
    res = augmentation_param['res']
    return cv2.resize(image, res, interpolation=cv2.INTER_AREA)


augmentation_mapping = {
    'noise': apply_noise_to_image,
    'blur': apply_blur_to_image,
    'contrast': apply_contrast_to_image,
    'brightness': apply_brightness_to_image,
    'rotation': apply_rotation_to_image,
    'flip_horizontal': apply_flip_horizontal,
    'rescale': apply_rescale_to_image,
}


def get_supported_noise_types():
    """
    :return: All supported noise types
    """
    return ['gaussian', 'speckle', 's&p', 'pepper', 'salt', 'poisson', 'localvar']


def get_random_noise_type():
    """

    :return: get random noise type from supported noise types
    """
    return random.choice(get_supported_noise_types())


def get_random_noise_setting():
    return {'noise_type': random_setting}


def get_noise_param_setting(type):
    return {'noise_type': type}


def get_random_blur_setting():
    return {'ksize': random_setting}


def get_random_contrast_setting():
    return {'contrast_value': random_setting}


def get_random_brightness_setting():
    return {'brightness_value': random_setting}


def get_random_rotation_setting():
    return {'angle': random_setting}


def get_random_rescale_setting():
    return {'res': random_setting}


def get_augmentation_setting_by_type(type=None):
    """
    Generate random augmentation param for the augmentation type given
    :param type: augmentation type
    :return: random augmentation param for the augmentation passed
    """
    noise_list = get_supported_noise_types()
    if type in noise_list:
        return None
    if type == 'noise':
        return get_random_noise_setting()
    if type == 'blur':
        return get_random_blur_setting()
    if type == 'contrast':
        return get_random_contrast_setting()
    if type == 'brightness':
        return get_random_brightness_setting()
    if type == 'rotation':
        return get_random_rotation_setting()
    if type == 'flip_horizontal':
        return None
    if type == 'rescale':
        return get_random_rescale_setting()

    raise Exception("Unknown type of augmentation given")


def get_random_augmentation(avoid_noise=False):
    """
    Get a random augmentation method and its parameter

    :param avoid_noise: mode 'noise' method if this is set to True
    :return: augmentation and augmentation_params
    """

    supported_augmentation_methods = get_supported_augmentation_methods()
    if avoid_noise:
        supported_augmentation_methods.remove('noise')
    augmentation_type = random.choice(supported_augmentation_methods)
    return augmentation_type, get_augmentation_setting_by_type(augmentation_type)


def prepare_augmentation_param(augmentation, augmentation_param, frame_num, res):
    """
    prepare augmentation params before applying the augmentation
    if 'random_setting' is set generate actual random values

    some random values are set for frame_num 0 only.

    :param augmentation: the method of augmentation
    :param augmentation_param: params
    :param frame_num: update param based on the frame num
    :param res: resolution of the frame
    :return: updated augmentation_param
    """
    if augmentation == 'noise':
        if frame_num == 0:
            if augmentation_param['noise_type'] == random_setting:
                augmentation_param['noise_type'] = get_random_noise_type()

    if augmentation == 'blur':
        if frame_num == 0:
            if augmentation_param['ksize'] == random_setting:
                augmentation_param['ksize'] = get_random_blur_value()

    if augmentation == 'contrast':
        if frame_num == 0:
            if augmentation_param['contrast_value'] == random_setting:
                augmentation_param['contrast_value'] = get_random_contrast_value()

    if augmentation == 'brightness':
        if frame_num == 0:
            if augmentation_param['brightness_value'] == random_setting:
                augmentation_param['brightness_value'] = get_random_contrast_value()

    if augmentation == 'rotation':
        if frame_num == 0:
            if augmentation_param['angle'] == random_setting:
                augmentation_param['angle'] = get_random_angle_value()

    if augmentation == 'rescale':
        if frame_num == 0:
            if augmentation_param['res'] == random_setting:
                augmentation_param['res'] = get_random_res_value()

    return augmentation_param


def apply_augmentation_to_videofile(input_video_filename, output_video_filename, augmentation=None,
                                    augmentation_param=None, save_intermdt_files=False, test_mode=False):
    """
    This is main driver API to apply augmentation to the input video
    :param input_video_filename: input file
    :param output_video_filename: output file
    :param augmentation: method of augmentation
    :param augmentation_param: params
    :param save_intermdt_files: save each frames as image
    :param test_mode: Dont process video, just for testing
    :return: updated augmentation_param for logging
    """
    # t = time.time()
    list_of_aug = get_supported_augmentation_methods()
    list_of_aug.extend(get_supported_noise_types())
    if augmentation in list_of_aug:
        augmentation_func = augmentation_mapping[augmentation]
    else:
        raise Exception("Unknown augmentation supplied")

    capture = cv2.VideoCapture(input_video_filename)
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    org_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    org_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    res = (org_width, org_height)
    org_fps = int(capture.get(cv2.CAP_PROP_FPS))

    if augmentation_param is None:
        augmentation_param = dict()
    augmentation_param['image_width'] = res[0]
    augmentation_param['image_height'] = res[1]

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
            augmentation_param = prepare_augmentation_param(augmentation, augmentation_param, i, res)
            if augmentation == 'rescale':
                res = augmentation_param['res']
            frame = augmentation_func(image=frame, augmentation_param=augmentation_param)

            if save_intermdt_files:
                out_image_name = os.path.join(out_images_path, "{}.jpg".format(i))
                # print(f'saving {out_image_name}')
                cv2.imwrite(out_image_name, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])

            frames.append(frame)

        create_video_from_images(frames, output_video_filename, fps=org_fps, res=res)
    # print('Done in', (time.time() - t))
    # print(output_video_filename)
    augmentation_param['input_file'] = input_video_filename
    augmentation_param['out_file'] = output_video_filename
    augmentation_param['augmentation'] = augmentation
    return augmentation_param
