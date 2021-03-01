import multiprocessing

from tqdm import tqdm

import data_utils.face_detection as fd
import json
import numpy as np
import os
from PIL import Image
import cv2
from scipy.ndimage import binary_erosion, binary_dilation
import math
import dlib
from skimage.metrics import structural_similarity
from scipy.spatial import ConvexHull
from skimage import measure
import skimage.draw
import random
from glob import glob
from imutils import face_utils
from utils import *
import pandas as pd
from sklearn.model_selection import train_test_split
from data_utils.utils import *


def get_structural_similarity(image1, image2):
    return structural_similarity(image1, image2, multichannel=True, full=True,
                                 gaussian_weights=True, use_sample_covariance=False, sigma=1.5)


def gen_mri(image1_path, image2_path, mri_path=None, res=(256, 256)):
    image1 = cv2.imread(image1_path, cv2.IMREAD_COLOR)
    image1 = cv2.resize(image1, res, interpolation=cv2.INTER_AREA)
    image2 = cv2.imread(image2_path, cv2.IMREAD_COLOR)
    image2 = cv2.resize(image2, res, interpolation=cv2.INTER_AREA)

    sim_index, sim = get_structural_similarity(image1, image2)

    mri = 1 - sim
    mri = (mri * 255).astype(np.uint8)
    if mri_path is not None:
        cv2.imwrite(mri_path, mri)
    return sim_index


def gen_face_mri_per_folder(real_dir=None, fake_dir=None, mri_basedir=None, overwrite=False):
    dest_folder = os.path.join(mri_basedir, os.path.basename(fake_dir))
    if not overwrite and os.path.isdir(dest_folder):
        return None

    r_all_files = glob(real_dir + "/*")
    os.makedirs(dest_folder, exist_ok=True)
    df = pd.DataFrame(columns=['real_image', 'fake_image', 'mri_image'])
    for r_file in r_all_files:
        r_file_base = os.path.basename(r_file)
        f_file = os.path.join(fake_dir, r_file_base)
        if os.path.isfile(f_file):
            mri_path = os.path.join(dest_folder, r_file_base)
            gen_mri(r_file, f_file, mri_path)
            item = {'real_image': r_file,
                    'fake_image': f_file,
                    'mri_image': mri_path}
            df = df.append(item, ignore_index=True)

    return df


def generate_MRI_dataset_from_dfdc(overwrite=True):
    metadata_csv_file = ConfigParser.getInstance().get_dfdc_mri_metadata_csv_path()

    if not overwrite and os.path.isfile(metadata_csv_file):
        return pd.read_csv(metadata_csv_file)

    pairs = get_dfdc_training_real_fake_pairs(ConfigParser.getInstance().get_dfdc_train_data_path())
    pairs_len = len(pairs)

    mri_basedir = ConfigParser.getInstance().get_dfdc_mri_path()
    crops_path = ConfigParser.getInstance().get_dfdc_crops_train_path()
    results = []
    df = pd.DataFrame(columns=['real_image', 'fake_image', 'mri_image'])
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        jobs = []
        for pid in tqdm(range(pairs_len), desc="Scheduling jobs"):
            item = pairs[pid]
            reals = os.path.join(crops_path, item[0])
            fakes = os.path.join(crops_path, item[1])
            jobs.append(pool.apply_async(gen_face_mri_per_folder, (reals, fakes, mri_basedir,)))

        for job in tqdm(jobs, desc="Generating MRIs for DFDC training dataset"):
            results.append(job.get())

    for r in tqdm(results, desc='Consolidating results'):
        if r is not None:
            df = df.append(r, ignore_index=True)

    df.to_csv(metadata_csv_file)

    return df


def generate_celeb_df_v2_real_fake_mapping():
    print('Generating real-fake mapping')
    real_path = ConfigParser.getInstance().get_celeb_df_v2_real_path()
    fake_path = ConfigParser.getInstance().get_celeb_df_v2_fake_path()

    real_all_files = glob(real_path + "/*")
    real_fake_dict = dict()
    out_file = ConfigParser.getInstance().get_celeb_df_v2_real_fake_mapping_json()
    for real_f in real_all_files:
        a = os.path.splitext(os.path.basename(real_f))[0].split('_')
        search_str = '/' + a[0] + '_*_' + a[1] + '*'
        fake_f = glob(fake_path + search_str)
        real_f = os.path.basename(real_f)
        fake_f = list(map(os.path.basename, fake_f))
        real_fake_dict[real_f] = fake_f

    with open(out_file, "w") as f:
        json.dump(real_fake_dict, f)


def generate_celeb_df_v2_real_fake_comb(overwrite=True):
    celeb_df_v2_metadata = ConfigParser.getInstance().get_celeb_df_v2_real_fake_mapping_json()
    if overwrite or not os.path.isfile(celeb_df_v2_metadata):
        generate_celeb_df_v2_real_fake_mapping()

    with open(celeb_df_v2_metadata, 'r') as jf:
        metadata = json.load(jf)

    mkeys = metadata.keys()

    rf_comb = list()
    for mk in mkeys:
        values = metadata[mk]
        mk = os.path.splitext(mk)[0]
        for v in values:
            v = os.path.splitext(v)[0]
            rf_comb.append((mk, v))

    return rf_comb


def generate_MRI_dataset_from_celeb_df_v2(overwrite=True):
    metadata_csv_file = ConfigParser.getInstance().get_celeb_df_v2_mri_metadata_csv_path()

    if not overwrite and os.path.isfile(metadata_csv_file):
        return pd.read_csv(metadata_csv_file)

    real_fake_comb = generate_celeb_df_v2_real_fake_comb(overwrite=overwrite)
    comb_len = len(real_fake_comb)
    print(f'Generated real_fake_comb of len {comb_len}')
    crops_path = ConfigParser.getInstance().get_celeb_df_v2_crops_train_path()
    mri_basedir = ConfigParser.getInstance().get_celeb_df_v2_mri_path()

    results = []
    df = pd.DataFrame(columns=['real_image', 'fake_image', 'mri_image'])

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        jobs = []
        for pid in tqdm(range(comb_len), desc="Scheduling jobs"):
            item = real_fake_comb[pid]
            reals = os.path.join(crops_path, item[0])
            fakes = os.path.join(crops_path, item[1])
            jobs.append(pool.apply_async(gen_face_mri_per_folder, (reals, fakes, mri_basedir,)))

        for job in tqdm(jobs, desc="Generating MRIs for Celeb-df-V2 dataset"):
            results.append(job.get())

    for r in tqdm(results, desc='Consolidating results'):
        if r is not None:
            df = df.append(r, ignore_index=True)

    df.to_csv(metadata_csv_file)

    return df


def generate_MRI_dataset(test_size=0.2, dfdc_fract=0.5):
    print(f'\t generating DFDC MRI dataset')
    dfdc_df = generate_MRI_dataset_from_dfdc(overwrite=False)
    # take dfdc_fract % of samples from DFDC dataset to train the MRI GAN.
    # print(f'before trunc len dfdc_df = {len(dfdc_df)}')
    if dfdc_fract < 1.0:
        dfdc_df, _ = train_test_split(dfdc_df, train_size=dfdc_fract)
    # print(f'after trunc len dfdc_df = {len(dfdc_df)}')

    print(f'\t generating celeb-df-v2 MRI dataset')
    celeb_v2_df = generate_MRI_dataset_from_celeb_df_v2(overwrite=False)
    df_combined = dfdc_df.append(celeb_v2_df, ignore_index=True)

    # convert ['real_image', 'fake_image', 'mri_image'] to ['face_image', 'mri_image']
    # where if image is real => use blank image as mri
    #       else use the mri(real_image, fake_image)

    dff = pd.DataFrame(columns=['face_image', 'mri_image', 'class'])
    dff['face_image'] = df_combined['fake_image'].unique()
    dff['mri_image'] = df_combined['mri_image']
    dff_len = len(dff)
    dff['class'][0:dff_len] = 'fake'
    dff = dff.set_index('face_image')

    dfr_base = pd.DataFrame(columns=['face_image', 'mri_image', 'class'])
    dfr_base['face_image'] = df_combined['real_image'].unique()
    dfr_base['mri_image'] = os.path.abspath(ConfigParser.getInstance().get_blank_imagepath())
    dfr_base_len = len(dfr_base)
    dfr_base['class'][0:dfr_base_len] = 'real'

    print(f'\t generating other real MRI dataset')
    other_real_crops_paths = [ConfigParser.getInstance().get_fdf_crops_path(),
                              ConfigParser.getInstance().get_ffhq_crops_path()]

    other_real_samples = list()

    for r in other_real_crops_paths:
        other_real_samples.extend(glob(r + '/*'))

    dfr_others = pd.DataFrame(columns=['face_image', 'mri_image', 'class'])
    dfr_others['face_image'] = other_real_samples
    dfr_others['mri_image'] = os.path.abspath(ConfigParser.getInstance().get_blank_imagepath())
    dfr_others['class'] = 'real'
    dfr = pd.concat([dfr_base, dfr_others])
    dfr_len = len(dfr)
    dfr = dfr.set_index('face_image')

    real_train, real_test = train_test_split(dfr, test_size=test_size)
    real_train.to_csv(ConfigParser.getInstance().get_mri_train_real_dataset_csv_path())
    real_test.to_csv(ConfigParser.getInstance().get_mri_test_real_dataset_csv_path())

    fake_train, fake_test = train_test_split(dff, test_size=test_size)
    fake_train.to_csv(ConfigParser.getInstance().get_mri_train_fake_dataset_csv_path())
    fake_test.to_csv(ConfigParser.getInstance().get_mri_test_fake_dataset_csv_path())

    total_samples = dfr_len + dff_len

    print(f'Total Fake samples {dff_len}')
    print(f'Total Real samples {dfr_len}')
    print(f'Fake Train samples {len(fake_train)}')
    print(f'Real Train samples {len(real_train)}')
    print(f'Fake Test samples {len(fake_test)}')
    print(f'Real Test samples {len(real_test)}')
    print(
        f'Total samples {total_samples}, real={round(dfr_len / total_samples, 2)}% fake={round(dff_len / total_samples, 2)}%')
