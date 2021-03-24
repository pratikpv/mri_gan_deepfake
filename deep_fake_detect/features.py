from glob import glob
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from mri_gan.model import *
import multiprocessing
from tqdm import tqdm
import pandas as pd
from data_utils.utils import *


def predict_mri_using_MRI_GAN(crops_path, mri_path, vid, imsize, overwrite=False):
    vid_path = os.path.join(crops_path, vid)
    vid_mri_path = os.path.join(mri_path, vid)
    if not overwrite and os.path.isdir(vid_mri_path):
        return
    batch_size = 8
    mri_generator = get_MRI_GAN(pre_trained=True).cuda()
    os.makedirs(vid_mri_path, exist_ok=True)
    frame_names = glob(vid_path + '/*.png')
    num_frames_detected = len(frame_names)
    batches = list()

    transforms_ = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    for i in range(0, num_frames_detected, batch_size):
        end = i + batch_size
        if end > num_frames_detected:
            end = num_frames_detected
        batches.append(frame_names[i:end])

    for j, frame_names_b in enumerate(batches):
        frames = []
        for k, fname in enumerate(frame_names_b):
            frames.append(transforms_(Image.open(frame_names[k])))

        frames = torch.stack(frames)
        frames = frames.cuda()
        mri_images = mri_generator(frames)
        b = mri_images.shape[0]
        for l in range(b):
            save_path = os.path.join(vid_mri_path, os.path.basename(frame_names_b[l]))
            save_image(mri_images[l], save_path)


def predict_mri_using_MRI_GAN_batch(crops_path, mri_path):
    print(f'Crops dir {crops_path}')
    print(f'MRI dir {mri_path}')

    video_ids_path = glob(crops_path + "/*")
    video_ids_len = len(video_ids_path)
    imsize = 256

    keep_trying = True
    MAX_RETRIES = 0
    retry_count = MAX_RETRIES
    processes = 8
    while keep_trying:
        try:
            with multiprocessing.Pool(processes=processes) as pool:
                jobs = []
                results = []
                for vidx in tqdm(range(video_ids_len), desc="Scheduling jobs"):
                    jobs.append(pool.apply_async(predict_mri_using_MRI_GAN,
                                                 (crops_path, mri_path,
                                                  os.path.basename(video_ids_path[vidx]), imsize)
                                                 )
                                )

                for job in tqdm(jobs, desc="Generating MRI data"):
                    results.append(job.get())

                keep_trying = False
        except RuntimeError:
            if retry_count <= 0:
                processes = max(processes - 1, 1)
                print(f'Retry with lower number of processes. processes = {processes}')
                retry_count = MAX_RETRIES
            else:
                print(f'Retry with same number of processes. retry_count = {retry_count}')
                retry_count -= 1
            pass


def generate_frame_label_csv(mode=None, dataset=None):
    if mode == 'train':
        originals_, fakes_ = get_training_reals_and_fakes()
        if dataset == 'plain':
            csv_file = ConfigParser.getInstance().get_dfdc_train_frame_label_csv_path()
            crop_path = ConfigParser.getInstance().get_dfdc_crops_train_path()
        elif dataset == 'mri':
            csv_file = ConfigParser.getInstance().get_train_mriframe_label_csv_path()
            crop_path = ConfigParser.getInstance().get_train_mrip2p_png_data_path()
        else:
            raise Exception('Bad dataset')
    elif mode == 'valid':
        originals_, fakes_ = get_valid_reals_and_fakes()
        if dataset == 'plain':
            csv_file = ConfigParser.getInstance().get_dfdc_valid_frame_label_csv_path()
            crop_path = ConfigParser.getInstance().get_dfdc_crops_valid_path()
        elif dataset == 'mri':
            csv_file = ConfigParser.getInstance().get_valid_mriframe_label_csv_path()
            crop_path = ConfigParser.getInstance().get_valid_mrip2p_png_data_path()
        else:
            raise Exception('Bad dataset')

    elif mode == 'test':
        originals_, fakes_ = get_test_reals_and_fakes()
        if dataset == 'plain':
            csv_file = ConfigParser.getInstance().get_dfdc_test_frame_label_csv_path()
            crop_path = ConfigParser.getInstance().get_dfdc_crops_test_path()
        elif dataset == 'mri':
            csv_file = ConfigParser.getInstance().get_test_mriframe_label_csv_path()
            crop_path = ConfigParser.getInstance().get_test_mrip2p_png_data_path()
        else:
            raise Exception('Bad dataset')
    else:
        raise Exception('Bad mode in generate_frame_label_csv')

    originals = [os.path.splitext(video_filename)[0] for video_filename in originals_]
    fakes = [os.path.splitext(video_filename)[0] for video_filename in fakes_]

    print(f'mode {mode}, csv file : {csv_file}')
    df = pd.DataFrame(columns=['video_id', 'frame', 'label'])

    crop_ids = glob(crop_path + '/*')
    results = []
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        jobs = []
        for cid in tqdm(crop_ids, desc='Scheduling jobs to label frames'):
            jobs.append(pool.apply_async(get_video_frame_labels_mapping, (cid, originals, fakes,)))

        for job in tqdm(jobs, desc="Labeling frames"):
            r = job.get()
            results.append(r)

    for r in tqdm(results, desc='Consolidating results'):
        df = df.append(r, ignore_index=True)
    df.set_index('video_id', inplace=True)
    df.to_csv(csv_file)


def generate_frame_label_csv_files():
    modes = ['train', 'valid', 'test']
    datasets = ['plain', 'mri']
    for d in datasets:
        print(f'Generating frame_label csv for dataset {d}')
        for m in modes:
            print(f'Generating frame_label csv for processed {m} samples')
            generate_frame_label_csv(mode=m, dataset=d)


def generate_DFDC_MRIs():
    print('Generating MRIs of DFDC train using trained MRI-GAN')
    predict_mri_using_MRI_GAN_batch(ConfigParser.getInstance().get_dfdc_crops_train_path(),
                                    ConfigParser.getInstance().get_train_mrip2p_png_data_path())
    print('Generating MRIs of DFDC valid using trained MRI-GAN')
    predict_mri_using_MRI_GAN_batch(ConfigParser.getInstance().get_dfdc_crops_valid_path(),
                                    ConfigParser.getInstance().get_valid_mrip2p_png_data_path())
    print('Generating MRIs of DFDC test using trained MRI-GAN')
    predict_mri_using_MRI_GAN_batch(ConfigParser.getInstance().get_dfdc_crops_test_path(),
                                    ConfigParser.getInstance().get_test_mrip2p_png_data_path())
