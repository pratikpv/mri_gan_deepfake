import argparse
from data_utils.face_mri import *
from data_utils.face_detection import *
from deep_fake_detect.features import *
import data_utils.augmentation as augmentation
import data_utils.distractions as distractions
import pickle
import traceback
import sys


def test_data_augmentation(input_file, output_folder):
    """
    Test all kinds of data augmentation. Applies supported augmentation to input_file
    and save individual output videos in output_folder
    :param input_file:
    :param output_folder:
    :return:
    """
    os.makedirs(output_folder, exist_ok=True)

    augmentation_methods = augmentation.get_supported_augmentation_methods()
    if 'noise' in augmentation_methods:
        augmentation_methods.remove('noise')
    augmentation_param = [augmentation.get_augmentation_setting_by_type(m) for m in augmentation_methods]
    noise_methods = augmentation.get_supported_noise_types()
    noise_methods_param = [augmentation.get_noise_param_setting(m) for m in noise_methods]
    augmentation_methods.extend(['noise'] * len(noise_methods))
    augmentation_param.extend(noise_methods_param)

    augmentation_plan = list(zip(augmentation_methods, augmentation_param))

    out_id = os.path.splitext(os.path.basename(input_file))[0]
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        jobs = []
        results = []
        for aug in augmentation_plan:
            aug_func, aug_param = aug
            if aug_func == 'noise':
                suffix = out_id + '_' + aug_func + '_' + aug_param['noise_type']
            else:
                suffix = out_id + '_' + aug_func
            outfile = os.path.join(output_folder, suffix + '.mp4')
            jobs.append(pool.apply_async(augmentation.apply_augmentation_to_videofile,
                                         (input_file, outfile,),
                                         dict(augmentation=aug_func, augmentation_param=aug_param,
                                              save_intermdt_files=True)
                                         )
                        )

        for job in tqdm(jobs, desc="Applying augmentation"):
            results.append(job.get())


def test_data_distraction(input_file, output_folder):
    """
    Test all kinds of data distraction. Applies supported distraction to input_file
    and save individual output videos in output_folder
    :param input_file:
    :param output_folder:
    :return:
    """
    os.makedirs(output_folder, exist_ok=True)

    distraction_methods = distractions.get_supported_distraction_methods()
    distraction_params = [distractions.get_distractor_setting_by_type(m) for m in distraction_methods]
    distraction_plan = list(zip(distraction_methods, distraction_params))
    out_id = os.path.splitext(os.path.basename(input_file))[0]

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        jobs = []
        results = []
        for distract in distraction_plan:
            distract_func, distract_param = distract
            outfile = os.path.join(output_folder, out_id + '_' + distract_func + '.mp4')
            jobs.append(pool.apply_async(distractions.apply_distraction_to_videofile,
                                         (input_file, outfile,),
                                         dict(distraction=distract_func, distraction_param=distract_param,
                                              save_intermdt_files=True)
                                         ))

        for job in tqdm(jobs, desc="Applying distraction"):
            results.append(job.get())

    return results


def generate_data_augmentation_plan(root_dir=None, plan_pkl_file=None):
    # Apply various kinds of data augmentation to 30 % of whole training set.
    # Sample without replacement in this case and each below case.
    # Form these randomly selected video files,
    #
    # apply distractions to 35% of files
    # distractions and random noise to 35%
    # distractions, random noise, and augmentation to 15%
    # noise to 5%
    # augmentation and noise to 5%
    # augmentation to 5%
    #

    results = get_dfdc_training_video_filepaths(root_dir=root_dir)
    polulation_size = len(results)
    random.shuffle(results)

    samples_size = int(polulation_size * 0.30)
    distr_samples_size = int(samples_size * 0.35)
    dist_noise_sample_size = int(samples_size * 0.35)
    dist_noise_aug_size = int(samples_size * 0.15)
    noise_sample_size = int(samples_size * 0.05)
    aug_noise_sample_size = int(samples_size * 0.05)
    aug_sample_size = int(samples_size * 0.05)

    print(f'Total data count {polulation_size}')
    print(f'Total samples count {samples_size}')

    samples = random.sample(results, samples_size)

    distr_samples = random.sample(samples, distr_samples_size)
    samples = list(filter(lambda i: i not in distr_samples, samples))

    dist_noise_samples = random.sample(samples, dist_noise_sample_size)
    samples = list(filter(lambda i: i not in dist_noise_samples, samples))

    dist_noise_aug_samples = random.sample(samples, dist_noise_aug_size)
    samples = list(filter(lambda i: i not in dist_noise_aug_samples, samples))

    noise_samples = random.sample(samples, noise_sample_size)
    samples = list(filter(lambda i: i not in noise_samples, samples))

    aug_noise_samples = random.sample(samples, aug_noise_sample_size)
    samples = list(filter(lambda i: i not in aug_noise_samples, samples))

    aug_samples = random.sample(samples, aug_sample_size)

    assert len(distr_samples) == distr_samples_size
    assert len(dist_noise_samples) == dist_noise_sample_size
    assert len(dist_noise_aug_samples) == dist_noise_aug_size
    assert len(noise_samples) == noise_sample_size
    assert len(aug_noise_samples) == aug_noise_sample_size
    assert len(aug_samples) == aug_sample_size

    distr_samples_exec_plan = []
    for i in distr_samples:
        plan = list()
        plan.append({'distraction': distractions.get_random_distractor()})
        entry = i, plan
        distr_samples_exec_plan.append(entry)
    # pprint(distr_samples_exec_plan)

    dist_noise_samples_exec_plan = []
    for i in dist_noise_samples:
        plan = list()
        plan.append({'distraction': distractions.get_random_distractor()})
        noise_type = augmentation.get_random_noise_type()
        noise_param = augmentation.get_noise_param_setting(noise_type)
        plan.append({'augmentation': ('noise', noise_param)})
        entry = i, plan
        dist_noise_samples_exec_plan.append(entry)
    # pprint(dist_noise_samples_exec_plan)

    dist_noise_aug_exec_plan = []
    for i in dist_noise_aug_samples:
        plan = list()
        plan.append({'distraction': distractions.get_random_distractor()})
        noise_type = augmentation.get_random_noise_type()
        noise_param = augmentation.get_noise_param_setting(noise_type)
        plan.append({'augmentation': ('noise', noise_param)})
        plan.append({'augmentation': augmentation.get_random_augmentation(avoid_noise=True)})
        entry = i, plan
        dist_noise_aug_exec_plan.append(entry)
    # pprint(dist_noise_aug_exec_plan)

    noise_samples_exe_plan = []
    for i in noise_samples:
        plan = list()
        noise_type = augmentation.get_random_noise_type()
        noise_param = augmentation.get_noise_param_setting(noise_type)
        plan.append({'augmentation': ('noise', noise_param)})
        entry = i, plan
        noise_samples_exe_plan.append(entry)
    # pprint(noise_samples_exe_plan)

    aug_noise_samples_exec_plan = []
    for i in aug_noise_samples:
        plan = list()
        plan.append({'augmentation': augmentation.get_random_augmentation(avoid_noise=True)})
        noise_type = augmentation.get_random_noise_type()
        noise_param = augmentation.get_noise_param_setting(noise_type)
        plan.append({'augmentation': ('noise', noise_param)})
        entry = i, plan
        aug_noise_samples_exec_plan.append(entry)
    # pprint(aug_noise_samples_exec_plan)

    aug_samples_exec_plan = []
    for i in aug_samples:
        plan = list()
        plan.append({'augmentation': augmentation.get_random_augmentation(avoid_noise=True)})
        entry = i, plan
        aug_samples_exec_plan.append(entry)
    # pprint(aug_samples_exec_plan)

    data_augmentation_plan = list()
    data_augmentation_plan.extend(distr_samples_exec_plan)
    data_augmentation_plan.extend(dist_noise_samples_exec_plan)
    data_augmentation_plan.extend(dist_noise_aug_exec_plan)
    data_augmentation_plan.extend(noise_samples_exe_plan)
    data_augmentation_plan.extend(aug_noise_samples_exec_plan)
    data_augmentation_plan.extend(aug_samples_exec_plan)

    print(f'Saving plan to {plan_pkl_file}')
    with open(plan_pkl_file, 'wb') as f:
        pickle.dump(data_augmentation_plan, f)

    return data_augmentation_plan


def check_plan_already_executed(filename, plan, metadata_folder):
    id = os.path.basename(filename)
    file_search = metadata_folder + '/{}*'.format(id)
    pe_files = glob(file_search)
    for f in pe_files:
        df = pd.read_csv(f, index_col=0)
        types = ['distraction', 'augmentation']
        for t in types:
            if t in plan.keys() and t in df.index:
                if plan[t][0] in df.loc[t].values:
                    return True

    return False


def execute_data_augmentation_plan(data_augmentation_plan_filename, metadata_folder):
    max_num_plans = 4
    with open(data_augmentation_plan_filename, 'rb') as f:
        data_augmentation_plan = pickle.load(f)

    os.makedirs(metadata_folder, exist_ok=True)

    # i wish to apply plan in sequence for each file, but if the file has multiple plans
    # later plans may get started before earlier one finishes.
    # to workaround this, execute plan[0] for each file, then plan[1] for each file
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        jobs = []
        results = []
        for plan_id in range(max_num_plans):
            desc = "Queuing Jobs for id {}".format(plan_id)
            for filename, plan in tqdm(data_augmentation_plan, desc=desc):
                if len(plan) > plan_id:
                    p = plan[plan_id]
                    if check_plan_already_executed(filename, p, metadata_folder):
                        continue
                    if 'augmentation' in p.keys():
                        distr = p['augmentation']
                        jobs.append(pool.apply_async(augmentation.apply_augmentation_to_videofile,
                                                     (filename, filename,),
                                                     dict(augmentation=distr[0], augmentation_param=distr[1])
                                                     )
                                    )
                    elif 'distraction' in p.keys():
                        distr = p['distraction']
                        jobs.append(pool.apply_async(distractions.apply_distraction_to_videofile,
                                                     (filename, filename,),
                                                     dict(distraction=distr[0], distraction_param=distr[1])
                                                     ))

        for job in tqdm(jobs, desc="Executing data augmentation plan"):
            r = job.get()
            try:
                df = pd.Series(r).reset_index().set_index('index')
                rfilename = os.path.basename(r['input_file']) + \
                            '_' + str(datetime.now().strftime("%d-%b-%Y_%H_%M_%S")) + '.csv'
                df.to_csv(os.path.join(metadata_folder, rfilename), header=False)
            except Exception as e:
                print_line()
                print('Got exception')
                print_line()
                print(r)
                print_line()
                print(traceback.print_exc())
                print_line()
                print(sys.exc_info()[0])
                print_line()
            results.append(r)

    return results


def main():
    if args.apply_aug_to_sample:
        # sample_test_file = os.path.join('dfdc_train_part_30', 'ajxcpxpmof.mp4')
        # sample_test_file = os.path.join(ConfigParser.getInstance().get_dfdc_train_data_path(), sample_test_file)
        sample_test_file = '4000.mp4'
        print(f'Applying augmentation and distraction to sample file {sample_test_file}')
        out_root = ConfigParser.getInstance().get_assets_path()
        aug_output_folder = os.path.join(out_root, 'sample_augmentation')
        print(f'aug_output_folder: {aug_output_folder}')
        print(f'Sample file: {sample_test_file}')

        os.makedirs(aug_output_folder, exist_ok=True)
        shutil.copy(sample_test_file, os.path.join(aug_output_folder, os.path.basename(sample_test_file)))

        test_data_augmentation(sample_test_file, aug_output_folder)
        test_data_distraction(sample_test_file, aug_output_folder)

    if args.gen_aug_plan:
        print('Generating augmentation plan')
        generate_data_augmentation_plan(ConfigParser.getInstance().get_dfdc_train_data_path(),
                                        ConfigParser.getInstance().get_data_aug_plan_pkl_filename())

    if args.apply_aug_to_all:
        print('Executing augmentation plan')
        execute_data_augmentation_plan(ConfigParser.getInstance().get_data_aug_plan_pkl_filename(),
                                       ConfigParser.getInstance().get_aug_metadata_path())

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

    parser.add_argument('--apply_aug_to_sample', action='store_true',
                        help='Apply augmentation and distractions to a file',
                        default=False)
    parser.add_argument('--gen_aug_plan', action='store_true',
                        help='Generate augmentation and distractions plan',
                        default=False)
    parser.add_argument('--apply_aug_to_all', action='store_true',
                        help='Apply augmentation and distractions to all samples, per plan',
                        default=False)
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
