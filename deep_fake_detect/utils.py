import multiprocessing

import torch
import sys
from utils import *
from data_utils.utils import *
from functools import partial
from timm.models.efficientnet import tf_efficientnet_b0_ns, tf_efficientnet_b7_ns
import torch.nn as nn
import os
import pickle
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import pandas as pd
import seaborn as sns
import numpy as np
import torch
import sys
from utils import *
from deep_fake_detect.checkpoint import *
from sklearn.metrics import roc_curve, auc, roc_auc_score
from tqdm import tqdm

encoder_params = {
    "tf_efficientnet_b0_ns": {
        "flat_features_dim": 1280,
        "imsize": 224,
        "init_op": partial(tf_efficientnet_b0_ns, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b7_ns": {
        "flat_features_dim": 2560,
        "imsize": 600,
        "init_op": partial(tf_efficientnet_b7_ns, pretrained=True, drop_path_rate=0.2)
    }
}


def get_encoder(name=None, pretrained=True):
    if name in encoder_params.keys():
        encoder = encoder_params[name]["init_op"](pretrained=pretrained)
        print(f'Returning {name}, pretrained = {pretrained}')
        return encoder
    else:
        raise Exception("Unknown encoder")


def get_encoder_params(name=None):
    if name in encoder_params.keys():
        return encoder_params[name]["flat_features_dim"], encoder_params[name]["imsize"]
    else:
        raise Exception("Unknown encoder")


class DeepFakeEncoder(nn.Module):
    def __init__(self, encoder_name='tf_efficientnet_b7_ns'):
        super().__init__()
        self.encoder = get_encoder(encoder_name)

    def forward(self, x):
        x = self.encoder.forward_features(x)
        return x


def print_batch_item(index, item, all_frames=False, simple=True):
    if simple:
        print_line()
        if item is None:
            print(f'{index} | None')
            return
        else:
            v_ids, frames, labels = item
            print(f'{index} | {v_ids} |frames={len(frames)}, shape={frames[0].shape} | {labels}')
            print_line()
            print(frames[0])
            print_line()

        print_line()


    else:
        print_line()
        print(f'index={index}')
        if item is None:
            print('None data')
            return
        v_ids, frames, labels = item
        print_line()
        print(f'v_ids={v_ids}')
        print_line()
        if all_frames:
            print(f'frames len = {len(frames)}')
            for f in frames:
                print(f'\t{f.shape}')
        else:
            print(f'frames len = {len(frames)}, shape = {frames[0].shape}')
        print_line()
        print(f'labels = {labels}')
        print_line()


def global_minibatch_number(epoch, batch_id, batch_size):
    """
    get global counter of iteration for smooth plotting
    @param epoch: epoch
    @param batch_id: the batch number
    @param batch_size: batch size
    @return: global counter of iteration
    """
    return epoch * batch_size + batch_id


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    """
    for i, b in enumerate(batch):
        print_batch_item(i, b)
    """
    batch = tuple(zip(*batch))
    return batch


def get_predictions(output):
    # return torch.argmax(output, dim=1)
    return torch.round(torch.sigmoid(output))


def get_probability(output):
    return torch.sigmoid(output)


def save_model_results_to_log(epoch=0, model=None, model_params=None, losses=None, accuracies=None, predicted=None,
                              ground_truth=None, misc_data=None, sample_names=None, log_dir=None, log_kind=None,
                              report_type=None, probabilities=None):
    log_params = ConfigParser.getInstance().get_log_params()
    model_name = model_params['model_name']
    num_of_classes = 2
    # real = 0, fake = 1
    class_names = ['Real', 'Fake']
    if log_kind:
        log_dir_kind = os.path.join(log_dir, log_kind)
    else:
        log_dir_kind = os.path.join(log_dir)
    model_log_dir = os.path.join(log_dir_kind, model_name, report_type)
    os.makedirs(model_log_dir, exist_ok=True)
    model_log_file = os.path.join(model_log_dir, log_params['model_info_log'])
    model_train_losses_log_file = os.path.join(model_log_dir, log_params['model_loss_info_log'])
    model_train_accuracy_log_file = os.path.join(model_log_dir, log_params['model_acc_info_log'])
    model_conf_mat_csv = os.path.join(model_log_dir, log_params['model_conf_matrix_csv'])
    model_conf_mat_png = os.path.join(model_log_dir, log_params['model_conf_matrix_png'])
    model_conf_mat_normalized_csv = os.path.join(model_log_dir, log_params['model_conf_matrix_normalized_csv'])
    model_conf_mat_normalized_png = os.path.join(model_log_dir, log_params['model_conf_matrix_normalized_png'])
    model_loss_png = os.path.join(model_log_dir, log_params['model_loss_png'])
    model_accuracy_png = os.path.join(model_log_dir, log_params['model_accuracy_png'])
    all_samples_pred_csv = os.path.join(model_log_dir, log_params['all_samples_pred_csv'])
    model_roc_png = os.path.join(model_log_dir, log_params['model_roc_png'])

    report = None
    if predicted is not None:
        df = pd.DataFrame([sample_names, ground_truth, predicted, probabilities]).T
        df.columns = ['sample_name', 'ground_truth', 'predictions', 'probability']
        df = df.set_index(['sample_name'])
        df.to_csv(all_samples_pred_csv)

        # generate and save confusion matrix
        plot_x_label = "Predictions"
        plot_y_label = "Actual"
        cmap = plt.cm.Blues
        # pred_class_indexes = sorted(np.unique(predicted))
        # pred_num_classes = len(pred_class_indexes)
        # target_class_names = [class_names[int(i)] for i in pred_class_indexes]

        cm = metrics.confusion_matrix(ground_truth, predicted)
        target_class_names = class_names  # TODO

        df_confusion = pd.DataFrame(cm)
        df_confusion.index = target_class_names
        df_confusion.columns = target_class_names
        df_confusion = df_confusion.round(4)
        df_confusion.to_csv(model_conf_mat_csv)
        fig = plt.figure(figsize=(5, 5))
        sns.heatmap(df_confusion, annot=True, cmap=cmap)
        plt.xlabel(plot_x_label)
        plt.ylabel(plot_y_label)
        plt.title('Confusion Matrix')
        plt.savefig(model_conf_mat_png)
        plt.close(fig)

        cm = metrics.confusion_matrix(ground_truth, predicted, normalize='true')
        df_confusion = pd.DataFrame(cm)
        df_confusion.index = target_class_names
        df_confusion.columns = target_class_names
        df_confusion = df_confusion.round(4)
        df_confusion.to_csv(model_conf_mat_normalized_csv)
        fig = plt.figure(figsize=(5, 5))
        sns.heatmap(df_confusion, annot=True, cmap=cmap)
        plt.xlabel(plot_x_label)
        plt.ylabel(plot_y_label)
        plt.title('Normalized Confusion Matrix')
        plt.savefig(model_conf_mat_normalized_png)
        plt.close(fig)

        report = metrics.classification_report(ground_truth, predicted, target_names=list(target_class_names))
        gen_roc(ground_truth, probabilities, model_roc_png)

    if losses is not None:
        fig = plt.figure(figsize=(8, 8))
        plt.plot(losses, label='Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xlim(0, epoch + 2)
        max_loss = np.max(losses)
        plt.ylim(0, max_loss + 0.20 * max_loss)
        plt.title(report_type + ' Loss')
        plt.legend()
        plt.savefig(model_loss_png)
        plt.close(fig)

        # save model training stats
        with open(model_train_losses_log_file, 'wb') as file:
            pickle.dump(losses, file)
            file.flush()

    if accuracies is not None:
        fig = plt.figure(figsize=(8, 8))
        plt.plot(accuracies, label='Accuracy', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.xlim(0, epoch + 2)
        plt.ylim(0, 100)
        plt.title(report_type + ' Accuracy')
        plt.legend()
        plt.savefig(model_accuracy_png)
        plt.close(fig)

        with open(model_train_accuracy_log_file, 'wb') as file:
            pickle.dump(accuracies, file)
            file.flush()

    # save model arch and params
    with open(model_log_file, 'w') as file:
        file.write('-' * log_params['line_len'] + '\n')
        file.write('model architecture' + '\n')
        file.write('-' * log_params['line_len'] + '\n')
        file.write(str(model) + '\n')
        file.write('-' * log_params['line_len'] + '\n')
        file.write('model params' + '\n')
        file.write('-' * log_params['line_len'] + '\n')
        file.write(str(model_params) + '\n')
        file.write('-' * log_params['line_len'] + '\n')
        file.write('-' * log_params['line_len'] + '\n')

        if misc_data is not None:
            file.write('misc data: ' + misc_data + '\n')
            file.write('-' * log_params['line_len'] + '\n')

        if report is not None:
            file.write(report_type + ' classification report' + '\n')
            file.write('-' * log_params['line_len'] + '\n')
            file.write(report + '\n')
            file.write('-' * log_params['line_len'] + '\n')
            file.write("roc_auc_score = {}\n".format(roc_auc_score(ground_truth, probabilities)))
            file.write('-' * log_params['line_len'] + '\n')

        if report_type == 'Test':
            if losses is not None:
                file.write('Mean loss: ' + str(np.mean(losses)) + '\n')
                file.write('-' * log_params['line_len'] + '\n')

            if accuracies is not None:
                file.write('Mean accuracy:' + str(np.mean(accuracies)) + '\n')
                file.write('-' * log_params['line_len'] + '\n')

    if model_params['batch_format'] == 'simple':
        print(
            f'all_samples_pred_csv:{all_samples_pred_csv}, log_dir={log_dir}, report_type={report_type},log_kind={log_kind}, model_params={model_params}')

        fake_best, fraction_best, _ = grid_search_for_per_frame_model(per_frame_csv=all_samples_pred_csv,
                                                                      log_dir=log_dir, report_type=report_type,
                                                                      log_kind=log_kind, model_params=model_params)
        gen_report_for_per_frame_model(per_frame_csv=all_samples_pred_csv, log_dir=log_dir, report_type=report_type,
                                       prob_threshold_fake=fake_best, prob_threshold_real=0.60,
                                       fake_fraction=fraction_best,
                                       log_kind=log_kind, model_params=model_params)

        ConfigParser.getInstance().copy_config(dest=model_log_dir)
        sys.stdout.flush()


def save_all_model_results(model=None, model_params=None, train_losses=None, train_accuracies=None, valid_losses=None,
                           valid_accuracies=None, valid_predicted=None, valid_ground_truth=None,
                           valid_sample_names=None, optimizer=None, criterion=None, epoch=0, log_dir=None,
                           log_kind=None, probabilities=None, amp_dict=None):
    report_type = 'Train'
    save_model_results_to_log(epoch=epoch, model=model, model_params=model_params,
                              losses=train_losses, accuracies=train_accuracies,
                              log_dir=log_dir, log_kind=log_kind, report_type=report_type)

    report_type = 'Validation'
    save_model_results_to_log(epoch=epoch, model=model, model_params=model_params,
                              losses=valid_losses, accuracies=valid_accuracies,
                              predicted=valid_predicted, ground_truth=valid_ground_truth,
                              sample_names=valid_sample_names,
                              log_dir=log_dir, log_kind=log_kind, report_type=report_type, probabilities=probabilities)

    save_checkpoint(epoch=epoch, model=model, model_params=model_params,
                    optimizer=optimizer, criterion=criterion, log_dir=log_dir, log_kind=log_kind,
                    amp_dict=amp_dict)


def get_per_video_stat(df, vid, prob_threshold_fake, prob_threshold_real):
    # number of frames detected as fake with at-least prob of prob_threshold
    df1 = df.loc[df['video'] == vid]
    fake_frames_prob = df1[df1['predictions'] == 1]['norm_probability'].values
    fake_frames_high_prob = fake_frames_prob[np.array(fake_frames_prob > prob_threshold_fake)]
    num_fake_frames = len(fake_frames_high_prob)
    if num_fake_frames == 0:
        fake_prob = 0
    else:
        fake_prob = sum(fake_frames_high_prob) / num_fake_frames

    real_frames_prob = df1[df1['predictions'] == 0]['norm_probability'].values
    real_frames_high_prob = real_frames_prob[np.array(real_frames_prob > prob_threshold_real)]
    num_real_frames = len(real_frames_high_prob)
    if num_real_frames == 0:
        real_prob = 0
    else:
        real_prob = sum(real_frames_high_prob) / num_real_frames

    total_number_frames = len(df1)
    ground_truth = df1['ground_truth'].values[0]

    return num_fake_frames, fake_prob, num_real_frames, total_number_frames, ground_truth


def split_video(val):
    return val.split('__')[0]


def split_frames(val):
    return val.split('__')[1]


def norm_probability(pred, prob):
    if pred == 0:
        return 1 - prob
    else:
        return prob


def pred_strategy(num_fake_frames, num_real_frames, total_number_frames, fake_fraction=0.10):
    """
    return True if there are atleast fake_fraction times total_number_frames are fake.
    e.g. if atleast 10% of total frames are fake then whole video is fake.

    :param num_fake_frames:
    :param num_real_frames:
    :param total_number_frames:
    :param fake_fraction:
    :return:
    """
    if num_fake_frames >= (fake_fraction * total_number_frames):
        return 1
    return 0


def gen_roc(ground_truth, probability, model_roc_png):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(ground_truth, probability)
        roc_auc[i] = auc(fpr[i], tpr[i])

    fig = plt.figure()
    plt.plot(fpr[1], tpr[1])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.savefig(model_roc_png)
    plt.close(fig)


def gen_report_for_per_frame_model(per_frame_csv=None, log_dir=None, report_type=None, prob_threshold_fake=0.50,
                                   prob_threshold_real=0.55, fake_fraction=0.10, log_kind=None, model_params=None):
    if not os.path.isfile(per_frame_csv):
        return
    df = pd.read_csv(per_frame_csv)

    df['video'] = df['sample_name'].apply(split_video)
    df['frames'] = df['sample_name'].apply(split_frames)
    df['norm_probability'] = df.apply(lambda x: norm_probability(x.predictions, x.probability), axis=1)
    all_videos = set(df['video'].values)

    final_df = pd.DataFrame(
        columns=['video', 'num_fake_frames', 'num_real_frames', 'total_number_frames', 'ground_truth'])

    for v in all_videos:
        num_fake_frames, fake_prob, num_real_frames, total_number_frames, \
        ground_truth = get_per_video_stat(df, v, prob_threshold_fake, prob_threshold_real)
        prediction = pred_strategy(num_fake_frames, num_real_frames, total_number_frames, fake_fraction=fake_fraction)
        final_df = final_df.append({'video': v, 'num_fake_frames': num_fake_frames,
                                    'num_real_frames': num_real_frames, 'total_number_frames': total_number_frames,
                                    'ground_truth': ground_truth, 'prediction': prediction, 'fake_prob': fake_prob},
                                   ignore_index=True)

    log_params = ConfigParser.getInstance().get_log_params()

    model_sub_dir = 'video_classi_' + str(prob_threshold_fake) + '_' + \
                    str(prob_threshold_real) + '_' + str(fake_fraction)

    model_log_dir = os.path.join(log_dir, log_kind, model_params['model_name'], report_type, model_sub_dir)
    os.makedirs(model_log_dir, exist_ok=True)

    model_conf_mat_csv = os.path.join(model_log_dir, log_params['model_conf_matrix_csv'])
    model_conf_mat_png = os.path.join(model_log_dir, log_params['model_conf_matrix_png'])
    model_conf_mat_normalized_csv = os.path.join(model_log_dir, log_params['model_conf_matrix_normalized_csv'])
    model_conf_mat_normalized_png = os.path.join(model_log_dir, log_params['model_conf_matrix_normalized_png'])
    model_roc_png = os.path.join(model_log_dir, log_params['model_roc_png'])
    all_samples_pred_csv = os.path.join(model_log_dir, log_params['all_samples_pred_csv'])
    model_log_file = os.path.join(model_log_dir, log_params['model_info_log'])

    final_df = final_df.set_index(['video'])
    final_df.to_csv(all_samples_pred_csv)

    # generate and save confusion matrix
    plot_x_label = "Predictions"
    plot_y_label = "Actual"
    cmap = plt.cm.Blues
    class_names = ['Real', 'Fake']

    cm = metrics.confusion_matrix(final_df['ground_truth'], final_df['prediction'])
    target_class_names = class_names

    df_confusion = pd.DataFrame(cm)
    df_confusion.index = target_class_names
    df_confusion.columns = target_class_names
    df_confusion = df_confusion.round(4)
    df_confusion.to_csv(model_conf_mat_csv)
    fig = plt.figure(figsize=(5, 5))
    sns.heatmap(df_confusion, annot=True, cmap=cmap)
    plt.xlabel(plot_x_label)
    plt.ylabel(plot_y_label)
    plt.title('Confusion Matrix')
    plt.savefig(model_conf_mat_png)
    plt.close(fig)

    cm = metrics.confusion_matrix(final_df['ground_truth'], final_df['prediction'], normalize='true')
    df_confusion = pd.DataFrame(cm)
    df_confusion.index = target_class_names
    df_confusion.columns = target_class_names
    df_confusion = df_confusion.round(4)
    df_confusion.to_csv(model_conf_mat_normalized_csv)
    fig = plt.figure(figsize=(5, 5))
    sns.heatmap(df_confusion, annot=True, cmap=cmap)
    plt.xlabel(plot_x_label)
    plt.ylabel(plot_y_label)
    plt.title('Normalized Confusion Matrix')
    plt.savefig(model_conf_mat_normalized_png)
    plt.close(fig)

    report = metrics.classification_report(final_df['ground_truth'], final_df['prediction'],
                                           target_names=list(target_class_names))

    ground_truth = final_df['ground_truth'].to_numpy()
    probability = final_df['fake_prob'].to_numpy()

    misc = "prob_threshold_fake = {}\nprob_threshold_real = {}\nfake_fraction = {}\n".format(prob_threshold_fake,
                                                                                             prob_threshold_real,
                                                                                             fake_fraction)
    gen_roc(ground_truth, probability, model_roc_png)
    with open(model_log_file, 'w') as file:
        if report is not None:
            file.write(report_type + ' classification report' + '\n')
            file.write('-' * log_params['line_len'] + '\n')
            file.write(report + '\n')
            file.write('-' * log_params['line_len'] + '\n')
            file.write("roc_auc_score = {}\n".format(roc_auc_score(ground_truth, probability)))
            file.write('-' * log_params['line_len'] + '\n')
            file.write(misc)
            file.write('-' * log_params['line_len'] + '\n')
            print(f'report saved at: {model_log_file}')


def get_classificiton_report_simple(prob_threshold_fake=None, prob_threshold_real=None, fake_fraction=None,
                                    all_videos=None,
                                    df=None):
    # print(f'processing for prob_threshold_fake={prob_threshold_fake}, fake_fraction={fake_fraction}')
    final_df = pd.DataFrame(
        columns=['video', 'num_fake_frames', 'num_real_frames', 'total_number_frames', 'ground_truth'])
    for v in all_videos:
        num_fake_frames, fake_prob, num_real_frames, total_number_frames, \
        ground_truth = get_per_video_stat(df, v, prob_threshold_fake, prob_threshold_real)
        prediction = pred_strategy(num_fake_frames, num_real_frames, total_number_frames, fake_fraction=fake_fraction)
        final_df = final_df.append({'video': v, 'num_fake_frames': num_fake_frames,
                                    'num_real_frames': num_real_frames, 'total_number_frames': total_number_frames,
                                    'ground_truth': ground_truth, 'prediction': prediction, 'fake_prob': fake_prob},
                                   ignore_index=True)

    final_df = final_df.set_index(['video'])
    class_names = ['Real', 'Fake']
    report = metrics.classification_report(final_df['ground_truth'], final_df['prediction'],
                                           target_names=list(class_names), output_dict=True)

    return prob_threshold_fake, prob_threshold_real, fake_fraction, report


def grid_search_for_per_frame_model(per_frame_csv=None, log_dir=None, report_type=None,
                                    log_kind=None, model_params=None):
    if not os.path.isfile(per_frame_csv):
        return None, None, None
    df = pd.read_csv(per_frame_csv)

    df['video'] = df['sample_name'].apply(split_video)
    df['frames'] = df['sample_name'].apply(split_frames)
    df['norm_probability'] = df.apply(lambda x: norm_probability(x.predictions, x.probability), axis=1)
    all_videos = set(df['video'].values)

    prob_threshold_fake_range = [0.50, 0.55, 0.60, 0.70, 0.80, 0.90]
    prob_threshold_real = 0.55  # unused
    fake_fraction_range = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]

    grid_search_df = pd.DataFrame(
        columns=['prob_threshold_fake', 'prob_threshold_real', 'fake_fraction', 'accuracy', 'report'])

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        jobs = []
        results = []
        print('Scheduling jobs')
        for prob_threshold_fake in prob_threshold_fake_range:
            for fake_fraction in fake_fraction_range:
                jobs.append(pool.apply_async(get_classificiton_report_simple,
                                             (prob_threshold_fake, prob_threshold_real, fake_fraction, all_videos, df,),
                                             )
                            )

        for job in tqdm(jobs, desc="Executing grid-search"):
            results.append(job.get())

    print('Parsing results')

    for r in results:
        prob_threshold_fake, prob_threshold_real, fake_fraction, report = r
        grid_search_df = grid_search_df.append(
            {'prob_threshold_fake': prob_threshold_fake, 'prob_threshold_real': prob_threshold_real,
             'fake_fraction': fake_fraction, 'accuracy': report['accuracy'], 'report': str(report)},
            ignore_index=True)
    top_gs = grid_search_df.nlargest(1, 'accuracy')
    prob_threshold_fake_best = top_gs.iloc[0]['prob_threshold_fake']
    fake_fraction_best = top_gs.iloc[0]['fake_fraction']
    accuracy_best = top_gs.iloc[0]['accuracy']
    print(
        f"top_gs best param: prob_threshold_fake {prob_threshold_fake_best}, fake_fraction_best {fake_fraction_best}, accuracy_best {accuracy_best}")

    model_sub_dir = 'grid_search'
    model_log_dir = os.path.join(log_dir, log_kind, model_params['model_name'], report_type, model_sub_dir)
    os.makedirs(model_log_dir, exist_ok=True)
    grid_search_df_path = os.path.join(model_log_dir, 'grid_search_df.csv')

    grid_search_df.to_csv(grid_search_df_path)

    return prob_threshold_fake_best, fake_fraction_best, accuracy_best
