import torch
from utils import *
import os
import pickle


def save_checkpoint(epoch=None, model=None, model_params=None,
                    optimizer=None, criterion=None, log_dir=None, log_kind=None, amp_dict=None):
    model_class_name = type(model).__name__
    checkpoint_root_path = os.path.join(log_dir, log_kind)
    os.makedirs(checkpoint_root_path, exist_ok=True)
    check_point_path = os.path.join(checkpoint_root_path, model_class_name + '.chkpt')

    check_point_dict = {
        'epoch': epoch,
        'model_class_name': model_class_name,
        'model_params': model_params,
        'criterion': criterion,
        'log_dir': log_dir,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'amp': amp_dict
    }

    torch.save(check_point_dict, check_point_path)


def load_checkpoint(model=None, optimizer=None, check_point_path=None):
    model_class_name = type(model).__name__
    check_point_dict = torch.load(check_point_path)

    if check_point_dict['model_class_name'] != model_class_name:
        raise Exception('Invalid checkpoint loading requested')
    model.load_state_dict(check_point_dict['model_state_dict'])
    optimizer.load_state_dict(check_point_dict['optimizer_state_dict'])
    return check_point_dict['epoch'], model, optimizer, check_point_dict['model_params'], \
           check_point_dict['log_dir'], check_point_dict['amp']


def load_acc_loss(model=None, log_dir=None, model_type=None):
    model_class_name = type(model).__name__
    log_params = ConfigParser.getInstance().get_log_params()

    report_type = 'Train'
    model_log_dir = os.path.join(log_dir, model_type, model_class_name, report_type)
    model_train_losses_log_file = os.path.join(model_log_dir, log_params['model_loss_info_log'])
    model_train_accuracy_log_file = os.path.join(model_log_dir, log_params['model_acc_info_log'])
    print(f'model_train_losses_log_file {model_train_losses_log_file}')
    print(f'model_train_accuracy_log_file {model_train_accuracy_log_file}')
    with open(model_train_losses_log_file, 'rb') as file:
        train_losses = pickle.load(file)
    with open(model_train_accuracy_log_file, 'rb') as file:
        train_accs = pickle.load(file)

    report_type = 'Validation'
    model_log_dir = os.path.join(log_dir, model_type, model_class_name, report_type)
    model_valid_losses_log_file = os.path.join(model_log_dir, log_params['model_loss_info_log'])
    model_valid_accuracy_log_file = os.path.join(model_log_dir, log_params['model_acc_info_log'])
    print(f'model_valid_losses_log_file {model_valid_losses_log_file}')
    print(f'model_valid_accuracy_log_file {model_valid_accuracy_log_file}')
    with open(model_valid_losses_log_file, 'rb') as file:
        valid_losses = pickle.load(file)
    with open(model_valid_accuracy_log_file, 'rb') as file:
        valid_accs = pickle.load(file)

    return train_accs, train_losses, valid_accs, valid_losses
