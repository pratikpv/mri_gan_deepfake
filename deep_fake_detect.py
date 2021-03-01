import argparse
from deep_fake_detect.utils import *
from data_utils.face_detection import *
from deep_fake_detect.training import *
from deep_fake_detect.testing import *


def test_saved_model(model_path, model_kind):
    print(f'Loading saved model {model_path} to test')
    check_point_dict = torch.load(model_path)
    model_params = check_point_dict['model_params']
    model = DeepFakeDetectModel(frame_dim=model_params['imsize'], encoder_name=model_params['encoder_name'])
    model.load_state_dict(check_point_dict['model_state_dict'])
    if check_point_dict['criterion'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise Exception("Unknown criterion in the saved model")
    print(f"Log override to {check_point_dict['log_dir']}")
    test_model(model, check_point_dict['model_params'], criterion, check_point_dict['log_dir'], model_kind)


def main():
    log_dir = print_banner()
    if args.train_from_scratch:
        print(f'Training from scratch')
        _, model_params, _, _ = train_model(log_dir)
        model_kind = 'highest_acc'
        model_path = os.path.join(log_dir, model_kind, model_params['model_name'] + '.chkpt')
        test_saved_model(model_path, model_kind)
    if args.train_resume_checkpoint:
        print(f'Resume training from checkpoint {args.train_resume_checkpoint}')
        _, model_params, _, log_dir = train_model(log_dir, train_resume_checkpoint=args.train_resume_checkpoint)
        model_kind = 'highest_acc'
        model_path = os.path.join(log_dir, model_kind, model_params['model_name'] + '.chkpt')
        test_saved_model(model_path, model_kind)

    if args.test_saved_model_path:
        model_kind = 'saved'
        test_saved_model(args.test_saved_model_path, model_kind)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and predict DeepFakes')

    parser.add_argument('--train_from_scratch', action='store_true', default=False, help='Train the model from scratch')
    parser.add_argument('--train_resume', dest='train_resume_checkpoint', default=False,
                        help='Resume the model training')
    parser.add_argument('--test_saved_model', dest='test_saved_model_path', default=False, help='Test the saved model')

    args = parser.parse_args()
    print(args)
    main()
