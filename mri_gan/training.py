import torch, torchvision
from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import os, sys, time, multiprocessing
import numpy as np
from tqdm import tqdm
from pytorch_msssim import ssim, SSIM
from mri_gan.model import *
from mri_gan.dataset import *
from data_utils.face_mri import get_structural_similarity
import pprint
import pickle
import matplotlib.pyplot as plt

pp = pprint.PrettyPrinter(indent=4)


def get_ssim_report(global_batch_num, model_params, imgs, generator, device, save_img=False):
    num_imgs = imgs["A"].shape[0]
    real_A = imgs["A"].to(device)
    fake_B = generator(real_A)
    scores = []
    real_B = imgs["B"].detach().cpu()
    fake_B = fake_B.detach().cpu()
    for i in range(num_imgs):
        image1 = np.transpose(real_B[i].numpy(), (1, 2, 0))
        image1 = denormalize(image1)
        image2 = np.transpose(fake_B[i].numpy(), (1, 2, 0))
        image2 = denormalize(image2)
        d, a = get_structural_similarity(image1, image2)
        scores.append(a)
        if save_img:
            impath = os.path.join(model_params['log_dir'], model_params['model_name'],
                                  'global_batches', str(global_batch_num))
            os.makedirs(impath, exist_ok=True)
            real_B_path = os.path.join(impath, 'real_B_{}.png'.format(i))
            fake_B_path = os.path.join(impath, 'fake_B_{}.png'.format(i))
            cv2.imwrite(real_B_path, image1)
            cv2.imwrite(fake_B_path, image2)

    return np.mean(scores)


def denormalize(img):
    img = (img + 1) / 2  # [-1, 1] => [0, 1]
    return img * 255


def generate_graphs(losses_file, ssim_report_file, model_params):
    losses = pickle.load(open(losses_file, "rb"))
    ssim_report = pickle.load(open(ssim_report_file, "rb"))

    plots_path = os.path.join(model_params['log_dir'], model_params['model_name'], 'plots')
    os.makedirs(plots_path, exist_ok=True)
    df_ssim = pd.DataFrame(data=ssim_report, columns=['epoch', 'local_batch_num', 'global_batch_num', 'mean_ssim'])
    df_ssim = df_ssim[['global_batch_num', 'mean_ssim']].set_index('global_batch_num')
    fig_ssim = df_ssim.plot(figsize=(16, 8), fontsize=16, alpha=0.8, title='SSIM score vs Global batches').get_figure()
    fig_ssim.savefig(os.path.join(plots_path, 'ssim_global_batch.png'))

    df_losses = pd.DataFrame(data=losses,
                             columns=['epoch', 'local_batch_num', 'global_batch_num',
                                      'loss_G', 'loss_GAN', 'loss_pixel', 'loss_ssim',
                                      'loss_D', 'loss_real', 'loss_fake'])
    df_losses = df_losses.drop(labels=['epoch', 'local_batch_num'], axis=1).set_index('global_batch_num')
    fig_losses = df_losses.plot(figsize=(16, 8), fontsize=16, alpha=0.8, title='All losses').get_figure()
    fig_losses.savefig(os.path.join(plots_path, 'all_losses.png'))

    df_GD = df_losses[['loss_D', 'loss_G']]
    fig_GD = df_GD.plot(figsize=(16, 8), fontsize=16, alpha=0.8,
                        title='Generator and Discriminator total loss').get_figure()
    fig_GD.savefig(os.path.join(plots_path, 'Gen_and_Dis_total_losses.png'))

    df_G = df_losses[['loss_GAN', 'loss_pixel', 'loss_ssim']]
    fig_G = df_G.plot(figsize=(16, 8), fontsize=16, alpha=0.8, title='Generator GAN, pixel and SSIM loss').get_figure()
    fig_G.savefig(os.path.join(plots_path, 'Generator_gan_pixel_ssim_losses.png'))

    df_D = df_losses[['loss_real', 'loss_fake']]
    fig_D = df_D.plot(figsize=(16, 8), fontsize=16, alpha=0.8, title='Discriminator fake and real loss').get_figure()
    fig_D.savefig(os.path.join(plots_path, 'Discriminator_fake_real_losses.png'))

    plt.close('all')


def train_MRI_GAN_model(log_dir=None, train_resume_dir=None):
    model_params = ConfigParser.getInstance().get_mri_gan_model_params()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    data_transforms = torchvision.transforms.Compose([
        transforms.Resize((model_params['imsize'], model_params['imsize'])),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Loss functions
    criterion_GAN = torch.nn.MSELoss().to(device)
    criterion_pixelwise = torch.nn.MSELoss().to(device)
    criterion_ssim = SSIM(win_size=11, win_sigma=1.5, data_range=255, size_average=True, channel=3,
                          nonnegative_ssim=True).to(device)

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, model_params['imsize'] // 2 ** 4, model_params['imsize'] // 2 ** 4)

    # Initialize generator and discriminator
    generator = GeneratorUNet().to(device)
    discriminator = Discriminator().to(device)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=model_params['lr'],
                                   betas=(model_params['b1'], model_params['b2']))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=model_params['lr'],
                                   betas=(model_params['b1'], model_params['b2']))
    losses = []
    ssim_report = []
    global_batches_done = 0
    start_epoch = 0
    mri_gan_metadata = dict()
    loss_D_lowest = float('inf')
    loss_G_lowest = float('inf')

    if train_resume_dir is not None:
        checkpoint_path = train_resume_dir
        print(f'Loading state dict {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        log_dir = checkpoint['log_dir']
        start_epoch = checkpoint['epoch'] + 1
        print(f'Override log dir {log_dir}')
        losses_file = os.path.join(log_dir, model_params['model_name'], model_params['losses_file'])
        metadata_file = os.path.join(log_dir, model_params['model_name'], model_params['metadata_file'])
        ssim_report_file = os.path.join(log_dir, model_params['model_name'], model_params['ssim_report_file'])

        losses = pickle.load(open(losses_file, "rb"))
        ssim_report = pickle.load(open(ssim_report_file, "rb"))
        mri_gan_metadata = pickle.load(open(metadata_file, "rb"))
        global_batches_done = mri_gan_metadata['global_batches_done']
        if len(losses) != global_batches_done + 1:
            print(f'losses len = {len(losses)}, mri_gan_metadata = {mri_gan_metadata}')
            print(f'Use data from metadata file')
            losses = losses[0:global_batches_done]
            # raise Exception('Bad metadata and saved states')
        loss_D_lowest = mri_gan_metadata['loss_D_lowest']
        loss_G_lowest = mri_gan_metadata['loss_G_lowest']
    else:
        print(f'Initializing weights')
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)
        checkpoint_path = os.path.join(log_dir, model_params['model_name'], 'checkpoint.chkpt')
        losses_file = os.path.join(log_dir, model_params['model_name'], model_params['losses_file'])
        metadata_file = os.path.join(log_dir, model_params['model_name'], model_params['metadata_file'])
        ssim_report_file = os.path.join(log_dir, model_params['model_name'], model_params['ssim_report_file'])

    model_params['log_dir'] = log_dir
    checkpoint_best_G_path = os.path.join(log_dir, model_params['model_name'], 'checkpoint_best_G.chkpt')
    checkpoint_best_D_path = os.path.join(log_dir, model_params['model_name'], 'checkpoint_best_D.chkpt')
    generated_samples_path = os.path.join(log_dir, model_params['model_name'], 'generated_samples')
    os.makedirs(generated_samples_path, exist_ok=True)

    print_line()
    print('model_params')
    pp.pprint(model_params)
    print_line()

    for e in range(model_params['n_epochs']):
        if e < start_epoch:
            print(f"Skipping epoch {e}")
            continue

        # we are creating dataloader at each epoch as we want to sample new fake image randomly
        # as each epoch. We have lesser real image and more fake. We can get more real images but
        # that would impose a lot of computation power need considering the dataset size we have.

        print(f'Creating data-loaders for epoch: {e}')
        train_dataset = MRIDataset(mode='train', transforms=data_transforms, frac=model_params['frac'])
        test_dataset = MRIDataset(mode='test', transforms=data_transforms, frac=model_params['frac'])

        num_workers = multiprocessing.cpu_count() - 2
        # num_workers = 0
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=model_params['batch_size'],
            shuffle=True,
            num_workers=num_workers,
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=model_params['batch_size'],
            shuffle=True,
            num_workers=num_workers,
        )

        desc = "Training MRI-GAN [e:{e}/{n_epochs}] [G_loss:{loss_G}] [D_loss:{loss_D}]".format(
            e=e, n_epochs=model_params['n_epochs'], loss_G='N/A', loss_D='N/A')
        pbar = tqdm(train_dataloader, desc=desc)

        for local_batch_num, batch in enumerate(pbar):

            generator.train()
            discriminator.train()
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            valid = torch.ones((real_A.size(0), *patch)).to(device)
            fake = torch.zeros((real_A.size(0), *patch)).to(device)

            #  Train Generator
            optimizer_G.zero_grad()

            # GAN loss
            fake_B = generator(real_A)
            pred_fake = discriminator(fake_B, real_A)
            loss_GAN = criterion_GAN(pred_fake, valid)
            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(fake_B, real_B)
            fake_B_dn = denormalize(fake_B)
            real_B_dn = denormalize(real_B)
            # SSIM loss
            loss_ssim = torch.sqrt(1 - criterion_ssim(fake_B_dn, real_B_dn))
            # Total generator loss
            loss_G = loss_GAN + model_params['lambda_pixel'] * (
                    model_params['tau'] * loss_pixel + (1 - model_params['tau']) * loss_ssim)

            loss_G.backward()
            optimizer_G.step()

            #  Train Discriminator
            optimizer_D.zero_grad()

            # Real loss
            pred_real = discriminator(real_B, real_A)
            loss_real = criterion_GAN(pred_real, valid)
            # Fake loss
            pred_fake = discriminator(fake_B.detach(), real_A)
            loss_fake = criterion_GAN(pred_fake, fake)
            # Total discriminator loss
            loss_D = 0.5 * (loss_real + loss_fake)

            loss_D.backward()
            optimizer_D.step()

            desc = "Training MRI-GAN [e:{e}/{n_epochs}] [G_loss:{loss_G}] [D_loss:{loss_D}]".format(
                e=e, n_epochs=model_params['n_epochs'], loss_G=loss_G.item(), loss_D=loss_D.item())

            pbar.set_description(desc=desc, refresh=True)

            losses.append(
                [e, local_batch_num, global_batches_done,
                 loss_G.item(), loss_GAN.item(), loss_pixel.item(), loss_ssim.item(),
                 loss_D.item(), loss_real.item(), loss_fake.item()]
            )

            mri_gan_metadata['global_batches_done'] = global_batches_done
            mri_gan_metadata['model_params'] = model_params

            global_batches_done += 1

            if global_batches_done % model_params['sample_gen_freq'] == 0:
                try:
                    # print(f'\nGenerating samples at {generated_samples_path}')
                    generator.eval()
                    imgs = next(iter(test_dataloader))
                    rand_start = random.randint(0, model_params['batch_size'] - model_params['test_sample_size'])
                    rand_end = rand_start + model_params['test_sample_size']
                    real_A = imgs["A"][rand_start:rand_end].to(device)
                    real_B = imgs["B"][rand_start:rand_end].to(device)
                    fake_B = generator(real_A)
                    img_sample = torch.cat((real_A.data, real_B.data, fake_B.data), -2)
                    os.makedirs(os.path.join(generated_samples_path, str(e)), exist_ok=True)
                    save_image(img_sample, "{}/{}/{}.png".format(generated_samples_path, e, local_batch_num),
                               nrow=int(np.sqrt(model_params['test_sample_size'])),
                               normalize=True)

                    mean_ssim = get_ssim_report(global_batches_done - 1, model_params, imgs, generator,
                                                device, save_img=False)
                    ssim_report.append([e, local_batch_num, global_batches_done, mean_ssim])
                    pickle.dump(ssim_report, open(ssim_report_file, "wb"))

                    generate_graphs(losses_file, ssim_report_file, model_params)
                except Exception as expn:
                    print(f'Exception {expn}')
                    pass

            if global_batches_done % model_params['chkpt_freq'] == 0:
                # print(f'\nSaving model checkpoint at {checkpoint_path}')
                check_point_dict = {
                    'epoch': e,
                    'model_params': model_params,
                    'log_dir': log_dir,
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'optimizer_G_state_dict': optimizer_G.state_dict(),
                    'optimizer_D_state_dict': optimizer_D.state_dict(),
                }
                torch.save(check_point_dict, checkpoint_path)

            if loss_D.item() < loss_D_lowest:
                loss_D_lowest = loss_D.item()

                check_point_dict = {
                    'epoch': e,
                    'model_params': model_params,
                    'log_dir': log_dir,
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'optimizer_G_state_dict': optimizer_G.state_dict(),
                    'optimizer_D_state_dict': optimizer_D.state_dict(),
                }
                torch.save(check_point_dict, checkpoint_best_D_path)

            if loss_G.item() < loss_G_lowest:
                loss_G_lowest = loss_G.item()

                check_point_dict = {
                    'epoch': e,
                    'model_params': model_params,
                    'log_dir': log_dir,
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'optimizer_G_state_dict': optimizer_G.state_dict(),
                    'optimizer_D_state_dict': optimizer_D.state_dict(),
                }
                torch.save(check_point_dict, checkpoint_best_G_path)

            mri_gan_metadata['loss_D_lowest'] = loss_D_lowest
            mri_gan_metadata['loss_G_lowest'] = loss_G_lowest
            pickle.dump(losses, open(losses_file, "wb"))
            pickle.dump(mri_gan_metadata, open(metadata_file, "wb"))
