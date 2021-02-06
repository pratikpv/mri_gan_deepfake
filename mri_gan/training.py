import time
from datetime import datetime, timedelta
import sys
import torch
import torchvision
from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from skimage.metrics import structural_similarity
from mri_gan.model import *
from mri_gan.dataset import *

def save_ssim_report(epoch, batch_num, imgs, generator, device, ssim_score_file):
    imgs_len = imgs["A"].shape[0]
    real_A = imgs["A"].to(device)
    real_B = imgs["B"]
    fake_B = generator(real_A)
    scores = []
    real_B = imgs["B"].detach().cpu().numpy()
    fake_B = fake_B.detach().cpu().numpy()
    for i in range(imgs_len):
        image1 = np.transpose(real_B[i], (1, 2, 0))
        image2 = np.transpose(fake_B[i], (1, 2, 0))
        d, a = structural_similarity(image1, image2, multichannel=True, full=True)
        scores.append(a)

    with open(ssim_score_file, 'a') as f:
        data = "epoch:{}, batch {}, mean ssim score {}\n".format(epoch, batch_num, np.mean(scores))
        f.write(data)


def train_MRI_GAN_model(log_dir=None, train_resume_dir=None):
    n_epochs = 100
    batch_size = 128
    test_sample_size = 16
    # Loss weight of L1 pixel-wise loss between translated image and real image
    lambda_pixel = 100
    b1 = 0.5
    b2 = 0.999
    lr = 0.0002

    model_params = {}
    model_params['imsize'] = 256
    model_params['model_name'] = 'MRI_GAN'
    model_params['logdir'] = log_dir
    model_params['n_epochs'] = n_epochs
    model_params['batch_size'] = batch_size

    print(f'model_params {model_params}')

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Configure dataloaders
    train_transforms = torchvision.transforms.Compose([
        transforms.Resize((model_params['imsize'], model_params['imsize'])),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    print(f'Creating data-loaders')
    train_dataset = MRIDataset(mode='train', transforms=train_transforms)
    test_dataset = MRIDataset(mode='test', transforms=train_transforms)

    #num_workers = multiprocessing.cpu_count() - 2
    num_workers = 0
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=model_params['batch_size'],
        shuffle=True,
        num_workers=num_workers,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    # Loss functions
    criterion_GAN = torch.nn.MSELoss().to(device)
    criterion_pixelwise = torch.nn.L1Loss().to(device)

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, model_params['imsize'] // 2 ** 4, model_params['imsize'] // 2 ** 4)

    # Initialize generator and discriminator
    generator = GeneratorUNet().to(device)
    discriminator = Discriminator().to(device)

    print(discriminator)
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    if train_resume_dir is not None:
        checkpoint_path = train_resume_dir
        print(f'Resuming training {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        log_dir = checkpoint['log_dir']
        model_params['logdir'] = log_dir
        start_epoch = checkpoint['epoch']
        print(f'Override logdir {log_dir}')
    else:
        print(f'Training from scratch')
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)
        checkpoint_path = os.path.join(log_dir, model_params['model_name'], 'checkpoint.chkpt')
        start_epoch = 0

    generated_samples_path = os.path.join(log_dir, model_params['model_name'], 'generated_samples')
    os.makedirs(generated_samples_path, exist_ok=True)
    ssim_score_file = os.path.join(log_dir, model_params['model_name'], 'ssim_score_file.txt')
    batches_done = 0
    prev_time = time.time()

    losses = []
    for e in range(n_epochs):
        if e < start_epoch:
            print(f"Skipping epoch {e}")
            continue
        for i, batch in enumerate(train_dataloader):
            # Model inputs
            # generator.train()
            # discriminator.train()
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            # Adversarial ground truths
            valid = torch.ones((real_A.size(0), *patch)).to(device)
            fake = torch.zeros((real_A.size(0), *patch)).to(device)

            # ------------------
            #  Train Generators
            # ------------------
            optimizer_G.zero_grad()

            # GAN loss
            fake_B = generator(real_A)
            pred_fake = discriminator(fake_B, real_A)
            print(f'pred_fake shape {pred_fake.shape}')
            loss_GAN = criterion_GAN(pred_fake, valid)
            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(fake_B, real_B)

            # Total loss
            loss_G = loss_GAN + lambda_pixel * loss_pixel

            loss_G.backward()

            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Real loss
            pred_real = discriminator(real_B, real_A)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = discriminator(fake_B.detach(), real_A)
            loss_fake = criterion_GAN(pred_fake, fake)

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)

            loss_D.backward()
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done += 1
            batches_left = n_epochs * len(train_dataloader) - batches_done
            time_left = timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[batches_done=%d] [Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
                % (batches_done, e, n_epochs, i, len(train_dataloader), loss_D.item(), loss_G.item(),
                   loss_pixel.item(), loss_GAN.item(), time_left))
            losses.append([(e, i, batches_done, loss_D.item(), loss_G.item(), loss_pixel.item(), loss_GAN.item())])
            if batches_done % 200 == 0:
                try:
                    print(f'\nGenerating samples at {generated_samples_path}')
                    # generator.eval()
                    imgs = next(iter(test_dataloader))
                    # imgs = imgs, test_sample_size, replace=False)
                    rand_start = random.randint(0, batch_size - test_sample_size)
                    rand_end = rand_start + test_sample_size
                    real_A = imgs["A"][rand_start:rand_end].to(device)
                    real_B = imgs["B"][rand_start:rand_end].to(device)
                    fake_B = generator(real_A)
                    img_sample = torch.cat((real_A.data, real_B.data, fake_B.data), -2)
                    os.makedirs(os.path.join(generated_samples_path, str(e)), exist_ok=True)
                    save_image(img_sample, "{}/{}/{}.png".format(generated_samples_path, e, i),
                               nrow=int(np.sqrt(test_sample_size)),
                               normalize=True)

                    save_ssim_report(e, i, imgs, generator, device, ssim_score_file)

                    np.save(os.path.join(log_dir, model_params['model_name'], 'losses.npy'), losses)
                except Exception as e:
                    print(f'Exception {e}')
                    pass
            if batches_done % 2000 == 0:
                print(f'\nSaving model checkpoint at {checkpoint_path}')
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
