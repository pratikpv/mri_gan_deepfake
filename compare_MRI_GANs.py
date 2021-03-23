import PIL
import torch, torchvision
from torchvision.transforms import transforms
from torchvision.utils import save_image
import os, sys, time, multiprocessing
from mri_gan.model import *
from mri_gan.dataset import *
import pandas as pd


def get_real_images(real_A_paths, real_B_paths):
    real_imgs_list = []
    fake_imgs_list = []
    celeb_dfdc = pd.read_csv(ConfigParser.getInstance().get_celeb_df_v2_mri_metadata_csv_path(), index_col=0)
    df_dfdc = pd.read_csv(ConfigParser.getInstance().get_dfdc_mri_metadata_csv_path(), index_col=0)
    for i, img in enumerate(real_B_paths):

        splitnames = img.split(os.path._get_sep(img))
        if splitnames[-1] == 'blank.png':
            # imgs_comp.append(imgs[i])
            real_img_path = real_A_paths[i]
            fake_img_path = None
        else:
            # find real image
            dataset = splitnames[-4]
            real_img_path = ""
            if dataset == 'dfdc':
                df = df_dfdc
            elif dataset == 'Celeb-DF-v2':
                df = celeb_dfdc
            else:
                raise Exception("Bad dataset name")
            real_img_path = df.loc[df['mri_image'] == img]['real_image'].tolist()[0]
            fake_img_path = df.loc[df['mri_image'] == img]['fake_image'].tolist()[0]
            #print(real_img_path)
            #real_img_path = real_A_paths[i]

        real_img = Image.open(real_img_path)
        fake_img = Image.open(fake_img_path) if fake_img_path is not None else None
        real_imgs_list.append(real_img)
        fake_imgs_list.append(fake_img)

        print(f'A:{real_img_path}, B:{img},  fake:{fake_img_path}')
    return real_imgs_list, fake_imgs_list


def get_demo_image_MRI_GAN_model(log_dir=None, model_params=None, imgs=None,
                                 rand_start=None, rand_end=None, device=None):
    # Initialize generator and discriminator
    generator = GeneratorUNet().to(device)
    checkpoint_path = os.path.join(log_dir, model_params['model_name'], 'checkpoint_best_G.chkpt')
    print(f'Loading state dict {checkpoint_path}')
    generated_samples_path = os.path.join(log_dir, model_params['model_name'], 'generated_samples_demo')
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator_state_dict'])

    print(f'\nGenerating samples at {generated_samples_path}')
    generator.eval()
    real_A = imgs["A"][rand_start:rand_end].to(device)
    fake_B = generator(real_A)
    return fake_B


log_dir_t03 = 'logs/23-Feb-2021_21_22_21'  # tau=0.3
log_dir_t05 = 'logs/02-Mar-2021_12_47_10'  # tau=0.5
log_dir_t07 = 'logs/04-Mar-2021_09_12_48'  # tau=0.7

use_cuda = torch.cuda.is_available()
use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

model_params = ConfigParser.getInstance().get_mri_gan_model_params()
data_transforms = torchvision.transforms.Compose([
    transforms.Resize((model_params['imsize'], model_params['imsize'])),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

test_dataset = MRIDataset(mode='test', transforms=data_transforms, frac=model_params['frac'], get_path=True)
num_workers = multiprocessing.cpu_count() - 2
test_dataloader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=num_workers,
)

for index, (imgs, imgs_path) in enumerate(test_dataloader):

    rand_start = random.randint(0, model_params['batch_size'] - model_params['test_sample_size'])
    rand_end = rand_start + 8

    all_logs = [log_dir_t03, log_dir_t05, log_dir_t07]
    fake_B_list = []
    for log_t in all_logs:
        fake_B = get_demo_image_MRI_GAN_model(log_dir=log_t, model_params=model_params, imgs=imgs, rand_start=rand_start,
                                              rand_end=rand_end, device=device)
        fake_B_list.append(fake_B)

    real_A = imgs["A"][rand_start:rand_end].to(device)
    real_B = imgs["B"][rand_start:rand_end].to(device)
    real_imgs_list, fake_imgs_list = get_real_images(imgs_path["A"][rand_start:rand_end], imgs_path["B"][rand_start:rand_end])
    real_imgs_tensor = []
    for i in real_imgs_list:
        if isinstance(i, PIL.PngImagePlugin.PngImageFile):
            real_imgs_tensor.append(data_transforms(i))
    fake_imgs_tensor = []
    for i in fake_imgs_list:
        if isinstance(i, PIL.PngImagePlugin.PngImageFile):
            fake_imgs_tensor.append(data_transforms(i))
        else:
            fake_imgs_tensor.append(data_transforms(Image.open("/home/therock/toshiba_nvme_3/my_code/mri_gan_deepfake/assets/blank.png")))
    #img_sample = torch.cat((real_A.data, torch.stack(real_imgs_tensor), real_B.data, fake_B_list[0].data,
    #                        fake_B_list[1].data, fake_B_list[2].data), -2)
    img_sample = torch.cat((real_A.data, torch.stack(real_imgs_tensor), real_B.data, fake_B_list[0].data,
                            fake_B_list[1].data, fake_B_list[2].data), -2)

    save_image(img_sample, "{}/{}_{}.png".format('assets', 'MRI_demo', index), nrow=8, normalize=True)
