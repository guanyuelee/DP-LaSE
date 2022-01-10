import os
import pickle as pkl

import argparse
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from derivable_models.derivable_generator import get_derivable_generator
from utils.file_utils import create_transformer_experiments_directory, get_generator_info, prepare_test_z
from utils.image_precossing import _tanh_to_sigmoid, resize_images, _sigmoid_to_tanh, post_process_image
import torchvision
import lpips
from utils.ortho_utils import torch_expm
import cv2
from utils.manipulate import convert_array_to_images


def easy_lpips(model, x, x_move):
    n = x.shape[0]
    dists = []
    for i in range(n):
        x_i = x[None, i]
        x_move_i = x_move[None, i]
        dist_i = model(x_i, x_move_i)
        dists.append(dist_i)

    return torch.cat(dists, dim=0)


def forward_generator(generator, z, layer, args):
    n = z.shape[0]
    batch_size = 6
    if n <= batch_size:
        y = torch.ones(size=(n, 1), dtype=torch.int64) * args.which_class
        f = generator([z, y], which_block=layer, truncation=1.0, pre_model=True)
        x = generator([z, y], which_block=layer, truncation=1.0, features=f, post_model=True)
        return x
    else:
        res = []
        for i in range(0, n, batch_size):
            start = i
            end = min(start + batch_size, n)
            y = torch.ones(size=(end - start, 1), dtype=torch.int64) * args.which_class
            f = generator([z[start: end], y], which_block=layer, truncation=1.0, pre_model=True)
            x = generator([z[start: end], y], which_block=layer, truncation=1.0, features=f, post_model=True)
            res.append(x.detach())
        res = torch.cat(res, dim=0)
        return res


def orthogonalize(directions, eps=1e-16):
    B, dim = directions.shape
    for i in range(B - 1):
        x1x2 = np.sum(directions[None, i] * directions[(i + 1):, :], axis=1, keepdims=True)
        x1x1 = np.sum(directions[None, i] * directions[None, i], axis=1, keepdims=True)
        a = x1x2 / (x1x1 + eps)
        directions[(i + 1):] = directions[(i + 1):, :] - a * directions[None, i]

    return directions


def orthogonal_direction(directions, eps=1e-16):
    d = directions.detach().cpu().numpy()
    d2 = orthogonalize(d, eps)
    return torch.from_numpy(d2 - d)


def get_label_mask(batch_size, labels):
    matrix = np.zeros(shape=(batch_size, batch_size), dtype=np.float32)
    for i in range(batch_size):
        matrix[i, labels == labels[i]] = 1.0
    return matrix


def density_preserving_traverse(z, d, alpha):
    z_move = z + alpha * d
    z_move_norm = torch.sqrt(1e-16 + torch.sum(z_move * z_move, dim=1, keepdim=True))
    z_move_n = z_move / z_move_norm
    z_norm = torch.sqrt(1e-16 + torch.sum(z * z, dim=1, keepdim=True))
    z_move_dpt = z_move_n * z_norm
    return z_move_dpt


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    eps = 1e-16
    dim_z = args.dim_z
    batch_size = args.batch_size
    os.makedirs(args.outputs, exist_ok=True)
    out_dir = args.save_dir
    generator = get_derivable_generator(args.gan_model, args.inversion_type, args)
    generator = torch.nn.DataParallel(generator)
    generator.cuda()
    generator.eval()

    fmap_size, fmap_ch, image_size, image_ch = get_generator_info(args, generator)
    print(fmap_size, fmap_ch, image_size, image_ch)

    # building skew symmetric D for orthogonal representation.
    state_dict = torch.load(args.load_dir)
    D0 = torch.from_numpy(state_dict['D0']).cuda()

    gan_type, image_type = args.gan_model.split("_")
    test_zs = prepare_test_z(args)

    D0_skew_symmetric = 0.5 * (D0 - torch.transpose(D0, 0, 1))
    D = torch_expm(D0_skew_symmetric.unsqueeze(0))

    test_z = test_zs.view(test_zs.shape[0], dim_z, 1, 1)
    frame_size = 256
    which_dir = args.dir_id

    for sample_i in range(test_z.shape[0]):
        print('Create GAN-Inversion video.')
        video = cv2.VideoWriter(
            filename=os.path.join(out_dir, 'class_%d_dir_%d_sample_%d_inversion.avi' % (args.which_class, args.dir_id, sample_i)),
            fourcc=cv2.VideoWriter_fourcc(*'XVID'),
            fps=args.fps,
            frameSize=(frame_size, frame_size))
        print('Save frames.')

        print('sample %d' % sample_i)
        z_alphas = []
        for i in range(args.n_interps):
            alpha = 2 * (i / (args.n_interps - 1) - 0.5) * args.t_scale
            # alpha = (- (i / (args.n_interps - 1))) * args.t_scale
            z_alpha = density_preserving_traverse(test_z[sample_i: sample_i + 1], D[which_dir: which_dir + 1].view(1, dim_z, 1, 1), alpha)
            z_alphas.append(z_alpha)
        z_alphas = torch.cat(z_alphas, dim=0)
        # z_alphas = torch.cat([z_alphas, torch.flip(z_alphas, dims=(0,))], dim=0)
        y = torch.ones(size=(args.batch_size, 1), dtype=torch.int64) * args.which_class

        print(z_alphas.shape)
        print(y.shape)
        count = 0
        for i in range(0, z_alphas.shape[0], batch_size):
            f = generator([z_alphas[i:i + batch_size], y], which_block=args.layer, pre_model=True,
                          truncation=args.truncation).detach()
            x = generator([z_alphas[i:i + batch_size], y], which_block=args.layer, post_model=True,
                          truncation=args.truncation, features=f).detach()
            x = post_process_image(x)
            print(x.shape)
            x_image = convert_array_to_images(x.cpu().numpy() * 2 - 1.0)
            for j in range(len(x_image)):
                print(count)
                video.write(cv2.cvtColor(x_image[j], cv2.COLOR_BGR2RGB))
                count += 1
        video.release()


if __name__ == '__main__':
    print('Working on applying transformer on unsupervised GAN discovery.')
    parser = argparse.ArgumentParser(description='GAN Transformer discovery.')
    parser.add_argument('-o', '--outputs', type=str, default='./TRAIN',
                        help='Directory to output corresponding images or loggings.')
    parser.add_argument('--exp_id', default='BigGAN', type=str,
                        help='experiment prefix for easy debugging. ')
    # Parameters for Multi-Code GAN Inversion
    parser.add_argument('--inversion_type', default='BigGANDeep',
                        help='Inversion type, "PGGAN-Multi-Z" for Multi-Code-GAN prior.')
    # Generator Setting, Check models/model_settings for available GAN models
    parser.add_argument('--gan_model', default='biggandeep256_imagenet', help='The name of model used.', type=str)
    parser.add_argument('--seed', default=0, help='The seed for the model. ', type=int)
    parser.add_argument('--which_class', default=239, type=int, help='The class of BigGAN.')
    parser.add_argument('--truncation', default=1.0, type=float, help='The number of samples pf visualization. ')
    parser.add_argument('--layer', default=1, type=int, help='which layer to plug transformer into.')
    parser.add_argument('--dim_z', default=128, type=int, help='experiment prefix for easy debugging. ')

    parser.add_argument('--total_iterations', default=50000, type=int, help='The total number of iterations.')
    parser.add_argument('--optim', default='Adam', type=str, help='The optimizer used.')
    parser.add_argument('--lr', default=1e-4, type=float, help='The learning rate of the optimizer.')
    parser.add_argument('--t_scale', default=0.0, type=float, help='The scale of scaling.')
    parser.add_argument('--n_dirs', default=20, type=int, help='The number of directions.')
    parser.add_argument('--batch_size', default=16, type=int, help='The batch size of the input')

    parser.add_argument('--report_value', default=10, type=int, help='The step of reporting value.')
    parser.add_argument('--report_metrics', default=1000, type=int, help='The step of reporting value.')
    parser.add_argument('--report_model', default=1000, type=int, help='The step of reporting value.')
    parser.add_argument('--report_image', default=1000, type=int, help='The step of reporting value.')

    parser.add_argument('--n_samples', default=3, type=int, help='The number of samples pf visualization. ')
    parser.add_argument('--n_dir_per_sheet', default=10, type=int, help='The number of samples pf visualization. ')
    parser.add_argument('--resize', default=256, type=int, help='The number of samples pf visualization. ')
    parser.add_argument('--distances', default='1-5-10-15-20', type=str, help='The distances of shift. ')
    parser.add_argument('--num_tests', default=20, type=int, help='The distances of shift. ')

    parser.add_argument('--wgt_pos', default=1.0, type=float, help='The learning rate of the optimizer.')
    parser.add_argument('--wgt_neg', default=0.1, type=float, help='The learning rate of the optimizer.')

    parser.add_argument('--n_interps', default=60 * 8, type=int, help='The number of interpolation of visualization. ')
    parser.add_argument('--fps', default=60, type=int, help='The frame per second. ')
    parser.add_argument('--load_dir', default='./none.pt', help='The directory of the save dir. ')
    parser.add_argument('--save_dir', default='./bin/vedio/aa', help='The directory')
    parser.add_argument('--dir_id', default=38, type=int, help='The id of the direction. ')

    args = parser.parse_args()

    if args.n_dir_per_sheet > args.n_dirs:
        args.n_dir_per_sheet = args.n_dirs

    main(args)

    # CUDA_VISIBLE_DEVICES=$CUDA_NUMBER python maximum_traverse_for_BigGAN.py --wgt_pos=1.0 --wgt_neg=0.5 --wgt_orth=100 --n_dirs=1
