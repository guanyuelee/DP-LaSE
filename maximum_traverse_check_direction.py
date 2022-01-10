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


def forward_generator(generator, z, layer):
    f = generator([z], which_block=layer, pre_model=True)
    x = generator([f], which_block=layer, post_model=True)
    return x


def orthogonalize(directions, eps=1e-16):
    with torch.no_grad():
        B, dim = directions.shape
        for i in range(B - 1):
            x1x2 = torch.sum(directions[None, i] * directions[(i + 1):, :], dim=1, keepdim=True)
            x1x1 = torch.sum(directions[None, i] * directions[None, i], dim=1, keepdim=True)
            a = x1x2 / (x1x1 + eps)
            directions[(i + 1):] = directions[(i + 1):, :] - a * directions[None, i]

        directions = directions / torch.relu(torch.sqrt(torch.sum(directions * directions, dim=1, keepdim=True) + eps))
        return directions


def get_norm(z, eps=1e-16):
    z_norm = torch.sqrt(eps + torch.sum(z * z, dim=1, keepdim=True))
    return z_norm


def main(args):
    eps = 1e-16
    dim_z = args.dim_z
    batch_size = args.batch_size
    os.makedirs(args.outputs, exist_ok=True)
    out_dir, exp_name = create_transformer_experiments_directory(args, args.exp_id)
    print('Experiment name: ', exp_name)
    generator = get_derivable_generator(args.gan_model, args.inversion_type, args)
    generator = torch.nn.DataParallel(generator)
    generator.cuda()

    fmap_size, fmap_ch, image_size, image_ch = get_generator_info(args, generator)

    restore_path = args.restore_path
    save_dicts = torch.load(restore_path)
    D = save_dicts['D']
    test_zs = prepare_test_z(args, n_test_zs=args.n_test_zs)

    print(D.shape)
    print(test_zs.shape)
    n_test_zs = test_zs.shape[0]
    n_interps = args.n_interps

    print(args.which_directions)
    for i in [int(item) for item in args.which_directions.split('_')]:
        for j in range(0, n_test_zs, batch_size):
            z = test_zs[j: min(j + batch_size, n_test_zs)].view(-1, dim_z, 1, 1)
            sheet = []
            for int_i in range(n_interps):
                alpha = (int_i / (n_interps - 1) - 0.5) * 2 * args.t_scale
                z_t = z + alpha * D[i].view(1, dim_z, 1, 1)

                if args.same_density:
                    z_t_norm = get_norm(z_t)
                    z_norm = get_norm(z)
                    z_t = z_t / z_t_norm * z_norm

                x_t = forward_generator(generator, z_t, layer=args.layer).detach()
                x_t = post_process_image(x_t).detach().cpu()
                if not args.gan_model.endswith('mnist'):
                    x_t = torch.from_numpy(resize_images(x_t.numpy(), args.resize))
                sheet.append(x_t)
            sheet = torch.cat(sheet, dim=3)
            save_path = os.path.join(out_dir, 'direction_%d_sample_%d-%d.png' % (i, j, j + batch_size))
            torchvision.utils.save_image(sheet, save_path, nrow=1)
            print('save to %s' % save_path)
        save_path2 = os.path.join(out_dir, 'direction_%d.pt' % i)
        torch.save(D[i].view(1, dim_z, 1, 1), save_path2)
        print('save to %s' % save_path2)


if __name__ == '__main__':
    print('Working on applying transformer on unsupervised GAN discovery.')
    parser = argparse.ArgumentParser(description='GAN Transformer discovery.')
    parser.add_argument('-o', '--outputs', type=str, default='./TRAIN', help='Directory to output corresponding images or loggings.')
    parser.add_argument('--exp_id', default='CheckDirections', type=str, help='experiment prefix for easy debugging. ')
    # Parameters for Multi-Code GAN Inversion
    parser.add_argument('--inversion_type', default='PGGAN-Layerwise',
                        help='Inversion type, "PGGAN-Multi-Z" for Multi-Code-GAN prior.')
    # Generator Setting, Check models/model_settings for available GAN models
    parser.add_argument('--gan_model', default='pggan_celebahq', help='The name of model used.', type=str)
    parser.add_argument('--which_class', default=239, type=int, help='The class of BigGAN.')
    parser.add_argument('--layer', default=3, type=int, help='which layer to plug transformer into.')
    parser.add_argument('--dim_z', default=512, type=int, help='experiment prefix for easy debugging. ')
    parser.add_argument('--report_value', default=10, type=int, help='The step of reporting value.')
    parser.add_argument('--report_model', default=1000, type=int, help='The step of reporting value.')
    parser.add_argument('--total_iterations', default=500000, type=int, help='The total number of iterations.')
    parser.add_argument('--optim', default='Adam', type=str, help='The optimizer used.')
    parser.add_argument('--lr', default=1e-4, type=float, help='The learning rate of the optimizer.')
    parser.add_argument('--t_scale', default=10.0, type=float, help='The scale of scaling.')
    parser.add_argument('--n_dirs', default=20, type=int, help='The number of directions.')
    parser.add_argument('--batch_size', default=16, type=int, help='The batch size of the input')

    # report images configuration.
    parser.add_argument('--report_image', default=500, type=int, help='The step of reporting value.')
    parser.add_argument('--n_interps', default=11, type=int, help='The number of interpolation of visualization. ')
    parser.add_argument('--n_samples', default=6, type=int, help='The number of samples pf visualization. ')
    parser.add_argument('--n_dir_per_sheet', default=10, type=int, help='The number of samples pf visualization. ')

    parser.add_argument('--resize', default=128, type=int, help='The number of samples pf visualization. ')
    parser.add_argument('--same_density', default=0, type=int, help='The number of samples pf visualization. ')

    parser.add_argument('--restore_path', default='./aa.pkl', help='The path to restore.')
    parser.add_argument('--which_directions', default='16', help='The path to restore.')
    parser.add_argument('--n_test_zs', default=50, type=int, help='The path to restore.')

    args = parser.parse_args()
    main(args)
