# This version is used for rebuttal. Compute LPIPS and LIM.

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


def factorize_weight(weight, layer_idx='all'):
    weight = weight.detach().cpu().numpy().T
    weight = weight[:128]
    weight = weight / np.linalg.norm(weight, axis=0, keepdims=True)
    eigen_values, eigen_vectors = np.linalg.eig(weight.dot(weight.T))

    return eigen_vectors.T, eigen_values


all_zs_dicts = {
    'stylegan2_ffhq1024': 'all_zs_stylegan2_ffhq.pkl',
    'stylegan2_cat256': 'all_zs_stylegan2_cat.pkl',
    'stylegan2_church256': 'all_zs_stylegan2_church.pkl',
    'stylegan2_car512': 'all_zs_stylegan2_car.pkl',
    'pggan_celebahq1024': 'all_zs_pggan_celebahq.pkl',
    'biggandeep256_imagenet': 'all_zs_biggandeep256_imagenet.pkl'
}


def easy_lpips(model, x, x_move):
    n = x.shape[0]
    dists = []
    for i in range(n):
        x_i = x[None, i]
        x_move_i = x_move[None, i]
        dist_i = model(x_i, x_move_i)
        dists.append(dist_i)

    return torch.cat(dists, dim=0)


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
    lpips_model = lpips.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, version='0.1')

    fmap_size, fmap_ch, image_size, image_ch = get_generator_info(args, generator)
    print(fmap_size, fmap_ch, image_size, image_ch)

    gen_z = generator.module.biggan.generator.gen_z
    weight = gen_z.weight
    print(gen_z)
    print(weight.shape)

    D, alpha = factorize_weight(weight)
    print(D.shape)

    save_D_path = os.path.join(out_dir, 'sefa_dog.pkl')
    with open(save_D_path, 'wb') as file_out:
        pkl.dump(D, file_out)
        print('Save factorized directions in path \'%s\'' % save_D_path)

    dist_list = [int(item) for item in args.distances.split('-')]
    print('We will calculate the distances list in ', dist_list)
    num_dirs_list = [int(item) for item in args.num_dirs.split('-')]
    print('We will calculate the number of dist, ', num_dirs_list)

    for i in range(len(num_dirs_list)):
        num_dirs = num_dirs_list[i]
        print('Compute %d directions. ' % num_dirs)

        for j in range(len(dist_list)):
            dists_for_each_test = []
            for k in range(args.num_tests):
                D_k = torch.from_numpy(D[:num_dirs].astype(np.float32)).cuda()
                z = torch.randn(size=(num_dirs, dim_z), dtype=torch.float32).cuda()
                z_move = z + dist_list[j] * D_k

                x = forward_generator(generator, z, layer=3, args=args).detach()
                x_move = forward_generator(generator, z_move, layer=3, args=args).detach()

                dist = easy_lpips(lpips_model, x, x_move)
                dists_for_each_test.append(dist)

            all_dists = torch.cat(dists_for_each_test, dim=0).detach().cpu().numpy()
            avg_dist = np.mean(all_dists)
            std_dist = np.std(all_dists)
            print('K=%d, dist=%.2f, LPIPS Mean: %.4f, STD: %.4f. ' % (num_dirs, dist_list[j], avg_dist, std_dist))


if __name__ == '__main__':
    print('Working on applying transformer on unsupervised GAN discovery.')
    parser = argparse.ArgumentParser(description='GAN Transformer discovery.')
    parser.add_argument('-o', '--outputs', type=str, default='./TRAIN',
                        help='Directory to output corresponding images or loggings.')
    parser.add_argument('--exp_id', default='BigGANSefa', type=str,
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
    parser.add_argument('--t_scale', default=10.0, type=float, help='The scale of scaling.')
    parser.add_argument('--n_dirs', default=20, type=int, help='The number of directions.')
    parser.add_argument('--batch_size', default=16, type=int, help='The batch size of the input')

    parser.add_argument('--report_value', default=10, type=int, help='The step of reporting value.')
    parser.add_argument('--report_model', default=1000, type=int, help='The step of reporting value.')
    parser.add_argument('--report_image', default=5000, type=int, help='The step of reporting value.')

    parser.add_argument('--n_interps', default=11, type=int, help='The number of interpolation of visualization. ')
    parser.add_argument('--n_samples', default=6, type=int, help='The number of samples pf visualization. ')
    parser.add_argument('--n_dir_per_sheet', default=10, type=int, help='The number of samples pf visualization. ')
    parser.add_argument('--resize', default=256, type=int, help='The number of samples pf visualization. ')

    parser.add_argument('--wgt_pos', default=1.0, type=float, help='The learning rate of the optimizer.')
    parser.add_argument('--wgt_neg', default=0.001, type=float, help='The learning rate of the optimizer.')
    parser.add_argument('--wgt_orth', default=1.0, type=float, help='The learning rate of the optimizer.')
    parser.add_argument('--step', default=11, type=int, help='The learning rate of the optimizer.')
    parser.add_argument('--distances', default='1-5-10-15-20', type=str, help='The distances of shift. ')
    parser.add_argument('--num_dirs', default='1-10-20-30-50', type=str, help='The distances of shift. ')
    parser.add_argument('--num_tests', default=20, type=int, help='The number of testing for averaging and std. ')
    args = parser.parse_args()

    if args.n_dir_per_sheet > args.n_dirs:
        args.n_dir_per_sheet = args.n_dirs

    main(args)
