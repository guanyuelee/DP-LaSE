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
    print(fmap_size, fmap_ch, image_size, image_ch)

    gen_z = generator.module.biggan.generator.gen_z
    weight = gen_z.weight
    print(gen_z)
    print(weight.shape)

    D, alpha = factorize_weight(weight)
    print(D.shape)

    with open(os.path.join('./bin/good_directions/sefa_dog.pkl'), 'wb') as file_out:
        pkl.dump(D, file_out)
        print('OK')

    with open(os.path.join('./bin/test_zs/', all_zs_dicts['biggandeep256_imagenet']), 'rb') as file_out:
        codes = np.reshape(pkl.load(file_out), [-1, 128])
        codes = torch.from_numpy(codes).cuda()
    codes = codes.detach().cpu().numpy()
    distances = np.linspace(10, -10, args.step)
    n_codes, n_chnl = codes.shape

    for direction_i in range(args.n_dirs):
        interp_images = []
        interp_codes = []
        d = D[direction_i: direction_i + 1]
        for step_i in range(args.step):
            tmp_codes = codes.copy()
            tmp_codes_move = tmp_codes + distances[step_i] * d
            tmp_codes_move_n = tmp_codes_move / np.sqrt(np.sum(tmp_codes_move * tmp_codes_move, axis=1, keepdims=True))
            tmp_codes_move_dpt = tmp_codes_move_n * np.sqrt(np.sum(tmp_codes * tmp_codes, axis=1, keepdims=True))
            tmp_codes_all = np.concatenate([tmp_codes_move, tmp_codes_move_dpt], axis=1).reshape([-1, 128])
            interp_codes.append(tmp_codes_all)

        interp_codes = np.concatenate(interp_codes, axis=1).reshape((2 * n_codes * args.step, n_chnl))
        # the results.
        print('Generating direction %d.' % direction_i)

        N = interp_codes.shape[0]
        bs = args.batch_size

        for cur_bs in range(0, N, bs):
            z_bs = torch.from_numpy(interp_codes[cur_bs: min(cur_bs + bs, N)].astype(np.float32)).cuda()
            y_bs = torch.ones(size=(z_bs.shape[0], 1), dtype=torch.int64) * args.which_class
            f_bs = generator([z_bs, y_bs], which_block=args.layer, pre_model=True, truncation=1.0).detach()
            x_bs = generator([z_bs, y_bs], which_block=args.layer, post_model=True, truncation=1.0, features=f_bs).detach()
            interp_images.append(post_process_image(x_bs).cpu())

            print('Direction %d: %d/%d' % (direction_i, cur_bs, N))

        interp_images = torch.cat(interp_images, dim=0)
        torchvision.utils.save_image(interp_images, os.path.join(out_dir, 'direction_%d.png' % direction_i), args.step)

    if 0 % args.report_model == 0:
        print('Saving transformer to disk. ')
        os.makedirs(os.path.join(out_dir, 'checkpoints'), exist_ok=True)
        save_path = os.path.join(out_dir, 'checkpoints', 'D_iter%d_which_layer%s.pt' % (iter + 1, args.layer))
        save_dicts = dict()
        save_dicts['D'] = D
        torch.save(save_dicts, save_path)


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
    args = parser.parse_args()

    if args.n_dir_per_sheet > args.n_dirs:
        args.n_dir_per_sheet = args.n_dirs

    main(args)
