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
import cv2
from utils.manipulate import convert_array_to_images


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


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    args.layer = 8              # we can only support the intermediate layer.
    eps = 1e-16
    dim_z = args.dim_z
    batch_size = args.batch_size
    os.makedirs(args.outputs, exist_ok=True)
    out_dir = args.save_dir
    generator = get_derivable_generator(args.gan_model, args.inversion_type, args)
    generator = torch.nn.DataParallel(generator)
    generator.cuda()

    fmap_size, fmap_ch, image_size, image_ch = get_generator_info(args, generator)
    print(fmap_size, fmap_ch, image_size, image_ch)

    print('Loading save directions...')
    D = torch.load(args.load_dir)['D']

    gan_type, image_type = args.gan_model.split("_")
    test_zs = prepare_test_z(args)

    start_layer, end_layer = args.which_layers.split('-')
    layer_list = [item for item in range(int(start_layer)-1, int(end_layer), 1)]

    dir_id = args.dir_id
    frame_size = 1024

    for sample_i in range(test_zs.shape[0]):
        print('Create GAN-Inversion video.')
        video = cv2.VideoWriter(
            filename=os.path.join(out_dir, 'dir_%d_sample_%d_inversion.avi' % (args.dir_id, sample_i)),
            fourcc=cv2.VideoWriter_fourcc(*'XVID'),
            fps=args.fps,
            frameSize=(frame_size, frame_size))

        print('Generating %d frames of testing images...')
        test_z = test_zs[sample_i].view(1, dim_z, 1, 1)
        codezs = generator([test_z], which_block=args.layer, pre_model=True).detach().cpu().numpy()
        n_interps = args.n_interps
        interp_sheet = []
        d_i = D[dir_id].view(1, dim_z, 1, 1)

        count = 0
        for int_i in range(n_interps):
            start = args.start
            end = args.end
            # alpha = (int_i / (n_interps - 1) - 0.5) * 2 * args.t_scale
            alpha = (start + (int_i / (n_interps - 1)) * (end - start)) * args.t_scale
            test_z_p = density_preserving_traverse(test_z, d_i, alpha)
            codezs_tmp = codezs.copy()
            codezs_move = generator([test_z_p], which_block=args.layer, pre_model=True).detach().cpu().numpy()
            codezs_tmp[:, layer_list, :] = 0.0
            codezs_tmp[:, layer_list, :] += codezs_move[:, layer_list, :]
            x_cur = generator([torch.from_numpy(codezs_tmp).cuda()], which_block=args.layer,
                              post_model=True).detach().cpu()

            x_image = convert_array_to_images(x_cur.cpu().numpy())

            print(x_image.shape)
            print(count)
            video.write(cv2.cvtColor(x_image[0], cv2.COLOR_BGR2RGB))
            count += 1
        video.release()
        print('OK')


if __name__ == '__main__':
    print('Working on applying transformer on unsupervised GAN discovery.')
    parser = argparse.ArgumentParser(description='GAN Transformer discovery.')
    parser.add_argument('-o', '--outputs', type=str, default='./TRAIN',
                        help='Directory to output corresponding images or loggings.')
    parser.add_argument('--exp_id', default='StyleGAN2', type=str,
                        help='experiment prefix for easy debugging. ')
    # Parameters for Multi-Code GAN Inversion
    parser.add_argument('--inversion_type', default='StyleGAN-Layerwise-z',
                        help='Inversion type, "PGGAN-Multi-Z" for Multi-Code-GAN prior.')
    # Generator Setting, Check models/model_settings for available GAN models
    parser.add_argument('--gan_model', default='stylegan2_ffhq', help='The name of model used.', type=str)
    parser.add_argument('--seed', default=0, help='The seed for the model. ', type=int)
    parser.add_argument('--which_class', default=239, type=int, help='The class of BigGAN.')
    parser.add_argument('--layer', default=8, type=int, help='which layer to plug transformer into.')
    parser.add_argument('--which_layers', default='1-3', type=str, help='which layers to use.')
    parser.add_argument('--dim_z', default=512, type=int, help='experiment prefix for easy debugging. ')

    parser.add_argument('--total_iterations', default=500000, type=int, help='The total number of iterations.')
    parser.add_argument('--optim', default='Adam', type=str, help='The optimizer used.')
    parser.add_argument('--lr', default=1e-4, type=float, help='The learning rate of the optimizer.')
    parser.add_argument('--t_scale', default=10.0, type=float, help='The scale of scaling.')
    parser.add_argument('--n_dirs', default=20, type=int, help='The number of directions.')
    parser.add_argument('--batch_size', default=16, type=int, help='The batch size of the input')

    parser.add_argument('--report_value', default=10, type=int, help='The step of reporting value.')
    parser.add_argument('--report_model', default=5000, type=int, help='The step of reporting value.')
    parser.add_argument('--report_image', default=5000, type=int, help='The step of reporting value.')
    parser.add_argument('--is_start_save', default=1, type=int, help='The step of reporting value.')

    parser.add_argument('--n_samples', default=6, type=int, help='The number of samples pf visualization. ')
    parser.add_argument('--n_dir_per_sheet', default=5, type=int, help='The number of samples pf visualization. ')
    parser.add_argument('--resize', default=256, type=int, help='The number of samples pf visualization. ')

    parser.add_argument('--wgt_pos', default=1.0, type=float, help='The learning rate of the optimizer.')
    parser.add_argument('--wgt_neg', default=0.1, type=float, help='The learning rate of the optimizer.')
    parser.add_argument('--wgt_orth', default=1.0, type=float, help='The learning rate of the optimizer.')

    parser.add_argument('--n_interps', default=60 * 6, type=int, help='The number of interpolation of visualization. ')
    parser.add_argument('--fps', default=60, type=int, help='The frame per second. ')
    parser.add_argument('--load_dir', default='./none.pt', help='The directory of the save dir. ')
    parser.add_argument('--save_dir', default='./bin/vedio/ffhq', help='The directory')
    parser.add_argument('--dir_id', default=18, type=int, help='The id of the direction. ')

    parser.add_argument('--start', default=-1, type=float, help='The directory')
    parser.add_argument('--end', default=1, type=float, help='The id of the direction. ')

    args = parser.parse_args()

    if args.n_dir_per_sheet > args.n_dirs:
        args.n_dir_per_sheet = args.n_dirs

    main(args)
