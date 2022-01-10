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

    D0 = torch.from_numpy(orthogonalize(np.random.randn(args.n_dirs, dim_z).astype(np.float32))).cuda()
    D0.requires_grad = True

    gan_type, image_type = args.gan_model.split("_")
    test_zs = prepare_test_z(args)

    if args.optim == 'Adam':
        optimizer = optim.Adam(params=[D0], lr=args.lr)
    elif args.optim == 'SGD':
        optimizer = optim.SGD(params=[D0], lr=args.lr)
    else:
        raise NotImplemented('We don\'t support this type of optimizer.')

    # D = D0 / torch.sqrt(eps + torch.sum(D0 * D0, dim=1, keepdim=True))
    # for visualizing
    D_copy = (D0 / torch.sqrt(eps + torch.sum(D0 * D0, dim=1, keepdim=True))).detach()

    for iter in range(args.total_iterations):
        D = D0 / torch.sqrt(eps + torch.sum(D0 * D0, dim=1, keepdim=True))
        z = torch.randn(size=(batch_size, dim_z, 1, 1), dtype=torch.float32).cuda()

        if args.same_density:
            z_norm = torch.sqrt(eps + torch.sum(z * z, dim=1, keepdim=True))

        t = _sigmoid_to_tanh(torch.rand((batch_size, 1, 1, 1), dtype=torch.float32).cuda()) * args.t_scale
        which_d = torch.zeros(size=(batch_size, args.n_dirs), dtype=torch.float32).cuda()
        labels = np.random.choice(args.n_dirs, size=batch_size).astype(np.int64)
        which_d[np.linspace(0, batch_size - 1, batch_size).astype(np.int64), labels] = 1.0
        d = torch.matmul(which_d, D).view(batch_size, dim_z, 1, 1)

        which_d2 = torch.zeros(size=(batch_size, args.n_dirs), dtype=torch.float32).cuda()
        labels2 = np.random.choice(args.n_dirs, size=batch_size).astype(np.int64)
        which_d2[np.linspace(0, batch_size - 1, batch_size).astype(np.int64), labels2] = 1.0
        d2 = torch.matmul(which_d2, D).view(batch_size, dim_z, 1, 1)

        z_t = z + t * d
        z_t2 = z + t * d2

        is_same = torch.from_numpy(np.array(labels == labels2, dtype=np.float32).reshape([-1, 1])).cuda()
        if args.same_density:
            z_t_norm = torch.sqrt(eps + torch.sum(z_t * z_t, dim=1, keepdim=True))
            z_t = z_t / z_t_norm * z_norm
            z_t2_norm = torch.sqrt(eps + torch.sum(z_t2 * z_t2, dim=1, keepdim=True))
            z_t2 = z_t2 / z_t2_norm * z_norm

        f = generator([z], which_block=args.layer, pre_model=True)
        f_t = generator([z_t], which_block=args.layer, pre_model=True)
        f_t2 = generator([z_t2], which_block=args.layer, pre_model=True)

        f_diff = torch.reshape(f_t - f, shape=[batch_size, -1])
        f_diff2 = torch.reshape(f_t2 - f, shape=[batch_size, -1])
        # mask = torch.from_numpy(get_label_mask(batch_size, labels)).cuda()

        if not args.reduce_mean:
            positive_distance = torch.sum(torch.pow(f - f_t, 2.0), dim=[1, 2, 3]) + \
                                torch.sum(torch.pow(f - f_t2, 2.0), dim=[1, 2, 3])
            # distance = torch.matmul(f_diff, torch.transpose(f_diff, 0, 1))
        else:
            # scales = np.sqrt(fmap_size * fmap_size * fmap_ch)
            positive_distance = torch.mean(torch.pow(f - f_t, 2.0), dim=[1, 2, 3]) + \
                                torch.mean(torch.pow(f - f_t2, 2.0), dim=[1, 2, 3])
            # distance = torch.matmul(f_diff, torch.transpose(f_diff, 0, 1)) / scales

        negative_distance = torch.pow(torch.sum(f_diff * f_diff2, dim=1, keepdim=True), 2.0) * (1 - is_same)
        negative_distance = torch.sum(negative_distance) / torch.sum(1 - is_same)
        positive_distance = torch.mean(positive_distance)
        orthogonal = torch.mean(torch.pow(torch.mm(D, torch.transpose(D, 0, 1)) - torch.eye(args.n_dirs).cuda(), 2.0))

        optimizer.zero_grad()
        # loss = - torch.mean(distance)
        loss = - args.wgt_pos * positive_distance + args.wgt_neg * negative_distance + args.wgt_orth * orthogonal
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()
        optimizer.step()

        if iter % args.report_value == 0:
            print('Iteration %d: loss = %.6f, positive_dist = %.6f, negative_dist = %.6f, orthogonal = %.6f. ' %
                  (iter, float(loss.item()), float(positive_distance.item()), float(negative_distance.item()),
                   float(orthogonal.item())))

        if iter % args.report_image == 0:
            n_samples = args.n_samples
            test_z = test_zs[np.random.choice(test_zs.shape[0], n_samples, replace=False)].view(n_samples, dim_z, 1, 1)
            n_interps = args.n_interps

            interp_sheet = []
            for dir_idx in range(args.n_dirs):
                interp_imgs = []
                d_i = D[dir_idx].view(1, dim_z, 1, 1)
                d_i_copy = D_copy[dir_idx].view(1, dim_z, 1, 1)
                for int_i in range(n_interps):
                    alpha = (int_i / (n_interps - 1) - 0.5) * 2 * args.t_scale
                    test_z_p = test_z + alpha * d_i
                    if args.same_density:
                        test_z_norm = torch.sqrt(eps + torch.sum(test_z * test_z, dim=1, keepdim=True))
                        test_z_p_norm = torch.sqrt(eps + torch.sum(test_z_p * test_z_p, dim=1, keepdim=True))
                        test_z_p = test_z_p / test_z_p_norm * test_z_norm
                    x_cur = post_process_image(forward_generator(generator, test_z_p, args.layer).detach()).cpu()

                    # for copy
                    test_z_p = test_z + alpha * d_i_copy
                    if args.same_density:
                        test_z_norm = torch.sqrt(eps + torch.sum(test_z * test_z, dim=1, keepdim=True))
                        test_z_p_norm = torch.sqrt(eps + torch.sum(test_z_p * test_z_p, dim=1, keepdim=True))
                        test_z_p = test_z_p / test_z_p_norm * test_z_norm
                    x_ori = post_process_image(forward_generator(generator, test_z_p, args.layer).detach()).cpu()

                    x_cont = torch.cat([x_cur, x_ori], dim=1).reshape((-1,) + x_cur.shape[1:]).numpy()
                    if not args.gan_model.endswith('mnist'):
                        x_cont = torch.from_numpy(resize_images(x_cont, resize=args.resize))
                    interp_imgs.append(x_cont)

                interp_imgs = torch.cat(interp_imgs, dim=3)
                interp_sheet.append(interp_imgs)

                if (dir_idx + 1) % args.n_dir_per_sheet == 0:
                    print('Saving checkpoint images to %s...' % out_dir)
                    interp_sheet = torch.cat(interp_sheet, dim=0)
                    image_path = os.path.join(out_dir, 'save_image_iter_%d_D_%d-%d.png' % (iter, dir_idx - args.n_dir_per_sheet + 1, dir_idx))
                    torchvision.utils.save_image(interp_sheet, image_path, nrow=1, padding=2)
                    interp_sheet = []
                    print('Save OK! ')

        if (iter + 1) % args.report_model == 0:
            print('Saving transformer to disk. ')
            os.makedirs(os.path.join(out_dir, 'checkpoints'), exist_ok=True)
            save_path = os.path.join(out_dir, 'checkpoints', 'fmap_transformer_iter_%d.pth' % (iter + 1))
            save_dicts = dict()
            save_dicts['D'] = D
            torch.save(save_dicts, save_path)


if __name__ == '__main__':
    print('Working on applying transformer on unsupervised GAN discovery.')
    parser = argparse.ArgumentParser(description='GAN Transformer discovery.')
    parser.add_argument('-o', '--outputs', type=str, default='./TRAIN', help='Directory to output corresponding images or loggings.')
    parser.add_argument('--exp_id', default='MaximumTraversingMask', type=str, help='experiment prefix for easy debugging. ')
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
    parser.add_argument('--reduce_mean', default=0, type=int, help='The number of samples pf visualization. ')

    parser.add_argument('--wgt_pos', default=0.1, type=float, help='The learning rate of the optimizer.')
    parser.add_argument('--wgt_neg', default=10.0, type=float, help='The learning rate of the optimizer.')
    parser.add_argument('--wgt_orth', default=100.0, type=float, help='The learning rate of the optimizer.')
    args = parser.parse_args()

    if args.n_dir_per_sheet > args.n_dirs:
        args.n_dir_per_sheet = args.n_dirs

    main(args)
