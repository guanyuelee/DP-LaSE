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


def main(args):
    args.layer = 8              # we can only support the intermediate layer.
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

    start_layer, end_layer = args.which_layers.split('-')
    layer_list = [item for item in range(int(start_layer)-1, int(end_layer), 1)]

    for iter in range(args.total_iterations):
        D = D0 / torch.sqrt(eps + torch.sum(D0 * D0, dim=1, keepdim=True))
        z = torch.randn(size=(batch_size, dim_z, 1, 1), dtype=torch.float32).cuda()

        t = _sigmoid_to_tanh(torch.rand((batch_size, 1, 1, 1), dtype=torch.float32).cuda()) * args.t_scale
        which_d = torch.zeros(size=(batch_size, args.n_dirs), dtype=torch.float32).cuda()
        labels = np.random.choice(args.n_dirs, size=batch_size).astype(np.int64)
        which_d[np.linspace(0, batch_size - 1, batch_size).astype(np.int64), labels] = 1.0
        d = torch.matmul(which_d, D).view(batch_size, dim_z, 1, 1)

        which_d2 = torch.zeros(size=(batch_size, args.n_dirs), dtype=torch.float32).cuda()
        labels2 = np.random.choice(args.n_dirs, size=batch_size).astype(np.int64)
        which_d2[np.linspace(0, batch_size - 1, batch_size).astype(np.int64), labels2] = 1.0
        d2 = torch.matmul(which_d2, D).view(batch_size, dim_z, 1, 1)

        z_t = density_preserving_traverse(z, d, t)
        z_t2 = density_preserving_traverse(z, d2, t)
        is_same = torch.from_numpy(np.array(labels == labels2, dtype=np.float32).reshape([-1, 1])).cuda()

        # we only select the needed layer.
        f = generator([z], which_block=args.layer, pre_model=True).reshape(tuple([batch_size, ] + list(fmap_ch)))[:, layer_list, :]
        f_t = generator([z_t], which_block=args.layer, pre_model=True).reshape(tuple([batch_size, ] + list(fmap_ch)))[:, layer_list, :]
        f_t2 = generator([z_t2], which_block=args.layer, pre_model=True).reshape(tuple([batch_size, ] + list(fmap_ch)))[:, layer_list, :]

        fs = []
        f_ts = []
        f_t2s = []
        for layer_idx in layer_list:
            style_net = generator.module.stylegan.synthesis.__getattr__('layer%d' % layer_idx).style
            f_s = style_net(f[:, layer_idx-int(start_layer)])
            f_t_s = style_net(f_t[:, layer_idx-int(start_layer)])
            f_t2_s = style_net(f_t2[:, layer_idx-int(start_layer)])
            fs.append(f_s)
            f_ts.append(f_t_s)
            f_t2s.append(f_t2_s)

        f = torch.reshape(torch.cat(fs, dim=1), shape=[batch_size, -1])
        f_t = torch.reshape(torch.cat(f_ts, dim=1), shape=[batch_size, -1])
        f_t2 = torch.reshape(torch.cat(f_t2s, dim=1), shape=[batch_size, -1])

        f_diff = f_t - f
        f_diff2 = f_t2 - f
        positive_distance = torch.mean(torch.pow(f_diff, 2.0), dim=1) + torch.mean(torch.pow(f_diff2, 2.0), dim=1)
        positive_distance = torch.mean(positive_distance)

        f_diff = f_diff / torch.sqrt(torch.sum(torch.pow(f_diff, 2.0), dim=1, keepdim=True) + eps)
        f_diff2 = f_diff2 / torch.sqrt(torch.sum(torch.pow(f_diff2, 2.0), dim=1, keepdim=True) + eps)
        negative_distance = torch.pow(torch.sum(f_diff * f_diff2, dim=1, keepdim=True), 2.0) * (1 - is_same)
        negative_distance = torch.sum(negative_distance) / torch.sum(1 - is_same)
        orthogonal = torch.mean(torch.pow(torch.mm(D, torch.transpose(D, 0, 1)) - torch.eye(args.n_dirs).cuda(), 2.0))

        optimizer.zero_grad()
        loss = - args.wgt_pos * positive_distance + args.wgt_neg * negative_distance + args.wgt_orth * orthogonal
        loss.backward()
        optimizer.step()

        if iter % args.report_value == 0:
            print('Iteration %d: loss = %.6f, positive_dist = %.6f, negative_dist = %.6f, orthogonal = %.6f. ' %
                  (iter, float(loss.item()), float(positive_distance.item()), float(negative_distance.item()),
                   float(orthogonal.item())))

        if (iter + args.is_start_save) % args.report_image == 0:
            n_samples = args.n_samples
            test_z = test_zs[np.random.choice(test_zs.shape[0], n_samples, replace=False)].view(n_samples, dim_z, 1, 1)
            codezs = generator([test_z], which_block=args.layer, pre_model=True).detach().cpu().numpy()
            n_interps = args.n_interps

            interp_sheet = []
            for dir_idx in range(args.n_dirs):
                interp_imgs = []
                d_i = D[dir_idx].view(1, dim_z, 1, 1)
                d_i_copy = D_copy[dir_idx].view(1, dim_z, 1, 1)
                for int_i in range(n_interps):
                    alpha = (int_i / (n_interps - 1) - 0.5) * 2 * args.t_scale
                    test_z_p = density_preserving_traverse(test_z, d_i, alpha)
                    codezs_tmp = codezs.copy()
                    codezs_move = generator([test_z_p], which_block=args.layer, pre_model=True).detach().cpu().numpy()
                    codezs_tmp[:, layer_list, :] = 0.0
                    codezs_tmp[:, layer_list, :] += codezs_move[:, layer_list, :]
                    x_cur = generator([torch.from_numpy(codezs_tmp).cuda()], which_block=args.layer,
                                      post_model=True).detach().cpu()
                    x_cur = post_process_image(x_cur)

                    # for copy
                    test_z_p = density_preserving_traverse(test_z, d_i_copy, alpha)

                    codezs_tmp = codezs.copy()
                    codezs_move = generator([test_z_p], which_block=args.layer, pre_model=True).detach().cpu().numpy()
                    codezs_tmp[:, layer_list, :] = 0.0
                    codezs_tmp[:, layer_list, :] += codezs_move[:, layer_list, :]
                    x_ori = generator([torch.from_numpy(codezs_tmp).cuda()], which_block=args.layer,
                                      post_model=True).detach().cpu()
                    x_ori = post_process_image(x_ori)

                    x_cont = torch.cat([x_cur, x_ori], dim=1).reshape((-1,) + x_cur.shape[1:]).numpy()
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

            print('Saving directions to disk. ')
            os.makedirs(os.path.join(out_dir, 'checkpoints'), exist_ok=True)
            save_path = os.path.join(out_dir, 'checkpoints', 'D_iter%d_which_layer%s.pt' % (iter + 1, args.which_layers))
            save_dicts = dict()
            save_dicts['D'] = D
            torch.save(save_dicts, save_path)

        D0.data = torch.from_numpy(orthogonalize(D0.detach().cpu().numpy())).cuda()


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

    parser.add_argument('--n_interps', default=5, type=int, help='The number of interpolation of visualization. ')
    parser.add_argument('--n_samples', default=6, type=int, help='The number of samples pf visualization. ')
    parser.add_argument('--n_dir_per_sheet', default=5, type=int, help='The number of samples pf visualization. ')
    parser.add_argument('--resize', default=256, type=int, help='The number of samples pf visualization. ')

    parser.add_argument('--wgt_pos', default=1.0, type=float, help='The learning rate of the optimizer.')
    parser.add_argument('--wgt_neg', default=0.1, type=float, help='The learning rate of the optimizer.')
    parser.add_argument('--wgt_orth', default=1.0, type=float, help='The learning rate of the optimizer.')
    args = parser.parse_args()

    if args.n_dir_per_sheet > args.n_dirs:
        args.n_dir_per_sheet = args.n_dirs

    main(args)
