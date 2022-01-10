import os
import pickle as pkl

import argparse
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from derivable_models.derivable_generator import get_derivable_generator
from utils.file_utils import create_transformer_experiments_directory, get_generator_info, prepare_test_z, post_proC, thrC
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


def density_preserving_traverse(z, d, alpha):
    z_move = z + alpha * d
    z_move_norm = torch.sqrt(1e-16 + torch.sum(z_move * z_move, dim=1, keepdim=True))
    z_move_n = z_move / z_move_norm
    z_norm = torch.sqrt(1e-16 + torch.sum(z * z, dim=1, keepdim=True))
    z_move_dpt = z_move_n * z_norm
    return z_move_dpt


def load_trained_directions(model_name):
    load_path = os.path.join('./bin/directions', '%s_directions.pt' % model_name)
    save_dict = torch.load(load_path)
    D = save_dict['D']
    print('Load D successfully.')
    print(D.shape)
    return D


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

    D0 = load_trained_directions(args.gan_model)
    D0 = torch.from_numpy(np.array(D0, dtype=np.float32)).cuda()
    d = D0[None, args.which_dir].view(1, dim_z, 1, 1)
    d = d / torch.sqrt(torch.sum(d * d, dim=1, keepdim=True))

    gan_type, image_type = args.gan_model.split("_")
    test_zs = prepare_test_z(args)

    S = torch.ones(size=[fmap_ch, fmap_ch], dtype=torch.float32).cuda() * 1e-4
    S.requires_grad = True

    if args.optim == 'Adam':
        optimizer = optim.Adam(params=[S], lr=args.lr)
    elif args.optim == 'SGD':
        optimizer = optim.SGD(params=[S], lr=args.lr)
    else:
        raise NotImplemented('We don\'t support this type of optimizer.')

    for iter in range(args.total_iterations):
        z = torch.from_numpy(np.random.randn(batch_size, args.dim_z, 1, 1).astype(np.float32)).cuda()
        alpha = 2 * (torch.rand(size=(batch_size, 1, 1, 1), dtype=torch.float32).cuda() - 0.5) * args.t_scale
        z_move = z + alpha * d

        S_masked = S * (1.0 - torch.eye(n=fmap_ch, m=fmap_ch).cuda())

        f_move = generator([z_move], which_block=args.layer, pre_model=True).detach()
        f_move_reshape = torch.reshape(torch.transpose(f_move, dim0=0, dim1=1), shape=[fmap_ch, -1])
        f_move_rec = torch.matmul(S_masked, f_move_reshape).reshape([fmap_ch, batch_size, fmap_size, fmap_size]).transpose(dim0=1, dim1=0)
        f_loss = torch.mean(torch.pow(f_move_rec - f_move, 2.0))
        x_move = generator([f_move], which_block=args.layer, post_model=True)
        x_move_rec = generator([f_move_rec], which_block=args.layer, post_model=True)
        d_loss = torch.mean(torch.pow(x_move_rec - x_move, 2.0))
        sparse_loss = torch.mean(torch.pow(S_masked, 2.0))

        optimizer.zero_grad()
        loss = args.beta_f * f_loss + args.beta_d * d_loss + args.beta_sparse * sparse_loss
        loss.backward()
        optimizer.step()

        if iter % args.report_value == 0:
            print('Iteration %d, loss=%.6f, f_loss=%.6f, d_loss=%.6f, sparse_loss=%.6f' %
                  (iter, float(loss.item()), float(f_loss.item()), float(d_loss.item()),
                   float(sparse_loss.item())))

        if iter % args.report_image == 0:
            n_samples = args.n_samples
            x_sample = x_move[0: n_samples]
            x_rec_sample = x_move_rec[0: n_samples]
            x_samples = torch.clamp(_tanh_to_sigmoid(torch.cat([x_sample, x_rec_sample], dim=0).detach().cpu()),
                                    min=0.0, max=1.0)
            torchvision.utils.save_image(x_samples, os.path.join(out_dir, 'rec_images_%d.png' % iter),
                                         nrow=n_samples)
            S_abs = torch.abs(S)
            S_normalize = (S_abs - S_abs.min()) / (S_abs.max() - S_abs.min())
            torchvision.utils.save_image(S_normalize.detach(), os.path.join(out_dir, 'matrix_pic_%d.png' % iter),
                                         nrow=1, normalize=True)
            S_val = np.abs(S.detach().cpu().numpy())
            S_val = thrC(S_val.T, args.alpha).T
            predict, L_val = post_proC(S_val, args.cluster_numbers, args.subspace_dimension, args.power)
            p_sum = [sum(predict == k) for k in range(1, args.cluster_numbers + 1, 1)]
            p_sum = np.array(p_sum)
            p_sort = np.argsort(p_sum)[::-1]
            predict_new = predict.copy()
            for i in range(1, args.cluster_numbers + 1, 1):
                predict_new[predict == (p_sort[i - 1] + 1)] = i
            predict = predict_new.copy()
            p_sum = [sum(predict == k) for k in range(1, args.cluster_numbers + 1, 1)]
            print(predict)
            print(p_sum)

            sel_idx = np.random.choice(test_zs.shape[0], size=[n_samples], replace=False)
            test_z = test_zs[sel_idx]
            test_f = generator([test_zs[sel_idx]], which_block=args.layer, pre_model=True).detach()
            features = test_f.detach().cpu().numpy()

            for class_i in range(1, args.cluster_numbers + 1, 1):
                ex_images = []
                int_images = []
                for ii in range(n_samples):
                    ex_rows = []
                    for jj in range(n_samples):
                        f_a = features[ii].copy()
                        f_a = f_a / np.sqrt(np.sum(f_a * f_a, axis=0, keepdims=True))
                        f_b = features[jj].copy()
                        f_b = f_b / np.sqrt(np.sum(f_b * f_b, axis=0, keepdims=True))
                        f_a[predict == class_i] = f_b[predict == class_i]
                        f_a = f_a.reshape((1, fmap_ch, fmap_size, fmap_size))
                        ex_rows.append(f_a)
                    ex_rows = np.concatenate(ex_rows, axis=0).astype(np.float32)
                    ex_rows = torch.from_numpy(ex_rows).cuda()
                    ex_ys = generator([ex_rows], which_block=args.layer, post_model=True).detach()
                    ex_ys = torch.clamp(_tanh_to_sigmoid(ex_ys), min=0.0, max=1.0)
                    ex_images.append(ex_ys)
                ex_images = torch.cat(ex_images, dim=0).detach().cpu().numpy()
                ex_images = torch.from_numpy(resize_images(ex_images, resize=args.resize))
                torchvision.utils.save_image(ex_images.detach().cpu(),
                                             os.path.join(out_dir, 'class_%d_iter_%d.png' % (class_i, iter)),
                                             nrow=n_samples)

                for interp_i in range(args.n_interps):
                    alpha2 = interp_i / (args.n_interps - 1)
                    alpha2 = args.v_scale * 2 * (alpha2 - 0.5)
                    test_z_move = test_z + alpha2 * d
                    test_f_move = generator([test_z_move], which_block=args.layer, pre_model=True).detach().cpu().numpy()
                    test_f_move = test_f_move / np.sqrt(np.sum(test_f_move * test_f_move, axis=1, keepdims=True))
                    test_f_copy = test_f.cpu().numpy().copy()
                    test_f_copy = test_f_copy / np.sqrt(np.sum(test_f_copy * test_f_copy, axis=1, keepdims=True))
                    test_f_copy[:, predict == class_i] = 0
                    test_f_copy[:, predict == class_i] += test_f_move[:, predict == class_i]
                    test_x_move = post_process_image(generator([torch.from_numpy(test_f_copy).cuda()], which_block=args.layer, post_model=True)).detach()
                    test_x_move = torch.from_numpy(resize_images(test_x_move.cpu().numpy(), args.resize))
                    int_images.append(test_x_move)

                int_images = torch.cat(int_images, dim=1).reshape(args.n_interps * n_samples, image_ch, args.resize, args.resize)
                torchvision.utils.save_image(int_images,
                                             os.path.join(out_dir, 'interp_class_%d_iter_%d.png' % (class_i, iter)),
                                             nrow=args.n_interps)
                print('subspace image: %d/%d' % (class_i, args.cluster_numbers))

            print('Saving transformer to disk. ')
            os.makedirs(os.path.join(out_dir, 'checkpoints'), exist_ok=True)
            save_path = os.path.join(out_dir, 'checkpoints', 'D_iter%d_which_layer%s.pt' % (iter + 1, args.layer))
            save_dicts = dict()
            save_dicts['S'] = S
            save_dicts['predict'] = predict
            torch.save(save_dicts, save_path)



if __name__ == '__main__':
    print('Working on applying transformer on unsupervised GAN discovery.')
    parser = argparse.ArgumentParser(description='GAN Transformer discovery.')
    parser.add_argument('-o', '--outputs', type=str, default='./TRAIN',
                        help='Directory to output corresponding images or loggings.')
    parser.add_argument('--exp_id', default='PGGANSubspace', type=str,
                        help='experiment prefix for easy debugging. ')
    # Parameters for Multi-Code GAN Inversion
    parser.add_argument('--inversion_type', default='PGGAN-Layerwise',
                        help='Inversion type, "PGGAN-Multi-Z" for Multi-Code-GAN prior.')
    # Generator Setting, Check models/model_settings for available GAN models
    parser.add_argument('--gan_model', default='pggan_celebahq', help='The name of model used.', type=str)
    parser.add_argument('--seed', default=0, help='The seed for the model. ', type=int)
    parser.add_argument('--which_class', default=239, type=int, help='The class of BigGAN.')
    parser.add_argument('--layer', default=8, type=int, help='which layer to plug transformer into.')
    parser.add_argument('--dim_z', default=512, type=int, help='experiment prefix for easy debugging. ')

    parser.add_argument('--total_iterations', default=50000, type=int, help='The total number of iterations.')
    parser.add_argument('--optim', default='Adam', type=str, help='The optimizer used.')
    parser.add_argument('--lr', default=1e-4, type=float, help='The learning rate of the optimizer.')
    parser.add_argument('--t_scale', default=6.0, type=float, help='The scale of scaling.')
    parser.add_argument('--v_scale', default=12.0, type=float, help='The scale of scaling.')
    parser.add_argument('--n_dirs', default=20, type=int, help='The number of directions.')
    parser.add_argument('--batch_size', default=16, type=int, help='The batch size of the input')

    parser.add_argument('--report_value', default=10, type=int, help='The step of reporting value.')
    parser.add_argument('--report_model', default=1000, type=int, help='The step of reporting value.')
    parser.add_argument('--report_image', default=5000, type=int, help='The step of reporting value.')

    parser.add_argument('--n_interps', default=11, type=int, help='The number of interpolation of visualization. ')
    parser.add_argument('--n_samples', default=6, type=int, help='The number of samples pf visualization. ')
    parser.add_argument('--n_dir_per_sheet', default=10, type=int, help='The number of samples pf visualization. ')
    parser.add_argument('--resize', default=256, type=int, help='The number of samples pf visualization. ')

    parser.add_argument('--beta_f', default=1.0, type=float, help='The weight for the feature loss. ')
    parser.add_argument('--beta_d', default=1.0, type=float, help='The weight for the data loss. ')
    parser.add_argument('--beta_sparse', default=0.5, type=float, help='the weight for the data loss. ')
    parser.add_argument('--which_dir', default=0, type=int, help='The index of the directions to use. ')
    parser.add_argument('--alpha', default=0.5, type=int, help='The alpha of the subspace clustering. ')

    parser.add_argument('--cluster_numbers', default=6, type=int, help='The cluster number of the subspace. ')
    parser.add_argument('--subspace_dimension', default=15, type=int, help='The susbpace dimension')
    parser.add_argument('--power', default=3.0, type=float, help='the power of the image')

    args = parser.parse_args()

    if args.n_dir_per_sheet > args.n_dirs:
        args.n_dir_per_sheet = args.n_dirs

    main(args)
