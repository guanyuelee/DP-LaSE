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

    # building skew symmetric D for orthogonal representation.
    D0 = torch.from_numpy(np.random.randn(dim_z, dim_z).astype(np.float32)).cuda()
    D0.requires_grad = True

    gan_type, image_type = args.gan_model.split("_")
    test_zs = prepare_test_z(args)

    if args.optim == 'Adam':
        optimizer = optim.Adam(params=[D0], lr=args.lr)
    elif args.optim == 'SGD':
        optimizer = optim.SGD(params=[D0], lr=args.lr)
    else:
        raise NotImplemented('We don\'t support this type of optimizer.')

    for iter in range(args.total_iterations):
        D0_skew_symmetric = 0.5 * (D0 - torch.transpose(D0, 0, 1))
        D = torch_expm(D0_skew_symmetric.unsqueeze(0))[:args.n_dirs]
        z = torch.randn(size=(args.batch_size, args.dim_z, 1, 1), dtype=torch.float32).cuda()
        y = torch.ones(size=(args.batch_size, 1), dtype=torch.int64) * args.which_class

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
        f = generator([z, y], which_block=args.layer, pre_model=True, truncation=args.truncation).reshape([batch_size, fmap_ch, fmap_size, fmap_size])
        f_t = generator([z_t, y], which_block=args.layer, pre_model=True, truncation=args.truncation).reshape([batch_size, fmap_ch, fmap_size, fmap_size])
        f_t2 = generator([z_t2, y], which_block=args.layer, pre_model=True, truncation=args.truncation).reshape([batch_size, fmap_ch, fmap_size, fmap_size])

        f = torch.reshape(f, shape=[batch_size, fmap_ch, -1])
        f_t = torch.reshape(f_t, shape=[batch_size, fmap_ch, -1])
        f_t2 = torch.reshape(f_t2, shape=[batch_size, fmap_ch, -1])

        f_diff = f_t - f                # [bs, 512, 16]
        f_diff2 = f_t2 - f              # [bs, 512, 16]
        f_diff_n = f_diff / torch.sqrt(eps + torch.sum(f_diff * f_diff, dim=2, keepdim=True))
        f_diff2_n = f_diff2 / torch.sqrt(eps + torch.sum(f_diff2 * f_diff2, dim=2, keepdim=True))

        positive_distance = torch.mean(torch.pow(f_diff, 2.0), dim=1, keepdim=True) + \
                            torch.mean(torch.pow(f_diff2, 2.0), dim=1, keepdim=True)
        positive_distance = torch.mean(positive_distance)
        negative_distance = torch.pow(torch.sum(f_diff_n * f_diff2_n, dim=2), 2.0) * (1 - is_same)
        negative_distance = torch.sum(negative_distance) / torch.sum(1 - is_same) / fmap_ch

        optimizer.zero_grad()
        loss = - args.wgt_pos * positive_distance + args.wgt_neg * negative_distance
        loss.backward()
        optimizer.step()

        if iter % args.report_value == 0:
            print(D0.grad)
            print('Iteration %d: loss = %.6f, positive_dist = %.6f, negative_dist = %.6f. ' %
                  (iter, float(loss.item()), float(positive_distance.item()), float(negative_distance.item())))

        if (iter+1) % args.report_image == 0:
            n_samples = args.n_samples
            test_z = test_zs[np.random.choice(test_zs.shape[0], n_samples, replace=False)].view(n_samples, dim_z, 1, 1)
            y = torch.ones(size=(args.n_samples, 1), dtype=torch.int64) * args.which_class
            n_interps = args.n_interps

            interp_sheet = []
            for dir_idx in range(args.n_dirs):
                interp_imgs = []
                d_i = D.detach()[dir_idx].view(1, dim_z, 1, 1)
                for int_i in range(n_interps):
                    alpha = (int_i / (n_interps - 1) - 0.5) * 2 * args.t_scale
                    test_z_p = density_preserving_traverse(test_z, d_i, alpha)
                    test_f_p = generator([test_z_p, y], which_block=args.layer, pre_model=True, truncation=args.truncation).detach()
                    x_cur = generator([test_z_p, y], which_block=args.layer, post_model=True,
                                      features=test_f_p, truncation=args.truncation).detach().cpu()
                    x_cur = post_process_image(x_cur).numpy()
                    x_cont = torch.from_numpy(resize_images(x_cur, resize=args.resize))
                    interp_imgs.append(x_cont)

                interp_imgs = torch.cat(interp_imgs, dim=3)
                interp_sheet.append(interp_imgs)

                if (dir_idx + 1) % args.n_dir_per_sheet == 0:
                    interp_sheet = torch.cat(interp_sheet, dim=0)
                    image_path = os.path.join(out_dir, 'save_image_iter_%d_D_%d-%d.png' % (iter, dir_idx - args.n_dir_per_sheet + 1, dir_idx))
                    torchvision.utils.save_image(interp_sheet, image_path, nrow=1, padding=0)
                    interp_sheet = []
                    print('Saving checkpoint images to %s...' % out_dir)

            print('Saving transformer to disk. ')
            os.makedirs(os.path.join(out_dir, 'checkpoints'), exist_ok=True)
            save_path = os.path.join(out_dir, 'checkpoints', 'D_iter%d_which_layer%s.pt' % (iter + 1, args.layer))
            save_dicts = dict()
            save_dicts['D0'] = D0.detach().cpu().numpy()
            save_dicts['D_skew'] = D0_skew_symmetric.detach().cpu().numpy()
            save_dicts['D'] = D.detach().cpu().numpy()
            torch.save(save_dicts, save_path)


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

    parser.add_argument('--n_interps', default=11, type=int, help='The number of interpolation of visualization. ')
    parser.add_argument('--n_samples', default=3, type=int, help='The number of samples pf visualization. ')
    parser.add_argument('--n_dir_per_sheet', default=10, type=int, help='The number of samples pf visualization. ')
    parser.add_argument('--resize', default=256, type=int, help='The number of samples pf visualization. ')
    parser.add_argument('--distances', default='1-5-10-15-20', type=str, help='The distances of shift. ')
    parser.add_argument('--num_tests', default=20, type=int, help='The distances of shift. ')

    parser.add_argument('--wgt_pos', default=1.0, type=float, help='The learning rate of the optimizer.')
    parser.add_argument('--wgt_neg', default=0.1, type=float, help='The learning rate of the optimizer.')
    parser.add_argument('--wgt_orth', default=1.0, type=float, help='The learning rate of the optimizer.')
    args = parser.parse_args()

    if args.n_dir_per_sheet > args.n_dirs:
        args.n_dir_per_sheet = args.n_dirs

    main(args)

    # CUDA_VISIBLE_DEVICES=$CUDA_NUMBER python maximum_traverse_for_BigGAN.py --wgt_pos=1.0 --wgt_neg=0.5 --wgt_orth=100 --n_dirs=1
