# This script is going to add subspace to each direction.

import os
import pickle as pkl

import argparse
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from derivable_models.derivable_generator import get_derivable_generator
from utils.file_utils import check_transformer_experiments_directory, get_generator_info, prepare_test_z, \
    create_subspace_distill_directory, restore_saved_checkpoints, get_sorted_subspace_prediction, thrC, post_proC, \
    get_dir_lists
from utils.image_precossing import _tanh_to_sigmoid, resize_images, _sigmoid_to_tanh, post_process_image
import torchvision


def forward_generator(generator, z, layer):
    f = generator([z], which_block=layer, pre_model=True)
    x = generator([f], which_block=layer, post_model=True)
    return x


def easy_forward(generator, z, layer, pre_model=False, post_model=False, batch_size=4):
    b, c, w, h = z.shape
    results = []
    for i in range(0, b, batch_size):
        z_i = z[i: min(i + batch_size, b)]
        x_i = generator([z_i], which_block=layer, pre_model=pre_model, post_model=post_model).detach()
        results.append(x_i)
    return torch.cat(results, dim=0)


def orthogonalize(directions, eps=1e-16):
    B, dim = directions.shape
    for i in range(B - 1):
        x1x2 = np.sum(directions[None, i] * directions[(i + 1):, :], axis=1, keepdims=True)
        x1x1 = np.sum(directions[None, i] * directions[None, i], axis=1, keepdims=True)
        a = x1x2 / (x1x1 + eps)
        directions[(i + 1):] = directions[(i + 1):, :] - a * directions[None, i]

    return directions


def main(args):
    eps = 1e-16

    dim_z = args.dim_z

    os.makedirs(args.outputs, exist_ok=True)
    out_dir, exp_name = check_transformer_experiments_directory(args, args.exp_id)
    out_dir2, exp_name2 = create_subspace_distill_directory(out_dir, args, args.exp_id2)

    print('Experiment name2: ', exp_name2)
    print('Output directory: ', out_dir2)
    generator = get_derivable_generator(args.gan_model, args.inversion_type, args)
    generator = torch.nn.DataParallel(generator)
    generator.cuda()

    fmap_size, fmap_ch, image_size, image_ch = get_generator_info(args, generator, which_layer=args.s_layer)

    # The layer to which the self-expressive layer attaches
    layer = args.s_layer
    # The batch size of training self-expressive layer.
    batch_size = args.s_batch_size
    test_zs = prepare_test_z(args)
    iterations = args.total_iterations

    D0 = restore_saved_checkpoints(os.path.join(out_dir, 'checkpoints'), args.restore_which_step).cuda()
    print(D0.shape)

    n_dirs = D0.shape[0]

    # for iter in range(iterations):
    # a_b_c+a:c+e

    if args.which_dirs == 'all':
        dir_lists = [x for x in range(n_dirs)]
    else:
        dir_lists = get_dir_lists(args.which_dirs)

    for dir_i in dir_lists:
        # create the self-expressive layer.
        S0 = torch.ones([fmap_ch, fmap_ch], dtype=torch.float32).cuda() * args.subspace_init_eps
        S0.requires_grad = True

        if args.optim == 'Adam':
            optimizer = optim.Adam(params=[S0], lr=args.lr)
        elif args.optim == 'SGD':
            optimizer = optim.SGD(params=[S0], lr=args.lr)
        else:
            raise NotImplemented('We don\'t support \'%s\' type of optimizer, please check it out. ' % args.optim)

        out_dir3 = os.path.join(out_dir2, 'direction_%d' % dir_i)
        os.makedirs(out_dir3, exist_ok=True)

        d = D0[dir_i].view(1, dim_z, 1, 1)

        print('Note: calculating Subspace for direction %d: ' % dir_i)
        for iter in range(iterations):
            z0 = torch.randn([batch_size, dim_z, 1, 1], dtype=torch.float32).cuda()
            alpha = _sigmoid_to_tanh(torch.rand(size=(batch_size, 1, 1, 1)).cuda()) * args.t_scale
            z = z0 + d * alpha
            f = generator([z], which_block=layer, pre_model=True)
            x = generator([f], which_block=layer, post_model=True)
            f_reshape = f.transpose(0, 1).reshape((fmap_ch, -1))
            S_k = S0 * (1.0 - torch.eye(n=fmap_ch, m=fmap_ch).cuda())
            f_rec = torch.matmul(S_k, f_reshape).reshape((fmap_ch, batch_size, fmap_size, fmap_size)).transpose(0, 1)
            x_rec = generator([f_rec], which_block=layer, post_model=True)

            optimizer.zero_grad()
            if args.sparse_type == 'L2':
                loss_sparse = torch.mean(torch.pow(S_k, 2.0))
            elif args.sparse_type == 'L1':
                loss_sparse = torch.mean(torch.abs(S_k))
            else:
                raise NotImplemented('Type not implemented.')

            loss_feature = torch.mean(torch.pow(f - f_rec, 2.0))
            loss_data = torch.mean(torch.pow(x - x_rec, 2.0))
            loss = args.wgt_f * loss_feature + args.wgt_x * loss_data + args.wgt_spa * loss_sparse
            loss.backward()
            optimizer.step()

            if iter % args.report_value == 0:
                procedure_remainder = 'direction (%d/%d) Iteration (%d/%d), ' % (dir_i, n_dirs, iter, iterations)
                loss_remainder = 'loss=%.4f, loss_feature=%.4f, loss_data=%.4f, loss_sparse=%.4f.' % \
                                 (float(loss.item()), float(loss_feature.item()), float(loss_data.item()),
                                  float(loss_sparse.item()))
                print(procedure_remainder + loss_remainder)

            if (iter + 1) % args.report_image == 0:
                n_samples = args.n_samples

                # save reconstruction images.
                x_show = resize_images(post_process_image(x).detach().cpu().numpy(), args.resize)
                x_show = torch.from_numpy(x_show)
                x_rec_show = resize_images(post_process_image(x_rec).detach().cpu().numpy(), args.resize)
                x_rec_show = torch.from_numpy(x_rec_show)
                x_show = torch.cat([x_show, x_rec_show], dim=0)
                rec_path = os.path.join(out_dir3, 'reconstruction_imgs_iter_%d_dir_%d.png' % (iter, dir_i))
                torchvision.utils.save_image(x_show, rec_path, nrow=x_rec_show.shape[0])
                print('Save reconstruction images to \'%s\'. ' % rec_path)
                # save reconstruction images.

                # visualize subspace.
                test_z = test_zs[np.random.choice(test_zs.shape[0], n_samples, replace=False)].view(n_samples, dim_z, 1, 1)

                S0_abs = torch.relu(S0)
                S0_val = S0_abs.detach().cpu().numpy()
                S_val = thrC(S0_val.copy().T, args.alpha).T
                predict, L_val = post_proC(S_val, args.n_subspaces, args.subspace_dim, args.power)

                predict, p_sum = get_sorted_subspace_prediction(predict, args)

                features = generator([test_z], which_block=layer, pre_model=True).detach()
                n_interps = args.n_interps

                for cls_i in range(1, args.n_subspaces + 1, 1):
                    exchanging_images = []
                    if p_sum[cls_i-1] > args.show_threshold:
                        for img_i in range(n_samples):
                            alpha = torch.linspace(-args.t_scale, args.t_scale, n_interps).view(n_interps, 1, 1, 1).cuda()
                            test_zi = test_z[None, img_i]
                            test_zi_m = test_zi + d * alpha

                            if args.same_density:
                                test_zi_norm = torch.sqrt(eps + torch.sum(test_zi * test_zi, dim=1, keepdim=True))
                                test_zi_m_norm = torch.sqrt(eps + torch.sum(test_zi_m * test_zi_m, dim=1, keepdim=True))
                                test_zi_m = test_zi_m / test_zi_m_norm * test_zi_norm

                            test_fi_m = easy_forward(generator, test_zi_m, layer, pre_model=True, post_model=False).detach()
                            test_fi_ms = [test_fi_m]
                            test_fi_m = test_fi_m.transpose(0, 1).cpu().numpy()
                            for img_j in range(n_samples):
                                feature_j = features[img_j].view(1, fmap_ch, fmap_size, fmap_size)
                                feature_j = torch.repeat_interleave(feature_j, repeats=n_interps, dim=0).transpose(0, 1).cpu().numpy()
                                feature_j[predict == cls_i] = test_fi_m[predict == cls_i]
                                test_fi_ms.append(torch.from_numpy(feature_j).transpose(0, 1).cuda())
                            test_fi_ms = torch.cat(test_fi_ms, dim=0).detach()
                            test_fi_ms = easy_forward(generator, test_fi_ms, layer, pre_model=False, post_model=True,
                                                      batch_size=4).detach()
                            test_fi_ms = resize_images(post_process_image(test_fi_ms).cpu().numpy(), args.resize)
                            exchanging_images.append(torch.from_numpy(test_fi_ms))

                        exchanging_images = torch.cat(exchanging_images, dim=0)
                        save_path = os.path.join(out_dir3, 'exchange_images_dir_%d_layer_%d_class_%d_iter_%d.png' %
                                                 (dir_i, layer, cls_i, iter))
                        torchvision.utils.save_image(exchanging_images, save_path, nrow=n_interps)
                        print('save image to %s. ' % out_dir3)
                        # the subspace mask of class_i, save it.
                        subspace_i = predict == cls_i
                        save_path = os.path.join(out_dir3, 'saved_subspace_dir_%d_layer_%d_class_%d_iter_%d.pkl' %
                                                 (dir_i, layer, cls_i, iter))
                        with open(save_path, 'wb') as file_out:
                            pkl.dump(subspace_i, file_out)
                            print('Save subspace images %s.' % save_path)


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
    parser.add_argument('--total_iterations', default=5000, type=int, help='The total number of iterations.')
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

    # configurations for subspace experiments.
    parser.add_argument('--exp_id2', default='SubspaceDiscovery', type=str, help='The number of subspaces to apply. ')
    parser.add_argument('--n_subspaces', default=6, type=int, help='The number of subspaces to apply. ')
    parser.add_argument('--s_layer', default=5, type=int, help='The layer to which the self-expressive layer attach. ')
    parser.add_argument('--s_batch_size', default=4, type=int, help='The batch size of the subspace learning. ')
    parser.add_argument('--subspace_init_eps', default=1e-4, type=int, help='The batch size of the subspace learning. ')
    parser.add_argument('--restore_which_step', default=-1, type=int, help='The step to restore. ')
    parser.add_argument('--sparse_type', default='L2', type=str, help='The type of sparsity to use. ')
    parser.add_argument('--wgt_f', default=1.0, type=float, help='The weights for feature space reconstruction loss. ')
    parser.add_argument('--wgt_x', default=1.0, type=float, help='The weights for data space reconstruction loss')
    parser.add_argument('--wgt_spa', default=0.1, type=float, help='The weights for sparsity term. ')

    # configuration for spectral clustering.
    parser.add_argument('--subspace_dim', type=int, default=12, help='The number of subspace dimension.')
    parser.add_argument('--power', type=float, default=3.0, help='The power of the alpha.')
    parser.add_argument('--alpha', type=float, default=0.2, help='The power of the alpha.')
    parser.add_argument('--which_dirs', type=str, default='all', help='The power of the alpha.')
    parser.add_argument('--show_threshold', type=int, default=40, help='The power of the alpha.')

    args = parser.parse_args()

    if args.n_dir_per_sheet > args.n_dirs:
        args.n_dir_per_sheet = args.n_dirs

    main(args)





