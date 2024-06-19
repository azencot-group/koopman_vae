import torch
import argparse
import os
from model import CDSVAE, classifier_Sprite_all
import utils
import numpy as np


def define_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--log_dir', default='./logs', type=str, help='base directory to save logs')

    parser.add_argument('--dataset', default='Sprite', type=str, help='dataset to train')
    parser.add_argument('--frames', default=8, type=int, help='number of frames, 8 for sprite, 15 for digits and MUGs')
    parser.add_argument('--channels', default=3, type=int, help='number of channels in images')
    parser.add_argument('--image_width', default=64, type=int, help='the height / width of the input image to network')

    parser.add_argument('--f_rnn_layers', default=1, type=int, help='number of layers (content lstm)')
    parser.add_argument('--rnn_size', default=256, type=int, help='dimensionality of hidden layer')
    parser.add_argument('--f_dim', default=256, type=int, help='dim of f')
    parser.add_argument('--z_dim', default=32, type=int, help='dimensionality of z_t')
    parser.add_argument('--g_dim', default=128, type=int,
                        help='dimensionality of encoder output vector and decoder input vector')

    parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
    parser.add_argument('--sche', default='cosine', type=str, help='scheduler')

    parser.add_argument('--model_epoch', type=int, default=None, help='ckpt epoch')
    parser.add_argument('--models_dir', type=str, default=None, help='ckpt directory')
    parser.add_argument('--model_name', type=str, default=None)

    return parser


def reorder(sequence):
    return sequence.permute(0, 1, 4, 2, 3)


def set_seed_device(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Use cuda if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


if __name__ == '__main__':
    # Define the different arguments and parser them.
    parser = define_args()
    args = parser.parse_args()

    # set PRNG seed
    args.device = set_seed_device(args.seed)

    # Define the CUDA gpu.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Load the model.
    if args.models_dir is not None:
        saved_model = torch.load(os.path.join(args.models_dir, args.model_name))
        models_dir = args.models_dir
        args.models_dir = models_dir
    else:
        raise ValueError('missing checkpoint')

    # Set the log files.
    log = os.path.join(args.log_dir, 'log.txt')
    os.makedirs('%s/gen/' % args.log_dir, exist_ok=True)
    os.makedirs('%s/plots/' % args.log_dir, exist_ok=True)
    dtype = torch.cuda.FloatTensor

    # Create the model.
    cdsvae = CDSVAE(args)
    cdsvae.load_state_dict(saved_model, strict=False)
    cdsvae.eval()
    cdsvae = cdsvae.cuda()

    # Load the data.
    data = np.load('/cs/cs_groups/azencot_group/inon/koopman_vae/dataset/batch1.npy', allow_pickle=True).item()
    data2 = np.load('/cs/cs_groups/azencot_group/inon/SKD_for_kvae/dataset/batch2.npy', allow_pickle=True).item()
    x, label_A, label_D = reorder(data['images']), data['A_label'][:, 0], data['D_label'][:, 0]

    X = x.to(args.device)

    cdsvae.swap(X, 0, 1, plot=True)


