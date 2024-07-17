import torch
import argparse
import os
from model import KoopmanVAE
from utils import general_utils
import numpy as np
import train_cdsvae

_PROJECT_WORKING_DIRECTORY = '/cs/cs_groups/azencot_group/inon/koopman_vae'

def define_args():
    # Define the arguments of the model.
    parser = train_cdsvae.define_args()

    parser.add_argument('--model_path', type=str, default=None, help='ckpt directory')
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
    if args.model_path is not None:
        saved_model = torch.load(args.model_path)
    else:
        raise ValueError('missing checkpoint')

    dtype = torch.cuda.FloatTensor

    # Create the model.
    model = KoopmanVAE(args)
    model.load_state_dict(saved_model, strict=False)
    model.eval()
    model = model.cuda()

    # Load the data.
    data = np.load('/cs/cs_groups/azencot_group/inon/koopman_vae/dataset/batch1.npy', allow_pickle=True).item()
    data2 = np.load('/cs/cs_groups/azencot_group/inon/koopman_vae/dataset/batch2.npy', allow_pickle=True).item()
    x, label_A, label_D = reorder(data['images']), data['A_label'][:, 0], data['D_label'][:, 0]

    X = x.to(args.device)

    # Pass the data through the model.
    z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, z_prior, z_post_koopman, z_post_dropout, Ct, koopman_recon_x, dropout_recon_x = model(X)

    # visualize
    index=1
    titles = ['Original image:', 'Reconstructed image koopman:', 'Original image:', 'Reconstructed image dropout:']
    utils.imshow_seqeunce([[x[index]], [koopman_recon_x[index]], [x[index]], [dropout_recon_x[index]]], titles=np.asarray([titles]).T, figsize=(50, 10), fontsize=50)

