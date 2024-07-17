import torch
import sys
import os
import numpy as np

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import train_cdsvae
from model import KoopmanVAE
from utils.general_utils import reorder, set_seed_device
from utils.koopman_utils import swap


def define_args():
    # Define the arguments of the model.
    parser = train_cdsvae.define_args()

    parser.add_argument('--model_path', type=str, default=None, help='ckpt directory')
    parser.add_argument('--model_name', type=str, default=None)

    return parser


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
    model = KoopmanVAE(args).to(device=args.device)
    model.load_state_dict(saved_model, strict=False)
    model.eval()

    # Load the data.
    data = np.load('/cs/cs_groups/azencot_group/inon/koopman_vae/dataset/batch1.npy', allow_pickle=True).item()
    data2 = np.load('/cs/cs_groups/azencot_group/inon/koopman_vae/dataset/batch2.npy', allow_pickle=True).item()

    # Reorder the data
    x, label_A, label_D = reorder(data['images']), data['A_label'][:, 0], data['D_label'][:, 0]
    x2 = reorder(data2['images'])

    X = x.to(args.device)
    X2 = x2.to(args.device)

    # First batch.
    outputs = model(X)
    dropout_recon_x, koopman_matrix, z_post = outputs[-1], outputs[-3], outputs[-4]

    # Second batch.
    outputs2 = model(X2)
    dropout_recon_x2, koopman_matrix2, z_post2 = outputs[-1], outputs[-3], outputs[-4]

    # Perform the swap.
    indices=[0, 1]
    swap(model, dropout_recon_x, z_post, koopman_matrix, indices, args.static_size, plot=True)

# # --------------- Performing multi-factor swap --------------- #
# """ Sprites have 4 factors of variation in the static subspace(appearance of the character):
#     hair, shirt, skin and pants colors. In multi-factor swapping therein, we show how we swap each of the 4^4
#     combinations from a target character to a source character"""
#
# from koopman_utils import swap_by_index
#
# indices = (2, 12)
#
# # 1_1 skin
# static_indexes = [32, 33]
# dynamic_indexes = np.delete(np.arange(Ct_te.shape[0]), static_indexes)
# swap_by_index(model, X_dec_te2, Z2, Ct_te2, indices, static_indexes, dynamic_indexes, plot=True)
#
# # 1_2 hair (2, 12)
# static_indexes = [38, 39, 35]
# dynamic_indexes = np.delete(np.arange(Ct_te.shape[0]), static_indexes)
# swap_by_index(model, X_dec_te2, Z2, Ct_te2, indices, static_indexes, dynamic_indexes, plot=True)
#
# # 1_3 pants (2, 12)
# static_indexes = [28]
# dynamic_indexes = np.delete(np.arange(Ct_te.shape[0]), static_indexes)
# swap_by_index(model, X_dec_te2, Z2, Ct_te2, indices, static_indexes, dynamic_indexes, plot=True)
#
# # 1_4 top (2, 12)
# static_indexes = [34, 35, 29, 36]
# dynamic_indexes = np.delete(np.arange(Ct_te.shape[0]), static_indexes)
# swap_by_index(model, X_dec_te2, Z2, Ct_te2, indices, static_indexes, dynamic_indexes, plot=True)
#
# # 2_1 skin hair (2, 12)
# static_indexes = [38, 39, 35, 32, 33]
# dynamic_indexes = np.delete(np.arange(Ct_te.shape[0]), static_indexes)
# swap_by_index(model, X_dec_te2, Z2, Ct_te2, indices, static_indexes, dynamic_indexes, plot=True)
#
# # 2_2 skin pants (2, 12)
# static_indexes = [28, 32, 33]
# dynamic_indexes = np.delete(np.arange(Ct_te.shape[0]), static_indexes)
# swap_by_index(model, X_dec_te2, Z2, Ct_te2, indices, static_indexes, dynamic_indexes, plot=True)
#
# # 2_3 skin and top (2, 12)
# static_indexes = [34, 35, 29, 36, 37, 32, 33]
# dynamic_indexes = np.delete(np.arange(Ct_te.shape[0]), static_indexes)
# swap_by_index(model, X_dec_te2, Z2, Ct_te2, indices, static_indexes, dynamic_indexes, plot=True)
#
# # 2_4 hair and pants (2, 12)
# static_indexes = [39, 38, 35, 28]
# dynamic_indexes = np.delete(np.arange(Ct_te.shape[0]), static_indexes)
# swap_by_index(model, X_dec_te2, Z2, Ct_te2, indices, static_indexes, dynamic_indexes, plot=True)
#
# # 2_5 hair and top (2, 12)
# static_indexes = [38, 39, 34, 35, 31, 29]
# dynamic_indexes = np.delete(np.arange(Ct_te.shape[0]), static_indexes)
# swap_by_index(model, X_dec_te2, Z2, Ct_te2, indices, static_indexes, dynamic_indexes, plot=True)
#
# # 2_6 pants and top (2, 12)
# static_indexes = [34, 35, 36, 28]
# dynamic_indexes = np.delete(np.arange(Ct_te.shape[0]), static_indexes)
# swap_by_index(model, X_dec_te2, Z2, Ct_te2, indices, static_indexes, dynamic_indexes, plot=True)
#
# # 3_1 pants hair top (2, 12)
# static_indexes = [28, 38, 39, 35, 34]
# dynamic_indexes = np.delete(np.arange(Ct_te.shape[0]), static_indexes)
# swap_by_index(model, X_dec_te2, Z2, Ct_te2, indices, static_indexes, dynamic_indexes, plot=True)
#
# # 3_2 pants hair skin (2, 12)
# static_indexes = [32, 33, 38, 39, 28]
# dynamic_indexes = np.delete(np.arange(Ct_te.shape[0]), static_indexes)
# swap_by_index(model, X_dec_te2, Z2, Ct_te2, indices, static_indexes, dynamic_indexes, plot=True)
#
# # 3_3 pants skin top (2, 12)
# static_indexes = [29, 34, 36, 37, 28, 33, 32, 30, 24]
# dynamic_indexes = np.delete(np.arange(Ct_te.shape[0]), static_indexes)
# swap_by_index(model, X_dec_te2, Z2, Ct_te2, indices, static_indexes, dynamic_indexes, plot=True)
#
# # 2_4 skin top hair (2, 12)
# static_indexes = [32, 33, 34, 29, 36, 38, 39, 35]
# dynamic_indexes = np.delete(np.arange(Ct_te.shape[0]), static_indexes)
# swap_by_index(model, X_dec_te2, Z2, Ct_te2, indices, static_indexes, dynamic_indexes, plot=True)
#
# # full
# static_indexes = [33, 32, 37, 36, 39, 38, 35, 34, 31, 28]
# dynamic_indexes = np.delete(np.arange(Ct_te.shape[0]), static_indexes)
# swap_by_index(model, X_dec_te2, Z2, Ct_te2, indices, static_indexes, dynamic_indexes, plot=True)
