import torch
import argparse
import os
from model import KoopmanVAE, classifier_Sprite_all
import utils
import numpy as np

_PROJECT_WORKING_DIRECTORY = '/cs/cs_groups/azencot_group/inon/koopman_vae'

def define_args():
    parser = argparse.ArgumentParser()

    # Training parameters.
    parser.add_argument('--lr', default=1.e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--epochs', default=1000, type=int, help='number of epochs to train for')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--evl_interval', default=10, type=int, help='evaluate every n epoch')
    parser.add_argument('--sche', default='cosine', type=str, help='scheduler')
    parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')

    # Technical parameters.
    parser.add_argument('--dataset_path', default='/cs/cs_groups/azencot_group/datasets/SPRITES_ICML/datasetICML',
                        type=str, help='dataset to train')
    parser.add_argument('--dataset', default='Sprite', type=str, help='dataset to train')
    project_working_directory = '/cs/cs_groups/azencot_group/inon/koopman_vae'
    parser.add_argument('--models_during_training_dir', default=f'{project_working_directory}/models_during_training',
                        type=str,
                        help='base directory to save the models during the training.')
    parser.add_argument('--checkpoint_dir', default=f'{project_working_directory}/checkpoints', type=str,
                        help='base directory to save the last checkpoint.')
    parser.add_argument('--final_models_dir', default=f'{project_working_directory}/final_models', type=str,
                        help='base directory to save the final models.')

    # Architecture parameters.
    parser.add_argument('--frames', default=8, type=int, help='number of frames, 8 for sprite, 15 for digits and MUGs')
    parser.add_argument('--channels', default=3, type=int, help='number of channels in images')
    parser.add_argument('--image_height', default=64, type=int, help='the height / width of the input image to network')
    parser.add_argument('--image_width', default=64, type=int, help='the height / width of the input image to network')
    parser.add_argument('--lstm', type=str, choices=['encoder', 'decoder', 'both'],
                        default='both',
                        help='Specify the LSTM type: "encoder", "decoder", or "both" (default: "both")')

    parser.add_argument('--conv_dim', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--k_dim', default=40, type=int,
                        help='Dimensionality of the Koopman module.')
    parser.add_argument('--hidden_size_koopman_multiplier', default=2, type=int,
                        help='Multiplier for the k_dim in order to set the hidden size.')

    # Koopman layer implementation parameters.
    parser.add_argument('--static_size', type=int, default=7)
    parser.add_argument('--static_mode', type=str, default='ball', choices=['norm', 'real', 'ball'])
    parser.add_argument('--dynamic_mode', type=str, default='real',
                        choices=['strict', 'thresh', 'ball', 'real', 'none'])
    parser.add_argument('--ball_thresh', type=float, default=0.6)  # related to 'ball' dynamic mode
    parser.add_argument('--dynamic_thresh', type=float, default=0.5)  # related to 'thresh', 'real'
    parser.add_argument('--eigs_thresh', type=float, default=.5)  # related to 'norm' static mode loss

    # Loss parameters.
    parser.add_argument('--weight_kl_z', default=1.0, type=float, help='Weight of KLD between prior and posterior.')
    parser.add_argument('--weight_x_pred', default=1.0, type=float, help='Weight of Koopman matrix leading to right '
                                                                         'decoding.')
    parser.add_argument('--weight_z_pred', default=1.0, type=float, help='Weight of Koopman matrix leading to right '
                                                                         'transformation in time.')
    parser.add_argument('--weight_spectral', default=1.0, type=float, help='Weight of the spectral loss.')

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

    # Calculate the size of the hidden dim.
    args.hidden_dim = args.k_dim * args.hidden_size_koopman_multiplier

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
    index=0
    titles = ['Original image:', 'Reconstructed image koopman:', 'Original image:', 'Reconstructed image dropout:']
    utils.imshow_seqeunce([[x[index]], [koopman_recon_x[index]], [x[index]], [dropout_recon_x[index]]], titles=np.asarray([titles]).T, figsize=(50, 10), fontsize=50)

