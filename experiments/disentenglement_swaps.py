import torch
import sys
import os
import numpy as np
from lightning.pytorch import seed_everything

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from input_parser.basic_input_parser import define_basic_args
from model import KoopmanVAE
from classifier import classifier_Sprite_all
from utils.general_utils import reorder, imshow_seqeunce, calculate_metrics
from datamodule.sprite_datamodule import SpriteDataModule
from utils.koopman_utils import swap


def define_args():
    # Define the basic arguments of the model.
    parser = define_basic_args()

    parser.add_argument('--model_path', type=str, default=None, help='ckpt directory')
    parser.add_argument('--model_name', type=str, default=None)

    return parser


def show_sampling(x, content_action_sampling_index, static_size=None):
    # Get the reconstruction and the fixed motion with sampled content.
    recon_x_sample_content, recon_x = model.forward_fixed_motion_for_classification(x, static_size=static_size)

    # Get the fixed content with sampled motion.
    recon_x_sample_motion, _ = model.forward_fixed_content_for_classification(x, static_size=static_size)

    # Set the titles and show the images.
    titles = ['Original image:', 'Reconstructed image :', 'Content Sampled:', 'Action Sampled:']
    imshow_seqeunce([[x[content_action_sampling_index]], [recon_x[content_action_sampling_index]],
                     [recon_x_sample_content[content_action_sampling_index]],
                     [recon_x_sample_motion[content_action_sampling_index]]],
                    titles=np.asarray([titles]).T)


def swap_within_batch(model, x, first_idx: int, second_idx: int, static_size, plot=True):
    # Transfer the data through the model.
    outputs = model(x)
    dropout_recon_x, koopman_matrix, z_post = outputs[-1], outputs[-3], outputs[-4]

    # Perform the swap.
    indices = [first_idx, second_idx]
    fig = swap(model, x, z_post, koopman_matrix, indices, static_size, plot=plot)

    return fig


if __name__ == '__main__':
    # Define the different arguments and parser them.
    parser = define_args()
    args = parser.parse_args()

    # Set seeds to all the randoms.
    seed_everything(args.seed)

    # Create and load the classifier.
    classifier = classifier_Sprite_all(args)
    loaded_dict = torch.load(args.classifier_path)
    classifier.load_state_dict(loaded_dict['state_dict'])
    classifier.cuda().eval()

    # Create the data module.
    data_module = SpriteDataModule(args.dataset_path, args.batch_size)
    data_module.setup("")
    validation_loader = data_module.val_dataloader()

    # Create the model.
    model = KoopmanVAE.load_from_checkpoint(args.model_path)
    model.eval()

    # Quantitative evaluation:
    calculate_metrics(model, classifier, validation_loader, should_print=True, fixed="content")
    calculate_metrics(model, classifier, validation_loader, should_print=True, fixed="motion")

    # Qualitative evaluation:
    # Get a new validation loader.
    validation_loader = data_module.val_dataloader()

    # Get a new batch from the validation loader.
    x = reorder(next(iter(validation_loader))['images']).to(model.device)

    # Sample content and action.
    show_sampling(x, content_action_sampling_index=1)

    # Swap between two batch images.
    swap_within_batch(model, x, first_idx=0, second_idx=1, static_size=args.static_size)

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
