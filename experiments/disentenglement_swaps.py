import torch
import sys
import os
import numpy as np
import lightning as L
from lightning.pytorch import seed_everything
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import train_cdsvae
from model import KoopmanVAE
from classifier import classifier_Sprite_all
from utils.general_utils import KL_divergence, entropy_Hyx, entropy_Hy, inception_score, reorder, ModelMetrics, \
    imshow_seqeunce
from datamodule.sprite_datamodule import SpriteDataModule
from utils.koopman_utils import swap


def define_args():
    # Define the arguments of the model.
    parser = train_cdsvae.define_args()

    parser.add_argument('--model_path', type=str, default=None, help='ckpt directory')
    parser.add_argument('--model_name', type=str, default=None)

    return parser


def calculate_metrics(model: KoopmanVAE,
                      classifier: nn.Module,
                      val_loader: DataLoader,
                      fixed: str = "content",
                      should_print: bool = False) -> ModelMetrics:
    e_values_action, e_values_skin, e_values_pant, e_values_top, e_values_hair = [], [], [], [], []
    mean_acc0, mean_acc1, mean_acc2, mean_acc3, mean_acc4 = 0, 0, 0, 0, 0
    mean_acc0_sample, mean_acc1_sample, mean_acc2_sample, mean_acc3_sample, mean_acc4_sample = 0, 0, 0, 0, 0
    pred1_all, pred2_all, label2_all = list(), list(), list()
    label_gt = list()

    for i, data in enumerate(val_loader):
        x, label_A, label_D = reorder(data['images']), data['A_label'][:, 0], data['D_label'][:, 0]
        x, label_A, label_D = x.to(model.device), label_A.to(model.device), label_D.to(model.device)

        if fixed == "content":
            recon_x_sample, recon_x = model.forward_fixed_content_for_classification(x)
        else:
            recon_x_sample, recon_x = model.forward_fixed_motion_for_classification(x)

        with torch.no_grad():
            pred_action1, pred_skin1, pred_pant1, pred_top1, pred_hair1 = classifier(x)
            pred_action2, pred_skin2, pred_pant2, pred_top2, pred_hair2 = classifier(recon_x_sample)
            pred_action3, pred_skin3, pred_pant3, pred_top3, pred_hair3 = classifier(recon_x)

            pred1 = F.softmax(pred_action1, dim=1)
            pred2 = F.softmax(pred_action2, dim=1)
            pred3 = F.softmax(pred_action3, dim=1)

        label1 = np.argmax(pred1.detach().cpu().numpy(), axis=1)
        label2 = np.argmax(pred2.detach().cpu().numpy(), axis=1)
        label3 = np.argmax(pred3.detach().cpu().numpy(), axis=1)
        label2_all.append(label2)

        pred1_all.append(pred1.detach().cpu().numpy())
        pred2_all.append(pred2.detach().cpu().numpy())
        label_gt.append(np.argmax(label_D.detach().cpu().numpy(), axis=1))

        # action
        acc0_sample = (np.argmax(pred_action2.detach().cpu().numpy(), axis=1)
                       == np.argmax(label_D.cpu().numpy(), axis=1)).mean()
        # skin
        acc1_sample = (np.argmax(pred_skin2.detach().cpu().numpy(), axis=1)
                       == np.argmax(label_A[:, 0].cpu().numpy(), axis=1)).mean()
        # pant
        acc2_sample = (np.argmax(pred_pant2.detach().cpu().numpy(), axis=1)
                       == np.argmax(label_A[:, 1].cpu().numpy(), axis=1)).mean()
        # top
        acc3_sample = (np.argmax(pred_top2.detach().cpu().numpy(), axis=1)
                       == np.argmax(label_A[:, 2].cpu().numpy(), axis=1)).mean()
        # hair
        acc4_sample = (np.argmax(pred_hair2.detach().cpu().numpy(), axis=1)
                       == np.argmax(label_A[:, 3].cpu().numpy(), axis=1)).mean()
        mean_acc0_sample += acc0_sample
        mean_acc1_sample += acc1_sample
        mean_acc2_sample += acc2_sample
        mean_acc3_sample += acc3_sample
        mean_acc4_sample += acc4_sample

    # Calculate the accuracy in percentage.
    action_acc = mean_acc0_sample / len(val_loader) * 100
    skin_acc = mean_acc1_sample / len(val_loader) * 100
    pant_acc = mean_acc2_sample / len(val_loader) * 100
    top_acc = mean_acc3_sample / len(val_loader) * 100
    hair_acc = mean_acc4_sample / len(val_loader) * 100

    if should_print:
        print(
            'Test sample: action_Acc: {:.2f}% skin_Acc: {:.2f}% pant_Acc: {:.2f}% top_Acc: {:.2f}% hair_Acc: {:.2f}% '.format(
                action_acc, skin_acc, pant_acc, top_acc, hair_acc))

    label2_all = np.hstack(label2_all)
    label_gt = np.hstack(label_gt)
    pred1_all = np.vstack(pred1_all)
    pred2_all = np.vstack(pred2_all)

    acc = (label_gt == label2_all).mean()
    kl = KL_divergence(pred2_all, pred1_all)

    nSample_per_cls = min([(label_gt == i).sum() for i in np.unique(label_gt)])
    index = np.hstack([np.nonzero(label_gt == i)[0][:nSample_per_cls] for i in np.unique(label_gt)]).squeeze()
    pred2_selected = pred2_all[index]

    IS = inception_score(pred2_selected)
    H_yx = entropy_Hyx(pred2_selected)
    H_y = entropy_Hy(pred2_selected)

    if should_print:
        print('acc: {:.2f}%, kl: {:.4f}, IS: {:.4f}, H_yx: {:.4f}, H_y: {:.4f}'.format(acc * 100, kl, IS, H_yx, H_y))

    return ModelMetrics(accuracy=acc, kl_divergence=kl, inception_score=IS, H_yx=H_yx, H_y=H_y,
                        action_accuracy=action_acc, skin_accuracy=skin_acc, pants_accuracy=pant_acc,
                        top_accuracy=top_acc, hair_accuracy=hair_acc)


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
                    titles=np.asarray([titles]).T, figsize=(50, 10), fontsize=50)


def swap_within_batch(x, first_idx: int, second_idx: int):
    # Transfer the data through the model.
    outputs = model(x)
    dropout_recon_x, koopman_matrix, z_post = outputs[-1], outputs[-3], outputs[-4]

    # Perform the swap.
    indices = [first_idx, second_idx]
    swap(model, x, z_post, koopman_matrix, indices, args.static_size, plot=True)


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
    swap_within_batch(x, 0, 1)

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
