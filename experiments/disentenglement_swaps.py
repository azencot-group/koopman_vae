import torch
import sys
import os
import numpy as np
from lightning.pytorch import seed_everything
import torch.nn.functional as F

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import train_cdsvae
from model import KoopmanVAE
from utils.general_utils import reorder
from utils.koopman_utils import swap


def define_args():
    # Define the arguments of the model.
    parser = train_cdsvae.define_args()

    parser.add_argument('--model_path', type=str, default=None, help='ckpt directory')
    parser.add_argument('--model_name', type=str, default=None)

    return parser


def calculate_metrics(model, classifier, val_loader, fixed="content"):
    e_values_action, e_values_skin, e_values_pant, e_values_top, e_values_hair = [], [], [], [], []
    mean_acc0, mean_acc1, mean_acc2, mean_acc3, mean_acc4 = 0, 0, 0, 0, 0
    mean_acc0_sample, mean_acc1_sample, mean_acc2_sample, mean_acc3_sample, mean_acc4_sample = 0, 0, 0, 0, 0
    pred1_all, pred2_all, label2_all = list(), list(), list()
    label_gt = list()

    for i, data in enumerate(val_loader):
        x, label_A, label_D = reorder(data['images']), data['A_label'][:, 0], data['D_label'][:, 0]
        x, label_A, label_D = x.to(model.device), label_A.to(model.device), label_D.to(model.device)

        if args.type_gt == "action":
            recon_x_sample, recon_x = model.forward_fixed_action_for_classification(x)
        else:
            recon_x_sample, recon_x = model.forward_fixed_content_for_classification(x)

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

        def count_D(pred, label, mode=1):
            return (pred // mode) == (label // mode)

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

    print(
        'Test sample: action_Acc: {:.2f}% skin_Acc: {:.2f}% pant_Acc: {:.2f}% top_Acc: {:.2f}% hair_Acc: {:.2f}% '.format(
            mean_acc0_sample / len(val_loader) * 100,
            mean_acc1_sample / len(val_loader) * 100, mean_acc2_sample / len(val_loader) * 100,
            mean_acc3_sample / len(val_loader) * 100, mean_acc4_sample / len(val_loader) * 100))

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

    print('acc: {:.2f}%, kl: {:.4f}, IS: {:.4f}, H_yx: {:.4f}, H_y: {:.4f}'.format(acc * 100, kl, IS, H_yx, H_y))

    e_values_action.append(mean_acc0_sample / len(val_loader) * 100)
    e_values_skin.append(mean_acc1_sample / len(val_loader) * 100)
    e_values_pant.append(mean_acc2_sample / len(val_loader) * 100)
    e_values_top.append(mean_acc3_sample / len(val_loader) * 100)
    e_values_hair.append(mean_acc4_sample / len(val_loader) * 100)

    print(
        'final | acc: {:.2f}% | acc: {:.2f}% |acc: {:.2f}% |acc: {:.2f}% |acc: {:.2f}%'.format(np.mean(e_values_action),
                                                                                               np.mean(e_values_skin),
                                                                                               np.mean(e_values_pant),
                                                                                               np.mean(e_values_top),
                                                                                               np.mean(e_values_hair)))


if __name__ == '__main__':
    # Define the different arguments and parser them.
    parser = define_args()
    args = parser.parse_args()

    # Set seeds to all the randoms.
    seed_everything(args.seed)

    # Create the model.
    model = KoopmanVAE.load_from_checkpoint(args.model_path)
    model.eval()

    # Load the data.
    data = np.load('/cs/cs_groups/azencot_group/inon/koopman_vae/dataset/batch1.npy', allow_pickle=True).item()
    data2 = np.load('/cs/cs_groups/azencot_group/inon/koopman_vae/dataset/batch2.npy', allow_pickle=True).item()
    x = reorder(data['images']).to(model.device)

    # First batch.
    outputs = model(x)
    dropout_recon_x, koopman_matrix, z_post = outputs[-1], outputs[-3], outputs[-4]

    # Perform the swap.
    indices=[0, 1]
    swap(model, x, z_post, koopman_matrix, indices, args.static_size, plot=True)

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
