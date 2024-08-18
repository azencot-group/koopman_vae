from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, fields, is_dataclass
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import TYPE_CHECKING, Tuple

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataloader.sprite import Sprite

if TYPE_CHECKING:
    from model import KoopmanVAE


# X, X, 64, 64, 3 -> # X, X, 3, 64, 64
def reorder(sequence):
    return sequence.permute(0, 1, 4, 2, 3)


def load_npy(path: str):
    with open(path, 'rb') as f:
        return np.load(f)


def load_dataset(dir_path: str):
    # Load the train and the test data.
    X_train = load_npy(os.path.join(dir_path, "sprites_X_train.npy"))
    X_test = load_npy(os.path.join(dir_path, "sprites_X_test.npy"))
    A_train = load_npy(os.path.join(dir_path, "sprites_A_train.npy"))
    A_test = load_npy(os.path.join(dir_path, "sprites_A_test.npy"))
    D_train = load_npy(os.path.join(dir_path, "sprites_D_train.npy"))
    D_test = load_npy(os.path.join(dir_path, "sprites_D_test.npy"))

    print("finish loading!")

    train_data = Sprite(data=X_train, A_label=A_train,
                        D_label=D_train)
    test_data = Sprite(data=X_test, A_label=A_test,
                       D_label=D_test)

    return train_data, test_data


def clear_progressbar():
    # moves up 3 lines
    print("\033[2A")
    # deletes the whole line, regardless of character position
    print("\033[2K")
    # moves up two lines again
    print("\033[2A")


def init_weights(model: nn.Module):
    for m in model.modules():
        if not m.training:
            continue
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


def entropy_Hy(p_yx, eps=1E-16):
    p_y = p_yx.mean(axis=0)
    sum_h = (p_y * np.log(p_y + eps)).sum() * (-1)
    return sum_h


def entropy_Hyx(p, eps=1E-16):
    sum_h = (p * np.log(p + eps)).sum(axis=1)
    # average over images
    avg_h = np.mean(sum_h) * (-1)
    return avg_h


def inception_score(p_yx, eps=1E-16):
    # calculate p(y)
    p_y = np.expand_dims(p_yx.mean(axis=0), 0)
    # kl divergence for each image
    kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = np.mean(sum_kl_d)
    # undo the logs
    is_score = np.exp(avg_kl_d)
    return is_score


def KL_divergence(P, Q, eps=1E-16):
    kl_d = P * (np.log(P + eps) - np.log(Q + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = np.mean(sum_kl_d)
    return avg_kl_d


def t_to_np(X):
    if torch.is_tensor(X):
        return X.detach().cpu().numpy()
    return np.asarray(X)


def np_to_t(X, device='cuda'):
    if torch.cuda.is_available() is False:
        device = 'cpu'

    from numpy import dtype
    if X.dtype in [dtype('float32'), dtype('float64')]:
        X = torch.from_numpy(X.astype(np.float32)).to(device)
    return X


def save_checkpoint(optimizer, model, epoch, checkpoint_path):
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()},
        checkpoint_path)


def load_checkpoint(model, optimizer, checkpoint_path):
    try:
        print("Loading Checkpoint from '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Resuming Training From Epoch {}".format(start_epoch))
        return start_epoch
    except:
        print("No Checkpoint Exists At '{}'.Start Fresh Training".format(checkpoint_path))
        return 0


def imshow_seqeunce(DATA, plot=True, titles=None, figsize=(50, 10), fontsize=50):
    rc = 2 * len(DATA[0])
    fig, axs = plt.subplots(rc, 2, figsize=figsize)

    for ii, data in enumerate(DATA):
        for jj, img in enumerate(data):

            img = t_to_np(img)
            tsz, csz, hsz, wsz = img.shape
            img = img.transpose((2, 0, 3, 1)).reshape((hsz, tsz * wsz, -1))

            ri, ci = jj * 2 + ii // 2, ii % 2
            axs[ri][ci].imshow(img)
            axs[ri][ci].set_axis_off()
            if titles is not None:
                axs[ri][ci].set_title(titles[ii][jj], fontsize=fontsize)

    plt.subplots_adjust(wspace=.05, hspace=0)

    if plot:
        plt.show()


from dataclasses import is_dataclass, fields
import torch


def dataclass_to_dict(instance):
    """
    Convert a dataclass instance to a dictionary.
    """
    if not is_dataclass(instance):
        raise TypeError("Provided instance is not a dataclass")

    return {f.name: getattr(instance, f.name) for f in fields(instance)}


@dataclass
class ModelSubMetrics:
    skin_accuracy: float
    pants_accuracy: float
    top_accuracy: float
    hair_accuracy: float


@dataclass
class ModelMetrics:
    accuracy: float
    kl_divergence: float
    inception_score: float
    H_yx: float
    H_y: float


def calculate_metrics(model: KoopmanVAE,
                      classifier: nn.Module,
                      val_loader: DataLoader,
                      fixed: str = "content",
                      should_print: bool = False) -> tuple[None, None] | tuple[ModelMetrics, ModelSubMetrics]:
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

        if recon_x is None:
            return None, None

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
    acc *= 100
    kl = KL_divergence(pred2_all, pred1_all)

    nSample_per_cls = min([(label_gt == i).sum() for i in np.unique(label_gt)])
    index = np.hstack([np.nonzero(label_gt == i)[0][:nSample_per_cls] for i in np.unique(label_gt)]).squeeze()
    pred2_selected = pred2_all[index]

    IS = inception_score(pred2_selected)
    H_yx = entropy_Hyx(pred2_selected)
    H_y = entropy_Hy(pred2_selected)

    if should_print:
        print('acc: {:.2f}%, kl: {:.4f}, IS: {:.4f}, H_yx: {:.4f}, H_y: {:.4f}'.format(acc, kl, IS, H_yx, H_y))

    return ModelMetrics(accuracy=acc, kl_divergence=kl, inception_score=IS, H_yx=H_yx, H_y=H_y), \
        ModelSubMetrics(skin_accuracy=skin_acc, pants_accuracy=pant_acc, top_accuracy=top_acc, hair_accuracy=hair_acc)
