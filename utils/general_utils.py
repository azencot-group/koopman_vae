from __future__ import annotations

import torch
from io import StringIO
import neptune
from neptune.types import File
import pandas as pd
import torch.nn as nn
import os
import sys
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, fields, is_dataclass
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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


def load_data_for_explore_and_test(device: torch.device, dataset_dir_path: str):
    # TODO: All the constants here should be fixed when the version that iterates over all the validation data will be
    # TODO: implemented.
    # Load the train and val data.
    train_data, val_data = load_dataset(dataset_dir_path)

    # Reduce the size of test data into k samples.
    k = 512
    idx = np.random.choice(2664, k, replace=False)
    val_data.data = val_data.data[idx]
    val_data.A_label = val_data.A_label[idx]
    val_data.D_label = val_data.D_label[idx]
    val_data.N = k
    val_loader = DataLoader(val_data,
                            num_workers=4,
                            batch_size=512,  # 1024
                            shuffle=False,
                            drop_last=True,
                            pin_memory=True)
    x_test, label_A_test, label_D_test = reorder(torch.tensor(val_data.data)), val_data.A_label[:, 0], val_data.D_label[
                                                                                                       :, 0]
    x_test = x_test.to(device)

    # Transfer label_a and label_d from one hot encoding to a single label.
    label_A_test = np.argmax(label_A_test, axis=-1)
    label_D_test = np.argmax(label_D_test, axis=1)

    # Append label_d to label_a.
    labels_test = np.column_stack((label_A_test, label_D_test))

    return x_test, labels_test, val_loader


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

    return fig


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


def check_cls_specific_indexes(model, classifier, test_loader, run, run_type, target_indexes, fix=False, swap=False,
                               label_name=""):
    for epoch in range(1):
        # print("Epoch", epoch)
        model.eval()
        mean_acc0, mean_acc1, mean_acc2, mean_acc3, mean_acc4 = 0, 0, 0, 0, 0
        mean_acc0_sample, mean_acc1_sample, mean_acc2_sample, mean_acc3_sample, mean_acc4_sample = 0, 0, 0, 0, 0
        pred1_all, pred2_all, label2_all = list(), list(), list()
        label_gt = list()
        for i, data in enumerate(test_loader):
            x, label_A, label_D = reorder(data['images']), data['A_label'][:, 0], data['D_label'][:, 0]
            x, label_A, label_D = x.cuda(), label_A.cuda(), label_D.cuda()

            if swap:
                # @todo implement function for swapping features in model
                recon_x_sample, recon_x, _ = model.forward_swap_specific_features_for_classification(target_indexes, x,
                                                                                                     label_A,
                                                                                                     fix=fix, run=run,
                                                                                                     filename="swap intervention",
                                                                                                     label_name=label_name)
            else:
                # @todo implement function for sampling features in model
                recon_x_sample, recon_x = model.forward_sample_specific_features_for_classification(target_indexes,
                                                                                                    x, fix=fix,
                                                                                                    run=run,
                                                                                                    filename="generation intervention",
                                                                                                    label_name=label_name)

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
            acc1_sample = (np.argmax(pred_skin2[0].detach().cpu().numpy(), axis=1)
                           == np.argmax(label_A[:, 0].cpu().numpy(), axis=1)).mean()
            # pant
            acc2_sample = (np.argmax(pred_pant2[0].detach().cpu().numpy(), axis=1)
                           == np.argmax(label_A[:, 1].cpu().numpy(), axis=1)).mean()
            # top
            acc3_sample = (np.argmax(pred_top2[0].detach().cpu().numpy(), axis=1)
                           == np.argmax(label_A[:, 2].cpu().numpy(), axis=1)).mean()
            # hair
            acc4_sample = (np.argmax(pred_hair2[0].detach().cpu().numpy(), axis=1)
                           == np.argmax(label_A[:, 3].cpu().numpy(), axis=1)).mean()
            mean_acc0_sample += acc0_sample
            mean_acc1_sample += acc1_sample
            mean_acc2_sample += acc2_sample
            mean_acc3_sample += acc3_sample
            mean_acc4_sample += acc4_sample

        # print(
        #     'action_Acc: {:.2f}% skin_Acc: {:.2f}% pant_Acc: {:.2f}% top_Acc: {:.2f}% hair_Acc: {:.2f}% '.format(
        #         mean_acc0_sample / len(test_loader) * 100,
        #         mean_acc1_sample / len(test_loader) * 100, mean_acc2_sample / len(test_loader) * 100,
        #         mean_acc3_sample / len(test_loader) * 100, mean_acc4_sample / len(test_loader) * 100))

        label2_all = np.hstack(label2_all)
        label_gt = np.hstack(label_gt)
        pred1_all = np.vstack(pred1_all)
        pred2_all = np.vstack(pred2_all)

        acc = (label_gt == label2_all).mean()
        # kl = KL_divergence(pred2_all, pred1_all)

        nSample_per_cls = min([(label_gt == i).sum() for i in np.unique(label_gt)])
        index = np.hstack([np.nonzero(label_gt == i)[0][:nSample_per_cls] for i in np.unique(label_gt)]).squeeze()
        pred2_selected = pred2_all[index]

        # IS = inception_score(pred2_selected)
        # H_yx = entropy_Hyx(pred2_selected)
        # H_y = entropy_Hy(pred2_selected)

        # print('acc: {:.2f}%, kl: {:.4f}, IS: {:.4f}, H_yx: {:.4f}, H_y: {:.4f}'.format(acc * 100, 0, 0, 0, 0))
    action_Acc = mean_acc0_sample / len(test_loader) * 100
    skin_Acc = mean_acc1_sample / len(test_loader) * 100
    pant_Acc = mean_acc2_sample / len(test_loader) * 100
    top_Acc = mean_acc3_sample / len(test_loader) * 100
    hair_Acc = mean_acc4_sample / len(test_loader) * 100

    return action_Acc, skin_Acc, pant_Acc, top_Acc, hair_Acc

def check_cls_for_consistency_swap(model, classifier, test_loader, run, run_type, target_indexes, fix=False, label_name=""):
    for epoch in range(1):
        model.eval()
        mean_acc1_sample, mean_acc2_sample, mean_acc3_sample, mean_acc4_sample = 0, 0, 0, 0

        for i, data in enumerate(test_loader):
            x, label_A, label_D = reorder(data['images']), data['A_label'][:, 0], data['D_label'][:, 0]
            x, label_A, label_D = x.cuda(), label_A.cuda(), label_D.cuda()
            T = x.size(dim=1)


            # Get the permuted labels
            # @todo implement function for swapping features in model and return the permuted order of labels
            recon_x_sample, recon_x, label_A_perm = model.forward_swap_specific_features_for_classification(target_indexes, x,
                                                                                                label_A, fix=fix, run=run,
                                                                                                filename="swap consistency",
                                                                                                label_name=label_name)


            with torch.no_grad():
                acc1_frames, acc2_frames, acc3_frames, acc4_frames = [], [], [], []

                # Get classifier predictions for each sample and each frame
                pred_action1, pred_skin2, pred_top2, pred_pant2, pred_hair2 = classifier(recon_x_sample)

                # Determine which labels to compare based on fix and target_indexes
                if fix:
                    # if the feature in label name is fixed, compare it to the original label. compare the rest to the
                    # labels of the swapped example (with the label from the permuted indices)

                    acc1_frame = (np.argmax(torch.stack(pred_skin2).permute(1, 0, 2).reshape(-1, 6).cpu(), axis=1).numpy()
                                  == np.argmax(label_A[:, 0].cpu().numpy(), axis=1).repeat(8)).mean() \
                        if label_name == "skin" else \
                        (np.argmax(torch.stack(pred_skin2).permute(1, 0, 2).reshape(-1, 6).cpu(), axis=1).numpy()
                         == np.argmax(label_A_perm[:, 0].cpu().numpy(), axis=1).repeat(8)).mean()

                    acc2_frame = (np.argmax(torch.stack(pred_top2).permute(1, 0, 2).reshape(-1, 6).cpu(), axis=1).numpy()
                                  == np.argmax(label_A[:, 1].cpu().numpy(), axis=1).repeat(8)).mean() \
                        if label_name == "top" else (np.argmax(torch.stack(pred_top2).permute(1, 0, 2).reshape(-1, 6).cpu(),
                               axis=1).numpy() == np.argmax(label_A_perm[:, 1].cpu().numpy(), axis=1).repeat(8)).mean()

                    acc3_frame = (np.argmax(torch.stack(pred_pant2).permute(1, 0, 2).reshape(-1, 6).cpu(),
                               axis=1).numpy() == np.argmax(label_A[:, 2].cpu().numpy(), axis=1).repeat(8)).mean() \
                        if label_name == "pant" else (np.argmax(torch.stack(pred_pant2).permute(1, 0, 2).reshape(-1, 6).cpu(),
                               axis=1).numpy() == np.argmax(label_A_perm[:, 2].cpu().numpy(), axis=1).repeat(8)).mean()

                    acc4_frame = (np.argmax(torch.stack(pred_hair2).permute(1, 0, 2).reshape(-1, 6).cpu(),
                               axis=1).numpy() == np.argmax(label_A[:, 3].cpu().numpy(), axis=1).repeat(8)).mean() \
                        if label_name == "hair" else (np.argmax(torch.stack(pred_hair2).permute(1, 0, 2).reshape(-1, 6).cpu(),
                               axis=1).numpy() == np.argmax(label_A_perm[:, 3].cpu().numpy(), axis=1).repeat(8)).mean()

                else:
                    acc1_frame = (np.argmax(torch.stack(pred_skin2).permute(1, 0, 2).reshape(-1, 6).cpu(), axis=1).numpy()
                                  == np.argmax(label_A_perm[:, 0].cpu().numpy(),axis=1).repeat(8)).mean() \
                        if label_name == "skin" else (
                                np.argmax(torch.stack(pred_skin2).permute(1, 0, 2).reshape(-1, 6).cpu(), axis=1).numpy()
                                == np.argmax(label_A[:, 0].cpu().numpy(), axis=1).repeat(8)).mean()

                    acc2_frame = (np.argmax(torch.stack(pred_top2).permute(1, 0, 2).reshape(-1, 6).cpu(), axis=1).numpy()
                                  == np.argmax(label_A_perm[:, 1].cpu().numpy(), axis=1).repeat(8)).mean() \
                        if label_name == "top" else (
                                np.argmax(torch.stack(pred_top2).permute(1, 0, 2).reshape(-1, 6).cpu(), axis=1).numpy()
                                == np.argmax(label_A[:, 1].cpu().numpy(), axis=1).repeat(8)).mean()

                    acc3_frame = (np.argmax(torch.stack(pred_pant2).permute(1, 0, 2).reshape(-1, 6).cpu(), axis=1).numpy()
                                  == np.argmax(label_A_perm[:, 2].cpu().numpy(), axis=1).repeat(8)).mean() \
                        if label_name == "pant" else (
                                np.argmax(torch.stack(pred_pant2).permute(1, 0, 2).reshape(-1, 6).cpu(), axis=1).numpy()
                                == np.argmax(label_A[:, 2].cpu().numpy(), axis=1).repeat(8)).mean()

                    acc4_frame = (np.argmax(torch.stack(pred_hair2).permute(1, 0, 2).reshape(-1, 6).cpu(), axis=1).numpy()
                                  == np.argmax(label_A_perm[:, 3].cpu().numpy(), axis=1).repeat(8)).mean() \
                        if label_name == "hair" else (
                                np.argmax(torch.stack(pred_hair2).permute(1, 0, 2).reshape(-1, 6).cpu(), axis=1).numpy()
                                == np.argmax(label_A[:, 3].cpu().numpy(), axis=1).repeat(8)).mean()

                # Append frame accuracies
                acc1_frames.append(acc1_frame)
                acc2_frames.append(acc2_frame)
                acc3_frames.append(acc3_frame)
                acc4_frames.append(acc4_frame)

                # Average accuracies over all frames for the sample
                mean_acc1_sample += np.mean(acc1_frames)
                mean_acc2_sample += np.mean(acc2_frames)
                mean_acc3_sample += np.mean(acc3_frames)
                mean_acc4_sample += np.mean(acc4_frames)


        # Print original accuracies
        # print('Accuracies of frames: skin_Acc: {:.2f}% pant_Acc: {:.2f}% top_Acc: {:.2f}% hair_Acc: {:.2f}% '.format(
        #     mean_acc1_sample / len(test_loader) * 100, mean_acc2_sample / len(test_loader) * 100,
        #     mean_acc3_sample / len(test_loader) * 100, mean_acc4_sample / len(test_loader) * 100))


    # Calculate overall accuracies
    skin_Acc = mean_acc1_sample / len(test_loader) * 100
    top_Acc = mean_acc2_sample / len(test_loader) * 100
    pant_Acc = mean_acc3_sample / len(test_loader) * 100
    hair_Acc = mean_acc4_sample / len(test_loader) * 100

    # Return both original and permuted accuracies
    return skin_Acc, top_Acc, pant_Acc, hair_Acc

def check_cls_for_consistency_gen(model, classifier, test_loader, run, run_type, target_indexes, fix=False, swap=False, label_name=""):
    for epoch in range(2):
        model.eval()
        mean_acc1_sample, mean_acc2_sample, mean_acc3_sample, mean_acc4_sample = 0, 0, 0, 0

        mean_global_acc1_sample, mean_global_acc2_sample, mean_global_acc3_sample, mean_global_acc4_sample = 0, 0, 0, 0
        mean_local_acc1_sample, mean_local_acc2_sample, mean_local_acc3_sample, mean_local_acc4_sample = 0, 0, 0, 0

        for i, data in enumerate(test_loader):
            x, label_A, label_D = reorder(data['images']), data['A_label'][:, 0], data['D_label'][:, 0]
            x, label_A, label_D = x.cuda(), label_A.cuda(), label_D.cuda()

            recon_x_sample, recon_x = model.forward_sample_specific_features_for_classification(target_indexes, x,
                                                                                       fix=fix, run=run,
                                                                                       filename="generation consistency",
                                                                                        label_name=label_name)

            with torch.no_grad():
                _, pred_skin2, pred_top2, pred_pant2, pred_hair2 = classifier(recon_x_sample)

                skin_labels = np.argmax(torch.stack(pred_skin2).permute(1, 0, 2).reshape(-1, 6).cpu(), axis=1)
                top_labels = np.argmax(torch.stack(pred_top2).permute(1, 0, 2).reshape(-1, 6).cpu(), axis=1)
                pant_labels = np.argmax(torch.stack(pred_pant2).permute(1, 0, 2).reshape(-1, 6).cpu(), axis=1)
                hair_labels = np.argmax(torch.stack(pred_hair2).permute(1, 0, 2).reshape(-1, 6).cpu(), axis=1)

                # Find the most common label in each series
                most_common_label_skin = [torch.mode(s)[0] for s in np.split(skin_labels, len(skin_labels) // 8)]
                most_common_label_top = [torch.mode(s)[0] for s in np.split(top_labels, len(top_labels) // 8)]
                most_common_label_pant = [torch.mode(s)[0] for s in np.split(pant_labels, len(pant_labels) // 8)]
                most_common_label_hair = [torch.mode(s)[0] for s in np.split(hair_labels, len(hair_labels) // 8)]

                # Global consistency - compare each frame label to the most common label in the series
                global_acc1 = np.mean(skin_labels.numpy() == np.repeat(most_common_label_skin, 8))
                global_acc2 = np.mean(top_labels.numpy() == np.repeat(most_common_label_top, 8))
                global_acc3 = np.mean(pant_labels.numpy() == np.repeat(most_common_label_pant, 8))
                global_acc4 = np.mean(hair_labels.numpy() == np.repeat(most_common_label_hair, 8))

                mean_global_acc1_sample += global_acc1
                mean_global_acc2_sample += global_acc2
                mean_global_acc3_sample += global_acc3
                mean_global_acc4_sample += global_acc4

                # Local consistency - check how many 'switches' were made between consecutive frames
                local_acc1 = np.mean((np.argmax(torch.stack(pred_skin2).permute(1, 0, 2).cpu(), axis=2)[:, :-1] ==
                                      np.argmax(torch.stack(pred_skin2).permute(1, 0, 2).cpu(), axis=2)[:, 1:]).numpy())
                local_acc2 = np.mean((np.argmax(torch.stack(pred_top2).permute(1, 0, 2).cpu(), axis=2)[:, :-1] ==
                                      np.argmax(torch.stack(pred_top2).permute(1, 0, 2).cpu(), axis=2)[:, 1:]).numpy())
                local_acc3 = np.mean((np.argmax(torch.stack(pred_pant2).permute(1, 0, 2).cpu(), axis=2)[:, :-1] ==
                                      np.argmax(torch.stack(pred_pant2).permute(1, 0, 2).cpu(), axis=2)[:, 1:]).numpy())
                local_acc4 = np.mean((np.argmax(torch.stack(pred_hair2).permute(1, 0, 2).cpu(), axis=2)[:, :-1] ==
                                      np.argmax(torch.stack(pred_hair2).permute(1, 0, 2).cpu(), axis=2)[:, 1:]).numpy())

                mean_local_acc1_sample += local_acc1
                mean_local_acc2_sample += local_acc2
                mean_local_acc3_sample += local_acc3
                mean_local_acc4_sample += local_acc4

                # Average accuracies over all frames for the sample
                mean_acc1_sample += np.mean(global_acc1)
                mean_acc2_sample += np.mean(global_acc2)
                mean_acc3_sample += np.mean(global_acc3)
                mean_acc4_sample += np.mean(global_acc4)

        global_skin_Acc = mean_global_acc1_sample / len(test_loader) * 100
        global_top_Acc = mean_global_acc2_sample / len(test_loader) * 100
        global_pant_Acc = mean_global_acc3_sample / len(test_loader) * 100
        global_hair_Acc = mean_global_acc4_sample / len(test_loader) * 100

        local_skin_Acc = mean_local_acc1_sample / len(test_loader) * 100
        local_top_Acc = mean_local_acc2_sample / len(test_loader) * 100
        local_pant_Acc = mean_local_acc3_sample / len(test_loader) * 100
        local_hair_Acc = mean_local_acc4_sample / len(test_loader) * 100

        return global_skin_Acc, global_top_Acc, global_pant_Acc, global_hair_Acc, \
            local_skin_Acc, local_top_Acc, local_pant_Acc, local_hair_Acc

def intervention_based_metrics(model, classifier, test_loader, map_label_to_idx, label_to_name_dict, run=None,
                               verbose=False):
    """
       Computation of intervention based metrics.
       Evaluates accuracy of each feature after fixing the feature from the row and
        swapping or generating the others.

        * This function assumes that the model has an implementation of swap and generation swap

       @params:
       - model: model to evaluate
       - classifier: classifier to evaluate the features of the dataset,
                    since the classifier returns static labels for each frame we use the label of the first frame
       - test_loader: latent codes of test dataset examples
       - map_label_to_idx: mapping of labels and their corresponding subset of latent code dimensions
       - label_to_name_dict: mapping of label indices to names

        The tables that are printed represent the scores, the columns are the accuracies of each feature
        and the rows represent the feature that was swapped/generated in the example.

       """

    if verbose:
        print('--- start multifactor INTERVENTION BASED classification --- \n')

    gen_df = pd.DataFrame(columns=['action', 'skin', 'pant', 'top', 'hair'],
                          index=['action', 'skin', 'pant', 'top', 'hair'])

    # --- Swap generation evaluation ---
    for label in map_label_to_idx:
        # print(f"Label {label_to_name_dict[label]}: {map_label_to_idx[label]}")
        action_Acc, skin_Acc, pant_Acc, top_Acc, hair_Acc = check_cls_specific_indexes(model, classifier, test_loader,
                                                                                       run, 'action',
                                                                                       map_label_to_idx[label],
                                                                                       fix=True,
                                                                                       label_name=label_to_name_dict[
                                                                                           label])
        gen_df.loc[label_to_name_dict[label]] = [action_Acc, skin_Acc, pant_Acc, top_Acc, hair_Acc]

    if verbose:
        print("generation swap")

    table = tabulate(gen_df, headers='keys', tablefmt='psql')

    if verbose:
        print(table)

    # calculate final score
    # the values in the diagonal should be close to 100 (since they should have stayed the same)
    # the other values should be close to random (since they generated)
    scores = gen_df.values
    scores_mask = np.zeros_like(scores)
    random_floor_dict = {'skin': 16.66, 'pant': 16.66, 'top': 16.66, 'hair': 16.66, 'action': 11.11}
    for k in random_floor_dict.keys():
        scores_mask[:, gen_df.columns.get_loc(k)] = random_floor_dict[k]
    # add on scores mask diagonal 100
    np.fill_diagonal(scores_mask, 100)

    off_diagonal = np.where(~np.eye(scores.shape[0], dtype=bool))
    off_diag_score = np.abs(scores[off_diagonal] - scores_mask[off_diagonal]).mean()
    diag_score = np.mean(100 - np.diag(scores))
    final_score = (off_diag_score + diag_score) / 2

    if verbose:
        print(f"Final score of generation swap: {100 - final_score}")

    if run is not None:
        run["metrics/intervention/generation"].append(value=File.from_content(table, extension='txt'))
        run["metrics/intervention/generation_overall_score"].append(100 - final_score)

        # save dataframe as csv
        csv_buffer = StringIO()
        gen_df.to_csv(csv_buffer, index=False)
        run["dataframes/intervention/generation_swap"].append(File.from_stream(csv_buffer, extension='csv'))

    #  --- Swap  evaluation !!! ---

    swap_df = pd.DataFrame(columns=['action', 'skin', 'pant', 'top', 'hair'],
                           index=['action', 'skin', 'pant', 'top', 'hair'])

    for label in map_label_to_idx:
        # print(f"Label {label_to_name_dict[label]}: {map_label_to_idx[label]}")
        action_Acc, skin_Acc, pant_Acc, top_Acc, hair_Acc = check_cls_specific_indexes(model, classifier, test_loader,
                                                                                       run, 'action',
                                                                                       map_label_to_idx[label],
                                                                                       fix=True, swap=True,
                                                                                       label_name=label_to_name_dict[
                                                                                           label])
        swap_df.loc[label_to_name_dict[label]] = [action_Acc, skin_Acc, pant_Acc, top_Acc, hair_Acc]

    if verbose:
        print("swap")

    table = tabulate(swap_df, headers='keys', tablefmt='psql')

    if verbose:
        print(table)

    # calculate final score
    # the values in the diagonal should be close to 100 (since they should have stayed the same)
    # the other values should be close to random (since they were swapped)
    scores = swap_df.values
    scores_mask = np.zeros_like(scores)
    random_floor_dict = {'skin': 16.66, 'pant': 16.66, 'top': 16.66, 'hair': 16.66, 'action': 11.11}
    for k in random_floor_dict.keys():
        scores_mask[:, swap_df.columns.get_loc(k)] = random_floor_dict[k]
    # add on scores mask diagonal 100
    np.fill_diagonal(scores_mask, 100)

    off_diagonal = np.where(~np.eye(scores.shape[0], dtype=bool))
    off_diag_score = np.abs(scores[off_diagonal] - scores_mask[off_diagonal]).mean()
    diag_score = np.mean(100 - np.diag(scores))
    final_score = (off_diag_score + diag_score) / 2

    if verbose:
        print(f"Final score with same weight in and out of diagonal: {100 - final_score}")

    if run is not None:
        run["metrics/intervention/swap"].append(value=File.from_content(table, extension='txt'))
        run["metrics/intervention/swap_overall_score"].append(100 - final_score)

        # save dataframe as csv
        csv_buffer = StringIO()
        swap_df.to_csv(csv_buffer, index=False)
        run["dataframes/intervention/swap"].append(File.from_stream(csv_buffer, extension='csv'))

def consistency_metrics(model, classifier, test_loader, map_label_to_idx, label_to_name_dict, run=None):
    """
       Evaluate the consistency between frames in each example

      @params:
       - model: model to evaluate
       - classifier: classifier to evaluate the features of the dataset
       - test_loader: latent codes of test dataset examples
       - map_label_to_idx: mapping of labels and their corresponding subset of latent code dimensions
       - label_to_name_dict: mapping of label indices to names

        The tables that are printed represent the scores, the columns are the accuracies of each feature
        and the rows represent the feature that was swapped/generated in the example.

        The swap consistency metric checks the accuracy of the label of each frame with the original label
        (if the feature was swapped then with the label from the swapped example, if it was fixed then with the original label)

        (The following metrics get a high score on random models as well,
        more of a sanity check that the frames are similiar to one another with static features)

        The global consistency metric checks the accuracy of the most common label that is
        predicted throughout the time series, when generated some of the features

        The local generation consistency metric checks the accuracy of successive frames, that the label of a
        frame is the same of the label of the next frame

       """

    print('\n--- start multifactor CONSISTENCY classification --- ')

    c_df = pd.DataFrame(columns=['skin', 'top', 'pant', 'hair'],
                        index=['skin', 'top', 'pant', 'hair'])

    #  --- consistency SWAP evaluation ---
    for label in map_label_to_idx:
        if label == 'action':
            continue
        # print(f"Label {label_to_name_dict[label]}: {map_label_to_idx[label]}")
        skin_Acc, top_Acc, pant_Acc, hair_Acc = check_cls_for_consistency_swap(
                                                model, classifier, test_loader,
                                                run, 'action',
                                                map_label_to_idx[label],
                                                fix=True, label_name=label_to_name_dict[label])
        c_df.loc[label_to_name_dict[label]] = [skin_Acc, top_Acc, pant_Acc, hair_Acc]

    print("consistency swap measure with regular labels")
    table = (tabulate(c_df, headers='keys', tablefmt='psql'))
    print(table)

    # overall score of whole table. the absolute value between the values and 100
    swap_consistency_penalty = np.abs(100 - c_df)
    swap_consistency_score = 100 - swap_consistency_penalty.mean().mean()

    if run is not None:
        run["metrics/consistency/swap_consistency_per_feature"].append(value=File.from_content(table, extension='txt'))
        run["metrics/consistency/swap_overall_score"].append(swap_consistency_score)
        print(f"Consistency swap score: {swap_consistency_score}")

        # save dataframe as csv
        csv_buffer = StringIO()
        c_df.to_csv(csv_buffer, index=False)
        run["dataframes/consistency/swap"].append(File.from_stream(csv_buffer, extension='csv'))


    #  --- consistency GENERATION evaluation ---

    loc_df = pd.DataFrame(columns=['skin', 'top', 'pant', 'hair'],
                        index=['skin', 'top', 'pant', 'hair'])

    for label in map_label_to_idx:
        if label == 'action':
            continue
        # print(f"Label {label_to_name_dict[label]}: {map_label_to_idx[label]}")
        glo_skin_Acc, glo_top_Acc, glo_pant_Acc, glo_hair_Acc, loc_skin_Acc, loc_top_Acc, loc_pant_Acc, loc_hair_Acc = \
                                                check_cls_for_consistency_gen(model, classifier, test_loader,
                                                                              run, 'action',
                                                                              map_label_to_idx[label],
                                                                              fix=True, label_name=label_to_name_dict[label])
        c_df.loc[label_to_name_dict[label]] = [glo_skin_Acc, glo_top_Acc, glo_pant_Acc, glo_hair_Acc]
        loc_df.loc[label_to_name_dict[label]] = [loc_skin_Acc, loc_top_Acc, loc_pant_Acc, loc_hair_Acc]

    print("consistency global generation measure")
    table1 = (tabulate(c_df, headers='keys', tablefmt='psql'))
    print(table1)

    gen_consistency_penalty = np.abs(100 - c_df)
    global_consistency_score = 100 - gen_consistency_penalty.mean().mean()
    print(f"Consistency global generation score: {global_consistency_score}")


    print("consistency local generation measure")
    table2 = (tabulate(loc_df, headers='keys', tablefmt='psql'))
    print(table2)

    gen_consistency_penalty = np.abs(100 - loc_df)
    local_consistency_score = 100 - gen_consistency_penalty.mean().mean()
    print(f"Consistency local generation score: {local_consistency_score}")

    if run is not None:
        run["metrics/consistency/global_generation_consistency_per_feature"].append(value=File.from_content(table1, extension='txt'))
        run["metrics/consistency/global_gen_overall_score"].append(global_consistency_score)
        run["metrics/consistency/local_generation_consistency_per_feature"].append(value=File.from_content(table2, extension='txt'))
        run["metrics/consistency/local_gen_overall_score"].append(local_consistency_score)

        # save dataframe as csv
        csv_buffer = StringIO()
        c_df.to_csv(csv_buffer, index=False)
        run["dataframes/consistency/global_gen"].append(File.from_stream(csv_buffer, extension='csv'))

        csv_buffer = StringIO()
        loc_df.to_csv(csv_buffer, index=False)
        run["dataframes/consistency/local_gen"].append(File.from_stream(csv_buffer, extension='csv'))


def compute_importance_gbt(x_train, y_train, x_test, y_test, classifier_type):
    """
    This  is a helper function that trains a classifier for each feature to predict from latent codes.
    The importance weights and accuracy from each classifier are extracted and returned

    @params:
    - x_train: latent codes of train dataset examples
    - y_train: labels of train dataset examples
    - x_test: latent codes of test dataset examples
    - y_test: labels of train dataset examples

    @returns
    - importance_matrix: importance weights of each feature and latent code dimension
    - train_acc: accuracy between predictions of each classifier and the real label in train set
    - test_acc: accuracy between predictions of each classifier and the real label in test set

    """
    num_factors = y_train.shape[1]
    num_codes = x_train.shape[1]
    importance_matrix = np.zeros(shape=[num_codes, num_factors], dtype=np.float64)
    train_acc = []
    test_acc = []

    for i in range(num_factors):
        # Select the type of classifier
        if classifier_type == 'linear':
            classifier = LogisticRegression()
        elif classifier_type == 'decision_tree':
            classifier = DecisionTreeClassifier(max_depth=8)
        elif classifier_type == 'random_forest':
            classifier = RandomForestClassifier(n_estimators=100, max_depth=8)
        elif classifier_type == 'gradient_boost':
            classifier = ensemble.GradientBoostingClassifier()
        else:
            raise ValueError("Unsupported classifier_type.")

        # train the classifier
        classifier.fit(x_train, y_train[:, i])

        # extract importance weights for modularity
        if classifier_type == 'linear':
            importance_matrix[:, i] = np.abs(classifier.coef_)
        else:
            importance_matrix[:, i] = np.abs(classifier.feature_importances_)

        # extract accuracy for explicitness
        train_acc.append(np.mean(classifier.predict(x_train) == y_train[:, i]))
        test_acc.append(np.mean(classifier.predict(x_test) == y_test[:, i]))

    return importance_matrix, train_acc, test_acc


def predictor_based_metrics(ZL, labels, map_label_to_idx, label_to_name_dict, classifier_type, run=None):
    """
     This function takes a map between labels (features) and a subset of latent codes corresponding to the label.
      For each label it trains a classifier to predict the label from the latent code.
      From each classifier, it collects its relative importance scores for computing the modularity and compactness scores
      and collects the accuracies of the classifiers for the explicitness score.

    @params
    - ZL: latent codes
    - labels: labels of dataset
    - map_label_to_idx: Map between labels to the subset of latent code that describes them
    - label_to_name_dict: map of labels to the name of the feature they describe
    - classifier_type: The type of classifier to use, either 'gradient_boost', 'linear' (Logistic Regression) or 'decision_tree'.
    """

    print('\n--- start multifactor PREDICTOR BASED classification --- ')

    # split the latent codes to train and test
    Z_train, Z_test, labels_train, labels_test = train_test_split(ZL, labels, test_size=0.2, random_state=42)

    # get importance matrix and accuracy from classifiers for each feature
    R, train_score, test_score = compute_importance_gbt(Z_train, labels_train, Z_test, labels_test,
                                                        classifier_type)

    factors, Lambda = zip(*map_label_to_idx.items())

    # --- computation of modularity score ---

    M = len(factors)  # number of factors
    d = sum(len(idx) for idx in Lambda)  # d = number of code dimensions in latent code
    modularity_score = 0
    modularity_data = []

    for label, idx in map_label_to_idx.items():
        modularity_j = 1
        for i in range(M):

            # compute p_iJ
            importance_weights = sum(R[j][i] for j in idx)
            importance_weights_all_codes = sum(R[j][k] for j in idx for k in range(M))
            p = importance_weights / importance_weights_all_codes

            if p <= 0: p = 1e-7  # not to do log of non-positive number

            # compute D_J
            modularity_j += (p * (np.log(p) / np.log(M)))  # get log with base M

        if run is not None:
            run[f"metrics/predictor/modularity_{label_to_name_dict[label]}"] = modularity_j

        modularity_data.append({
            "feature": label_to_name_dict[label],
            "modularity_j": modularity_j
        })

        # compute rho_j
        importance_weights = sum(R[j][i] for j in idx for i in range(M))
        all_importance_weights = sum(R[k][i] for k in range(d) for i in range(M))
        rho = importance_weights / all_importance_weights

        # compute weighted sum for modularity score
        modularity_score = modularity_score + (rho * modularity_j)

    modularity_df = pd.DataFrame(modularity_data)
    modularity_table = tabulate(modularity_df, headers='keys', tablefmt='psql')
    print("modularity score for each feature:")
    print(modularity_table)

    print(f'overall modularity score = {modularity_score}\n')
    if run is not None:
        run["metrics/predictor/modularity_per_feature"].append(value=File.from_content(modularity_table, extension='txt'))
        run["metrics/predictor/overall_modularity"].append(modularity_score)

        # save dataframe as csv
        csv_buffer = StringIO()
        modularity_df.to_csv(csv_buffer, index=False)
        run["dataframes/predictor/modularity_per_feature"].append(File.from_stream(csv_buffer, extension='csv'))


    # --- computation of completeness score ---
    compactness_data = []

    for label, _ in map_label_to_idx.items():
        compactness_i = 1
        i = label
        for _, J in map_label_to_idx.items():

            # compute p_iJ
            importance_weights = sum(R[j][i] for j in J)
            importance_weights_all_dims = sum(R[k][i] for k in range(d))
            p = importance_weights / importance_weights_all_dims
            if p <= 0: p = 1e-7  # not to do log of non-positive number
            # compute C_J
            compactness_i += (p * (np.log(p) / np.log(d)))  # get log with base M

        if run is not None:
            run[f"metrics/predictor/compactness_{label_to_name_dict[label]}"] = compactness_i

        compactness_data.append({
            "feature": label_to_name_dict[label],
            "compactness_i": compactness_i
        })

    # compute average for compactness score
    compactness_score = np.mean([j["compactness_i"] for j in compactness_data])

    compactness_df = pd.DataFrame(compactness_data)
    compactness_table = tabulate(compactness_df, headers='keys', tablefmt='psql')
    print("compactness score for each feature:")
    print(compactness_table)

    print(f'overall compactness score = {compactness_score}\n')

    if run is not None:
        run["metrics/predictor/compactness_per_feature"].append(value=File.from_content(compactness_table, extension='txt'))
        run["metrics/predictor/overall_compactness"].append(compactness_score)

        # save dataframe as csv
        csv_buffer = StringIO()
        compactness_df.to_csv(csv_buffer, index=False)
        run["dataframes/predictor/compactness_per_feature"].append(File.from_stream(csv_buffer, extension='csv'))

    # --- compute explicitness score ---
    # use losses from classifiers to compute explicitness score
    explicitness_data = []
    for label, _ in map_label_to_idx.items():
        train_explicitness_j = train_score[label]
        test_explicitness_j = test_score[label]

        # print(f"explicitness score for feature {label_to_name_dict[label]} = {explicitness_j}")
        if run is not None:
            run[f"metrics/predictor/train_explicitness_{label_to_name_dict[label]}"] = train_explicitness_j
            run[f"metrics/predictor/test_explicitness_{label_to_name_dict[label]}"] = test_explicitness_j

        explicitness_data.append({
            "feature": label_to_name_dict[label],
            "train_explicitness_j": train_explicitness_j,
            "test_explicitness_j": test_explicitness_j
        })

    explicitness_df = pd.DataFrame(explicitness_data)
    explicitness_table = tabulate(explicitness_df, headers='keys', tablefmt='psql')
    print("explicitness score for each feature")
    print(explicitness_table)
    print(f'train explicitness score = {np.mean(train_score)}')
    print(f'test explicitness score = {np.mean(test_score)}\n')

    if run is not None:
        run["metrics/predictor/explicitness_per_feature:"].append(value=File.from_content(explicitness_table, extension='txt'))
        run["metrics/predictor/train_explicitness_score"].append(np.mean(train_score))
        run["metrics/predictor/test_explicitness_score"].append(np.mean(test_score))

        # save dataframe as csv
        csv_buffer = StringIO()
        explicitness_df.to_csv(csv_buffer, index=False)
        run["dataframes/predictor/explicitness_per_feature"].append(File.from_stream(csv_buffer, extension='csv'))
