import torch
import socket
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataloader.sprite import Sprite

hostname = socket.gethostname()

# X, X, 64, 64, 3 -> # X, X, 3, 64, 64
def reorder(sequence):
    return sequence.permute(0, 1, 4, 2, 3)


def load_npy(path):
    with open(path, 'rb') as f:
        return np.load(f)

def load_dataset(dir_path):
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


import torch.nn as nn


def init_weights(model):
    for m in model.modules():
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
    if isinstance(X, np.ndarray):
        return X
    if X.dtype in [torch.float32, torch.float64]:
        X = X.detach().cpu().numpy()
    return X

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

@dataclass
class ModelMetrics:
    accuracy: float
    kl_divergence: float
    inception_score: float
    H_yx: float
    H_y: float
    action_accuracy: float
    skin_accuracy: float
    pants_accuracy: float
    top_accuracy: float
    hair_accuracy: float


