import os, sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import torch.nn.functional as F
from neptune.types import File
import pandas as pd
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from model import KoopmanVAE

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.general_utils import t_to_np, imshow_seqeunce, reorder


def get_unique_num(D, I, static_number):
    """ This function gets a parameter for number of unique components. Unique is a component with imag part of 0 or
        couple of conjugate couple """
    i = 0
    for j in range(static_number):
        index = len(I) - i - 1
        val = D[I[index]]

        if val.imag == 0:
            i = i + 1
        else:
            i = i + 2

    return i


def get_sorted_indices(D, pick_type):
    """ Return the indexes of the eigenvalues (D) sorted by the metric chosen by an hyperparameter"""

    if pick_type == 'real':
        I = np.argsort(np.real(D))
    elif pick_type == 'norm':
        I = np.argsort(np.abs(D))
    elif pick_type == 'ball' or pick_type == 'space_ball':
        Dr = np.real(D)
        Db = np.sqrt((Dr - np.ones(len(Dr))) ** 2 + np.imag(D) ** 2)
        I = np.argsort(Db)
    else:
        raise Exception("no such method")

    return I


def static_dynamic_split(D, I, pick_type, static_size):
    """Return the eigenvalues indexes of the static and dynamic factors"""

    static_size = get_unique_num(D, I, static_size)
    if pick_type == 'ball' or pick_type == 'space_ball':
        Is, Id = I[:static_size], I[static_size:]
    else:
        Id, Is = I[:-static_size], I[-static_size:]
    return Id, Is


def swap(model, X, Z, C, indices, static_size, plot=False, pick_type='norm'):
    """Swaps between two samples in a batch by the indices given
        :param model - the trained model to use in the swap
        :param X - the original samples, used for displaying the original
        :param Z - the latent representation
        :param C - The koopman matrix. Used to project into the subspaces
        :param indices - indexes for choosing a pair from the batch
        :param static_size - the number of eigenvalues that are dedicated to the static subspace.
         The rest will be for the dynamic subspace
        :param plot - plot with matplotlib
        :param pick_type - the metric to pick the static eigenvalues"""

    # swap a single pair in batch
    bsz, fsz = X.shape[0:2]
    device = X.device

    # swap contents of samples in indices
    X = t_to_np(X)
    Z = t_to_np(Z.reshape(bsz, fsz, -1))
    C = t_to_np(C)

    ii1, ii2 = indices[0], indices[1]

    S1, Z1 = X[ii1].squeeze(), Z[ii1].squeeze()
    S2, Z2 = X[ii2].squeeze(), Z[ii2].squeeze()

    # eig
    D, V = np.linalg.eig(C)
    U = np.linalg.inv(V)

    # project onto V
    Zp1, Zp2 = Z1 @ V, Z2 @ V

    # static/dynamic split
    I = get_sorted_indices(D, pick_type)
    Id, Is = static_dynamic_split(D, I, pick_type, static_size)

    # Plot the eigenvalues.
    eigenvalues_fig = plot_eigenvalues(D, Id, Is, plot=plot)

    # Zp* is in t x k
    Z1d, Z1s = Zp1[:, Id] @ U[Id], Zp1[:, Is] @ U[Is]
    Z2d, Z2s = Zp2[:, Id] @ U[Id], Zp2[:, Is] @ U[Is]

    Z1d2s = np.real(Z1d + Z2s)
    Z2d1s = np.real(Z2d + Z1s)

    # reconstruct
    S1d2s = model.decode(torch.from_numpy(Z1d2s.reshape((fsz, -1, 1, 1))).to(device))
    S2d1s = model.decode(torch.from_numpy(Z2d1s.reshape((fsz, -1, 1, 1))).to(device))

    # Get the swap image and visualize.
    titles = ['S{}'.format(ii1), 'S{}'.format(ii2), 'S{}d{}s'.format(ii1, ii2), 'S{}d{}s'.format(ii2, ii1)]
    swap_fig = imshow_seqeunce([[S1], [S2], [S1d2s.squeeze()], [S2d1s.squeeze()]],
                               plot=plot, titles=np.asarray([titles]).T)

    return eigenvalues_fig, swap_fig


def plot_eigenvalues(eigenvalues, Id, Is, plot=True):
    dynamic_eigenvalues = eigenvalues[Id]
    static_eigenvalues = eigenvalues[Is]

    # Create the plot
    fig = plt.figure(figsize=(8, 6))

    # Extract the real and imaginary parts of the eigenvalues
    for i, (eigvals_type, color) in enumerate([(dynamic_eigenvalues, "blue"), (static_eigenvalues, "red")]):
        real_parts = eigvals_type.real
        imaginary_parts = eigvals_type.imag

        plt.scatter(real_parts, imaginary_parts, color=color, marker='o', s=5)

    # Plot the unit circle
    unit_circle = plt.Circle((0, 0), 1, color='Black', fill=False, linestyle='-', label='Unit Circle')
    plt.gca().add_artist(unit_circle)

    # Add axis labels and a grid
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)

    # Set axis limits to show the entire unit circle
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)

    # Set the aspect ratio of the plot to be equal
    plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title('Eigenvalues on the Real-Imaginary Plane')

    # Show the plot
    if plot:
        plt.show()

    return fig


def swap_by_index(model, X, Z, C, indices, Sev_idx, Dev_idx, plot=False):
    """ Transfer specific features using static eigenvectors indices and dynamic eigenvectors indices
        Can be used for example to illustrate the multi-factor disentanglement
        indices - tuple of 2 samples
        Sev_idx - static eigenvectors indices
        Dev_idx - dynamic eigenvectors indices
        X - batch of samples
        Z - latent features of the batch """
    # swap a single pair in batch
    bsz, fsz = X.shape[0:2]
    device = X.device

    # swap contents of samples in indices
    X = t_to_np(X)
    Z = t_to_np(Z.reshape(bsz, fsz, -1))
    C = t_to_np(C)

    ii1, ii2 = indices[0], indices[1]
    S1, Z1 = X[ii1].squeeze(), Z[ii1].squeeze()
    S2, Z2 = X[ii2].squeeze(), Z[ii2].squeeze()

    # eig
    D, V = np.linalg.eig(C)
    U = np.linalg.inv(V)

    # project onto V
    Zp1, Zp2 = Z1 @ V, Z2 @ V

    # static/dynamic split
    Id, Is = Dev_idx, Sev_idx

    # Zp* is in t x k
    Z1d, Z1s = Zp1[:, Id] @ U[Id], Zp1[:, Is] @ U[Is]
    Z2d, Z2s = Zp2[:, Id] @ U[Id], Zp2[:, Is] @ U[Is]

    # swap
    Z1d2s = np.real(Z1d + Z2s)
    Z2d1s = np.real(Z2d + Z1s)

    # reconstruct
    S1d2s = model.decode(torch.from_numpy(Z1d2s.reshape((fsz, -1, 1, 1))).to(device))
    S2d1s = model.decode(torch.from_numpy(Z2d1s.reshape((fsz, -1, 1, 1))).to(device))

    # visualize
    if plot:
        titles = ['S{}'.format(ii1), 'S{}'.format(ii2), 'S{}d{}s'.format(ii1, ii2), 'S{}d{}s'.format(ii2, ii1),
                  'S{}s'.format(ii1), 'S{}s'.format(ii2), 'S{}d'.format(ii1), 'S{}d'.format(ii2)]
        imshow_seqeunce([[S1], [S2], [S1d2s.squeeze()], [S2d1s.squeeze()]],
                        plot=plot, titles=np.asarray([titles[:4]]).T)

    return S1d2s, S2d1s, Z1d2s, Z2d1s


def check_cls_specific_indexes(model, classifier, test_loader, target_indexes, fix=False, swap=False,
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
                                                                                                     fix=fix,
                                                                                                     filename="swap intervention",
                                                                                                     label_name=label_name)
            else:
                # @todo implement function for sampling features in model
                recon_x_sample, recon_x = model.forward_sample_specific_features_for_classification(target_indexes,
                                                                                                    x, fix=fix,
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


def check_cls_for_consistency_swap(model, classifier, test_loader, target_indexes, fix=False, label_name=""):
    for epoch in range(1):
        model.eval()
        mean_acc1_sample, mean_acc2_sample, mean_acc3_sample, mean_acc4_sample = 0, 0, 0, 0

        for i, data in enumerate(test_loader):
            x, label_A, label_D = reorder(data['images']), data['A_label'][:, 0], data['D_label'][:, 0]
            x, label_A, label_D = x.cuda(), label_A.cuda(), label_D.cuda()
            T = x.size(dim=1)

            # Get the permuted labels
            # @todo implement function for swapping features in model and return the permuted order of labels
            recon_x_sample, recon_x, label_A_perm = model.forward_swap_specific_features_for_classification(
                target_indexes, x,
                label_A, fix=fix,
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

                    acc1_frame = (
                            np.argmax(torch.stack(pred_skin2).permute(1, 0, 2).reshape(-1, 6).cpu(), axis=1).numpy()
                            == np.argmax(label_A[:, 0].cpu().numpy(), axis=1).repeat(8)).mean() \
                        if label_name == "skin" else \
                        (np.argmax(torch.stack(pred_skin2).permute(1, 0, 2).reshape(-1, 6).cpu(), axis=1).numpy()
                         == np.argmax(label_A_perm[:, 0].cpu().numpy(), axis=1).repeat(8)).mean()

                    acc2_frame = (
                            np.argmax(torch.stack(pred_top2).permute(1, 0, 2).reshape(-1, 6).cpu(), axis=1).numpy()
                            == np.argmax(label_A[:, 1].cpu().numpy(), axis=1).repeat(8)).mean() \
                        if label_name == "top" else (
                            np.argmax(torch.stack(pred_top2).permute(1, 0, 2).reshape(-1, 6).cpu(),
                                      axis=1).numpy() == np.argmax(label_A_perm[:, 1].cpu().numpy(), axis=1).repeat(
                        8)).mean()

                    acc3_frame = (np.argmax(torch.stack(pred_pant2).permute(1, 0, 2).reshape(-1, 6).cpu(),
                                            axis=1).numpy() == np.argmax(label_A[:, 2].cpu().numpy(), axis=1).repeat(
                        8)).mean() \
                        if label_name == "pant" else (
                            np.argmax(torch.stack(pred_pant2).permute(1, 0, 2).reshape(-1, 6).cpu(),
                                      axis=1).numpy() == np.argmax(label_A_perm[:, 2].cpu().numpy(), axis=1).repeat(
                        8)).mean()

                    acc4_frame = (np.argmax(torch.stack(pred_hair2).permute(1, 0, 2).reshape(-1, 6).cpu(),
                                            axis=1).numpy() == np.argmax(label_A[:, 3].cpu().numpy(), axis=1).repeat(
                        8)).mean() \
                        if label_name == "hair" else (
                            np.argmax(torch.stack(pred_hair2).permute(1, 0, 2).reshape(-1, 6).cpu(),
                                      axis=1).numpy() == np.argmax(label_A_perm[:, 3].cpu().numpy(), axis=1).repeat(
                        8)).mean()

                else:
                    acc1_frame = (
                            np.argmax(torch.stack(pred_skin2).permute(1, 0, 2).reshape(-1, 6).cpu(), axis=1).numpy()
                            == np.argmax(label_A_perm[:, 0].cpu().numpy(), axis=1).repeat(8)).mean() \
                        if label_name == "skin" else (
                            np.argmax(torch.stack(pred_skin2).permute(1, 0, 2).reshape(-1, 6).cpu(), axis=1).numpy()
                            == np.argmax(label_A[:, 0].cpu().numpy(), axis=1).repeat(8)).mean()

                    acc2_frame = (
                            np.argmax(torch.stack(pred_top2).permute(1, 0, 2).reshape(-1, 6).cpu(), axis=1).numpy()
                            == np.argmax(label_A_perm[:, 1].cpu().numpy(), axis=1).repeat(8)).mean() \
                        if label_name == "top" else (
                            np.argmax(torch.stack(pred_top2).permute(1, 0, 2).reshape(-1, 6).cpu(), axis=1).numpy()
                            == np.argmax(label_A[:, 1].cpu().numpy(), axis=1).repeat(8)).mean()

                    acc3_frame = (
                            np.argmax(torch.stack(pred_pant2).permute(1, 0, 2).reshape(-1, 6).cpu(), axis=1).numpy()
                            == np.argmax(label_A_perm[:, 2].cpu().numpy(), axis=1).repeat(8)).mean() \
                        if label_name == "pant" else (
                            np.argmax(torch.stack(pred_pant2).permute(1, 0, 2).reshape(-1, 6).cpu(), axis=1).numpy()
                            == np.argmax(label_A[:, 2].cpu().numpy(), axis=1).repeat(8)).mean()

                    acc4_frame = (
                            np.argmax(torch.stack(pred_hair2).permute(1, 0, 2).reshape(-1, 6).cpu(), axis=1).numpy()
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


def check_cls_for_consistency_gen(model, classifier, test_loader, target_indexes, fix=False, swap=False, label_name=""):
    for epoch in range(2):
        model.eval()
        mean_acc1_sample, mean_acc2_sample, mean_acc3_sample, mean_acc4_sample = 0, 0, 0, 0

        mean_global_acc1_sample, mean_global_acc2_sample, mean_global_acc3_sample, mean_global_acc4_sample = 0, 0, 0, 0
        mean_local_acc1_sample, mean_local_acc2_sample, mean_local_acc3_sample, mean_local_acc4_sample = 0, 0, 0, 0

        for i, data in enumerate(test_loader):
            x, label_A, label_D = reorder(data['images']), data['A_label'][:, 0], data['D_label'][:, 0]
            x, label_A, label_D = x.cuda(), label_A.cuda(), label_D.cuda()

            recon_x_sample, recon_x = model.forward_sample_specific_features_for_classification(target_indexes, x,
                                                                                                fix=fix,
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


def intervention_based_metrics(model: KoopmanVAE, classifier, test_loader, map_label_to_idx, label_to_name_dict,
                               verbose=False, should_log_files=False):
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
                                                                                       map_label_to_idx[label],
                                                                                       fix=True,
                                                                                       label_name=label_to_name_dict[
                                                                                           label])
        gen_df.loc[label_to_name_dict[label]] = [action_Acc, skin_Acc, pant_Acc, top_Acc, hair_Acc]

    table = tabulate(gen_df, headers='keys', tablefmt='psql')

    if verbose:
        print("generation swap")
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

    if model.trainer is not None:
        if should_log_files:
            # save dataframe as csv
            csv_buffer = StringIO()
            gen_df.to_csv(csv_buffer, index=False)
            model.trainer.logger.experiment['dataframes/intervention/generation_swap'].append(
                File.from_stream(csv_buffer, extension='csv'))

            model.trainer.logger.experiment['metrics/intervention/generation'].append(
                File.from_content(table, extension='txt'))

        model.log('metrics/intervention/generation_overall_score', 100 - final_score)

    #  --- Swap  evaluation !!! ---

    swap_df = pd.DataFrame(columns=['action', 'skin', 'pant', 'top', 'hair'],
                           index=['action', 'skin', 'pant', 'top', 'hair'])

    for label in map_label_to_idx:
        # print(f"Label {label_to_name_dict[label]}: {map_label_to_idx[label]}")
        action_Acc, skin_Acc, pant_Acc, top_Acc, hair_Acc = check_cls_specific_indexes(model, classifier, test_loader,
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

    if model.trainer is not None:
        if should_log_files:
            # save dataframe as csv
            csv_buffer = StringIO()
            swap_df.to_csv(csv_buffer, index=False)
            model.trainer.logger.experiment['dataframes/intervention/swap'].append(
                File.from_stream(csv_buffer, extension='csv'))

            model.trainer.logger.experiment['metrics/intervention/swap'].append(
                File.from_content(table, extension='txt'))

        model.log('metrics/intervention/swap_overall_score', 100 - final_score)


def consistency_metrics(model, classifier, test_loader, map_label_to_idx, label_to_name_dict, verbose=False,
                        should_log_files=False):
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

    if verbose:
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
            map_label_to_idx[label],
            fix=True, label_name=label_to_name_dict[label])
        c_df.loc[label_to_name_dict[label]] = [skin_Acc, top_Acc, pant_Acc, hair_Acc]

    table = (tabulate(c_df, headers='keys', tablefmt='psql'))

    if verbose:
        print("consistency swap measure with regular labels")
        print(table)

    # overall score of whole table. the absolute value between the values and 100
    swap_consistency_penalty = np.abs(100 - c_df)
    swap_consistency_score = 100 - swap_consistency_penalty.mean().mean()

    if verbose:
        print(f"Consistency swap score: {swap_consistency_score}")

    if model.trainer is not None:
        if should_log_files:
            # save dataframe as csv
            csv_buffer = StringIO()
            c_df.to_csv(csv_buffer, index=False)
            model.trainer.logger.experiment['dataframes/consistency/swap'].append(
                File.from_stream(csv_buffer, extension='csv'))
            model.trainer.logger.experiment['metrics/consistency/swap_consistency_per_feature'].append(
                File.from_content(table, extension='txt'))

        model.log('metrics/consistency/swap_overall_score', swap_consistency_score)

    #  --- consistency GENERATION evaluation ---

    loc_df = pd.DataFrame(columns=['skin', 'top', 'pant', 'hair'],
                          index=['skin', 'top', 'pant', 'hair'])

    for label in map_label_to_idx:
        if label == 'action':
            continue
        # print(f"Label {label_to_name_dict[label]}: {map_label_to_idx[label]}")
        glo_skin_Acc, glo_top_Acc, glo_pant_Acc, glo_hair_Acc, loc_skin_Acc, loc_top_Acc, loc_pant_Acc, loc_hair_Acc = \
            check_cls_for_consistency_gen(model, classifier, test_loader,
                                          map_label_to_idx[label],
                                          fix=True, label_name=label_to_name_dict[label])
        c_df.loc[label_to_name_dict[label]] = [glo_skin_Acc, glo_top_Acc, glo_pant_Acc, glo_hair_Acc]
        loc_df.loc[label_to_name_dict[label]] = [loc_skin_Acc, loc_top_Acc, loc_pant_Acc, loc_hair_Acc]

    table1 = (tabulate(c_df, headers='keys', tablefmt='psql'))

    if verbose:
        print("consistency global generation measure")
        print(table1)

    gen_consistency_penalty = np.abs(100 - c_df)
    global_consistency_score = 100 - gen_consistency_penalty.mean().mean()

    if verbose:
        print(f"Consistency global generation score: {global_consistency_score}")

    table2 = (tabulate(loc_df, headers='keys', tablefmt='psql'))

    if verbose:
        print("consistency local generation measure")
        print(table2)

    gen_consistency_penalty = np.abs(100 - loc_df)
    local_consistency_score = 100 - gen_consistency_penalty.mean().mean()

    if verbose:
        print(f"Consistency local generation score: {local_consistency_score}")

    if model.trainer is not None:
        if should_log_files:
            # save dataframe as csv
            csv_buffer = StringIO()
            c_df.to_csv(csv_buffer, index=False)
            model.trainer.logger.experiment['dataframes/consistency/global_gen'].append(
                File.from_stream(csv_buffer, extension='csv'))

            csv_buffer = StringIO()
            loc_df.to_csv(csv_buffer, index=False)
            model.trainer.logger.experiment['dataframes/consistency/local_gen'].append(
                File.from_stream(csv_buffer, extension='csv'))

            model.trainer.logger.experiment['metrics/consistency/global_generation_consistency_per_feature'].append(
                File.from_content(table1, extension='txt'))
            model.trainer.logger.experiment['metrics/consistency/local_generation_consistency_per_feature'].append(
                File.from_content(table2, extension='txt'))

        model.log('metrics/consistency/global_gen_overall_score', global_consistency_score)
        model.log('metrics/consistency/local_gen_overall_score', local_consistency_score)


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


def predictor_based_metrics(model: KoopmanVAE, ZL, labels, map_label_to_idx, label_to_name_dict, classifier_type,
                            verbose=False, should_log_files=False):
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

    if verbose:
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

        if model.trainer is not None:
            model.log(f"metrics/predictor/modularity_{label_to_name_dict[label]}", modularity_j)

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
    if verbose:
        print("modularity score for each feature:")
        print(modularity_table)
        print(f'overall modularity score = {modularity_score}\n')

    if model.trainer is not None:
        if should_log_files:
            # save dataframe as csv
            csv_buffer = StringIO()
            modularity_df.to_csv(csv_buffer, index=False)
            model.trainer.logger.experiment['dataframes/predictor/modularity_per_feature'].append(
                File.from_stream(csv_buffer, extension='csv'))
            model.trainer.logger.experiment['metrics/predictor/modularity_per_feature'].append(
                File.from_content(modularity_table, extension='txt'))

        model.log('metrics/predictor/overall_modularity', modularity_score)

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

        if model.trainer is not None:
            model.log(f"metrics/predictor/compactness_{label_to_name_dict[label]}", compactness_i)

        compactness_data.append({
            "feature": label_to_name_dict[label],
            "compactness_i": compactness_i
        })

    # compute average for compactness score
    compactness_score = np.mean([j["compactness_i"] for j in compactness_data])

    compactness_df = pd.DataFrame(compactness_data)
    compactness_table = tabulate(compactness_df, headers='keys', tablefmt='psql')

    if verbose:
        print("compactness score for each feature:")
        print(compactness_table)
        print(f'overall compactness score = {compactness_score}\n')

    if model.trainer is not None:
        if should_log_files:
            # save dataframe as csv
            csv_buffer = StringIO()
            compactness_df.to_csv(csv_buffer, index=False)
            model.trainer.logger.experiment['dataframes/predictor/compactness_per_feature'].append(
                File.from_stream(csv_buffer, extension='csv'))
            model.trainer.logger.experiment['metrics/predictor/compactness_per_feature'].append(
                File.from_content(compactness_table, extension='txt'))

        model.log('metrics/predictor/overall_compactness', compactness_score)

    # --- compute explicitness score ---
    # use losses from classifiers to compute explicitness score
    explicitness_data = []
    for label, _ in map_label_to_idx.items():
        train_explicitness_j = train_score[label]
        test_explicitness_j = test_score[label]

        # print(f"explicitness score for feature {label_to_name_dict[label]} = {explicitness_j}")
        if model.trainer is not None:
            model.log(f"metrics/predictor/train_explicitness_{label_to_name_dict[label]}", train_explicitness_j)
            model.log(f"metrics/predictor/test_explicitness_{label_to_name_dict[label]}", test_explicitness_j)

        explicitness_data.append({
            "feature": label_to_name_dict[label],
            "train_explicitness_j": train_explicitness_j,
            "test_explicitness_j": test_explicitness_j
        })

    explicitness_df = pd.DataFrame(explicitness_data)
    explicitness_table = tabulate(explicitness_df, headers='keys', tablefmt='psql')

    if verbose:
        print("explicitness score for each feature")
        print(explicitness_table)
        print(f'train explicitness score = {np.mean(train_score)}')
        print(f'test explicitness score = {np.mean(test_score)}\n')

    if model.trainer is not None:
        if should_log_files:
            # save dataframe as csv
            csv_buffer = StringIO()
            explicitness_df.to_csv(csv_buffer, index=False)
            model.trainer.logger.experiment['dataframes/predictor/explicitness_per_feature'].append(
                File.from_stream(csv_buffer, extension='csv'))
            model.trainer.logger.experiment['metrics/predictor/explicitness_per_feature:'].append(
                File.from_content(explicitness_table, extension='txt'))

        model.log('metrics/predictor/train_explicitness_score', np.mean(train_score))
        model.log('metrics/predictor/test_explicitness_score', np.mean(test_score))
