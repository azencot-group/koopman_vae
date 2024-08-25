import os, sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.general_utils import t_to_np, imshow_seqeunce


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
