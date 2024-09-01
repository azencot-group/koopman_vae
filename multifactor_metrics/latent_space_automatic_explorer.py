from typing import List
from sklearn import ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import pearsonr
from utils.general_utils import load_checkpoint, reorder, t_to_np, load_dataset

def explore_latent_space_and_return_mapping(latent_codes: List[np.ndarray], attributes: List[List[int]],
                                            multifactor_classifier_type: str) -> List[List[int]]:
    """
    This function takes a list of latent codes, and for each label in the attribute list, it trains a classifier
    to predict the label from the latent code. For each latent code coordinate, it collects its relative importance
    score for each label. Then it assigns each coordinate to the label with the highest importance score.

    Args:
    - latent_codes: A list of tensors, each tensor is a latent code.
    - attributes: A list of lists, where each list contains the attributes of the corresponding latent code.
    - multifactor_classifier_type: The type of classifier to use, either 'linear' (Logistic Regression) or 'decision_tree'.

    Returns:
    - mapping: A list of lists where each sublist corresponds to the most important label for each latent code coordinate.
    """

    # Convert latent codes to a numpy array
    latent_codes_np = np.array(latent_codes)

    # Convert attributes to numpy array
    attributes_np = np.array(attributes)

    num_latent_dims = latent_codes_np.shape[1]
    num_labels = attributes_np.shape[1]

    importance_scores = np.zeros((num_latent_dims, num_labels))

    for label_idx in range(num_labels):
        # Select the classifier
        if multifactor_classifier_type == 'linear':
            classifier = LogisticRegression()
        elif multifactor_classifier_type == 'decision_tree':
            classifier = DecisionTreeClassifier(max_depth=8)
        elif multifactor_classifier_type == 'gradient_boost':
            classifier = ensemble.GradientBoostingClassifier()
        else:
            raise ValueError("Unsupported multifactor_classifier_type.")

        # Train the classifier on the latent codes to predict the current label
        classifier.fit(latent_codes_np, attributes_np[:, label_idx])
        # print the accuracy
        print(f"Accuracy for label {label_idx}: {classifier.score(latent_codes_np, attributes_np[:, label_idx])}")

        # Collect the importance scores
        if multifactor_classifier_type == 'linear':
            # For linear classifier, importance is the absolute value of the coefficients
            normalized_coefs = (np.abs(classifier.coef_) / np.sum(np.abs(classifier.coef_))).sum(axis=0)
            importance_scores[:, label_idx] = normalized_coefs
        elif multifactor_classifier_type == 'decision_tree' or multifactor_classifier_type == 'gradient_boost':
            normalized_coefs = classifier.feature_importances_ / np.sum(classifier.feature_importances_)
            # For decision tree, importance is the feature_importances_ attribute
            importance_scores[:, label_idx] = normalized_coefs

    # Assign each coordinate to the label with the highest importance score
    mapping = np.argmax(importance_scores, axis=1)

    # if all importance scores are zero, assign the coordinate -1
    mapping[np.all(importance_scores == 0, axis=1)] = -1

    # Convert the mapping to a list of lists
    return mapping.tolist()

def unsupervised_latent_mapping(latent_codes: List[np.ndarray], group_num=5) -> List[List[int]]:
    """
    Given a list of latent codes, the function calculates the correlation between each coordinate in the latent codes,
    and then groups the coordinates into `group_num` groups based on the correlation between them.

    Args:
    - latent_codes: A list of latent codes where each latent code is a numpy array.
    - group_num: The number of groups to cluster the coordinates into.

    Returns:
    - A list of lists where each sublist contains the indices of the coordinates belonging to the same group.
    """

    # Convert latent codes to a numpy array
    latent_codes_np = np.array(latent_codes)

    # Calculate the correlation matrix between the latent code coordinates
    num_coordinates = latent_codes_np.shape[1]
    correlation_matrix = np.zeros((num_coordinates, num_coordinates))

    for i in range(num_coordinates):
        for j in range(num_coordinates):
            if i == j:
                correlation_matrix[i, j] = 1.0  # Correlation with itself is 1
            else:
                correlation_matrix[i, j] = pearsonr(latent_codes_np[:, i], latent_codes_np[:, j])[0]

    # Convert the correlation matrix to a distance matrix (1 - correlation)
    distance_matrix = 1 - np.abs(correlation_matrix)

    # Perform hierarchical clustering based on the distance matrix
    clustering = AgglomerativeClustering(n_clusters=group_num, affinity='precomputed', linkage='average')
    cluster_labels = clustering.fit_predict(distance_matrix)

    # Group the coordinates based on the clustering labels
    groups = [[] for _ in range(group_num)]
    for coord_idx, group_label in enumerate(cluster_labels):
        groups[group_label].append(coord_idx)

    return groups


def extract_latent_code(model, x):
    # Transfer the example through the model.
    outputs = model(x)

    # If there was error during the run - return None.
    if outputs is None:
        return None

    # Get the Koopman matrix and the latent representation.
    Ct_te, Z = outputs[-3], outputs[-4]
    Z = t_to_np(Z.reshape(x.shape[0], model.frames, model.k_dim))
    C = t_to_np(Ct_te)

    # Calculate the Eigen-Decomposition.
    D, V = np.linalg.eig(C)

    # project onto V in order to create a single latent code scaler for each sample.
    ZL = np.real((Z @ V).mean(axis=1))

    return ZL


def get_mappings(ZL, labels, multifactor_exploration_type, multifactor_classifier_type):
    #  --- Latent exploration !!! ---
    # todo - maybe add more options or classifiers
    if multifactor_exploration_type == 'supervised':
        mappings = explore_latent_space_and_return_mapping(latent_codes=list(ZL),
                                                           attributes=[list(a) for a in list(labels)],
                                                           multifactor_classifier_type=multifactor_classifier_type)
    else:
        # todo - if using unsupervised - need to align the table with the labels
        mappings = unsupervised_latent_mapping(latent_codes=list(ZL), group_num=5)

    if multifactor_exploration_type == 'supervised':
        map_idx_to_label = {}
        for idx, label in enumerate(mappings):
            if label != -1:
                map_idx_to_label[idx] = label

        map_label_to_idx = {}
        for idx, label in map_idx_to_label.items():
            if label not in map_label_to_idx:
                map_label_to_idx[label] = []
            map_label_to_idx[label].append(idx)

    else:
        map_label_to_idx = {}
        for label, group in enumerate(mappings):
            map_label_to_idx[label] = group

    return map_label_to_idx
