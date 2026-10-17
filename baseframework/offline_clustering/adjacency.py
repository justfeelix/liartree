
import numpy as np
from .. import utils

"""
    Adjacency-based clustering

    Clustering by merging similar instances

    Parameters
    ----------
    
    X: np.ndarray
        Either given dataset or distance matrix denoting distance between all instances in dataset (must be square and symmetric)
    
    metric: str
        Metric to use, "precomputed" if X is already a distance matrix
    
    status: np.ndarray
        Status of each instance, array containing ["dense", "normal", "noise_normal, "noise"] for each element in X
        "dense" instances propagate labels to similar 
        "normal" instances receive labels, but can not propagate them
        "noise_normal" instances are labeled as -1, but they can receive labels from a similar "dense" instance
        "noise" instances are always labeled as -1
        
    similarity_thresholds: np.ndarray | float 
        Either individual distance cutoff for each element in X when they are  similar to other instances, 
        or fixed similarity cutoff value for all elements;
        Similarity between two elements requires dist(Xi, Xj) <= similarity_thresholds[i] + similarity_thresholds[j]
        
    weight: np.ndarray
        Weight for each element in X, set to None if all instances are weighted the same (default: None)
    
    weight_threshold: float
        Weight threshold below which a cluster is considered to be a noise cluster (default: None)

"""
def adjacency_clustering(X: np.ndarray, metric: str, status: np.ndarray, similarity_thresholds: np.ndarray | float, weight: np.ndarray = None, weight_threshold: float = None):
    distance_matrix = get_distance_matrix(X, metric)

    assert X.shape[0] == status.shape[0], "Status needed for all elements in X"

    if type(similarity_thresholds) is np.ndarray:
        assert X.shape[0] == similarity_thresholds.shape[0], "Similarity thresholds needed for all elements in X"
    else:
        similarity_thresholds = np.full(X.shape[0], similarity_thresholds)

    if weight is None:
        weight = np.ones(X.shape[0])
    else:
        assert X.shape[0] == weight.shape[0], "Weight needed for all elements in X"

    threshold_pair = np.array([similarity_thresholds + similarity_thresholds[i] for i in range(len(X))])
    connected = np.where(distance_matrix <= threshold_pair, True, False)
    partitions = np.arange(len(X))  # get connected
    for i in range(len(partitions)):
        if status[i] != "noise":
            for j in range(i):
                if status[j] != "noise":
                    if connected[i, j]:
                        if partitions[i] != partitions[j]:
                            if status[i] == "dense" and status[j] == "dense":
                                key = min(partitions[i], partitions[j])
                                other_key = max(partitions[i], partitions[j])
                                partitions[np.where(partitions == other_key)[0]] = key
                            elif (status[i] == "normal" or status[i] == "noise_normal") and status[j] == "dense":
                                partitions[i] = partitions[j]
                            elif status[i] == "dense" and (status[j] == "normal" or status[j] == "noise_normal"):
                                partitions[j] = partitions[i]
        else:
            partitions[i] = -1

    for i in range(len(partitions)-1):
        if status[i] == "noise_normal":
            if partitions[i] == i:
                partitions[i] = -1

    if weight_threshold is not None:
        clusters_ids = np.unique(partitions)
        for cluster_id in clusters_ids:
            if cluster_id != -1:
                cluster_indices = np.where(partitions == cluster_id)[0]
                is_noise = np.sum(weight[cluster_indices]) < weight_threshold
                if is_noise:
                    partitions[cluster_indices] = -1

    partitions_noise = [-1]
    partitions_noise.extend(partitions)
    _, partitions_unique = np.unique(partitions_noise, return_inverse=True)
    partitions_unique -= 1
    return partitions_unique[1:]






def get_distance_matrix(X: np.ndarray, metric: str) -> np.ndarray:
    if metric == 'precomputed':
        assert X.shape[0] == X.shape[1], "Precompued distance matrix must be square"
        # from https://stackoverflow.com/questions/42908334/checking-if-a-matrix-is-symmetric-in-numpy
        assert np.allclose(X, X.T, rtol=1e-05, atol=1e-08), "Precomputed distance matrix must be symmetric"
        distance_matrix = X.copy()
    else:
        distance_matrix = utils.distance_calc(X, X, metric)
    return distance_matrix
