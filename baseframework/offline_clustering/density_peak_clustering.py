import math

from numpy import sort
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from scipy.spatial.distance import pdist, squareform

from .. import utils


def _density_peak_clustering(X: np.ndarray, weights: np.ndarray | None, metric: str, distance_threshold: float | None,
                             gaussian: bool, min_rho: float | None, min_delta: float, hard_assign: bool, halo: bool,
                             halo_noise: bool, halo_avg: bool, allow_duplicate: bool):
    if weights is None:
        weights = np.ones(len(X))
    distance_matrix = get_distance_matrix(X, metric=metric)
    rho = get_rho(distance_matrix, distance_threshold, gaussian, weights)
    rho_order = np.argsort(rho, kind='stable')
    delta, nn = get_delta(distance_matrix, rho_order, rho, allow_duplicate)

    #automatic rho and delta selection undefined, use 2% as with automatic dc selection for min_rho (i.e., more than average within dc) and dc for min_delta (i.e., local peak)
    if min_rho is None:
        min_rho = math.floor(0.02 * len(X))

    labels = np.ones(X.shape[0]) * -1
    min_label = 0
    peaks = []
    pbs = {}
    for i in range(len(labels)):
        if rho[i] > min_rho and delta[i] > min_delta:
            labels[i] = min_label
            if halo:
                pbs[min_label] = 0
            min_label += 1
            peaks.append(i)
    for i in range(len(labels)):
        j = rho_order[-i - 1]  # go through in reverse order
        if labels[j] == -1:
            if not hard_assign or delta[j] <= distance_threshold:
                labels[j] = labels[nn[j]]  # use label of closest higher peak

    if halo:
        border = np.zeros(len(labels))
        for i in range(len(labels) - 1):
            for j in range(i + 1, len(labels)):
                if labels[j] != labels[i]:
                    if distance_matrix[i, j] <= distance_threshold:
                        border[i] = 1
                        border[j] = 1
                        if halo_avg:  # use average of rho of different cluster dps for halo cutoff
                            rho_avg = (rho[i] + rho[j]) / 2
                            if labels[i] != -1:
                                if rho_avg > pbs[labels[i]]:
                                    pbs[labels[i]] = rho_avg
                            if labels[j] != -1:
                                if rho_avg > pbs[labels[j]]:
                                    pbs[labels[j]] = rho_avg
        if not halo_avg:
            for i in range(len(labels)):
                if border[i] == 1 and labels[i] != -1:
                    if rho[i] > pbs[labels[i]]:
                        pbs[labels[i]] = rho[i]
        for i in range(len(labels)):
            if labels[i] != -1:
                if rho[i] <= pbs[labels[i]]:  # halo point
                    if halo_noise:
                        labels[i] = -1
                    else:
                        labels[i] = -1 * labels[i] - 2  # invert label, but skip -1 and 0: 0 -> -2, 1-> -3, etc.

    return np.unique(labels), labels, np.array(peaks), rho, delta





def get_rho(distance_matrix: np.ndarray, dc: float, gaussian: bool, weights: np.ndarray) -> np.ndarray:
    rho = np.zeros(len(distance_matrix))
    if gaussian:
        for i in range(len(distance_matrix)):
            for j in range(i+1, len(distance_matrix)):
                if dc == 0:
                    dc = 1e-8
                addon = math.exp(-(distance_matrix[i, j] / dc) * (distance_matrix[i, j] / dc))
                rho[i] += addon * weights[j]
                rho[j] += addon * weights[i]
            rho[i] += weights[i]
    else:
        for i in range(len(distance_matrix)):
            for j in range(i+1, len(distance_matrix)):
                if distance_matrix[i, j] <= dc:
                    rho[i] += weights[j]
                    rho[j] += weights[i]
            rho[i] += weights[i]
    return rho


def get_delta(distance_matrix: np.ndarray, rho_order: np.ndarray, rho: np.ndarray, allow_dupliate: bool) -> tuple[np.ndarray, np.ndarray]:
    max_dist = np.max(distance_matrix[rho_order[-1]])

    delta = np.ones(len(distance_matrix)) * max_dist
    nn = np.ones(len(distance_matrix), dtype=int) * -1
    k = 0
    for i in rho_order:
        for j in rho_order[k + 1:]:
            if distance_matrix[i, j] < delta[i] and (rho[i] < rho[j] or allow_dupliate):
                delta[i] = distance_matrix[i, j]
                nn[i] = j
        k += 1
    return delta, nn


def get_distance_matrix(X: np.ndarray, metric: str) -> np.ndarray:
    if metric == 'precomputed':
        assert X.shape[0] == X.shape[1], "Precompued distance matrix must be square"
        # from https://stackoverflow.com/questions/42908334/checking-if-a-matrix-is-symmetric-in-numpy
        assert np.allclose(X, X.T, rtol=1e-05, atol=1e-08), "Precomputed distance matrix must be symmetric"
        distance_matrix = X.copy()
    else:
        distance_matrix = utils.distance_calc(X, X, metric)
    return distance_matrix


class DensityPeakClustering(ClusterMixin, BaseEstimator):
    """
        Density Peak Clustering algorithm
        Implementation done referencing the original Matlab implementation of DPC from the supplementary materials of the original paper  under consideration of other ClustPy implementations

        Parameters
        ----------
        X : np.ndarray
            Given data set / alternatively: distance matrix if metric='precomputed'

        metric: str
            Metric used to compute the distances. Can be "euclidean", "l1", "l2", "manhattan", "cosine", or "precomputed" (see scipy.spatial.distance.pdist) (default: "precomputed")

        distance_threshold: float | None
            Value for dc if float; if None, dc is set to a value where the average neighborhood contains 2% of the full datatset (computed as 2% of all distances between any point, including self-distance) (default:None)

        gaussian: bool
            Whether to use a Gaussian kernel with variance dc for calculation of rho, otherwise uses count of points within radius dc (default: True)

        min_rho: float | None
            Determines the rho cutoff for peaks (only points with rho above min_rho can be peaks); if None, 2% of the dataset size is used (as expected average number of neighbors) (default: None)

        min_delta: float | None
            Determines the delta cutoff for peaks (only points with delta above min_delta can be peaks); if None, the value will be set to dc (default: None)

        hard_assign: bool
            Whether to limit shared cluster assignment across neighbors only to more dense data poits within the dc range (default: False)

        halo: bool
            Whether to determine a halo region for each cluster, where border regions where data points of different clusters are close to each other are determined and the density within the border region is used as a threshold to reassign less dense points within the cluster to the halo region (default: True)

        halo_noise: bool
            Whether the halo regions hould be considered noise or each halo should receive their own cluster depending on their original cluster (-1*clu_id -2) (default: True)

        halo_avg: bool
            Whether to use the average of the rhos of two points of different clusters within the radius dc of each other (i.e., border region points) when calculating the halo threshold; if False, uses the maximal rho within the halo region of the cluster instead (default: True)

        allow_duplicate: bool
            Whether to enforce that rho needs to be strictly smaller

        Attributes
        ----------
        n_clusters_ : int
            The identified number of clusters
        labels_ : np.ndarray
            The final labels
        density_peaks_ : np.ndarray
            The indices of density peaks of the clusters
        rho: np.ndarray
            The rho values for each data point
        delta: np.ndarray
            The delta values for each data point

        References
        ----------

        Alex Rodriguez and Alessandro Laio. "Clustering by fast search and find of density peaks."
        Science 344.6191, 1492 (2014)
        """

    def __init__(self, metric: str = 'precomputed', distance_threshold: float | None = None, gaussian=True,
                 min_rho: float | None = None, min_delta: float | None = None, hard_assign: bool = False,
                 halo: bool = True, halo_noise: bool = True, halo_avg: bool = True, allow_duplicate: bool = False):
        self.metric = metric
        self.distance_threshold = distance_threshold
        self.gaussian = gaussian
        self.min_rho = min_rho
        if min_delta is None:
            self.min_delta = distance_threshold
        else:
            self.min_delta = min_delta
        self.hard_assign = hard_assign
        self.halo = halo
        self.halo_noise = halo_noise
        self.halo_avg = halo_avg
        self.allow_duplicate = allow_duplicate

    def fit(self, X: np.ndarray, y: np.ndarray = None, sample_weight: np.ndarray = None) -> 'DensityPeakClustering':
        """
        Initiate the actual clustering process on the input data set.
        The resulting cluster labels will be stored in the labels_ attribute.

        Parameters
        ----------
        X : np.ndarray
            the given data set
        y : np.ndarray
            the labels (can be ignored)

        Returns
        -------
        self : DensityPeakClustering
            this instance of the Density Peak Clustering algorithm
        """
        n_clusters, labels, density_peaks, rho, delta = _density_peak_clustering(X, sample_weight, self.metric,
                                                                                 self.distance_threshold,
                                                                                 self.gaussian, self.min_rho,
                                                                                 self.min_delta,
                                                                                 self.hard_assign, self.halo,
                                                                                 self.halo_noise, self.halo_avg,
                                                                                 self.allow_duplicate)
        self.n_clusters_ = n_clusters
        self.labels_ = labels
        self.density_peaks_ = density_peaks
        self.rho_ = rho
        self.delta_ = delta
        return self

