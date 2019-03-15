# Author: Craig.C.Li

from scipy.stats import scoreatpercentile
from sklearn.neighbors import LocalOutlierFactor
from warnings import warn
import numpy as np


class IntrinsicDimensionality(LocalOutlierFactor):
    """ Unsupervised Outlier Detection using Intrinsic Dimensional Outlier
        Detection (IDOS)

        The anomaly score of each sample is called Intrinsic Dimensionality
        Outlier Score. It works better than LOF when data in high dimension.

        It performs same outlier ranking function of Local Outlier Factor(LOF)
        by replacing the local density with the local estimation of intrinsic
        dimensionality.

         References
        ----------: [1]Intrinsic Dimensional Outlier Detection in
        High-Dimensional Data, Jonathan von Brünken, Michael E. Houle, and
        Arthur Zimek (2015, Mar)

        Parameters: same as parameters as LOF
        """
    # overwrite fit function
    def fit(self, X, y=None):
        if not (0. < self.contamination <= .5):
            raise ValueError("contamination must be in (0, 0.5]")

        super(IntrinsicDimensionality, self).fit(X)

        n_samples = self._fit_X.shape[0]
        if self.n_neighbors > n_samples:
            warn("n_neighbors (%s) is greater than the "
                 "total number of samples (%s). n_neighbors "
                 "will be set to (n_samples - 1) for estimation."
                 % (self.n_neighbors, n_samples))
        self.n_neighbors_ = max(1, min(self.n_neighbors, n_samples - 1))

        self._distances_fit_X_, _neighbors_indices_fit_X_ = (
            self.kneighbors(None, n_neighbors=self.n_neighbors_))
        self._id_q = self._intrinsic_dimensionality(self._distances_fit_X_)
        ID_N = np.mean(self._id_q[_neighbors_indices_fit_X_], axis=1)
        self._id = self._id_q / ID_N
        id_ratios_array = (self._id[_neighbors_indices_fit_X_] /
                            self._id[:, np.newaxis])

        self.id_negative_outlier_factor_ = -np.mean(id_ratios_array, axis=1)

        self.id_threshold_ = -scoreatpercentile(
            -self.id_negative_outlier_factor_, 100. * (1. - self.contamination))

    #calculate intrinsic dimensionality score
    def _intrinsic_dimensionality(self, distances_X):
        if self.n_neighbors_ > 100:
            ID_x = -1 / (np.sum(np.log(distances_X / (distances_X[:, -1][:, None])), axis=1) / self.n_neighbors_)
            return ID_x
        else:
            n_samples = distances_X.shape[0]
            ID_q = np.zeros((n_samples, self.n_neighbors_))
            ID_q[:, 1] = -1 / (np.log(distances_X[:, 0] / distances_X[:, 1]) / 2)
            for j in range(2, self.n_neighbors_):
                ID_q[:, j] = (j / (j - 1) / (1 / ID_q[:, j - 1] + np.log(distances_X[:, j] / distances_X[:, j - 1])))
            ID_q[:, 0] += 1e-10
            w = np.zeros(self.n_neighbors_)
            for i in range(len(w)):
                w[i] = i + 1
            k = self.n_neighbors_ * (self.n_neighbors_ - 1)
            w = (2 * w - 2) / k
            ID_t = 1 / ID_q * w[None, :]
            ID = 1 / np.sum(ID_t, axis=1)
        return ID

    def _predict(self, X=None):
        if X is not None:
            is_inlier = np.ones(X.shape[0], dtype=int)
            is_inlier[self._decision_function(X) <= self.id_threshold_] = -1
        else:
            is_inlier = np.ones(self._fit_X.shape[0], dtype=int)
            is_inlier[self._id <= self.id_threshold_] = -1

        return is_inlier

    def _decision_function(self, X):
        distances_X, neighbors_indices_X = (
            self.kneighbors(X, n_neighbors=self.n_neighbors_))
        X_ID = self._intrinsic_dimensionality(distances_X)
        ID_N = np.mean(self._id_q[neighbors_indices_X], axis=1)
        id = X_ID / ID_N
        return id
