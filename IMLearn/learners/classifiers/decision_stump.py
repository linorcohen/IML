from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a decision stump to the given data. That is, finds the best feature and threshold by
        which to split

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        n_samples, n_features = X.shape
        min_loss = np.inf
        for j in range(n_features):
            for sign in [-1, 1]:
                jth_column = X[:, j]
                threshold, threshold_loss = self._find_threshold(jth_column, y, sign)
                if threshold_loss < min_loss:
                    self.threshold_ = threshold
                    self.j_ = j
                    self.sign_ = sign
                    min_loss = threshold_loss

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict sign responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        result = np.zeros(X.shape[0], dtype=int)
        for idx, val in enumerate(X[:, self.j_]):
            if val < self.threshold_:
                result[idx] = -self.sign_
            else:
                result[idx] = self.sign_
        return result

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float,
    float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign`
        whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        sorted_index = np.argsort(values)
        loss = np.sum(np.abs(labels[sorted_index])[np.sign(labels[sorted_index]) == sign])
        loss = np.append(loss, loss - np.cumsum(labels[sorted_index] * sign))
        min_index = np.argmin(loss)
        min_loss = loss[min_index]
        sorted_values = values[sorted_index]
        if min_index == len(sorted_values):
            return np.inf, min_loss
        if min_index == 0:
            return -np.inf, min_loss
        return sorted_values[min_index], min_loss

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y_true=y, y_pred=self._predict(X))
