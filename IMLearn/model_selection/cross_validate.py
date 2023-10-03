from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> \
        Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    x_rows_idx = np.arange(X.shape[0])
    folds_idx = np.array_split(x_rows_idx, cv, axis=0)
    folds = [X[idx.tolist()] for idx in folds_idx]

    train_score, validation_score = 0.0, 0.0
    for i in range(cv):
        x_i = np.concatenate(folds[:i] + folds[i + 1:], axis=0)
        y_i = y[np.concatenate(folds_idx[:i] + folds_idx[i + 1:])]
        h_i = deepcopy(estimator).fit(x_i, y_i)
        error_i = h_i.predict(x_i)
        train_score += scoring(y_i, error_i)
        validation_score += scoring(y[folds_idx[i]], h_i.predict(folds[i]))
    train_score /= cv
    validation_score /= cv
    return train_score, validation_score
