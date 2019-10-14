import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


def sample_from_ball(cnt=1, dim=1, radius=2):
    points = np.random.normal(size=(cnt, dim))
    points /= np.expand_dims(np.linalg.norm(points, axis=1), axis=1)
    scales = np.power(np.random.uniform(size=(cnt, 1)), 1 / dim)
    points *= scales * radius

    return points


def get_importance_sampling(train_y, valid_y, pred_type):
    if pred_type == "reg":
        n_bins = 100
        intervals = pd.cut(train_y, n_bins).cat.categories
        min_y, max_y = train_y.min(), train_y.max()
        valid_y = valid_y.clip(min_y, max_y)
        train_y = train_y.apply(intervals.get_loc)
        valid_y = valid_y.apply(intervals.get_loc)

    train_dist = train_y.value_counts()
    valid_dist = valid_y.value_counts()
    dist = (valid_dist / train_dist).fillna(0)

    def get_weight(y):
        if pred_type == "reg":
            y = np.clip(y, min_y, max_y)
            y = intervals.get_loc(y)
        elif pred_type == "cls":
            if y not in train_dist.index:
                return 0
        return dist.loc[y]

    return get_weight


def get_instance_weights(emb_x, alpha, y, c=1):
    """Get the MOEW weights for each instance
    Args:
        emb_x (np.ndarray): embedding vector of feature x.
        alpha (np.ndarray): an array of alpha.
        y (pd.Series): target.
        c (float): a constant for weights.
    """

    def softmax(x):
        return 1 / (1 + np.exp(-x))

    importance = pd.Series(c * y.apply(get_importance_sampling))
    weights = pd.DataFrame(softmax(emb_x.dot(alpha.T)))
    weights = (weights * importance).sum(axis=1).values

    return weights
