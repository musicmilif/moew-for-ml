import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

from ..utils import sample_from_ball, get_instance_weights
from .. import models


class Alpha(object):
    def __init__(self, model_name, alpha_dim, n_iters=100, radius=2, **kwargs):
        self.model = models.__dict__[model_name](**kwargs)
        self.alpha_dim = alpha_dim
        self.n_iters = n_iters
        self.radius = radius
        self.kernel = RBF(
            length_scale=self.radius,
            length_scale_bounds=(self.radius * 1e-3, self.radius * 1e3),
        ) * ConstantKernel(1.0, (1e-3, 1e3))
        self.valid_metrics = []
        self.alphas = np.zeros(shape=(0, self.alpha_dim))

    def search_alpha(self, train_X, valid_X, train_y, valid_y):
        for idx in range(self.n_iters):
            if idx == 0:
                # Initial alpha by sampling from a ball
                next_alpha = sample_from_ball(1, self.alpha_dim, self.radius)
            else:
                # Use UCB to generate candidates
                candidates = sample_from_ball(1000, self.alpha_dim, self.radius)
                gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.1)
                gp.fit(self.alphas, self.valid_metrics)

                metrics_mle, metrics_std = gp.predict(candidates, return_std=True)
                metrics_lcbs = metrics_mle - 1.0 * metrics_std

                best_idx = np.argmin(metrics_lcbs)
                next_alpha = np.expand_dims(candidates[best_idx], axis=0)
                next_metric_ucb = metric_mles[best_idx] + 1.0 * metric_stds[best_idx]

        # Assign instance weights to each data then train model
        weights = get_instance_weights(embedding, next_alpha, train_y)
        clr = self.model.fit(train_X, train_y, sample_weight=weights)

        # Append alpha and metrics
        alphas = np.concatenate([alphas, alpha_batch])
        parallel_validation_metrics = [[clr.best_score["valid_1"]["auc"]]]
        validation_metrics.extend(parallel_validation_metrics)
