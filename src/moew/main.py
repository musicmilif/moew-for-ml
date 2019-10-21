from torch import nn

from . import models
from .sample_weights.base import Alpha
from .embedding.base import Embedding
from .embedding.loss import FocalLoss


class Moew(object):
    def __init__(self, model_name, alpha_dim):
        """Inplement Metrics-Optimize Example Weights (MOEW)
        Args:
            model_name (string): name of the base model. 
            alpha_dim (int): the dimension for embedding.
        """
        self.model_name = model_name
        self.alpha_dim = alpha_dim

    def fit(self, train_X, train_y, valid_X=None, valid_y=None):
        # Train auto encoder
        self.auto_encoder = Embedding(nn.MSELoss(), FocalLoss(), self.alpha_dim)
        auto_encoder.fit(train_X, train_y)

        # Train alpha

        # Train Model
        # self.model = models.__dict__[self.model_name](**kwargs)

    def predict(self, test_X):
        pass
