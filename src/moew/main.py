import .models
from .base import Alpha
from .utils.model import TrainEmbedding


class MOEW(object):
    def __init__(self, model_name, alpha_dim, **kwargs):
        """Inplement Metrics-Optimize Example Weights (MOEW)
        Args:
            model_name (string): name of the base model. 
            alpha_dim (int): the dimension for embedding.
        """
        self.model_name = model_name
        self.alpha_dim = alpha_dim

    def fit(self, train_X, valid_X, train_y, valid_y):
        # Train auto encoder
        self.auto_encoder = TrainEmbedding(
            train_X.shape[1], self.alpha_dim, self.num_classes
        )

self, model_name, alpha_dim, n_iters=100, radius=2):


        # Train Model
        self.model = models.__dict__[self.model_name](**kwargs)

    def predict(self, test_X):
        pass
