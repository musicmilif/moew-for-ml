import numpy as np
from pandas.api.types import is_numeric_dtype
import torch
from torch.utils.data import DataLoader

from .model import AutoEncoder
from .dataset import AutoEncoderDataset


class Embedding(object):
    def __init__(self, x_loss, y_loss, alpha_dim, lambda_=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.x_criterion = x_loss
        self.y_criterion = y_loss
        self.alpha_dim = alpha_dim
        self.lambda_ = lambda_

    def fit(self, X, y, n_epochs=100, batch_size=128, **kwargs):
        num_classes = 1 if is_numeric_dtype(y) else y.nunique()
        self.model = AutoEncoder(X.shape[1], self.alpha_dim, num_classes).to(
            self.device
        )
        dataset = AutoEncoderDataset(X, y)
        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1
        )
        optimizer = torch.optim.Adam(self.model.parameters(), **kwargs)

        for _ in range(n_epochs):
            self.model.train()
            for _, Xy in enumerate(data_loader):
                Xy.to(self.device)
                X_, y_ = self.model(Xy)

                x_loss = self.x_criterion(X_, Xy[:-1])
                y_loss = self.y_criterion(y_, Xy[-1])
                loss = lambda_ * x_loss + (1 - lambda_) * y_loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    def get_embbed(self, X, y, batch_size=128):
        emb_Xy = np.zeros(shape=(0, self.model.hidden_dim))
        dataset = AutoEncoderDataset(X, y)
        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1
        )

        self.model.eval()
        with torch.no_grad():
            for _, Xy in enumerate(data_loader):
                Xy.to(self.device)
                emb = self.model.extract_feats(Xy)
                emb_Xy = np.concatenate((emb_Xy, emb))

        return emb_Xy.cpu().numpy()
