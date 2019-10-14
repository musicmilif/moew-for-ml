import numpy as np
import torch
from torch import nn
from .data import AutoEncoderDataset
from torch.utils.data import DataLoader


class TrainEmbedding(object):
    def __init__(self, model, x_loss, y_loss, lambda_=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.x_criterion = x_loss
        self.y_criterion = y_loss
        self.lambda_ = lambda_

    def fit(self, X, y, n_epochs=100, batch_size=128, **kwargs):
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


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, embedd_dim, num_classes=1):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim + num_classes
        self.hidden_dim = max(2, self.input_dim // 2)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.sigmoid(),
            nn.Linear(self.hidden_dim, self.embedd_dim),
            nn.sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.embedd_dim, self.hidden_dim),
            nn.sigmoid(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )

    def forward(self, x):
        embbed = self.extract_feats(x)
        output = self.decoder(embbed)
        x, y = output[:, :-1], output[:, -1]

        return x, y

    def extract_feats(self, x):
        feats = self.encoder(x)
        return feats
