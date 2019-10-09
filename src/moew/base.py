from torch import nn


def train_ae(train_X, train_y):
    pass


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
            nn.Linear(self.hidden_dim, self.input_dim)
        )

    def forward(self, x):
        embbed = self.extract_feats(x)
        output = self.decoder(embbed)
        x = output[:, :-self.num_classes]
        y = output[:, -self.num_classes:]

        return x, y

    def extract_feats(self, x):
        feats = self.encoder(x)
        return feats
