from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, embedd_dim, num_classes=1):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim + num_classes
        self.hidden_dim = max(2, self.input_dim // 2)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Sigmoid(),
            nn.Linear(self.hidden_dim, self.embedd_dim),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.embedd_dim, self.hidden_dim),
            nn.Sigmoid(),
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
