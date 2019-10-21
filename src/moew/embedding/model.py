from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, embedd_dim, num_classes=1):
        super(AutoEncoder, self).__init__()
        self.num_classes = num_classes
        input_dim = input_dim + num_classes
        hidden_dim = max(2, input_dim // 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, embedd_dim),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedd_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        embbed = self.extract_feats(x)
        output = self.decoder(embbed)
        x, y = output[:, : -self.num_classes], output[:, -self.num_classes :]

        return x, y

    def extract_feats(self, x):
        feats = self.encoder(x)
        return feats
