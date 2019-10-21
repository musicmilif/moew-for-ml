from torch import nn
from .loss import FocalLoss
from torch.utils.data import Dataset


class AutoEncoderDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        X = self.X.iloc[idx].values
        y = self.y.iloc[idx]
        return np.append(X, y)

    def __len__(self):
        return len(self.y)
