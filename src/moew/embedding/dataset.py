import numpy as np
from torch import nn
from torch.utils.data import Dataset


class AutoEncoderDataset(Dataset):
    def __init__(self, X, y, num_classes):
        """Pytorch dataset for auto encoder
        Args:
            X (pd.DataFrame): features dataframe.
            y (pd.Series): target series.
            num_classes (int): number of classes in target, 1 for regression.
        """
        self.X = X
        self.y = y
        self.num_classes = num_classes

    def __getitem__(self, idx):
        X = self.X.iloc[idx].values
        y = int(self.y.iloc[idx])
        y_ohe = np.zeros(self.num_classes, dtype=np.float16)
        y_ohe[y] = 1.0

        return np.append(X, y_ohe), y

    def __len__(self):
        return len(self.y)
