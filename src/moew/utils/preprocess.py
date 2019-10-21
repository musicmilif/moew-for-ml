import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from pandas.api.types import is_numeric_dtype


class TargetPreProcess(BaseEstimator, TransformerMixin):
    def fit(self, y):
        self.is_numeric_dtype = is_numeric_dtype(y)
        if self.is_numeric_dtype:
            self.avg = y.mean()
            self.std = y.std()
            y = (y - self.avg) / self.std
        else:
            y = y.astype("category")
            self.mapping = {v: k for k, v in enumerate(y.cat.categories)}

        return self

    def transform(self, y, norm=True):
        if self.is_numeric_dtype:
            y = (y - self.avg) / self.std if norm else y
        else:
            y = y.map(self.mapping)

        return y


class FeaturePreProcess(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.norm = dict()
        self.label_encode = dict()

    def fit(self, X):
        # TODO: datetime features and proper dealing with categorical features
        self.col_order = X.columns
        self.num_cols = X.select_dtypes(include=["number"]).columns
        self.cat_cols = X.select_dtypes(include=["object", "category"]).columns

        for col in X.columns:
            if col in self.num_cols:
                avg, std = X[col].mean(), X[col].std()
                self.norm[col] = [avg, std]
            elif col in self.cat_cols:
                X[col] = X[col].astype("category")
                self.label_encode[col] = {
                    v: k + 1 for k, v in enumerate(y.cat.categories)
                }
                avg, std = X[col].mean(), X[col].std()
                self.norm[col] = [avg, std]
            else:
                raise NotImplementedError(f"Got data type: {X[col].dtype}")

        return self

    def transform(self, X, norm=True, fillna=False):
        # TODO: better filling missing data strategy
        for col in self.num_cols:
            avg, std = self.norm[col]
            X[col] = X[col].fillna(0) if fillna else X[col]
            X[col] = (X[col] - avg) / std if norm else X[col]
        for col in self.cat_cols:
            avg, std = self.norm[col]
            X[col] = X[col].map(self.label_encode[col])
            X[col] = X[col].fillna(0) if fillna else X[col]
            X[col] = (X[col] - avg) / std if norm else X[col]

        return X
