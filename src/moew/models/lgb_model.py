import lightgbm as lgb


class LGBMRegressor(lgb.LGBMRegressor):
    def __init__(self, train_size=0.9, **kwargs):
        super().__init__(**kwargs)
        self.train_size = train_size

    def fit(self, X, y, sample_weight=None, eval_set=None, **kwargs):
        if not eval_set:
            train_size = int(len(X) * self.train_size)
            train_X, train_y = X.iloc[:train_size], y.iloc[:train_size]
            valid_X, valid_y = X.iloc[train_size:], y.iloc[train_size:]
            eval_set = [(valid_X, valid_y)]
        else:
            train_X, train_y = X, y
        super().fit(train_X, train_y, sample_weight=sample_weight, eval_set=eval_set)

        return self

    def predict(self, X):
        return super().predict(X)


class LGBMClassifier(lgb.LGBMClassifier):
    def __init__(self, train_size=0.9, **kwargs):
        super().__init__(**kwargs)
        self.train_size = train_size

    def fit(self, X, y, sample_weight=None, eval_set=None, **kwargs):
        if not eval_set:
            train_size = int(len(X) * self.train_size)
            train_X, train_y = X.iloc[:train_size], y.iloc[:train_size]
            valid_X, valid_y = X.iloc[train_size:], y.iloc[train_size:]
            eval_set = [(valid_X, valid_y)]
        else:
            train_X, train_y = X, y
        super().fit(train_X, train_y, sample_weight=sample_weight, eval_set=eval_set)

        return self

    def predict(self, X):
        return super().predict(X)

    def predict_proba(self, X):
        return super().predict_proba(X)
