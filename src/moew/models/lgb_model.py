import lightgbm as lgb


class LGBMRegressor(lgb.LGBMRegressor):
    def __init__(self, train_size=0.9, **kwargs):
        super().__init__(**kwargs)
        self.train_size = train_size

    def fit(self, X, y, sample_weight=None, valid_X=None, valid_y=None, **kwargs):
        if not valid_X:
            train_size = int(len(X) * self.train_size)
            train_X, train_y = X.iloc[:train_size], y.iloc[:train_size]
            valid_X, valid_y = X.iloc[train_size:], y.iloc[train_size:]
        else:
            train_X, train_y = X, y

        eval_set = [(valid_X, valid_y)]
        super().fit(train_X, train_y, sample_weight=sample_weight, eval_set=eval_set)

        return self


class LGBMClassifier(lgb.LGBMClassifier):
    def __init__(self, train_size=0.9, **kwargs):
        super().__init__(**kwargs)
        self.train_size = train_size

    def fit(self, X, y, sample_weight=None, valid_X=None, valid_y=None, **kwargs):
        if not valid_X:
            train_size = int(len(X) * self.train_size)
            train_X, train_y = X.iloc[:train_size], y.iloc[:train_size]
            valid_X, valid_y = X.iloc[train_size:], y.iloc[train_size:]
        else:
            train_X, train_y = X, y

        eval_set = [(valid_X, valid_y)]
        super().fit(train_X, train_y, sample_weight=sample_weight, eval_set=eval_set)

        return self
