import lightgbm as lgb


class LightGBMWrapper(object):
    def __init__(
        self, model, n_rounds=1000, earlystop_rounds=100, verbose_rounds=0, **kwargs
    ):
        self.params = kwargs
        self.n_rounds = n_rounds
        self.earlystop_rounds = earlystop_rounds
        self.verbose_rounds = verbose_rounds

    def fit(self, train_X, valid_X, train_y, valid_y, weights):
        dtrain = lgb.Dataset(train_X, train_y, w=weights)
        dvalid = lgb.Dataset(valid_X, valid_y)
        self.model = lgb.train(
            self.params,
            train_set=dtrain,
            num_boost_round=self.n_rounds,
            valid_sets=[dtrain, dvalid],
            early_stopping_rounds=self.earlystop_rounds,
            verbose_eval=self.verbose_rounds,
        )

    def predict(self, test_X):
        return self.model.predict(test_X)
