import lightgbm as lgb


class LGBMRegressor(lgb.LGBMRegressor):
    def __init__(
        self,
        train_size=0.9,
        init_score=None,
        eval_set=None,
        eval_names=None,
        eval_sample_weight=None,
        eval_init_score=None,
        eval_metric=None,
        early_stopping_rounds=None,
        verbose=True,
        feature_name="auto",
        categorical_feature="auto",
        callbacks=None,
        **kwargs
    ):
        super(LGBMRegressor, self).__init__(**kwargs)
        self.train_size = train_size
        self.init_score = init_score
        self.eval_set = eval_set
        self.eval_names = eval_names
        self.eval_sample_weight = eval_sample_weight
        self.eval_init_score = eval_init_score
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.feature_name = feature_name
        self.categorical_feature = categorical_feature
        self.callbacks = callbacks

    def fit(self, X, y, sample_weight=None):
        if not self.eval_set:
            train_size = int(len(X) * self.train_size)
            train_X, train_y = X.iloc[:train_size], y.iloc[:train_size]
            valid_X, valid_y = X.iloc[train_size:], y.iloc[train_size:]
            self.eval_set = [(valid_X, valid_y)]

        super().fit(
            train_X,
            train_y,
            sample_weight=sample_weight,
            init_score=self.init_score,
            eval_set=self.eval_set,
            eval_names=self.eval_names,
            eval_sample_weight=self.eval_sample_weight,
            eval_init_score=self.eval_init_score,
            eval_metric=self.eval_metric,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=self.verbose,
            feature_name=self.feature_name,
            categorical_feature=self.categorical_feature,
            callbacks=self.callbacks,
        )
        return self
    
    def predict(self, X):
        return self.model.predict(X)


class LGBMClassifier(lgb.LGBMClassifier):
    def __init__(
        self,
        train_size=0.9,
        init_score=None,
        eval_set=None,
        eval_names=None,
        eval_sample_weight=None,
        eval_init_score=None,
        eval_metric=None,
        early_stopping_rounds=None,
        verbose=True,
        feature_name="auto",
        categorical_feature="auto",
        callbacks=None,
        **kwargs
    ):
        super(LGBMClassifier, self).__init__(**kwargs)
        self.train_size = train_size
        self.init_score = init_score
        self.eval_set = eval_set
        self.eval_names = eval_names
        self.eval_sample_weight = eval_sample_weight
        self.eval_init_score = eval_init_score
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.feature_name = feature_name
        self.categorical_feature = categorical_feature
        self.callbacks = callbacks
        # self.model = lgb.LGBMClassifier(**kwargs)

    def fit(self, X, y, sample_weight=None):
        if not self.eval_set:
            train_size = int(len(X) * self.train_size)
            train_X, train_y = X.iloc[:train_size], y.iloc[:train_size]
            valid_X, valid_y = X.iloc[train_size:], y.iloc[train_size:]
            self.eval_set = [(valid_X, valid_y)]

        super().fit(
            train_X,
            train_y,
            sample_weight=sample_weight,
            init_score=self.init_score,
            eval_set=self.eval_set,
            eval_names=self.eval_names,
            eval_sample_weight=self.eval_sample_weight,
            eval_init_score=self.eval_init_score,
            eval_metric=self.eval_metric,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=self.verbose,
            feature_name=self.feature_name,
            categorical_feature=self.categorical_feature,
            callbacks=self.callbacks,
        )
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
