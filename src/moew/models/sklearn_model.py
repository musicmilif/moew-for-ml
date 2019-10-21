from collections import defaultdict, OrderedDict
from sklearn import svm, tree, ensemble, linear_model
from sklearn.metrics import log_loss, mean_squared_error

__all__ = [
    "SVC",
    "SVR",
    "DecisionTreeRegressor",
    "DecisionTreeClassifier",
    "ExtraTreeRegressor",
    "ExtraTreeClassifier",
    "ExtraTreesRegressor",
    "ExtraTreesClassifier",
    "RandomForestRegressor",
    "RandomForestClassifier",
    "AdaBoostRegressor",
    "AdaBoostClassifier",
    "GradientBoostingRegressor",
    "GradientBoostingClassifier",
    "LinearRegression",
    "LogisticRegression",
]


class BaseRegressor(object):
    def __init__(self, model, train_size=0.9, **kwargs):
        self.train_size = train_size
        self.model = model(**kwargs)
        self.best_score_ = defaultdict(OrderedDict)

    def fit(self, X, y, sample_weight=None, valid_X=None, valid_y=None, **kwargs):
        if not valid_X:
            train_size = int(len(X) * self.train_size)
            train_X, train_y = X.iloc[:train_size], y.iloc[:train_size]
            valid_X, valid_y = X.iloc[train_size:], y.iloc[train_size:]
        else:
            train_X, train_y = X, y

        self.model.fit(train_X, train_y, sample_weight=sample_weight)
        valid_y_ = self.model.predict(valid_X)
        l2 = mean_squared_error(valid_y, valid_y_)
        self.best_score_["valid_0"] = [("l2", l2)]

        return self

    def predict(self, X):
        return self.model.predict(X)


class BaseClassifier(object):
    def __init__(self, model, train_size=0.9, **kwargs):
        self.train_size = train_size
        self.model = model(**kwargs)
        self.best_score_ = defaultdict(OrderedDict)

    def fit(self, X, y, sample_weight=None, valid_X=None, valid_y=None, **kwargs):
        if not valid_X:
            train_size = int(len(X) * self.train_size)
            train_X, train_y = X.iloc[:train_size], y.iloc[:train_size]
            valid_X, valid_y = X.iloc[train_size:], y.iloc[train_size:]
        else:
            train_X, train_y = X, y

        self.model.fit(train_X, train_y, sample_weight=sample_weight)
        valid_y_ = self.model.predict_proba(valid_X)
        logloss = log_loss(valid_y, valid_y_)

        if train_y.nunique() == 2:
            self.best_score_["valid_0"] = [("binary_logloss", logloss)]
        else:
            self.best_score_["valid_0"] = [("multi_logloss", logloss)]

        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class SVR(BaseRegressor):
    def __init__(self, train_size=0.9, **kwargs):
        super(SVR, self).__init__(svm.SVR, train_size=0.9, **kwargs)


class SVC(BaseClassifier):
    def __init__(self, train_size=0.9, **kwargs):
        kwargs["probability"] = True
        super(SVC, self).__init__(svm.SVC, train_size=0.9, **kwargs)


class DecisionTreeRegressor(BaseRegressor):
    def __init__(self, train_size=0.9, **kwargs):
        super(DecisionTreeRegressor, self).__init__(
            tree.DecisionTreeRegressor, train_size=0.9, **kwargs
        )


class DecisionTreeClassifier(BaseClassifier):
    def __init__(self, train_size=0.9, **kwargs):
        super(DecisionTreeClassifier, self).__init__(
            tree.DecisionTreeClassifier, train_size=0.9, **kwargs
        )


class ExtraTreeRegressor(BaseRegressor):
    def __init__(self, train_size=0.9, **kwargs):
        super(ExtraTreeRegressor, self).__init__(
            tree.ExtraTreeRegressor, train_size=0.9, **kwargs
        )


class ExtraTreeClassifier(BaseClassifier):
    def __init__(self, train_size=0.9, **kwargs):
        super(ExtraTreeClassifier, self).__init__(
            tree.ExtraTreeClassifier, train_size=0.9, **kwargs
        )


class ExtraTreesRegressor(BaseRegressor):
    def __init__(self, train_size=0.9, **kwargs):
        super(ExtraTreesRegressor, self).__init__(
            ensemble.ExtraTreesRegressor, train_size=0.9, **kwargs
        )


class ExtraTreesClassifier(BaseClassifier):
    def __init__(self, train_size=0.9, **kwargs):
        super(ExtraTreesClassifier, self).__init__(
            ensemble.ExtraTreesClassifier, train_size=0.9, **kwargs
        )


class RandomForestRegressor(BaseRegressor):
    def __init__(self, train_size=0.9, **kwargs):
        super(RandomForestRegressor, self).__init__(
            ensemble.RandomForestRegressor, train_size=0.9, **kwargs
        )


class RandomForestClassifier(BaseClassifier):
    def __init__(self, train_size=0.9, **kwargs):
        super(RandomForestClassifier, self).__init__(
            ensemble.RandomForestClassifier, train_size=0.9, **kwargs
        )


class AdaBoostRegressor(BaseRegressor):
    def __init__(self, train_size=0.9, **kwargs):
        super(AdaBoostRegressor, self).__init__(
            ensemble.AdaBoostRegressor, train_size=0.9, **kwargs
        )


class AdaBoostClassifier(BaseClassifier):
    def __init__(self, train_size=0.9, **kwargs):
        super(AdaBoostClassifier, self).__init__(
            ensemble.AdaBoostClassifier, train_size=0.9, **kwargs
        )


class GradientBoostingRegressor(BaseRegressor):
    def __init__(self, train_size=0.9, **kwargs):
        super(GradientBoostingRegressor, self).__init__(
            ensemble.GradientBoostingRegressor, train_size=0.9, **kwargs
        )


class GradientBoostingClassifier(BaseClassifier):
    def __init__(self, train_size=0.9, **kwargs):
        super(GradientBoostingClassifier, self).__init__(
            ensemble.GradientBoostingClassifier, train_size=0.9, **kwargs
        )


class LinearRegression(BaseRegressor):
    def __init__(self, train_size=0.9, **kwargs):
        super(LinearRegression, self).__init__(
            linear_model.LinearRegression, train_size=0.9, **kwargs
        )


class LogisticRegression(BaseClassifier):
    def __init__(self, train_size=0.9, **kwargs):
        super(LogisticRegression, self).__init__(
            linear_model.LogisticRegression, train_size=0.9, **kwargs
        )
