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


class SVR(object):
    def __init__(self, train_size=0.9, **kwargs):
        self.train_size = train_size
        self.model = svm.SVR(**kwargs)
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


class SVC(object):
    def __init__(self, train_size=0.9, **kwargs):
        self.train_size = train_size
        self.model = svm.SVC(probability=True, **kwargs)
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

        if train_y.nunique() > 2:
            self.best_score_["valid_0"] = [("multi_logloss", logloss)]
        else:
            self.best_score_["valid_0"] = [("binary_logloss", logloss)]

        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
