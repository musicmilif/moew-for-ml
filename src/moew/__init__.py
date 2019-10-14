from sklearn.svm import *
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.linear_model import *
from .lgb_model import LightGBMWrapper

regressor_models = {
    # Linear models
    "linear_regression": LinearRegression,
    # Tree models
    "decision_tree": DecisionTreeRegressor,
    "extra_tree": ExtraTreeRegressor,
    # SVM models
    "svm": SVR,
    # Ensemble models
    "random_forest": RandomForestRegressor,
    "extra_trees": ExtraTreesRegressor,
    "ada_boost": AdaBoostRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "lightgbm": LightGBMWrapper,
    # TODO: Neural Network models
}

classifier_models = {
    # Linear models
    "logistic_regression": LogisticRegression,
    # Tree models
    "decision_tree": DecisionTreeClassifier,
    "extra_tree": ExtraTreeClassifier,
    # SVM models
    "svm": SVC,
    # Ensemble models
    "random_forest": RandomForestClassifier,
    "extra_trees": ExtraTreesClassifier,
    "ada_boost": AdaBoostClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "lightgbm": LightGBMWrapper,
    # TODO: Neural Network models
}
