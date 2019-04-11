import numpy as np
from math import sqrt


def accuracy_score(y_true, y_predict):
    """计算y_true和y_predict之间的准确率(分类任务)"""

    assert y_true.shape[0] == y_predict.shape[0], \
        "the size of y_true must be equal to the size of y_predict"

    return sum(y_true == y_predict) / len(y_true)

    # 使用scikit-learn中的accuacy_score方法
    # from sklearn.neighbors import KNeighborsClassifier
    # from sklearn.metrics import accuracy_score
    #
    # knn_clf = KNeighborsClassifier(n_neighbors = 3)
    # knn_clf.fit(X_train, y_train)
    # y_predict = knn_clf.predict(X_test)
    # accuracy_score(y_test, y_predict)
    #
    # 使用使用scikit-learn中的score方法
    # knn_clf.score(X_test, y_test)


def mean_squared_error(y_true, y_predict):
    """计算y_true和y_predict之间的MSE(线性回归任务)"""

    assert len(y_true) == len(y_predict), \
        "the size of y_true must be equal to the size of y_predict"

    return np.sum((y_true - y_predict) ** 2) / len(y_true)


def root_mean_squared_error(y_true, y_predict):
    """计算y_true和y_predict之间的RMSE(线性回归任务)"""

    assert len(y_true) == len(y_predict), \
        "the size of y_true must be equal to the size of y_predict"

    return sqrt(mean_squared_error(y_true, y_predict))


def mean_absolute_error(y_true, y_predict):
    """计算Y_true和y_predict之间的MAE(线性回归任务)"""

    assert len(y_true) == len(y_predict), \
        "the size of y_true must be equal to the size of y_predict"

    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)


def r2_score(y_true, y_predict):
    """计算y_true和y_predict之间的R Square"""

    return 1 - mean_squared_error(y_true, y_predict) / np.var(y_true)

    # 使用scikit-learn中的MSE和MAE
    # from sklearn.metrics import mean_absolute_error
    # from sklearn.metrics import mean_absolute_error
    # from sklearn.metrics import r2_score
    #
    # mean_squared_error(y_test, y_predict)
    # mean_absolute_error(y_test, y_predict)
    # r2_score(y_test, y_predict)
