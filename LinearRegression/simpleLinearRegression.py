import numpy as np


class SimpleLinearRegression1:
    """简单线性回归的循环实现"""

    def __init__(self):
        """初始化SimpleLinearRegression模型"""

        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        """根据训练数据集x_train, y_train训练SimpleLinearRegression模型"""

        assert x_train.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training data."
        assert len(x_train) == len(y_train), \
            "the size of x_train must be equal to the size of y_train."

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        molecular = 0.0  # 分子
        denominator = 0.0  # 分母

        for x, y in zip(x_train, y_train):
            molecular += (x - x_mean) * (y - y_mean)
            denominator += (x - x_mean) ** 2

        self.a_ = molecular / denominator
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        """给定待测数据集x_predict,返回表示x_predict的结果向量"""

        assert x_predict.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training data."
        assert self.a_ is not None and self.b_ is not None, \
            "must fit before predict!"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        """给定单个待测数据x_single，返回x的预测结果值"""

        return self.a_ * x_single + self.b_

    def __repr__(self):
        return "SimpleLinearRegression1()"
