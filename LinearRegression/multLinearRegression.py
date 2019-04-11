import numpy as np
from ModelTest.metrics import r2_score


class MultLinearRegression:
    """多元线性回归的实现"""

    def __init__(self):
        """初始化Multielement Linear Regression模型"""

        self.coef_ = None  # 系数
        self.interception_ = None  # 截距
        self._theta = None  # 计算结果

    def fit_normal(self, X_train, y_train):
        """根据训练集数据X_train，y_train训练Multielement Linear Regression模型"""

        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])  # 给截距增加一列
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4):
        """根据训练集数据X_train, y_train，使用梯度下降法训练Multielement Linear Regression模型"""

        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        def J(theta, X_b, y):
            """计算点theta在损失函数J下的取值"""

            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
            except:
                return float(inf)  # 返回float最大值

        def dJ(theta, X_b, y):
            """对损失函数J在向量theta进行求导"""

            res = np.empty(len(theta))
            res[0] = np.sum(X_b.dot(theta) - y)  # 目标函数的截距求偏导

            for i in range(1, len(theta)):
                res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])

            return res * 2 / len(X_b)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
            """寻找损失函数的最小值点,eta是学习率，epsilon是下降幅度的界限，n_iters循环限制次数"""

            theta = initial_theta  # 起始点
            cur_iter = 0  # 当前循环次数

            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)  # 记录随点变化对应的导数值变化
                last_theta = theta
                theta = theta - eta * gradient  # 移动点

                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break

                cur_iter += 1

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""

        assert self.interception_ is not None and self.coef_ is not None, \
            "must fit before predict"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])

        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        """根据测试数据集X_test和y_test确定当前模型的准确率"""

        y_predict = self.predict(X_test)

        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "MultLinearRegression()"
