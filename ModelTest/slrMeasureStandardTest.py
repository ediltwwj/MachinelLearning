import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn import datasets
from ModelTest.metrics import *
from KNN.modelSplit import train_test_split
from LinearRegression.simpleLinearRegressionS import SimpleLinearRegression2

if __name__ == "__main__":
    """衡量线性回归算法的标准测试"""

    boston = datasets.load_boston()

    x = boston.data[:, 5]  # 只使用房间数量这个特征
    y = boston.target

    x = x[y < 50.0]
    y = y[y < 50.0]

    x_train, x_test, y_train, y_test = train_test_split(x, y, seed=666)

    reg = SimpleLinearRegression2()
    reg.fit(x_train, y_train)

    y_predict = reg.predict(x_test)

    print("MSE : {}".format(mean_squared_error(y_test, y_predict)))
    print("RMSE : {}".format(root_mean_squared_error(y_test, y_predict)))
    print("MAE : {}".format(mean_absolute_error(y_test, y_predict)))
    print("R Squared : {}".format(r2_score(y_test, y_predict)))
