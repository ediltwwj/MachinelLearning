import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from KNN.modelSplit import train_test_split
from LinearRegression.multLinearRegression import MultLinearRegression

if __name__ == "__main__":
    """多元线性回归方程的测试"""

    boston = datasets.load_boston()

    X = boston.data
    y = boston.target

    X = X[y < 50.0]
    y = y[y < 50.0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)

    reg = MultLinearRegression()
    reg.fit_normal(X_train, y_train)

    print("MultLinearRegression's Predict Score : {}".format(reg.score(X_test, y_test)))

    # reg1 = MultLinearRegression()
    # reg1.fit_gd(X_train, y_train) # 使用梯度下降法拟合模型
    # print("MultLinearRegression's Predict Score(Use gd) : {}".format(reg1.score(X_test, y_test)))
    # 输出nan
