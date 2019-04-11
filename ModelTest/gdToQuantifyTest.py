import numpy as np
from sklearn import datasets
from KNN.modelSplit import train_test_split
from LinearRegression.multLinearRegression import MultLinearRegression
from LinearRegression.multLinearRegressionS import MultLinearRegressionS
from sklearn.preprocessing import StandardScaler

"""梯度下降法的向量化测试"""

if __name__ == "__main__":
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target

    X = X[y < 50.0]
    y = y[y < 50.0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)

    lin_reg1 = MultLinearRegression()
    lin_reg1.fit_normal(X_train, y_train)
    print("MultLinearRegression's Predict Score : {}".format(lin_reg1.score(X_test, y_test)))

    # 报错nan, 如果使用默认eta
    lin_reg2 = MultLinearRegressionS()
    lin_reg2.fit_gd(X_train, y_train, eta=0.000001, n_iters=1e6)
    # score很低，最好在使用梯度下降法前进行数据归一化
    print("MultLinearRegression's Predict Score : {}".format(lin_reg2.score(X_test, y_test)))

    # 进行数值归一化
    standardScaler = StandardScaler()
    standardScaler.fit(X_train)
    X_train_standard = standardScaler.transform(X_train)  # 归一化处理
    X_test_standard = standardScaler.transform(X_test)
    lin_reg3 = MultLinearRegressionS()
    lin_reg3.fit_gd(X_train_standard, y_train)
    print("MultLinearRegression's Predict Score : {}".format(lin_reg3.score(X_test_standard, y_test)))
