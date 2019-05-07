import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


# 过拟合：就是训练时的结果很好，但是在预测时结果不好的情况。
# 欠拟合: 模型没有很好地捕捉到数据特征，不能够很好地拟合数据的情况。

def PolynomialRegression(degree):
    """封装数据处理的步骤"""

    return Pipeline([
        ("Poly", PolynomialFeatures(degree=degree)),
        ("std_scaler", StandardScaler()),  # 归一化
        ("lin_reg", LinearRegression())
    ])


if __name__ == "__main__":
    np.random.seed(666)

    x = np.random.uniform(-3, 3, size=100)
    X = x.reshape(-1, 1)

    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)

    # 曲线太过简单，不能很好的概括模型，欠拟合
    # 线性回归
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    y_predict = lin_reg.predict(X)
    print("线性回归的均方误差 : {}".format(mean_squared_error(y, y_predict)))  # 3.075

    # 随着特征项变多，均值方差更接近0，但是过多特征项会导致曲线太过复杂，产生过拟合
    # 多项式回归（2次幂）
    poly2_reg = PolynomialRegression(degree=2)
    poly2_reg.fit(X, y)
    y2_predict = poly2_reg.predict(X)
    print("多项式回归（2）的均方误差 : {}".format(mean_squared_error(y, y2_predict)))  # 1.098

    plt.scatter(x, y)
    plt.plot(np.sort(x), y2_predict[np.argsort(x)],
             color='r')
    plt.show()

    # 多项式回归（10次幂）
    poly10_reg = PolynomialRegression(degree=10)
    poly10_reg.fit(X, y)
    y10_predict = poly10_reg.predict(X)
    print("多项式回归（10）的均方误差 : {}".format(mean_squared_error(y, y10_predict)))  # 1.050

    plt.scatter(x, y)
    plt.plot(np.sort(x), y10_predict[np.argsort(x)],
             color='r')
    plt.show()

    # 多项式回归（100次幂）
    poly100_reg = PolynomialRegression(degree=100)
    poly100_reg.fit(X, y)
    y100_predict = poly100_reg.predict(X)
    print("多项式回归（100）的均方误差 : {}".format(mean_squared_error(y, y100_predict)))  # 0.687

    plt.scatter(x, y)
    plt.plot(np.sort(x), y100_predict[np.argsort(x)],
             color='r')
    plt.show()
