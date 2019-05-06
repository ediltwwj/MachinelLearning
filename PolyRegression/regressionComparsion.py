import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    x = np.random.uniform(-3, 3, size=100)
    X = x.reshape(-1, 1)

    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)

    # plt.scatter(x, y)
    # plt.show()

    # 使用线性回归拟合生成的数据
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)

    y_predict = lin_reg.predict(X)

    plt.scatter(x, y)
    plt.plot(x, y_predict, color='r')
    plt.show()

    # 采用多项式回归拟合生成的数据
    X2 = np.hstack([X, X ** 2])  # 给样本X再引入一个特征
    lin_reg2 = LinearRegression()
    lin_reg2.fit(X2, y)

    y_predict2 = lin_reg2.predict(X2)

    plt.scatter(x, y)
    plt.plot(np.sort(x), y_predict2[np.argsort(x)],
             color='r')  # 因为x是无序的，为了画出如下图平滑的线条，需要先将x进行排序，y_predict2按照x从的大小的顺序进行取值
    plt.show()

    print("X的系数和X^2的系数分别为 : {}".format(lin_reg2.coef_))  # 比对自己模拟数据的公式系数
