import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    x = np.random.uniform(-3, 3, size=100)
    X = x.reshape(-1, 1)

    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)

    poly = PolynomialFeatures(degree=2)  # 表示添加2次幂的特征，给定不同特征，样本特征呈指数级增长
    poly.fit(X)
    X2 = poly.transform(X)  # 标准化数字

    lin_reg2 = LinearRegression()
    lin_reg2.fit(X2, y)

    y_predict2 = lin_reg2.predict(X2)

    plt.scatter(x, y)
    plt.plot(np.sort(x), y_predict2[np.argsort(x)],
             color='r')  # 因为x是无序的，为了画出如下图平滑的线条，需要先将x进行排序，y_predict2按照x从的大小的顺序进行取值
    plt.show()

    print("X的系数和X^2的系数分别为 : {}".format(lin_reg2.coef_))  # 这个时候x2有三个特征项，因为在第1列加入1列1，并加入了x^2项
