import numpy as np
import matplotlib.pyplot as plt

"""在线性回归模型中使用梯度下降法(向量化实现)"""


def J(theta, X_b, y):
    """计算点theta在损失函数J下的取值"""

    try:
        return np.sum((y - X_b.dot(theta)) ** 2) / len(X_b)
    except:
        return float(inf)  # 返回float最大值


def dJ(theta, X_b, y):
    """对损失函数J在向量theta进行求导"""

    # res = np.empty(len(theta))
    # res[0] = np.sum(X_b.dot(theta) - y)  # 目标函数的截距求偏导
    #
    # for i in range(1, len(theta)):
    #     res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])
    #
    # return res * 2 / len(X_b)

    return X_b.T.dot(X_b.dot(theta) - y) * 2.0 / len(X_b)


def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
    """寻找损失函数的最小值点,eta是学习率，epsilon是下降幅度的界限，n_iters循环限制次数"""

    theta = initial_theta  # 起始点
    i_iter = 0  # 当前循环次数

    while i_iter < n_iters:
        gradient = dJ(theta, X_b, y)  # 记录随点变化对应的导数值变化
        last_theta = theta
        theta = theta - eta * gradient  # 移动点

        if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
            break

        i_iter += 1

    return theta


if __name__ == "__main__":
    np.random.seed(666)

    x = 2 * np.random.random(size=100)
    y = x * 3. + 4. + np.random.normal(size=100)

    X = x.reshape(-1, 1)

    X_b = np.hstack([np.ones((len(x), 1)), x.reshape(-1, 1)])

    initial_theta = np.zeros(X_b.shape[1])
    eta = 0.01  # 学习率

    theta = gradient_descent(X_b, y, initial_theta, eta)

    print(theta)  # 截距, 斜率
