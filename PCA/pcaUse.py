import numpy as np
import matplotlib.pyplot as plt

"""使用梯度上升法求解主成分"""


def demean(X):
    """均值归零处理"""

    return X - np.mean(X, axis=0)


def f(w, X):
    """求解目标函数"""

    return np.sum((X.dot(w) ** 2)) / len(X)


def df_math(w, X):
    """求解目标函数的导函数"""

    return X.T.dot(X.dot(w)) * 2. / len(X)


def df_debug(w, X, epsilon=0.0001):
    """测试导函数是否正确"""

    res = np.empty(len(w))

    for i in range(len(w)):
        w_1 = w.copy()
        w_1[i] += epsilon
        w_2 = w.copy()
        w_2[i] -= epsilon
        res[i] = (f(w_1, X) - f(w_2, X)) / (2 * epsilon)

    return res


def direction(w):
    """使w转换为单位向量"""

    return w / np.linalg.norm(w)


def gradient_ascent(df, X, initial_w, eta, n_iters=1e4, epsilon=1e-8):
    """寻找目标函数最大斜率,使得样本点到该直线的方差最大"""

    cur_iter = 0
    w = direction(initial_w)

    while cur_iter < n_iters:
        gradient = df(w, X)
        last_w = w
        w = w + eta * gradient
        w = direction(w)  # 每次计算后都应该将w转换为单位向量

        if (abs(f(w, X) - f(last_w, X)) < epsilon):
            break;

        cur_iter += 1

    return w


if __name__ == "__main__":

    X = np.empty((100, 2))
    X[:, 0] = np.random.uniform(0., 100., size=100)
    X[:, 1] = 0.75 * X[:, 0] + 3. + np.random.normal(0, 10., size=100)

    initial_w = np.random.random(X.shape[1])  # 不能使用零向量开始，零向量本身是一个极值点
    eta = 0.0001
    X_demean = demean(X)

    w1 = gradient_ascent(df_debug, X_demean, initial_w, eta)  # 不能使用StandardScaler标准化数据
    w2 = gradient_ascent(df_math, X_demean, initial_w, eta)
    print("Vector(df_debug) ： {}".format(w1))
    print("Vector(df_math) ： {}".format(w2))

    plt.scatter(X_demean[:, 0], X_demean[:, 1])  # 可视化图例
    plt.plot([0, w2[0] * 50], [0, w2[1] * 50], color='r')
    plt.show()
