import numpy as np
import matplotlib.pyplot as plt

"""获得前n个主成分"""


def demean(X):
    """均值归零处理"""

    return X - np.mean(X, axis=0)


def f(w, X):
    """求解目标函数"""

    return np.sum((X.dot(w) ** 2)) / len(X)


def df(w, X):
    """求解目标函数的导函数"""

    return X.T.dot(X.dot(w)) * 2. / len(X)


def direction(w):
    """使w转换为单位向量"""

    return w / np.linalg.norm(w)


def first_component(X, initial_w, eta, n_iters=1e4, epsilon=1e-8):
    """寻找目标函数的第一个主成分"""

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


def first_n_components(n, X, eta=0.01, n_iters=1e4, epsilon=1e-8):
    """获取X的前n个主成分"""

    X_pca = X.copy()
    X_pca = demean(X_pca)

    res = []

    for i in range(n):
        initial_w = np.random.random(X_pca.shape[1])
        w = first_component(X_pca, initial_w, eta)
        res.append(w)

        X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w

    return res


if __name__ == "__main__":
    X = np.empty((100, 2))
    X[:, 0] = np.random.uniform(0., 100., size=100)
    X[:, 1] = 0.75 * X[:, 0] + 3. + np.random.normal(0, 10., size=100)

    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

    initial_w = np.random.random(X.shape[1])  # 不能使用零向量开始，零向量本身是一个极值点
    eta = 0.0001
    X_demean = demean(X)

    w = first_component(X, initial_w, eta)  # 求第一个主成分
    print("First Component : {}".format(w))

    X2 = np.empty(X.shape)

    # for i in range(len(X)):  # 去掉在第一主成分上的分量
    #     X2[i] = X[i] - X[i].dot(w) * w  # *w是的坐标数值发生变化

    X2 = X - X.dot(w).reshape(-1, 1) * w  # 向量化去除在第一主成分的分量

    plt.scatter(X2[:, 0], X2[:, 1])
    plt.show()

    w2 = first_component(X2, initial_w, eta)  # 求第二个主成分
    print("Second Component : {}".format(w2))

    print("w.dot(w2) : {}".format(w.dot(w2)))  # 互相垂直

    w3 = first_n_components(2, X)

    print("N Component : {}".format(w3))
