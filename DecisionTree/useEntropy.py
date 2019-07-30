import numpy as np
from sklearn import datasets
from collections import Counter
from math import log

"""模拟使用信息熵进行划分"""


def split(X, y, d, value):
    """X表示数据集, y表示结果集, d表示哪个维度, value表示维度上的阈值"""

    index_a = (X[:, d] <= value)
    index_b = (X[:, d] > value)

    return X[index_a], X[index_b], y[index_a], y[index_b]

def entropy(y):
    """计算信息熵"""
    counter = Counter(y)
    res = 0.0
    for num in counter.values():
        p = num / len(y)
        res += -p * log(p)
    return res

def try_split(X, y):
    """找出使得划分信息熵最小的值"""

    best_entropy = float('inf')
    best_d, best_v = -1, -1
    for d in range(X.shape[1]):
        sorted_index = np.argsort(X[:, d])
        for i in range(1, len(X)):
            if (X[sorted_index[i-1], d] != X[sorted_index[i], d]):
                v = (X[sorted_index[i-1], d] + X[sorted_index[i], d]) / 2
                X_l, X_r, y_l, y_r = split(X, y, d, v)
                e = entropy(y_l) + entropy(y_r)

                if e < best_entropy:
                    best_entropy, best_d, best_v = e, d, v

    return best_entropy, best_d, best_v


if __name__ == "__main__":

    iris = datasets.load_iris()
    X = iris.data[:, 2: ]
    y = iris.target

    # 第一次划分
    best_entropy, best_d, best_v = try_split(X, y)
    print("best_entropy = ", best_entropy)
    print("best_d = ", best_d)
    print("best_v = ", best_v)

    # 第一次划分结果
    X1_l, X1_r, y1_l, y1_r = split(X, y, best_d, best_v)
    print("左信息熵 = ", entropy(y1_l))
    print("右信息熵 = ", entropy(y1_r))

    print()

    # 第二次划分,第一次左信息熵为0，无需再划分
    best_entropy2, best_d2, best_v2 = try_split(X1_r, y1_r)
    print("best_entropy2 = ", best_entropy2)
    print("best_d2 = ", best_d2)
    print("best_v2 = ", best_v2)

    # 第二次划分结果
    X2_l, X2_r, y2_l, y2_r = split(X1_r, y1_r, best_d2, best_v2)
    print("左信息熵 = ", entropy(y2_l))
    print("右信息熵 = ", entropy(y2_r))
