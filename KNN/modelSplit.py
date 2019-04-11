import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


def train_test_split(X, y, test_ratio=0.2, seed=None):
    """将数据X和y按照test_ratio分割成X_train, X_test, y_train, y_test"""

    assert X.shape[0] == y.shape[0], \
        "the size of X must be equal to the size of y"
    assert 0.0 <= test_ratio <= 1.0, \
        "test_ratio must be valid"

    if seed:
        np.random.seed(seed)

    shuffle_indexes = np.random.permutation(len(X))  # 打乱数据集X
    test_size = int(len(X) * test_ratio)
    test_indexes = shuffle_indexes[: test_size]  # 测试数据集索引
    train_indexes = shuffle_indexes[test_size:]  # 训练数据集索引

    X_train = X[train_indexes]  # 训练数据集
    y_train = y[train_indexes]
    X_test = X[test_indexes]  # 测试数据集
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test

"""
    # 使用scikit-learn中的train_test_split方法
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X, y) # test_size默认为0.2.想复现拆分传入随机种子random_state
"""