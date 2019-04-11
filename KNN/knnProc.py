import numpy as np
from math import sqrt
from collections import Counter

"""kNN过程模拟"""


def kNN_classify(k, X_train, y_train, x):
    assert 1 <= k <= X_train.shape[0], "k must be valid"
    assert X_train.shape[0] == y_train.shape[0], \
        "the size of X_train must equal to the size of y_train"
    assert X_train.shape[1] == x.shape[0], \
        "the feature number of x must be equal to X_train"

    distance = [sqrt(np.num((x_train - x) ** 2)) for x_train in X_train]
    nearest = np.argsort(distance)

    topK_y = [y_train[i] for i in nearest[: k]]  # y_train[i]就是X_train对应的类别
    votes = Counter(topK_y)

    return votes.most_common(1)[0][0]
