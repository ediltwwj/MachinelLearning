import numpy as np
import matplotlib
import matplotlib.pyplot as plt

"""均值方差归一化的实现过程"""

X = np.random.randint(0, 100, (50, 2))
X = np.array(X, dtype='float64')  # X = X.astype(np.float64)

X[:, 0] = (X[:, 0] - np.mean(X[:, 0])) / np.std(X[:, 0])  # 先处理第一列
X[:, 1] = (X[:, 1] - np.mean(X[:, 1])) / np.std(X[:, 1])  # 先处理第一列

print(X)
print("均值分别为{},{}".format(np.mean(X[:, 0]), np.mean(X[:, 1])))
print("标准差分别为{},{}".format(np.std(X[:, 0]), np.std(X[:, 1])))

plt.scatter(X[:, 0], X[:, 1])
plt.show()
