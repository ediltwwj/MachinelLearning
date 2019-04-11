import numpy as np
import matplotlib
import matplotlib.pyplot as plt

"""最值归一化的实现过程"""

X = np.random.randint(0, 100, size=100)
X = (X - np.min(X)) / (np.max(X) - np.min(X))
print(X)

Y = np.random.randint(0, 100, (50, 2))
Y = Y.astype(np.float64)
Y[:, 0] = (Y[:, 0] - np.min(Y[:, 0])) / (np.max(Y[:, 0]) - np.min(Y[:, 0]))  # 先处理第一列
Y[:, 1] = (Y[:, 1] - np.min(Y[:, 1])) / (np.max(Y[:, 1]) - np.min(Y[:, 1]))  # 再处理第二列

print(Y)
print("均值分别为{},{}".format(np.mean(Y[:, 0]), np.mean(Y[:, 1])))
print("标准差分别为{},{}".format(np.std(Y[:, 0]), np.std(Y[:, 1])))

plt.scatter(Y[:, 0], Y[:, 1])
plt.show()
