import numpy as np
import matplotlib.pyplot as plt


def sigmoid(t):
    """返回Sigmoid函数结果值"""

    return 1 / (1 + np.exp(-t))


if __name__ == "__main__":

    # 绘制Sigmoid函数曲线
    x = np.linspace(-10, 10, 500)
    y = sigmoid(x)

    plt.plot(x, y)
    plt.show()