import numpy as np
import matplotlib.pyplot as plt



def entropy(p):
    """信息熵 二分类"""
    return -p * np.log(p) - (1 - p) * np.log(1 - p)

if __name__ == "__main__":

    x = np.linspace(0.01, 0.99, 200)
    plt.plot(x, entropy(x))
    plt.show()