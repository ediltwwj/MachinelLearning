import numpy as np
import matplotlib.pyplot as plt
from PCA.pca import PCA

"""自定义PCA的测试"""

if __name__ == "__main__":
    x = np.empty((100, 2))
    x[:, 0] = np.random.uniform(0., 100., size=100)
    x[:, 1] = 0.75 * x[:, 0] + 3. + np.random.normal(0, 10., size=100)  # 这个数据集和上面使用的是一样的数据

    pca = PCA(n_components=1)
    pca.fit(x)
    x_reduction = pca.transform(x)
    x_restore = pca.inverse_transform(x_reduction)  # 即将一维的数据重新映射回二维，仅仅是展示用
    plt.scatter(x[:, 0], x[:, 1], color='b', alpha=0.5)
    plt.scatter(x_restore[:, 0], x_restore[:, 1], color='r', alpha=0.5)
    plt.show()  # 可以发现重新映射回原来矩阵形式时，只剩下主成分方向上的数据，restore的功能只是在高维的空间表达低维的数据
