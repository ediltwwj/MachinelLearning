import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

"""使用PCA对数据进行降噪"""


def plot_digits(data):
    """绘制数字"""

    fig, axes = plt.subplots(10, 10, figsize=(10, 10),
                             subplot_kw={'xticks': [], 'yticks': []},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))

    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8, 8),
                  cmap='binary', interpolation='nearest', clim=(0, 16))

    plt.show()


if __name__ == "__main__":

    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    noisy_digits = X + np.random.normal(0, 4, size=X.shape)  # 对数据添加噪音

    example_digits = noisy_digits[y == 0, :][:10]
    for num in range(1, 10):
        X_num = noisy_digits[y == num, :][:10]
        example_digits = np.vstack([example_digits, X_num])

    plot_digits(example_digits)  # 有噪音的数字集

    pca = PCA(0.5)
    pca.fit(noisy_digits)
    print("降维到哪个维度 : {}".format(pca.n_components_))

    components = pca.transform(example_digits)  # 降维处理
    filtered_digits = pca.inverse_transform(components)  # 再复原特征
    plot_digits(filtered_digits)
