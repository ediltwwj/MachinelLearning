import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA

"""人脸识别与特征脸"""


def plot_face(faces):
    """绘制人脸图像"""

    fig, axes = plt.subplots(6, 6, figsize=(10, 10),
                             subplot_kw={'xticks': [], 'yticks': []},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))

    for i, ax in enumerate(axes.flat):
        ax.imshow(faces[i].reshape(62, 47), cmap='bone')

    plt.show()


if __name__ == "__main__":
    faces = fetch_lfw_people(data_home="D:\dataset")
    random_indexes = np.random.permutation(len(faces.data))  # 打乱顺序，但不改变原数组
    X = faces.data[random_indexes]

    example_faces = X[:36, :]
    plot_face(example_faces)

    pca = PCA(svd_solver="randomized")  # 丢弃奇异值
    pca.fit(X)
