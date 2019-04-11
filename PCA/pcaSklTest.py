import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

"""使用scikit_learn当中的PCA,"""

if __name__ == "__main__":

    digits = datasets.load_digits()  # 导入手写数字集
    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_train)

    print("KNN : {}".format(knn_clf.score(X_test, y_test)))

    # 使用PCA降维后，准确率大幅下降，但运算时间显著提高
    pca = PCA(n_components=2)
    pca.fit(X_train)

    X_train_reduction = pca.transform(X_train)
    X_test_reduction = pca.transform(X_test)

    knn_clf_pca = KNeighborsClassifier()
    knn_clf_pca.fit(X_train_reduction, y_train)

    print("After PCA（2） : {}".format(knn_clf_pca.score(X_test_reduction, y_test)))

    # 特征维度的方差比例,从大到小排列
    print("特征维度的方差比例(二维) : {}".format(pca.explained_variance_ratio_))

    pca_all = PCA(n_components=X_train.shape[1])
    pca_all.fit(X_train)

    print("特征维度的方差比例(所有) : {}".format(pca_all.explained_variance_ratio_))

    # 绘制折线图，查看方差比例累加和变化趋势
    plt.plot([i for i in range(X_train.shape[1])],
             [np.sum(pca_all.explained_variance_ratio_[: i + 1]) for i in range(X_train.shape[1])])
    plt.show()

    pca_suit = PCA(0.95)  # 保留0.95的特征
    pca_suit.fit(X_train)
    print("保留手写数字数据集%95的特征需要多少维度 : {}".format(pca_suit.n_components_))

    X_train_reduction = pca_suit.transform(X_train)
    X_test_reduction = pca_suit.transform(X_test)

    knn_clf_suit = KNeighborsClassifier()
    knn_clf_suit.fit(X_train_reduction, y_train)
    print("After PCA(28)， Suit : {}".format(knn_clf_suit.score(X_test_reduction, y_test)))

    # X = np.empty((100, 2))
    # X[:, 0] = np.random.uniform(0., 100., size=100)
    # X[:, 1] = 0.75 * X[:, 0] + 3. + np.random.normal(0, 10., size=100)
    #
    # pca = PCA(n_components=1)
    # pca.fit(X)
    #
    # print(pca.components_)
    #
    # X_reduction = pca.transform(X)
    # print(X_reduction.shape)
    #
    # X_restore = pca.inverse_transform(X_reduction)
    # print(X_restore.shape)
    #
    # plt.scatter(X[:, 0], X[:, 1], color='b', alpha=0.5)
    # plt.scatter(X_restore[:, 0], X_restore[:, 1], color='r', alpha=0.5)
    # plt.show()
