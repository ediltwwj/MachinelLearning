import numpy as np
from sklearn.datasets import fetch_mldata  # 从官方网站下载数据集
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

"""使用MNIST数据集"""

if __name__ == "__main__":

    mnist = fetch_mldata("MNIST original", data_home="D:\dataset") # 自己先下载数据集放在scikit-learn的路径下
    X, y = mnist['data'], mnist['target'] # 28X28维

    X_train = np.array(X[: 60000], dtype=float) # 数据集是int类型
    y_train = np.array(y[:60000], dtype=float)
    X_test = np.array(X[60000 :], dtype=float)
    y_test = np.array(y[60000 :], dtype=float)

    # 不管是拟合还是测试都巨慢无比(几分钟)
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_train)

    print("Score : {}".format(knn_clf.score(X_test, y_test)))

    # PCA降维
    pca = PCA(0.9)
    pca.fit(X_train)
    X_train_reduction = pca.transform(X_train) # 返回降维后的数据（87维度）
    X_test_reduction = pca.transform(X_test)

    # PCA降维之后丢失数据，但是准确率提高
    # 这是因为PCA进行降噪，使得图片特征更加明显
    knn_clf_pca = KNeighborsClassifier()
    knn_clf_pca.fit(X_train_reduction, y_train)
    print("After PCA Score : {}".format(knn_clf_pca.score(X_test_reduction, y_test)))


