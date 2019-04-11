import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

"""使用scikit-learn提供的数值归一化"""

if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)

    standardScaler = StandardScaler()
    standardScaler.fit(X_train)

    print("特征均值 : {}".format(standardScaler.mean_))  # 各个特征的均值
    print("特征方差 : {}".format(standardScaler.scale_))  # 各个特征标准差

    X_train = standardScaler.transform(X_train)  # 归一化处理
    X_test = standardScaler.transform(X_test)

    knn_clf = KNeighborsClassifier(n_neighbors=3)
    knn_clf.fit(X_train, y_train)

    print("归一化处理后的准确率 : {}".format(knn_clf.score(X_test, y_test)))
