import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from KNN.modelSplit import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from LogisticRegression.drawDecisionBoundary import plot_decision_boundary

if __name__ == "__main__":

    iris = datasets.load_iris()

    X = iris.data
    y = iris.target

    X = X[y<2, :2]
    y = y[y<2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)

    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_train)

    print("Precit score : {}".format(knn_clf.score(X_test, y_test)))

    # 绘制决策边界,两个类别
    plot_decision_boundary(knn_clf, axis=[4, 7.5, 1.5, 4.5])
    plt.scatter(X[y == 0, 0], X[y == 0, 1])
    plt.scatter(X[y == 1, 0], X[y == 1, 1])

    plt.show()

    # 绘制决策边界，全类别
    knn_clf_all = KNeighborsClassifier(n_neighbors=50)
    knn_clf_all.fit(iris.data[:, :2], iris.target)

    plot_decision_boundary(knn_clf_all, axis=[4, 8, 1.5, 4.5])
    plt.scatter(iris.data[iris.target==0, 0], iris.data[iris.target==0, 1])
    plt.scatter(iris.data[iris.target==1, 0], iris.data[iris.target==1, 1])
    plt.scatter(iris.data[iris.target==2, 0], iris.data[iris.target==2, 1])

    plt.show()

