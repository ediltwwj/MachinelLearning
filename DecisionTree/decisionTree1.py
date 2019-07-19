import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from LogisticRegression.drawDecisionBoundary import plot_decision_boundary

if __name__ == "__main__":

    iris = datasets.load_iris()
    X = iris.data[:, 2: ]
    y = iris.target

    plt.scatter(X[y==0, 0], X[y==0, 1])
    plt.scatter(X[y==1, 0], X[y==1, 1])
    plt.scatter(X[y==2, 0], X[y==2, 1])
    plt.show()

    dt_clf = DecisionTreeClassifier(max_depth=2, criterion="entropy")
    dt_clf.fit(X, y)

    plot_decision_boundary(dt_clf, axis=[0.5, 7.5, 0, 3])
    plt.scatter(X[y==0, 0], X[y==0, 1])
    plt.scatter(X[y==1, 0], X[y==1, 1])
    plt.scatter(X[y==2, 0], X[y==2, 1])
    plt.show()


