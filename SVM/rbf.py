import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from LogisticRegression.drawDecisionBoundary import plot_decision_boundary

def RBFKernelSVC(gamma=1.0):
    return Pipeline([
        ("std_scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", gamma=gamma))
    ])


"""gamma调整的是拟合的复杂程度"""
if __name__ == "__main__":
    X, y = datasets.make_moons(noise=0.15, random_state=666)

    plt.scatter(X[y==0, 0], X[y==0, 1])
    plt.scatter(X[y==1, 0], X[y==1, 1])
    plt.show()

    svc = RBFKernelSVC(gamma=1.0)
    svc.fit(X, y)

    plot_decision_boundary(svc, axis=[-1.5, 2.5, -1.0, 1.5])
    plt.scatter(X[y==0, 0], X[y==0, 1])
    plt.scatter(X[y==1, 0], X[y==1, 1])
    plt.show()


    # gamma越大高斯分布越窄，分布越集中，产生过拟合
    svc_gamma100 = RBFKernelSVC(gamma=100)
    svc_gamma100.fit(X, y)

    plot_decision_boundary(svc_gamma100, axis=[-1.5, 2.5, -1.0, 1.5])
    plt.scatter(X[y==0, 0], X[y==0, 1])
    plt.scatter(X[y==1, 0], X[y==1, 1])
    plt.show()
