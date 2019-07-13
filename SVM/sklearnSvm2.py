import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from LogisticRegression.drawDecisionBoundary import plot_decision_boundary
from sklearn.svm import SVC


def PolynomiaSVC(degree, C=1.0):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('linearSVC', LinearSVC(C=C))
    ])

def PolynomialKernelSVC(degree, C=1.0):
    return Pipeline([
        ('std_scaler', StandardScaler()),
        ('kernelSVC', SVC(kernel='poly', degree=degree, C=C))
    ])

if __name__ == "__main__":

    # 使用skleran生成数据
    X, y = datasets.make_moons(noise=0.15, random_state=666)

    # 绘制数据集图像
    plt.scatter(X[y==0, 0], X[y==0, 1])
    plt.scatter(X[y==1, 0], X[y==1, 1])
    plt.show()

    # 使用多项式特征的SVM
    poly_svc = PolynomiaSVC(degree=3)
    poly_svc.fit(X, y)

    # 绘制决策边界
    plot_decision_boundary(poly_svc, axis=[-1.5, 2.5, -1.0, 1.5])
    plt.scatter(X[y==0, 0], X[y==0, 1])
    plt.scatter(X[y==1, 0], X[y==1, 1])
    plt.show()

    # 使用多项式核函数的SVM
    poly_kernel_svc = PolynomiaSVC(degree=3)
    poly_kernel_svc.fit(X, y)

    # 绘制决策边界
    plot_decision_boundary(poly_kernel_svc, axis=[-1.5, 2.5, -1.0, 1.5])
    plt.scatter(X[y==0, 0], X[y==0, 1])
    plt.scatter(X[y==1, 0], X[y==1, 1])
    plt.show()