import numpy as np
import matplotlib.pyplot as plt
from LogisticRegression.logisticRegression import LogisticRegression
from LogisticRegression.drawDecisionBoundary import plot_decision_boundary
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


def PolynomialLogisticRegression(degree):
    """为逻辑回归添加多项式特征"""

    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('log_reg', LogisticRegression())
    ])


if __name__ == "__main__":

    np.random.seed(666)
    X = np.random.normal(0, 1, size=(200, 2))
    y = np.array(X[:, 0] + X[:, 1] ** 2 < 1.5, dtype='int')

    # plt.scatter(X[y==0, 0], X[y==0, 1])
    # plt.scatter(X[y==1, 0], X[y==1, 1])
    # plt.show()

    log_reg = LogisticRegression()
    log_reg.fit(X, y)

    print("Score Value : {}".format(log_reg.score(X, y)))

    # 预测准确度很糟糕
    plot_decision_boundary(log_reg, axis=[-4, 4, -4, 4])

    plt.scatter(X[y==0, 0], X[y==0, 1])
    plt.scatter(X[y==1, 0], X[y==1, 1])
    plt.show()


    ploy_log_reg = PolynomialLogisticRegression(degree=2)
    ploy_log_reg.fit(X, y)

    print("Score Value : {}".format(ploy_log_reg.score(X, y)))

    # 预测准确度还不错， degree取太大会产生过拟合，使得边界不明确
    plot_decision_boundary(ploy_log_reg, axis=[-4, 4, -4, 4])

    plt.scatter(X[y==0, 0], X[y==0, 1])
    plt.scatter(X[y==1, 0], X[y==1, 1])
    plt.show()