import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from LogisticRegression.drawDecisionBoundary import plot_decision_boundary
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


def PolynomialLogisticRegression(degree, C=1):
    """为逻辑回归添加多项式特征, C正比控制正则化强弱"""

    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('log_reg', LogisticRegression(C=C))
    ])


if __name__ == "__main__":

    np.random.seed(666)
    X = np.random.normal(0, 1, size=(200, 2))
    y = np.array(X[:, 0] + X[:, 1] ** 2 < 1.5, dtype='int')

    # 随机在样本当中选取20个点， 使其分类结果为1
    for _ in range(20):
        y[np.random.randint(200)] = 1

    plt.scatter(X[y==0, 0], X[y==0, 1])
    plt.scatter(X[y==1, 0], X[y==1, 1])
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

    # 使用scikit-learn当中的逻辑回归
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    print("Train data score Value : {}".format(log_reg.score(X_train, y_train)))
    print("Test data score Value : {}".format(log_reg.score(X_test, y_test)))

    # 绘制决策边界
    plot_decision_boundary(log_reg, axis=[-4, 4, -4, 4])
    plt.scatter(X[y==0, 0], X[y==0, 1])
    plt.scatter(X[y==1, 0], X[y==1, 1])
    plt.show()

    # 使用多项式逻辑回归
    poly_log_reg = PolynomialLogisticRegression(degree=2)
    poly_log_reg.fit(X_train, y_train)

    print()
    print("Train data score Value : {}".format(poly_log_reg.score(X_train, y_train)))
    print("Test data score Value : {}".format(poly_log_reg.score(X_test, y_test)))

    # 绘制决策边界
    plot_decision_boundary(poly_log_reg, axis=[-4, 4, -4, 4])
    plt.scatter(X[y==0, 0], X[y==0, 1])
    plt.scatter(X[y==1, 0], X[y==1, 1])
    plt.show()


    poly_log_reg2 = PolynomialLogisticRegression(degree=20, C=0.1)
    poly_log_reg2.fit(X_train, y_train)

    print()
    print("Train data score Value : {}".format(poly_log_reg2.score(X_train, y_train)))
    print("Test data score Value : {}".format(poly_log_reg2.score(X_test, y_test)))

    plot_decision_boundary(poly_log_reg2, axis=[-4, 4, -4, 4])
    plt.scatter(X[y==0, 0], X[y==0, 1])
    plt.scatter(X[y==1, 0], X[y==1, 1])
    plt.show()