import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from KNN.modelSplit import train_test_split
from LogisticRegression.logisticRegression import LogisticRegression
from LogisticRegression.drawDecisionBoundary import plot_decision_boundary


def x2(x1):
    """以p=0.5为边界推导决策边界函数"""

    return (-log_reg.coef_[0] * x1 - log_reg.interception_) / log_reg.coef_[1]


if __name__ == "__main__":

    iris = datasets.load_iris()

    X = iris.data
    y = iris.target

    X = X[y<2, :2]
    y = y[y<2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)

    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    print("Precit score : {}".format(log_reg.score(X_test, y_test)))

    # 绘制决策边界,有一个样本分类错误
    x1_plot = np.linspace(4, 8, 1000)
    x2_plot = x2(x1_plot)

    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue')
    plt.plot(x1_plot, x2_plot)

    plt.show()

    # 绘制不规则决策边界
    plot_decision_boundary(log_reg, axis=[4, 7.5, 1.5, 4.5])
    plt.scatter(X[y == 0, 0], X[y == 0, 1])
    plt.scatter(X[y == 1, 0], X[y == 1, 1])

    plt.show()



