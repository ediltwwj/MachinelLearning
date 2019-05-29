import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


def PolynomialRegression(degree):
    """封装数据处理的步骤"""

    return Pipeline([
        ("Poly", PolynomialFeatures(degree=degree)),
        ("std_scaler", StandardScaler()),  # 归一化
        ("lin_reg", LinearRegression())
    ])


def plot_learning_curve(algo, X_train, X_test, y_train, y_test):
    """绘制学习曲线,观察预测结果误差变化,algo是机器学习算法模型"""

    train_score = []
    test_score = []

    for i in range(1, len(X_train) + 1):  # 每次多取一组进行模型训练
        algo.fit(X_train[:i], y_train[:i])

        y_train_predict = algo.predict(X_train[:i])  # 训练数据集测试结果
        train_score.append(mean_squared_error(y_train[:i], y_train_predict))
        y_test_predict = algo.predict(X_test)  # 测试数据集测试结果
        test_score.append(mean_squared_error(y_test, y_test_predict))

    plt.plot([i for i in range(1, len(X_train) + 1)], np.sqrt(train_score), label="train")
    plt.plot([i for i in range(1, len(X_train) + 1)], np.sqrt(test_score), label="test")
    plt.legend()
    plt.axis([0, len(X_train) + 1, 0, 4])
    plt.show()


if __name__ == "__main__":
    np.random.seed(666)
    x = np.random.uniform(-3.0, 3.0, size=100)
    X = x.reshape(-1, 1)
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)

    plt.scatter(x, y)
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

    # 观察曲线稳定值以及两条曲线的间距
    plot_learning_curve(LinearRegression(), X_train, X_test, y_train, y_test)  # 线性回归学习曲线

    poly2_reg = PolynomialRegression(degree=2)
    plot_learning_curve(poly2_reg, X_train, X_test, y_train, y_test)  # 多项式回归学习曲线

    poly20_reg = PolynomialRegression(degree=20)
    plot_learning_curve(poly20_reg, X_train, X_test, y_train, y_test)  # 多项式回归学习曲线

    # train_score = []
    # test_score = []
    #
    # for i in range(1, 76):  # 训练集就75组，每次多取一组进行模型训练
    #     lin_reg = LinearRegression()
    #     lin_reg.fit(X_train[:i], y_train[:i])
    #
    #     y_train_predict = lin_reg.predict(X_train[:i])  # 训练数据集测试结果
    #     train_score.append(mean_squared_error(y_train[:i], y_train_predict))
    #
    #     y_test_predict = lin_reg.predict(X_test)  # 测试数据集测试结果
    #     test_score.append(mean_squared_error(y_test, y_test_predict))
    #
    # # 绘制学习曲线,误差变化
    # plt.plot([i for i in range(1, 76)], np.sqrt(train_score), label="train")
    # plt.plot([i for i in range(1, 76)], np.sqrt(test_score), label="test")
    # plt.legend()
    # plt.show()
