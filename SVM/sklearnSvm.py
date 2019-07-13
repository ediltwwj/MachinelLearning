import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC # 使用SVM进行分类任务
from LogisticRegression.drawDecisionBoundary import plot_decision_boundary

"""线性数据"""


def plot_svc_decision_boundary(model, axis):

    # meshgrid函数用两个坐标轴上的点在平面上画格，返回坐标矩阵
    X0, X1 = np.meshgrid(
        # 随机两组数，起始值和密度由坐标轴的起始值决定
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1),
    )
    # ravel()方法将高维数组降为一维数组，c_[]将两个数组以列的形式拼接起来，形成矩阵
    X_new = np.c_[X0.ravel(), X1.ravel()]

    # 通过训练好的逻辑回归模型，预测平面上这些点的分类
    y_predict = model.predict(X_new)
    zz = y_predict.reshape(X0.shape)

    # 设置色彩表
    from matplotlib.colors import ListedColormap
    my_colormap = ListedColormap(['#0000CD', '#40E0D0', '#FFFF00'])

    # 绘制等高线，并且填充等高区域的颜色
    plt.contourf(X0, X1, zz, linewidth=5, cmap=my_colormap)

    w = model.coef_[0]
    b = model.intercept_[0]

    # 决策边界直线方程
    # w0 * x0 + w1 * x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    plot_x = np.linspace(axis[0], axis[1], 200)
    up_y = -w[0]/w[1] * plot_x - b/w[1] + 1/w[1]
    down_y = -w[0]/w[1] * plot_x - b/w[1] - 1/w[1]

    up_index = (up_y >= axis[2]) & (up_y <= axis[3])
    down_index = (down_y >= axis[2]) & (down_y <= axis[3])

    plt.plot(plot_x[up_index], up_y[up_index], color='black')
    plt.plot(plot_x[down_index], down_y[down_index], color='black')



if __name__ == "__main__":

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X = X[y<2, :2]
    y = y[y<2]

    plt.scatter(X[y==0, 0], X[y==0, 1], color='red')
    plt.scatter(X[y==1, 0], X[y==1, 1], color='blue')
    plt.show()

    # 数据归一化处理
    standardScaler = StandardScaler()
    standardScaler.fit(X)
    X_standardScaler = standardScaler.transform(X)

    # C越大容错能力越差，越是硬间隔支撑向量机
    svc = LinearSVC(C=1e9)
    svc.fit(X_standardScaler, y)

    # 绘制决策边界
    plot_decision_boundary(svc, axis=[-3, 3, -3, 3])
    plt.scatter(X_standardScaler[y==0, 0], X_standardScaler[y==0, 1])
    plt.scatter(X_standardScaler[y==1, 0], X_standardScaler[y==1, 1])
    plt.show()

    # 绘制决策边界和margin线
    plot_svc_decision_boundary(svc, axis=[-3, 3, -3, 3])
    plt.scatter(X_standardScaler[y==0, 0], X_standardScaler[y==0, 1])
    plt.scatter(X_standardScaler[y==1, 0], X_standardScaler[y==1, 1])
    plt.show()

    # svc2 = LinearSVC(C=0.01)
    # svc2.fit(X_standardScaler, y)
    #
    # plot_decision_boundary(svc2, axis=[-3, 3, -3, 3])
    # plt.scatter(X_standardScaler[y==0, 0], X_standardScaler[y==0, 1])
    # plt.scatter(X_standardScaler[y==1, 0], X_standardScaler[y==1, 1])
    # plt.show()