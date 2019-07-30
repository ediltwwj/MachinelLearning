import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_boundary(model, axis):
    """绘制不规则决策边界"""

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
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])

    # 绘制等高线，并且填充等高区域的颜色
    plt.contourf(X0, X1, zz, linewidth=5, cmap=custom_cmap)



