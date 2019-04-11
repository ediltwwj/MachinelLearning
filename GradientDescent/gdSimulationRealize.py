import numpy as np
import matplotlib.pyplot as plt

"""梯度下降法模拟实现"""


def dJ(theta):
    """对损失函数J在点theta进行求导"""

    return 2 * (theta - 2.5)


def J(theta):
    """计算点theta在损失函数J下的取值"""

    return (theta - 2.5) ** 2 - 1


def gradient_descent(initial_theta, eta, n_iters=1e4, epsilon=1e-8):
    """寻找损失函数的最小值点,eta是学习率，epsilon是下降幅度的界限，n_iters循环限制次数"""

    theta = initial_theta  # 起始点
    theta_history.append(initial_theta)
    i_iter = 0  # 当前循环次数

    while i_iter < n_iters:
        gradient = dJ(theta)  # 记录随点变化对应的导数值变化
        last_theta = theta
        theta = theta - eta * gradient  # 移动点
        theta_history.append(theta)  # 记录theta值的变化

        if (abs(J(theta) - J(last_theta)) < epsilon):
            break

        i_iter += 1

    return gradient, theta, i_iter


def plot_theta_history():
    """显示点再函数曲线上的移动情况"""

    plt.plot(plot_x, J(plot_x))
    plt.plot(np.array(theta_history), J(np.array(theta_history)), color='r', marker='+')
    plt.show()


if __name__ == "__main__":
    plot_x = np.linspace(-1, 6, 141)  # 构建损失函数
    plot_y = (plot_x - 2.5) ** 2 - 1

    theta = 0.00
    eta = 0.01

    theta_history = []

    gradient_value, theta_value, iter_value = gradient_descent(theta, eta)
    plot_theta_history()

    print("最小值的点导数是 ： {}, 坐标为 ： {}, 循环次数 : {}".format(gradient_value, theta_value, iter_value))
