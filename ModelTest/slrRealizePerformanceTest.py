import numpy as np
import matplotlib.pyplot as plt
from time import clock
from ModelTest.metrics import *
from LinearRegression.simpleLinearRegression import SimpleLinearRegression1
from LinearRegression.simpleLinearRegressionS import SimpleLinearRegression2

"""对自定义的简单线性回归模型进行测试"""

if __name__ == "__main__":
    # 模型预测结果测试
    x = np.array([1., 2., 3., 4., 5.])
    y = np.array([1., 3., 2., 3., 5.])

    reg1 = SimpleLinearRegression1()
    reg1.fit(x, y)

    x_predict = np.array([2., 3., 2.5])
    y_predict = reg1.predict(x_predict)

    print("a is : {}, b is : {}".format(reg1.a_, reg1.b_))
    print("简单线性回归预测结果 : {}".format(y_predict))

    y_hat1 = reg1.predict(x)
    plt.scatter(x, y)
    plt.plot(x, y_hat1, color='r')
    plt.show()

    # 算法性能对比测试
    size = 1000000
    big_x = np.random.random(size=size)
    big_y = big_x * 2.0 + 3.0 + np.random.normal()

    reg2 = SimpleLinearRegression2()

    time_begin = clock()
    reg1.fit(big_x, big_y)
    time_end = clock()

    print("循环实现线性回归模型的耗时 : {}".format((time_end - time_begin)))  # 循环实现

    time_begin = clock()
    reg2.fit(big_x, big_y)
    time_end = clock()

    print("向量化实现线性回归模型的耗时 : {}".format((time_end - time_begin)))  # 向量化实现


