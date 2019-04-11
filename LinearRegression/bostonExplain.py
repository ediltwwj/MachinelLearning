import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression

"""利用线性回归算法对波士顿放假影响因素的程度进行解释"""

if __name__ == "__main__":
    boston = datasets.load_boston()

    X = boston.data
    y = boston.target

    X = X[y < 50.0]
    y = y[y < 50.0]

    lin_reg = LinearRegression()
    lin_reg.fit(X, y)

    print("打印特征系数 ： ")
    print(lin_reg.coef_)  # 正负分别代表正负相关，绝对值反映影响程度

    print("排列特征系数 : ")
    print(boston.feature_names[np.argsort(lin_reg.coef_)])  # 参照数据集具体信息

    print("打印数据集信息 : ")
    print(boston.DESCR)
