import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor



"""决策树解决回归问题"""

if __name__ == "__main__":

    boston = datasets.load_boston()
    X = boston.data
    y = boston.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

    dt_reg = DecisionTreeRegressor()
    dt_reg.fit(X_train, y_train)
    # 训练数据集1.0，过拟合
    print("默认参数 = ",dt_reg.score(X_test, y_test))