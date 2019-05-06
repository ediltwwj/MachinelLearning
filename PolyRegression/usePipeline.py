import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    x = np.random.uniform(-3, 3, size=100)
    X = x.reshape(-1, 1)

    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)

    # 这里将三个处理步骤进行了封装，将数据传入poly_reg之后，将会智能地沿着该管道进行处理
    # (对象名，实例)
    poly_reg = Pipeline([
        ("Poly", PolynomialFeatures(degree=2)),
        ("std_scaler", StandardScaler()),  # 归一化
        ("lin_reg", LinearRegression())
    ])

    poly_reg.fit(X, y)
    y_predict = poly_reg.predict(X)

    plt.scatter(x, y)
    plt.plot(np.sort(x), y_predict[np.argsort(x)],
             color='r')
    plt.show()



