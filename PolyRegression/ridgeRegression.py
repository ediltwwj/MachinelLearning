import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


def RidgeRegression(degree, alpha):
    """岭回归过程封装"""

    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("std_scaler", StandardScaler()),
        ("ridge_reg", Ridge(alpha=alpha))
    ])


if __name__ == "__main__":
    np.random.seed(42)
    x = np.random.uniform(-3.0, 3.0, size=100)
    X = x.reshape(-1, 1)
    y = 0.5 * x + 3 + np.random.normal(0, 1, size=100)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    ridge = Ridge(alpha=1)

    ridge1_reg = RidgeRegression(20, 0.001)
    ridge1_reg.fit(X_train, y_train)
    y1_predict = ridge1_reg.predict(X_test)
    print("MSE VALUE : {}".format(mean_squared_error(y_test, y1_predict)))
