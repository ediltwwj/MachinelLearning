import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def StandardLinearSVR(epsilon=0.1):
    return Pipeline([
        ("std_scaler", StandardScaler()),
        ("linearSVR", LinearSVR(epsilon=epsilon))
    ])


"""SVM思想解决回归问题"""

if __name__ == "__main__":

    boston = datasets.load_boston()
    X = boston.data
    y = boston.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

    svr = StandardLinearSVR()
    svr.fit(X_train, y_train)
    print("linearSVR'score = {}".format(svr.score(X_test, y_test)))