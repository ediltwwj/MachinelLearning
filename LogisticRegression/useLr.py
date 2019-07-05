import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from KNN.modelSplit import train_test_split
from LogisticRegression.logisticRegression import LogisticRegression

if __name__ == "__main__":

    iris = datasets.load_iris()

    X = iris.data
    y = iris.target

    X = X[y<2, :2]
    y = y[y<2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)

    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    print("Precit score : {}".format(log_reg.score(X_test, y_test)))


