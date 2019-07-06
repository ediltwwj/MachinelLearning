import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from LogisticRegression.drawDecisionBoundary import plot_decision_boundary
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier

if __name__ == "__main__":

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

    # 默认OVR
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    print("Score Value : {}".format(log_reg.score(X_test, y_test)))

    plt.show()

    # 使用OVO
    log_reg_ovo = LogisticRegression(multi_class='multinomial', solver="newton-cg") # 使用ovo必须添加solver参数
    log_reg_ovo.fit(X_train, y_train)
    print("Score Value : {}".format(log_reg_ovo.score(X_test, y_test)))

    print()

    # 使用OVR
    ovr = OneVsRestClassifier(log_reg)
    ovr.fit(X_train, y_train)

    print("Score Value : {}".format(ovr.score(X_test, y_test)))

    # 使用OVO
    ovo = OneVsOneClassifier(log_reg)
    ovo.fit(X_train, y_train)

    print("Score Value : {}".format(ovo.score(X_test, y_test)))